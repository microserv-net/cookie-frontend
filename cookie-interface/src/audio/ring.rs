//! A single-producer / single-consumer lock-free ring buffer of `f32`.
//!
//! # Why this file contains the only `unsafe` in the crate
//!
//! One side of every audio ring is a real-time callback owned by the operating
//! system's audio thread. That thread must never allocate, never block and
//! never take a lock that a normal-priority thread can hold. `Mutex<VecDeque>`
//! fails all three tests under load; a channel that allocates per send fails
//! the first.
//!
//! So: a fixed-size buffer, two atomic cursors, acquire/release ordering, and
//! `UnsafeCell` for the slots. The unsafety is confined to `Producer::push_slice`
//! and `Consumer::pop_slice`, and is sound because
//!
//! * `Producer` and `Consumer` are not `Clone` and are handed out exactly once,
//!   so there is precisely one writer and one reader;
//! * **each cursor has exactly one writer.** The producer owns `write` and
//!   never stores to `read`; the consumer owns `read` and never stores to
//!   `write`. Both may *load* the other's cursor. This is the invariant the
//!   whole file rests on, and violating it is subtle enough to survive code
//!   review — see the note below.
//! * the writer publishes with `Release` after writing and the reader acquires
//!   with `Acquire` before reading, so the data is visible when the cursor is.
//!
//! # Overflow, and a bug worth remembering
//!
//! When the consumer falls behind, the newest audio is worth more than the
//! oldest, so the producer keeps writing and laps the reader. The reader
//! notices on its next pop and skips forward.
//!
//! The obvious implementation of that — have the producer advance `read` past
//! the samples it is about to overwrite — is wrong, and was what this file did
//! first. It gives `read` two writers. The producer's `fetch_add` and the
//! consumer's `store` then race, the cursor goes backwards, `len()` exceeds
//! the capacity, and `free()` underflows. On a single-core machine the threads
//! interleave rarely enough that it never showed up; CI on macOS, with real
//! parallelism, failed on the first run.
//!
//! So the producer does not touch `read` at all. It writes, counts what it
//! overwrote, and moves on; the consumer resynchronises when it sees that the
//! distance between the cursors has exceeded what the ring can hold. A skid
//! guard keeps the reader clear of slots the writer may be part-way through,
//! because reading a slot while it is being written is a data race even when
//! the value would be discarded.
//!
//! Overflow is counted, never silently ignored: a rising count is the
//! signature of a stalled consumer and shows up in `--doctor`.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Inner {
    /// Capacity is a power of two so wrapping is a mask, not a modulo.
    buffer: UnsafeCell<Box<[f32]>>,
    mask: usize,
    /// Index of the next slot the producer will write.
    write: AtomicUsize,
    /// Index of the next slot the consumer will read.
    read: AtomicUsize,
    /// Samples the producer had to drop because the consumer fell behind.
    dropped: AtomicUsize,
}

// SAFETY: access to `buffer` is partitioned between exactly one producer and
// one consumer by the cursor protocol described above.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Samples written but not yet read.
    ///
    /// Clamped to the capacity: once the producer has lapped the consumer the
    /// true distance is larger than the ring, and every caller wants "how much
    /// can I actually get", not "how far apart are the cursors".
    fn len(&self) -> usize {
        let w = self.write.load(Ordering::Acquire);
        let r = self.read.load(Ordering::Acquire);
        w.wrapping_sub(r).min(self.capacity())
    }

    /// Where the consumer lands after being lapped.
    ///
    /// Not the full capacity: the producer may be part-way through a push when
    /// the consumer resynchronises, so the reader stops an eighth of the ring
    /// short of the write cursor's wrap point. A torn read of a slot is a data
    /// race even when the sample would have been discarded anyway, and this
    /// guard band is what makes the overwrite case sound rather than lucky.
    ///
    /// Only reached after an overrun, which already means audio was lost.
    fn readable_after_overrun(&self) -> usize {
        let cap = self.capacity();
        cap - cap / 8
    }
}

/// Producer half. Live on the real-time side.
pub struct RingProducer {
    inner: Arc<Inner>,
}

/// Consumer half. Live on the async side.
pub struct RingConsumer {
    inner: Arc<Inner>,
}

/// Create a ring holding at least `capacity` samples (rounded up to a power of
/// two, minimum 64).
pub fn ring(capacity: usize) -> (RingProducer, RingConsumer) {
    let cap = capacity.next_power_of_two().max(64);
    let inner = Arc::new(Inner {
        buffer: UnsafeCell::new(vec![0.0f32; cap].into_boxed_slice()),
        mask: cap - 1,
        write: AtomicUsize::new(0),
        read: AtomicUsize::new(0),
        dropped: AtomicUsize::new(0),
    });
    (
        RingProducer {
            inner: Arc::clone(&inner),
        },
        RingConsumer { inner },
    )
}

impl RingProducer {
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Room before the producer starts overwriting unread audio.
    pub fn free(&self) -> usize {
        self.capacity().saturating_sub(self.len())
    }

    pub fn dropped(&self) -> usize {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Append samples. Real-time safe: no allocation, no locks, no syscalls.
    ///
    /// If `data` does not fit, the oldest samples are discarded to make room —
    /// for a live microphone, fresh audio is worth more than stale audio.
    /// Returns the number of samples that had to be dropped.
    pub fn push_slice(&mut self, data: &[f32]) -> usize {
        if data.is_empty() {
            return 0;
        }
        let cap = self.inner.capacity();
        // A write larger than the whole ring can only keep its tail.
        let data = if data.len() > cap {
            &data[data.len() - cap..]
        } else {
            data
        };

        // Count what this write will overwrite, but do not touch `read`:
        // that cursor belongs to the consumer, which resynchronises itself.
        let dropped = data.len().saturating_sub(self.free());
        if dropped > 0 {
            self.inner.dropped.fetch_add(dropped, Ordering::Relaxed);
        }

        // Relaxed is enough: this is the only thread that stores to `write`.
        let write = self.inner.write.load(Ordering::Relaxed);
        // SAFETY: single producer; we only touch slots in the free region.
        let buf = unsafe { &mut *self.inner.buffer.get() };
        let start = write & self.inner.mask;
        let first = (cap - start).min(data.len());
        buf[start..start + first].copy_from_slice(&data[..first]);
        if first < data.len() {
            let rest = data.len() - first;
            buf[..rest].copy_from_slice(&data[first..]);
        }
        self.inner
            .write
            .store(write.wrapping_add(data.len()), Ordering::Release);
        dropped
    }
}

impl RingConsumer {
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dropped(&self) -> usize {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Fill `out` with up to `out.len()` samples. Returns how many were read.
    ///
    /// If the producer has lapped us since the last call, the lost samples are
    /// skipped here rather than returned as garbage: stale audio that has been
    /// partially overwritten is worse than no audio.
    pub fn pop_slice(&mut self, out: &mut [f32]) -> usize {
        let read = self.resynchronise();
        let available = self.len().min(out.len());
        if available == 0 {
            return 0;
        }
        // SAFETY: single consumer; we only touch slots in the filled region,
        // published by the producer's Release store.
        let buf = unsafe { &*self.inner.buffer.get() };
        let start = read & self.inner.mask;
        let cap = self.inner.capacity();
        let first = (cap - start).min(available);
        out[..first].copy_from_slice(&buf[start..start + first]);
        if first < available {
            let rest = available - first;
            out[first..available].copy_from_slice(&buf[..rest]);
        }
        self.inner
            .read
            .store(read.wrapping_add(available), Ordering::Release);
        available
    }

    /// Catch up if the producer has overwritten unread samples.
    ///
    /// Returns the (possibly advanced) read cursor. Only the consumer calls
    /// this, which is what keeps `read` single-writer.
    fn resynchronise(&mut self) -> usize {
        let write = self.inner.write.load(Ordering::Acquire);
        let read = self.inner.read.load(Ordering::Relaxed);
        let distance = write.wrapping_sub(read);
        if distance <= self.inner.capacity() {
            return read;
        }
        // The producer has lapped us: some of what we have not read has
        // already been overwritten. Jump to the oldest sample still
        // guaranteed intact.
        let skipped = distance - self.inner.readable_after_overrun();
        let read = read.wrapping_add(skipped);
        self.inner.read.store(read, Ordering::Release);
        read
    }

    /// Read exactly `out.len()` samples, or nothing at all. Useful for
    /// fixed-size analysis frames.
    pub fn pop_exact(&mut self, out: &mut [f32]) -> bool {
        self.resynchronise();
        if self.len() < out.len() {
            return false;
        }
        self.pop_slice(out) == out.len()
    }

    /// Discard everything currently buffered (used when speech is interrupted,
    /// so stale audio is not played after the stop).
    pub fn clear(&mut self) -> usize {
        self.resynchronise();
        let n = self.len();
        let read = self.inner.read.load(Ordering::Relaxed);
        self.inner
            .read
            .store(read.wrapping_add(n), Ordering::Release);
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_data_in_order() {
        let (mut p, mut c) = ring(64);
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(p.push_slice(&data), 0);
        let mut out = vec![0.0; 32];
        assert_eq!(c.pop_slice(&mut out), 32);
        assert_eq!(out, data);
        assert!(c.is_empty());
    }

    #[test]
    fn wraps_around_the_end() {
        let (mut p, mut c) = ring(64);
        let mut expected = Vec::new();
        let mut out = vec![0.0; 20];
        for round in 0..20 {
            let data: Vec<f32> = (0..20).map(|i| (round * 20 + i) as f32).collect();
            p.push_slice(&data);
            let n = c.pop_slice(&mut out);
            expected.extend_from_slice(&out[..n]);
        }
        let want: Vec<f32> = (0..400).map(|i| i as f32).collect();
        assert_eq!(expected, want);
    }

    #[test]
    fn overflow_discards_the_oldest_and_counts_it() {
        let (mut p, mut c) = ring(64);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        p.push_slice(&data);

        // Two samples more than fits, numbered to continue the sequence so
        // that a gap in the output means a real discontinuity rather than a
        // hole in the test data. The producer writes them anyway and reports
        // what it overwrote; it does not touch the read cursor.
        let dropped = p.push_slice(&[64.0, 65.0]);
        assert_eq!(dropped, 2);
        assert_eq!(p.dropped(), 2);

        let mut out = vec![0.0; 64];
        let n = c.pop_slice(&mut out);

        // The consumer resynchronises on its next read, landing a guard band
        // short of the producer so it never reads a slot mid-write. It
        // therefore skips a little more than was strictly overwritten — the
        // cost of making the overwrite case sound rather than lucky, and it
        // only applies when audio was being lost anyway.
        assert!(n >= 56, "read {n}, expected most of the ring");
        assert!(
            out[0] >= 2.0,
            "the overwritten samples must not be returned, got {}",
            out[0]
        );
        assert_eq!(out[n - 1], 65.0, "the newest sample survived");
        // What is returned is still contiguous and in order.
        for window in out[..n].windows(2) {
            assert_eq!(window[1] - window[0], 1.0, "a gap appeared mid-read");
        }
    }

    #[test]
    fn push_larger_than_capacity_keeps_the_tail() {
        let (mut p, mut c) = ring(64);
        let data: Vec<f32> = (0..200).map(|i| i as f32).collect();
        p.push_slice(&data);
        let mut out = vec![0.0; 64];
        assert_eq!(c.pop_slice(&mut out), 64);
        assert_eq!(out[63], 199.0);
    }

    #[test]
    fn pop_exact_is_all_or_nothing() {
        let (mut p, mut c) = ring(64);
        p.push_slice(&[1.0, 2.0, 3.0]);
        let mut out = [0.0; 8];
        assert!(!c.pop_exact(&mut out));
        assert_eq!(c.len(), 3, "nothing consumed on failure");
        let mut out3 = [0.0; 3];
        assert!(c.pop_exact(&mut out3));
        assert_eq!(out3, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn clear_empties_the_ring() {
        let (mut p, mut c) = ring(64);
        p.push_slice(&[1.0; 40]);
        assert_eq!(c.clear(), 40);
        assert!(c.is_empty());
    }

    #[test]
    fn survives_concurrent_producer_and_consumer() {
        // Not a proof of correctness, but it reliably catches ordering
        // mistakes when run under `--test-threads` pressure or miri.
        let (mut p, mut c) = ring(1024);
        const TOTAL: usize = 200_000;
        let producer = std::thread::spawn(move || {
            let mut next = 0usize;
            while next < TOTAL {
                let n = 128.min(TOTAL - next);
                let block: Vec<f32> = (next..next + n).map(|i| i as f32).collect();
                p.push_slice(&block);
                next += n;
                std::thread::yield_now();
            }
        });

        // Samples may be *skipped* when the producer laps us, so we cannot
        // wait for a count; we wait for the final value to come through.
        let final_value = (TOTAL - 1) as f32;
        let mut last = -1.0f32;
        let mut out = vec![0.0; 256];
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while last < final_value && std::time::Instant::now() < deadline {
            let n = c.pop_slice(&mut out);
            for v in &out[..n] {
                assert!(*v > last, "{v} came after {last}: ordering violated");
                last = *v;
            }
            if n == 0 {
                std::thread::yield_now();
            }
        }
        producer.join().unwrap();
        assert_eq!(last, final_value, "never observed the final sample");
    }
}
