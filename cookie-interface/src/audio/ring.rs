//! A single-producer / single-consumer lock-free ring buffer of `f32`.
//!
//! # Why this file exists
//!
//! One side of every audio ring is a real-time callback owned by the operating
//! system's audio thread. That thread must never allocate, never block and
//! never take a lock that a normal-priority thread can hold. `Mutex<VecDeque>`
//! fails all three tests under load; a channel that allocates per send fails
//! the first.
//!
//! So: a fixed-size buffer, two atomic cursors, and acquire/release ordering.
//!
//! # Why there is no `unsafe` here
//!
//! There was, and the history is worth keeping, because both versions looked
//! obviously correct and both were wrong.
//!
//! The first version used `UnsafeCell` slots and had the producer advance the
//! *read* cursor past samples it was about to overwrite. That gives `read` two
//! writers: the producer's `fetch_add` races the consumer's `store`, the
//! cursor moves backwards, and `free()` underflows. CI on macOS failed on its
//! first run; a single-core machine never reproduced it.
//!
//! The second version gave each cursor exactly one writer — correct, and still
//! not enough. When the producer is allowed to overwrite, it can write into
//! the very slots the consumer is *part-way through copying*, so the consumer
//! returns a mixture of old and new samples. That is an ordering violation in
//! the output and, in Rust's memory model, a data race regardless of whether
//! the value is later discarded.
//!
//! Both problems come from the same source: an overwriting producer and a
//! reading consumer genuinely can touch the same slot at the same time. So the
//! slots are `AtomicU32` holding `f32` bit patterns, and the consumer
//! validates after copying that the region it read was not overwritten while
//! it was reading; if it was, the data is discarded and re-read. Relaxed
//! atomic loads and stores of `u32` compile to ordinary loads and stores on
//! every architecture this targets, so the cost is nil and the `unsafe` block
//! is gone.
//!
//! # The invariants
//!
//! * `Producer` and `Consumer` are not `Clone` and are handed out exactly
//!   once, so there is one writer and one reader.
//! * **Each cursor has exactly one writer.** The producer owns `write`, the
//!   consumer owns `read`. Both may load the other's.
//! * The producer publishes with `Release` after filling slots; the consumer
//!   acquires before reading, so data is visible when the cursor is.
//! * The consumer never returns samples it cannot prove were intact for the
//!   whole copy.
//!
//! Overflow drops the oldest audio — fresh audio is worth more than stale on a
//! live microphone — and is counted, never silently ignored. A rising count is
//! the signature of a stalled consumer and shows up in `--doctor`.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

/// How many times the consumer will re-read a region the producer overwrote
/// mid-copy before giving up for this call.
///
/// Retrying forever would let a fast producer starve the consumer entirely.
/// Giving up simply means returning nothing this time; the next call
/// resynchronises and makes progress.
const MAX_REREADS: usize = 4;

struct Inner {
    /// `f32` bit patterns. Atomic so that a producer overwriting a slot while
    /// the consumer reads it is defined behaviour rather than a data race.
    buffer: Box<[AtomicU32]>,
    /// Capacity is a power of two so wrapping is a mask, not a modulo.
    mask: usize,
    /// Next position the producer will write. Producer-owned.
    write: AtomicUsize,
    /// Next position the consumer will read. Consumer-owned.
    read: AtomicUsize,
    /// Samples overwritten before the consumer could read them.
    dropped: AtomicUsize,
}

impl Inner {
    fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Samples written but not yet read, clamped to the capacity.
    ///
    /// Once the producer has lapped the consumer the true distance exceeds the
    /// ring, and every caller wants "how much can I actually get".
    fn len(&self) -> usize {
        let w = self.write.load(Ordering::Acquire);
        let r = self.read.load(Ordering::Acquire);
        w.wrapping_sub(r).min(self.capacity())
    }

    /// Where the consumer lands after being lapped.
    ///
    /// Not the full capacity: landing exactly a ring behind the producer means
    /// the very next push collides again, and the consumer would spend its
    /// time re-reading. Dropping an extra eighth buys a margin, and only
    /// applies after an overrun, when audio was being lost anyway.
    fn readable_after_overrun(&self) -> usize {
        let cap = self.capacity();
        cap - cap / 8
    }
}

/// Producer half. Lives on the real-time side.
pub struct RingProducer {
    inner: Arc<Inner>,
}

/// Consumer half. Lives on the async side.
pub struct RingConsumer {
    inner: Arc<Inner>,
}

/// Create a ring holding at least `capacity` samples (rounded up to a power of
/// two, minimum 64).
pub fn ring(capacity: usize) -> (RingProducer, RingConsumer) {
    let cap = capacity.next_power_of_two().max(64);
    let buffer = (0..cap).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
    let inner = Arc::new(Inner {
        buffer: buffer.into_boxed_slice(),
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
    /// If they do not fit, the oldest unread audio is overwritten. Returns how
    /// many samples that cost.
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

        // Count what will be overwritten, but never touch `read`: that cursor
        // belongs to the consumer, which notices and resynchronises itself.
        let dropped = data.len().saturating_sub(self.free());
        if dropped > 0 {
            self.inner.dropped.fetch_add(dropped, Ordering::Relaxed);
        }

        // Relaxed: this is the only thread that stores to `write`.
        let write = self.inner.write.load(Ordering::Relaxed);
        for (i, sample) in data.iter().enumerate() {
            let slot = write.wrapping_add(i) & self.inner.mask;
            self.inner.buffer[slot].store(sample.to_bits(), Ordering::Relaxed);
        }
        // Release: publishes every slot store above to whoever acquires this.
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
    /// Samples the producer overwrote before we reached them are skipped
    /// rather than returned: partially overwritten audio is worse than none.
    /// What is returned is always contiguous and in order.
    pub fn pop_slice(&mut self, out: &mut [f32]) -> usize {
        for _ in 0..MAX_REREADS {
            let read = self.resynchronise();
            let available = self.len().min(out.len());
            if available == 0 {
                return 0;
            }
            for (i, slot) in out[..available].iter_mut().enumerate() {
                let index = read.wrapping_add(i) & self.inner.mask;
                *slot = f32::from_bits(self.inner.buffer[index].load(Ordering::Relaxed));
            }
            // Did the producer wrap into the region we just copied? If so the
            // samples are a mixture of old and new, so throw them away and try
            // again rather than handing back audio that goes backwards.
            if self.overwritten_since(read) {
                continue;
            }
            self.inner
                .read
                .store(read.wrapping_add(available), Ordering::Release);
            return available;
        }
        // A producer this far ahead of us means the machine is in trouble;
        // returning nothing lets the caller try again rather than spin here.
        0
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

    /// Catch up if the producer has overwritten unread samples.
    ///
    /// Returns the (possibly advanced) read cursor. Only the consumer calls
    /// this, which is what keeps `read` single-writer.
    fn resynchronise(&mut self) -> usize {
        let read = self.inner.read.load(Ordering::Relaxed);
        if !self.overwritten_since(read) {
            return read;
        }
        let write = self.inner.write.load(Ordering::Acquire);
        let skipped = write.wrapping_sub(read) - self.inner.readable_after_overrun();
        let read = read.wrapping_add(skipped);
        self.inner.read.store(read, Ordering::Release);
        read
    }

    /// True when the producer has moved more than a full ring past `read`,
    /// meaning some unread samples no longer exist.
    fn overwritten_since(&self, read: usize) -> bool {
        let write = self.inner.write.load(Ordering::Acquire);
        write.wrapping_sub(read) > self.inner.capacity()
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
