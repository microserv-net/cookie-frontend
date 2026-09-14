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
//! * the writer only ever writes to slots in `[head, tail)` (the free region)
//!   and the reader only ever reads `[tail, head)` (the filled region), and the
//!   two regions are disjoint by construction;
//! * the writer publishes with `Release` after writing and the reader acquires
//!   with `Acquire` before reading, so the data is visible when the cursor is.
//!
//! Overflow drops the *oldest* audio on the capture side and is counted, never
//! silently ignored: a rising overflow count is the signature of a stalled
//! consumer and shows up in `--doctor`.

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

    fn len(&self) -> usize {
        let w = self.write.load(Ordering::Acquire);
        let r = self.read.load(Ordering::Acquire);
        w.wrapping_sub(r)
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

    pub fn free(&self) -> usize {
        self.capacity() - self.len()
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

        let mut dropped = 0;
        let free = self.free();
        if data.len() > free {
            let need = data.len() - free;
            // Advance the read cursor past the oldest `need` samples. Safe for
            // the consumer: it re-reads the cursor before every pop.
            self.inner.read.fetch_add(need, Ordering::AcqRel);
            dropped = need;
            self.inner.dropped.fetch_add(need, Ordering::Relaxed);
        }

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
    pub fn pop_slice(&mut self, out: &mut [f32]) -> usize {
        let available = self.len().min(out.len());
        if available == 0 {
            return 0;
        }
        let read = self.inner.read.load(Ordering::Relaxed);
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

    /// Read exactly `out.len()` samples, or nothing at all. Useful for
    /// fixed-size analysis frames.
    pub fn pop_exact(&mut self, out: &mut [f32]) -> bool {
        if self.len() < out.len() {
            return false;
        }
        self.pop_slice(out) == out.len()
    }

    /// Discard everything currently buffered (used when speech is interrupted,
    /// so stale audio is not played after the stop).
    pub fn clear(&mut self) -> usize {
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
    fn overflow_drops_oldest_and_counts_it() {
        let (mut p, mut c) = ring(64);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        p.push_slice(&data);
        let dropped = p.push_slice(&[100.0, 101.0]);
        assert_eq!(dropped, 2);
        assert_eq!(p.dropped(), 2);
        let mut out = vec![0.0; 64];
        c.pop_slice(&mut out);
        assert_eq!(out[0], 2.0, "oldest two were discarded");
        assert_eq!(out[63], 101.0, "newest survived");
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
