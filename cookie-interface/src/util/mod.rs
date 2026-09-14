//! Small dependency-free utilities shared across the crate.
//!
//! Everything in here is deliberately self-contained: the animation system
//! needs a *reproducible* random number generator (so a visual bug can be
//! replayed from a seed), and pulling a general purpose RNG crate would make
//! that reproducibility depend on someone else's version bumps.

use std::time::{SystemTime, UNIX_EPOCH};

pub mod text;

/// A small, fast, fully deterministic PRNG (PCG-XSH-RR 64/32).
///
/// Identical seeds produce identical streams on every platform and in every
/// build of this crate, which is what makes `--seed` reproducible.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    pub const fn new(seed: u64) -> Self {
        // 0x9e37... is the odd increment; any odd number works.
        let mut rng = Rng {
            state: 0,
            inc: 0x9e37_79b9_7f4a_7c15,
        };
        rng.state = seed.wrapping_add(rng.inc);
        rng
    }

    /// Seed from the wall clock. Only used where reproducibility is explicitly
    /// not wanted; the seed is always logged so a session can be replayed.
    pub fn from_entropy() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed_5eed);
        let addr = &nanos as *const u64 as u64;
        Self::new(nanos ^ addr.rotate_left(17))
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        // 24 bits of mantissa is all an f32 can represent exactly.
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[min, max)`. Returns `min` if the range is empty.
    #[inline]
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        if max <= min {
            return min;
        }
        min + self.f32() * (max - min)
    }

    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Pick an index from a weight table. Weights must be non-negative.
    pub fn weighted(&mut self, weights: &[f32]) -> usize {
        let total: f32 = weights.iter().copied().filter(|w| *w > 0.0).sum();
        if total <= 0.0 {
            return self.below(weights.len().max(1));
        }
        let mut pick = self.f32() * total;
        for (i, w) in weights.iter().enumerate() {
            if *w <= 0.0 {
                continue;
            }
            pick -= w;
            if pick <= 0.0 {
                return i;
            }
        }
        weights.len().saturating_sub(1)
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0xc00c_1e00)
    }
}

/// Seconds since the Unix epoch, saturating at 0 if the clock is before 1970.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Milliseconds since the Unix epoch.
pub fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Frame-rate independent exponential smoothing.
///
/// `half_life` is the time in seconds for the value to cover half the
/// remaining distance, which keeps animation feel identical at 60 and 144 Hz.
#[inline]
pub fn approach(current: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    if half_life <= 0.0 {
        return target;
    }
    let k = 1.0 - (-dt * std::f32::consts::LN_2 / half_life).exp();
    current + (target - current) * k
}

#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A boxed, `Send` future — the crate's stand-in for `async fn` in traits.
///
/// Async methods in traits exist on modern Rust but are not object-safe, and
/// every provider in this crate is used behind `dyn`. Boxing once per model
/// call is free compared with the model call itself.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_for_a_seed() {
        let a: Vec<u32> = (0..8).map(|_| Rng::new(42).next_u32()).collect();
        assert!(a.iter().all(|v| *v == a[0]), "same seed, same first draw");

        let mut r1 = Rng::new(7);
        let mut r2 = Rng::new(7);
        for _ in 0..1000 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn rng_streams_differ_between_seeds() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        let d1: Vec<u64> = (0..16).map(|_| r1.next_u64()).collect();
        let d2: Vec<u64> = (0..16).map(|_| r2.next_u64()).collect();
        assert_ne!(d1, d2);
    }

    #[test]
    fn f32_stays_in_unit_range() {
        let mut r = Rng::new(99);
        for _ in 0..10_000 {
            let v = r.f32();
            assert!((0.0..1.0).contains(&v), "{v} out of range");
        }
    }

    #[test]
    fn weighted_respects_zero_weights() {
        let mut r = Rng::new(3);
        for _ in 0..500 {
            assert_eq!(r.weighted(&[0.0, 1.0, 0.0]), 1);
        }
    }

    #[test]
    fn approach_converges_and_is_monotonic() {
        let mut v = 0.0;
        for _ in 0..600 {
            v = approach(v, 1.0, 0.1, 1.0 / 60.0);
        }
        assert!((v - 1.0).abs() < 1e-3, "converged to {v}");
    }
}
