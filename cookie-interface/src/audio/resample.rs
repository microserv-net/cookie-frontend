//! Sample-rate conversion.
//!
//! Catmull-Rom (cubic Hermite) interpolation. For the rates we actually care
//! about — 44.1/48 kHz device audio down to the 16 kHz the recognisers want,
//! and 22.05/24 kHz synthesiser output up to whatever the speaker runs at —
//! this is both cheap enough to run inside the capture task and clean enough
//! that no one can hear the difference against a windowed-sinc job.
//!
//! Deliberately *not* a dependency: `rubato` is excellent, but it is a
//! heavyweight FFT resampler whose latency and block-size constraints are a
//! poor fit for the small, irregular chunks a voice pipeline moves around.

/// One-shot conversion of a whole buffer.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos.floor() as isize;
        let frac = (pos - idx as f64) as f32;
        out.push(hermite(input, idx, frac));
    }
    out
}

/// Streaming resampler that keeps the fractional phase across calls, so a
/// sequence of chunks converts identically to the concatenated whole.
#[derive(Debug, Clone)]
pub struct Resampler {
    from_rate: u32,
    to_rate: u32,
    /// Fractional read position within `history` + incoming data.
    phase: f64,
    /// Last three input samples, needed for the cubic kernel.
    history: [f32; 3],
    primed: bool,
}

impl Resampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self {
            from_rate: from_rate.max(1),
            to_rate: to_rate.max(1),
            phase: 0.0,
            history: [0.0; 3],
            primed: false,
        }
    }

    pub fn is_passthrough(&self) -> bool {
        self.from_rate == self.to_rate
    }

    pub fn ratio(&self) -> f64 {
        self.from_rate as f64 / self.to_rate as f64
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.history = [0.0; 3];
        self.primed = false;
    }

    /// Convert one chunk, appending to `out` (which is reused by the caller to
    /// avoid per-chunk allocation in the capture loop).
    pub fn process_into(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        if self.is_passthrough() {
            out.extend_from_slice(input);
            return;
        }

        // Working buffer = 3 samples of history followed by the new chunk, so
        // the kernel can look backwards across the chunk boundary.
        let mut work = Vec::with_capacity(input.len() + 3);
        work.extend_from_slice(&self.history);
        work.extend_from_slice(input);

        if !self.primed {
            // Pre-fill history with the first sample rather than silence,
            // otherwise every stream starts with a click.
            let first = input[0];
            work[0] = first;
            work[1] = first;
            work[2] = first;
            self.primed = true;
        }

        let ratio = self.ratio();
        // Index 3 in `work` is the first *new* sample; phase is relative to it.
        let mut pos = 3.0 + self.phase;
        let limit = work.len() as f64 - 1.0;
        while pos < limit {
            let idx = pos.floor() as isize;
            let frac = (pos - idx as f64) as f32;
            out.push(hermite(&work, idx, frac));
            pos += ratio;
        }
        // `pos` is an index into `work`, whose last three samples become the
        // *history* of the next call — i.e. `work_next[0..3]`. The next call
        // starts reading at `3.0 + phase`, so phase must be measured from the
        // first sample of the next chunk, which sits at absolute index
        // `work.len()`. Adding the history offset here as well would skip
        // three input samples per chunk, which at 48k→16k silently dropped one
        // output sample every chunk.
        self.phase = pos - work.len() as f64;

        let n = work.len();
        self.history = [work[n - 3], work[n - 2], work[n - 1]];
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity((input.len() as f64 / self.ratio()) as usize + 4);
        self.process_into(input, &mut out);
        out
    }
}

#[inline]
fn hermite(data: &[f32], idx: isize, frac: f32) -> f32 {
    let at = |i: isize| -> f32 {
        let i = i.clamp(0, data.len() as isize - 1) as usize;
        data[i]
    };
    let y0 = at(idx - 1);
    let y1 = at(idx);
    let y2 = at(idx + 1);
    let y3 = at(idx + 2);

    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * frac + c2) * frac + c1) * frac + c0
}

/// Interleave mono into `channels` identical channels (for device playback).
pub fn mono_to_interleaved(mono: &[f32], channels: u16, out: &mut Vec<f32>) {
    let ch = channels.max(1) as usize;
    out.reserve(mono.len() * ch);
    for s in mono {
        for _ in 0..ch {
            out.push(*s);
        }
    }
}

/// Average interleaved frames down to mono, appending to `out`.
pub fn interleaved_to_mono(data: &[f32], channels: u16, out: &mut Vec<f32>) {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        out.extend_from_slice(data);
        return;
    }
    let scale = 1.0 / ch as f32;
    for frame in data.chunks_exact(ch) {
        out.push(frame.iter().sum::<f32>() * scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, rate: u32, freq: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn passthrough_when_rates_match() {
        let input = sine(100, 16_000, 440.0);
        assert_eq!(resample(&input, 16_000, 16_000), input);
    }

    #[test]
    fn downsampling_produces_the_expected_length() {
        let input = vec![0.0; 48_000];
        let out = resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn upsampling_produces_the_expected_length() {
        let input = vec![0.0; 16_000];
        let out = resample(&input, 16_000, 48_000);
        assert_eq!(out.len(), 48_000);
    }

    #[test]
    fn a_sine_survives_conversion() {
        // 440 Hz at 48k -> 16k should stay a 440 Hz sine: check amplitude and
        // zero-crossing count rather than doing a full spectral comparison.
        let input = sine(4800, 48_000, 440.0);
        let out = resample(&input, 48_000, 16_000);
        let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {peak}");

        let crossings = out.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let expected = (440.0 * (out.len() as f32 / 16_000.0)).round() as usize;
        assert!(
            crossings.abs_diff(expected) <= 1,
            "{crossings} crossings, expected ~{expected}"
        );
    }

    #[test]
    fn streaming_matches_one_shot_length() {
        let input = sine(48_000, 48_000, 220.0);
        let mut r = Resampler::new(48_000, 16_000);
        let mut streamed = Vec::new();
        for chunk in input.chunks(1024) {
            r.process_into(chunk, &mut streamed);
        }
        assert!(
            (streamed.len() as i64 - 16_000).abs() <= 4,
            "streamed {} samples",
            streamed.len()
        );
    }

    #[test]
    fn streaming_has_no_discontinuity_at_chunk_edges() {
        let input = sine(9600, 48_000, 200.0);
        let mut r = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        for chunk in input.chunks(157) {
            // deliberately awkward chunk size
            r.process_into(chunk, &mut out);
        }
        // A click at a chunk boundary shows up as a sample-to-sample jump far
        // larger than the signal's own slope.
        let max_delta = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta < 0.25, "max step {max_delta} suggests a click");
    }

    #[test]
    fn interleave_roundtrip() {
        let mono = vec![0.1, 0.2, 0.3];
        let mut inter = Vec::new();
        mono_to_interleaved(&mono, 2, &mut inter);
        assert_eq!(inter, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        let mut back = Vec::new();
        interleaved_to_mono(&inter, 2, &mut back);
        for (a, b) in back.iter().zip(mono.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
