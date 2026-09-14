//! Lightweight spectral analysis that drives the orb.
//!
//! Everything here runs once per 20 ms frame on both the capture and playback
//! paths, so it has a strict budget: one 512-point FFT, a handful of sums, no
//! allocation after construction. The output is deliberately *perceptual*
//! rather than raw — `envelope`, `onset`, band ratios and voicing are what an
//! animation wants, whereas a raw magnitude spectrum would push all the
//! interpretation work into the shader.
//!
//! Scaling a whole orb by volume is the obvious thing and it looks cheap, so
//! the feature set is designed to let the animation react to *character*:
//! sibilance lands in `high`, vowels in `low`/`mid`, plosives in `onset`, and
//! `voiced` separates speech from a fan or a keyboard.

use serde::{Deserialize, Serialize};

const FFT_SIZE: usize = 512;
const BINS: usize = FFT_SIZE / 2;

/// A frame of analysis. Cheap to clone; published on a `watch` channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// Root-mean-square amplitude of this frame, 0..1.
    pub rms: f32,
    /// Largest absolute sample in this frame, 0..1.
    pub peak: f32,
    /// dBFS of `rms`, floored at -100.
    pub level_db: f32,
    /// Fast-attack / slow-release follower. This is what you animate with;
    /// raw `rms` flickers.
    pub envelope: f32,
    /// Energy below 300 Hz, normalised 0..1. Vowels and chest resonance.
    pub low: f32,
    /// 300 Hz - 2 kHz. Where most of the intelligibility lives.
    pub mid: f32,
    /// Above 2 kHz. Sibilance and consonants.
    pub high: f32,
    /// Spectral centroid mapped to 0..1 (brightness).
    pub centroid: f32,
    /// Positive spectral flux — how much the spectrum just changed.
    pub flux: f32,
    /// Transient detector, 0..1, with a decay. Plosives spike this.
    pub onset: f32,
    /// Zero-crossing rate 0..1. High for fricatives, low for vowels.
    pub zcr: f32,
    /// Crude voicing estimate 0..1: loud, low-ZCR, low-flux frames score high.
    pub voiced: f32,
    /// Estimated noise floor in dB, adapted continuously.
    pub noise_floor_db: f32,
    /// Milliseconds of audio analysed since the analyser was created.
    pub elapsed_ms: u64,
    /// Which side of the pipeline produced this frame.
    pub source: FeatureSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSource {
    /// Microphone.
    Input,
    /// Synthesised speech on its way to the speaker.
    Output,
    /// Nothing is running.
    Silent,
}

impl AudioFeatures {
    pub fn silent() -> Self {
        Self {
            rms: 0.0,
            peak: 0.0,
            level_db: -100.0,
            envelope: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
            centroid: 0.0,
            flux: 0.0,
            onset: 0.0,
            zcr: 0.0,
            voiced: 0.0,
            noise_floor_db: -100.0,
            elapsed_ms: 0,
            source: FeatureSource::Silent,
        }
    }

    /// How far above the adapted noise floor this frame sits, in dB.
    pub fn snr_db(&self) -> f32 {
        self.level_db - self.noise_floor_db
    }

    /// Interpolate towards another frame. The renderer uses this to stay
    /// smooth when analysis frames arrive slower than it draws.
    pub fn lerp(&self, other: &AudioFeatures, t: f32) -> AudioFeatures {
        use crate::util::lerp;
        AudioFeatures {
            rms: lerp(self.rms, other.rms, t),
            peak: lerp(self.peak, other.peak, t),
            level_db: lerp(self.level_db, other.level_db, t),
            envelope: lerp(self.envelope, other.envelope, t),
            low: lerp(self.low, other.low, t),
            mid: lerp(self.mid, other.mid, t),
            high: lerp(self.high, other.high, t),
            centroid: lerp(self.centroid, other.centroid, t),
            flux: lerp(self.flux, other.flux, t),
            onset: lerp(self.onset, other.onset, t),
            zcr: lerp(self.zcr, other.zcr, t),
            voiced: lerp(self.voiced, other.voiced, t),
            noise_floor_db: lerp(self.noise_floor_db, other.noise_floor_db, t),
            elapsed_ms: other.elapsed_ms,
            source: other.source,
        }
    }
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self::silent()
    }
}

/// Streaming analyser. One per audio direction.
pub struct AudioAnalyzer {
    sample_rate: u32,
    source: FeatureSource,
    fft: Fft,
    window: Vec<f32>,
    /// Sliding input window; new frames shift in from the right.
    history: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    magnitude: Vec<f32>,
    prev_magnitude: Vec<f32>,
    envelope: f32,
    onset: f32,
    noise_floor_db: f32,
    samples_seen: u64,
    band_edges: [usize; 4],
}

impl AudioAnalyzer {
    pub fn new(sample_rate: u32, source: FeatureSource) -> Self {
        let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
        let bin_of = |hz: f32| ((hz / bin_hz).round() as usize).clamp(1, BINS - 1);
        Self {
            sample_rate,
            source,
            fft: Fft::new(FFT_SIZE),
            window: hann(FFT_SIZE),
            history: vec![0.0; FFT_SIZE],
            re: vec![0.0; FFT_SIZE],
            im: vec![0.0; FFT_SIZE],
            magnitude: vec![0.0; BINS],
            prev_magnitude: vec![0.0; BINS],
            envelope: 0.0,
            onset: 0.0,
            noise_floor_db: -60.0,
            samples_seen: 0,
            band_edges: [bin_of(60.0), bin_of(300.0), bin_of(2000.0), BINS - 1],
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = 0.0);
        self.prev_magnitude.iter_mut().for_each(|s| *s = 0.0);
        self.envelope = 0.0;
        self.onset = 0.0;
    }

    /// Analyse one frame of mono audio. Frame length is free — it is shifted
    /// into a fixed 512-sample analysis window.
    pub fn process(&mut self, frame: &[f32]) -> AudioFeatures {
        if frame.is_empty() {
            return self.decay_only();
        }
        self.samples_seen += frame.len() as u64;

        // --- time domain -------------------------------------------------
        let mut sum_sq = 0.0f32;
        let mut peak = 0.0f32;
        let mut crossings = 0usize;
        let mut prev = self.history[FFT_SIZE - 1];
        for s in frame {
            sum_sq += s * s;
            peak = peak.max(s.abs());
            if (prev < 0.0 && *s >= 0.0) || (prev >= 0.0 && *s < 0.0) {
                crossings += 1;
            }
            prev = *s;
        }
        let rms = (sum_sq / frame.len() as f32).sqrt();
        let zcr = (crossings as f32 / frame.len() as f32).min(1.0);
        let level_db = super::amplitude_to_db(rms);

        // --- slide the analysis window ------------------------------------
        let take = frame.len().min(FFT_SIZE);
        self.history.copy_within(take.., 0);
        let start = FFT_SIZE - take;
        self.history[start..].copy_from_slice(&frame[frame.len() - take..]);

        // --- frequency domain ---------------------------------------------
        for i in 0..FFT_SIZE {
            self.re[i] = self.history[i] * self.window[i];
            self.im[i] = 0.0;
        }
        self.fft.forward(&mut self.re, &mut self.im);

        let mut flux = 0.0f32;
        let mut total = 0.0f32;
        let mut weighted = 0.0f32;
        for k in 0..BINS {
            let mag = (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt();
            let diff = mag - self.prev_magnitude[k];
            if diff > 0.0 {
                flux += diff;
            }
            self.prev_magnitude[k] = mag;
            self.magnitude[k] = mag;
            total += mag;
            weighted += mag * k as f32;
        }

        let band = |from: usize, to: usize, mags: &[f32]| -> f32 {
            mags[from..=to.min(BINS - 1)].iter().sum::<f32>()
        };
        let low_raw = band(self.band_edges[0], self.band_edges[1], &self.magnitude);
        let mid_raw = band(self.band_edges[1] + 1, self.band_edges[2], &self.magnitude);
        let high_raw = band(self.band_edges[2] + 1, self.band_edges[3], &self.magnitude);

        // Normalise bands against a loudness reference rather than against
        // each other alone: the animation wants to know both *where* the
        // energy is and *how much* there is.
        let denom = (low_raw + mid_raw + high_raw).max(1e-6);
        let loudness = (rms * 6.0).clamp(0.0, 1.0);
        let low = (low_raw / denom) * loudness;
        let mid = (mid_raw / denom) * loudness;
        let high = (high_raw / denom) * loudness;

        let centroid = if total > 1e-6 {
            (weighted / total / BINS as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let flux_norm = (flux / BINS as f32 * 20.0).clamp(0.0, 1.0);

        // --- followers -----------------------------------------------------
        // Fast attack so a consonant registers immediately, slow release so
        // the orb settles rather than strobing.
        let target = rms.clamp(0.0, 1.0);
        let coeff = if target > self.envelope { 0.55 } else { 0.08 };
        self.envelope += (target - self.envelope) * coeff;

        let onset_impulse = if flux_norm > 0.18 && self.envelope > 0.01 {
            flux_norm
        } else {
            0.0
        };
        self.onset = (self.onset * 0.82).max(onset_impulse);

        // Noise floor tracks quiet frames quickly and loud frames very slowly,
        // so a long sentence does not drag the floor up with it.
        let floor_coeff = if level_db < self.noise_floor_db {
            0.25
        } else {
            0.0015
        };
        self.noise_floor_db += (level_db - self.noise_floor_db) * floor_coeff;
        self.noise_floor_db = self.noise_floor_db.clamp(-100.0, -10.0);

        let snr = level_db - self.noise_floor_db;
        let voiced = {
            let loud = crate::util::smoothstep(3.0, 14.0, snr);
            let tonal = 1.0 - crate::util::smoothstep(0.18, 0.45, zcr);
            let bodied = crate::util::smoothstep(0.02, 0.15, low_raw / denom);
            (loud * (0.45 + 0.35 * tonal + 0.2 * bodied)).clamp(0.0, 1.0)
        };

        AudioFeatures {
            rms,
            peak,
            level_db,
            envelope: self.envelope,
            low,
            mid,
            high,
            centroid,
            flux: flux_norm,
            onset: self.onset,
            zcr,
            voiced,
            noise_floor_db: self.noise_floor_db,
            elapsed_ms: self.samples_seen * 1000 / self.sample_rate.max(1) as u64,
            source: self.source,
        }
    }

    /// Advance the followers without new audio (playback gap, mic muted).
    pub fn decay_only(&mut self) -> AudioFeatures {
        self.envelope *= 0.9;
        self.onset *= 0.8;
        let mut f = AudioFeatures::silent();
        f.envelope = self.envelope;
        f.onset = self.onset;
        f.noise_floor_db = self.noise_floor_db;
        f.elapsed_ms = self.samples_seen * 1000 / self.sample_rate.max(1) as u64;
        f.source = self.source;
        f
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / n as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

/// Iterative in-place radix-2 FFT with precomputed twiddles and bit-reversal
/// table. 512 points costs ~4.6k butterflies, which is nothing next to the
/// 320 samples of audio it analyses.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two");
        let mut cos = Vec::with_capacity(n / 2);
        let mut sin = Vec::with_capacity(n / 2);
        for i in 0..n / 2 {
            let angle = -2.0 * std::f32::consts::PI * i as f32 / n as f32;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| (i as u32).reverse_bits() >> (32 - bits))
            .map(|i| i as usize)
            .collect();
        Self { n, cos, sin, rev }
    }

    fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        debug_assert_eq!(re.len(), self.n);
        for i in 0..self.n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= self.n {
            let step = self.n / len;
            let half = len / 2;
            let mut i = 0;
            while i < self.n {
                for k in 0..half {
                    let t = k * step;
                    let (wr, wi) = (self.cos[t], self.sin[t]);
                    let a = i + k;
                    let b = a + half;
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
                i += len;
            }
            len <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, rate: u32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn silence_yields_silent_features() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let f = a.process(&vec![0.0; 320]);
        assert_eq!(f.rms, 0.0);
        assert_eq!(f.level_db, -100.0);
        assert!(f.envelope < 1e-3);
        assert!(f.voiced < 0.01);
    }

    #[test]
    fn a_low_tone_lands_in_the_low_band() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let signal = tone(150.0, 16_000, 16_000, 0.5);
        let mut last = AudioFeatures::silent();
        for frame in signal.chunks(320) {
            last = a.process(frame);
        }
        assert!(last.low > last.high, "low {} high {}", last.low, last.high);
        assert!(last.centroid < 0.15, "centroid {}", last.centroid);
    }

    #[test]
    fn a_high_tone_lands_in_the_high_band() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let signal = tone(5000.0, 16_000, 16_000, 0.5);
        let mut last = AudioFeatures::silent();
        for frame in signal.chunks(320) {
            last = a.process(frame);
        }
        assert!(last.high > last.low, "low {} high {}", last.low, last.high);
        assert!(last.centroid > 0.4, "centroid {}", last.centroid);
    }

    #[test]
    fn envelope_attacks_fast_and_releases_slowly() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let loud = tone(400.0, 16_000, 3200, 0.8);
        let mut env_up = 0.0;
        for frame in loud.chunks(320) {
            env_up = a.process(frame).envelope;
        }
        assert!(env_up > 0.3, "envelope did not rise: {env_up}");

        let mut env_down = env_up;
        let silence = vec![0.0; 320];
        let mut frames = 0;
        while env_down > env_up * 0.5 && frames < 100 {
            env_down = a.process(&silence).envelope;
            frames += 1;
        }
        assert!(frames > 3, "release was too fast ({frames} frames to half)");
    }

    #[test]
    fn onset_fires_on_a_transient() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        for _ in 0..20 {
            a.process(&vec![0.0; 320]);
        }
        let burst = tone(1200.0, 16_000, 320, 0.9);
        let f = a.process(&burst);
        assert!(f.onset > 0.05, "onset {} on a hard transient", f.onset);
    }

    #[test]
    fn zcr_separates_noise_from_tone() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let tonal = a.process(&tone(200.0, 16_000, 320, 0.5)).zcr;

        let mut rng = crate::util::Rng::new(1);
        let noise: Vec<f32> = (0..320).map(|_| rng.range(-0.5, 0.5)).collect();
        let noisy = a.process(&noise).zcr;
        assert!(noisy > tonal * 3.0, "tone {tonal} vs noise {noisy}");
    }

    #[test]
    fn noise_floor_adapts_downwards_quickly() {
        let mut a = AudioAnalyzer::new(16_000, FeatureSource::Input);
        let quiet = vec![0.0005f32; 320];
        let mut f = AudioFeatures::silent();
        for _ in 0..50 {
            f = a.process(&quiet);
        }
        assert!(f.noise_floor_db < -55.0, "floor {}", f.noise_floor_db);
        let loud = tone(300.0, 16_000, 320, 0.6);
        let f2 = a.process(&loud);
        assert!(f2.snr_db() > 20.0, "snr {}", f2.snr_db());
    }

    #[test]
    fn features_interpolate() {
        let a = AudioFeatures::silent();
        let mut b = AudioFeatures::silent();
        b.envelope = 1.0;
        b.source = FeatureSource::Output;
        let mid = a.lerp(&b, 0.5);
        assert!((mid.envelope - 0.5).abs() < 1e-6);
        assert_eq!(mid.source, FeatureSource::Output);
    }

    #[test]
    fn fft_matches_a_naive_dft() {
        let n = 64;
        let fft = Fft::new(n);
        let mut rng = crate::util::Rng::new(5);
        let input: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut re = input.clone();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);

        for k in [0usize, 1, 7, 31] {
            let mut sr = 0.0f32;
            let mut si = 0.0f32;
            for (t, x) in input.iter().enumerate() {
                let ang = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                sr += x * ang.cos();
                si += x * ang.sin();
            }
            assert!((re[k] - sr).abs() < 1e-3, "bin {k}: {} vs {sr}", re[k]);
            assert!((im[k] - si).abs() < 1e-3, "bin {k}: {} vs {si}", im[k]);
        }
    }
}
