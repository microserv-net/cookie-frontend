//! Audio capture, playback, analysis and voice activity detection.
//!
//! Layering, from the metal up:
//!
//! ```text
//!   cpal callback  ──push──▶  SpscRing  ──pull──▶  async capture task
//!   (real-time)               (lock-free)          (resample, analyse, VAD)
//!
//!   async playback task ──push──▶ SpscRing ──pull──▶ cpal callback
//! ```
//!
//! The rings are the only thing the real-time callbacks touch. No allocation,
//! no locks, no syscalls, no `tracing` macro ever runs inside a callback: an
//! audio callback that blocks produces an audible glitch, and a callback that
//! takes a mutex the async side also holds produces a priority inversion that
//! shows up as a mysterious stutter under load.

use serde::{Deserialize, Serialize};

pub mod features;
pub mod input;
pub mod output;
pub mod resample;
pub mod ring;
pub mod vad;
pub mod wav;

pub use features::{AudioAnalyzer, AudioFeatures, FeatureSource};
pub use input::{AudioInput, CaptureHandle, MockInput};
pub use output::{AudioOutput, MockOutput, PlaybackHandle};
pub use ring::{ring, RingConsumer, RingProducer};
pub use vad::{EnergyVad, Vad, VadDecision, VadEvent};

#[cfg(feature = "audio-io")]
pub use input::CpalInput;
#[cfg(feature = "audio-io")]
pub use output::CpalOutput;

/// Interleaved PCM, always `f32` in `[-1, 1]` once inside the pipeline.
///
/// Device-native formats (i16, u16, i32, f64...) are converted exactly once,
/// at the device boundary. Everything above that boundary is f32 mono at
/// `AudioConfig::sample_rate`, which removes an entire class of "why is it
/// chipmunk speed" bugs.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    pub fn silent(duration_ms: u64, sample_rate: u32) -> Self {
        let n = (sample_rate as u64 * duration_ms / 1000) as usize;
        Self::mono(vec![0.0; n], sample_rate)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.frames() as u64 * 1000) / self.sample_rate as u64
    }

    /// Average all channels down to one. No-op if already mono.
    pub fn to_mono(&self) -> AudioBuffer {
        if self.channels <= 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let scale = 1.0 / ch as f32;
        let samples = self
            .samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() * scale)
            .collect();
        AudioBuffer::mono(samples, self.sample_rate)
    }

    /// Resample to `rate` (cubic Hermite). Returns `self` unchanged if rates match.
    pub fn resampled(&self, rate: u32) -> AudioBuffer {
        if self.sample_rate == rate || self.samples.is_empty() {
            return AudioBuffer {
                samples: self.samples.clone(),
                sample_rate: rate.max(1),
                channels: self.channels,
            };
        }
        let mono = self.to_mono();
        let out = resample::resample(&mono.samples, mono.sample_rate, rate);
        AudioBuffer::mono(out, rate)
    }

    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        (self.samples.iter().map(|s| s * s).sum::<f32>() / self.samples.len() as f32).sqrt()
    }

    /// Scale in place, with hard clipping so a bad gain setting distorts
    /// rather than producing values that make downstream maths explode.
    pub fn apply_gain(&mut self, gain: f32) {
        if (gain - 1.0).abs() < f32::EPSILON {
            return;
        }
        for s in &mut self.samples {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}

/// What a device is and whether we can actually use it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
    /// `"cpal"`, `"mock"`, ...
    pub backend: String,
}

impl AudioDeviceInfo {
    pub fn mock(name: &str, sample_rate: u32) -> Self {
        Self {
            name: name.to_string(),
            is_default: true,
            sample_rate,
            channels: 1,
            backend: "mock".into(),
        }
    }
}

/// Linear amplitude to dBFS, floored so silence is a number rather than -inf.
#[inline]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    const FLOOR_DB: f32 = -100.0;
    if amplitude <= 1e-10 {
        FLOOR_DB
    } else {
        (20.0 * amplitude.log10()).max(FLOOR_DB)
    }
}

#[inline]
pub fn db_to_amplitude(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_downmix_averages_channels() {
        let b = AudioBuffer {
            samples: vec![1.0, 0.0, 0.5, 0.5],
            sample_rate: 48_000,
            channels: 2,
        };
        let m = b.to_mono();
        assert_eq!(m.samples, vec![0.5, 0.5]);
        assert_eq!(m.channels, 1);
    }

    #[test]
    fn duration_is_computed_from_frames() {
        let b = AudioBuffer::silent(1000, 16_000);
        assert_eq!(b.samples.len(), 16_000);
        assert_eq!(b.duration_ms(), 1000);
    }

    #[test]
    fn resampling_preserves_duration_within_a_frame() {
        let src = AudioBuffer::mono(vec![0.0; 48_000], 48_000);
        let out = src.resampled(16_000);
        assert!(
            (out.samples.len() as i64 - 16_000).abs() <= 2,
            "got {}",
            out.samples.len()
        );
        assert_eq!(out.sample_rate, 16_000);
    }

    #[test]
    fn gain_clips_instead_of_exploding() {
        let mut b = AudioBuffer::mono(vec![0.9, -0.9], 16_000);
        b.apply_gain(4.0);
        assert_eq!(b.samples, vec![1.0, -1.0]);
    }

    #[test]
    fn db_conversion_roundtrips() {
        for amp in [1.0f32, 0.5, 0.1, 0.01] {
            let db = amplitude_to_db(amp);
            assert!((db_to_amplitude(db) - amp).abs() < 1e-4);
        }
        assert_eq!(amplitude_to_db(0.0), -100.0);
    }
}
