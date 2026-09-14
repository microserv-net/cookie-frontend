//! Voice activity detection.
//!
//! The job is narrow: decide when a human started and stopped talking, so the
//! recogniser is handed an utterance instead of an endless stream, and so the
//! orb can acknowledge that it *heard* something before it has understood
//! anything.
//!
//! `EnergyVad` is an adaptive energy + voicing detector. It deliberately does
//! not use a neural VAD by default: a 2 MB model that needs downloading before
//! the microphone works is a bad first-run experience, and this is accurate
//! enough for near-field speech. The `Vad` trait means a Silero/sherpa-onnx
//! detector can be dropped in later without touching the pipeline.
//!
//! Two details that matter more than the threshold maths:
//!
//! * **pre-roll** — audio from *before* the trigger is kept, because by the
//!   time energy crosses a threshold the first consonant is already gone;
//! * **hangover** — a short silence inside a sentence ("the answer is … 42")
//!   must not end the utterance, so silence has to persist before it counts.

use std::collections::VecDeque;

use super::features::AudioFeatures;
use crate::config::VadConfig;

/// What the detector concluded about one frame.
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    /// Speech began. Carries the level that triggered it.
    SpeechStart { level_db: f32 },
    /// Speech ended. Carries the complete utterance including pre-roll.
    SpeechEnd {
        audio: Vec<f32>,
        duration_ms: u64,
        /// True when the utterance was cut short by `max_utterance_ms`.
        truncated: bool,
    },
    /// The utterance was discarded for being too short to be speech.
    Discarded { duration_ms: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadDecision {
    /// Whether this frame is considered speech.
    pub speech: bool,
    /// Whether an utterance is currently open.
    pub in_utterance: bool,
    pub event: Option<VadEvent>,
}

impl VadDecision {
    fn quiet() -> Self {
        Self {
            speech: false,
            in_utterance: false,
            event: None,
        }
    }
}

/// Replaceable detector interface.
pub trait Vad: Send {
    /// Feed one frame of mono audio plus its analysis.
    fn push(&mut self, frame: &[f32], features: &AudioFeatures) -> VadDecision;
    /// Forget all state (device change, interruption, config reload).
    fn reset(&mut self);
    /// End any open utterance immediately and return it.
    fn flush(&mut self) -> Option<VadEvent>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Silence,
    /// Energy is up but not yet for long enough to be believed.
    Onset,
    Speech,
    /// Below threshold, waiting out the hangover before closing.
    Hangover,
}

pub struct EnergyVad {
    cfg: VadConfig,
    sample_rate: u32,
    phase: Phase,
    /// Circular pre-roll of recent audio, capped at `preroll_ms`.
    /// Milliseconds of actually-voiced audio in the current utterance.
    /// Distinct from the buffer length, which also holds pre-roll and the
    /// hangover tail — counting those would let a 200 ms blip masquerade as a
    /// long utterance and defeat `min_utterance_ms` entirely.
    voiced_ms: u64,
    preroll: VecDeque<f32>,
    preroll_cap: usize,
    utterance: Vec<f32>,
    speech_ms: u32,
    silence_ms: u32,
    frame_ms: u32,
    /// Smoothed evidence that this is speech, 0..1. Hysteresis lives here
    /// rather than in a raw threshold comparison, which is what stops the
    /// detector chattering at the boundary.
    evidence: f32,
}

impl EnergyVad {
    pub fn new(cfg: VadConfig, sample_rate: u32, frame_ms: u32) -> Self {
        let preroll_cap = (sample_rate as usize * cfg.preroll_ms as usize) / 1000;
        Self {
            cfg,
            sample_rate,
            phase: Phase::Silence,
            voiced_ms: 0,
            preroll: VecDeque::with_capacity(preroll_cap + 1),
            preroll_cap,
            utterance: Vec::new(),
            speech_ms: 0,
            silence_ms: 0,
            frame_ms: frame_ms.max(1),
            evidence: 0.0,
        }
    }

    pub fn is_speaking(&self) -> bool {
        matches!(self.phase, Phase::Speech | Phase::Hangover)
    }

    fn utterance_ms(&self) -> u64 {
        (self.utterance.len() as u64 * 1000) / self.sample_rate.max(1) as u64
    }

    fn remember(&mut self, frame: &[f32]) {
        if self.preroll_cap == 0 {
            return;
        }
        for s in frame {
            if self.preroll.len() == self.preroll_cap {
                self.preroll.pop_front();
            }
            self.preroll.push_back(*s);
        }
    }

    fn open_utterance(&mut self) {
        self.utterance.clear();
        self.utterance.reserve(self.sample_rate as usize * 3);
        self.utterance.extend(self.preroll.iter().copied());
        // The onset frames that got us here were speech; count them.
        self.voiced_ms = self.speech_ms as u64;
    }

    fn close_utterance(&mut self, truncated: bool) -> VadEvent {
        let audio = std::mem::take(&mut self.utterance);
        let duration_ms = (audio.len() as u64 * 1000) / self.sample_rate.max(1) as u64;
        self.phase = Phase::Silence;
        self.speech_ms = 0;
        self.silence_ms = 0;
        self.evidence = 0.0;
        let voiced_ms = std::mem::take(&mut self.voiced_ms);
        if voiced_ms < self.cfg.min_utterance_ms as u64 {
            return VadEvent::Discarded { duration_ms };
        }
        VadEvent::SpeechEnd {
            audio,
            duration_ms,
            truncated,
        }
    }
}

impl Vad for EnergyVad {
    fn push(&mut self, frame: &[f32], features: &AudioFeatures) -> VadDecision {
        if !self.cfg.enabled {
            return VadDecision::quiet();
        }

        // Evidence combines "louder than the noise floor" with "sounds voiced".
        // Energy alone opens on a door slam; voicing alone is too twitchy at
        // low levels.
        let snr = features.snr_db();
        let loud = crate::util::smoothstep(
            self.cfg.threshold_db * 0.5,
            self.cfg.threshold_db * 1.5,
            snr,
        );
        let target = (0.65 * loud + 0.35 * features.voiced).clamp(0.0, 1.0);
        // Rise faster than we fall: catching the start matters more.
        let coeff = if target > self.evidence { 0.5 } else { 0.2 };
        self.evidence += (target - self.evidence) * coeff;

        let speechy = self.evidence > 0.5;
        let mut event = None;

        match self.phase {
            Phase::Silence => {
                self.remember(frame);
                if speechy {
                    self.phase = Phase::Onset;
                    self.speech_ms = self.frame_ms;
                }
            }
            Phase::Onset => {
                self.remember(frame);
                if speechy {
                    self.speech_ms += self.frame_ms;
                    if self.speech_ms >= self.cfg.speech_ms {
                        self.open_utterance();
                        self.phase = Phase::Speech;
                        self.silence_ms = 0;
                        event = Some(VadEvent::SpeechStart {
                            level_db: features.level_db,
                        });
                    }
                } else {
                    self.phase = Phase::Silence;
                    self.speech_ms = 0;
                }
            }
            Phase::Speech | Phase::Hangover => {
                self.utterance.extend_from_slice(frame);
                if speechy {
                    self.phase = Phase::Speech;
                    self.silence_ms = 0;
                    self.voiced_ms += self.frame_ms as u64;
                } else {
                    self.phase = Phase::Hangover;
                    self.silence_ms += self.frame_ms;
                    if self.silence_ms >= self.cfg.silence_ms {
                        // Trim the trailing silence we only kept in case
                        // speech resumed.
                        let trim =
                            (self.sample_rate as usize * self.silence_ms as usize) / 1000 / 2;
                        let keep = self.utterance.len().saturating_sub(trim);
                        self.utterance.truncate(keep.max(1));
                        event = Some(self.close_utterance(false));
                    }
                }
                if event.is_none() && self.utterance_ms() >= self.cfg.max_utterance_ms as u64 {
                    event = Some(self.close_utterance(true));
                }
            }
        }

        // Keep the pre-roll current even while speaking, so a new utterance
        // starting right after this one still has context.
        if matches!(self.phase, Phase::Speech | Phase::Hangover) {
            self.remember(frame);
        }

        VadDecision {
            speech: speechy,
            in_utterance: matches!(self.phase, Phase::Speech | Phase::Hangover),
            event,
        }
    }

    fn reset(&mut self) {
        self.phase = Phase::Silence;
        self.preroll.clear();
        self.voiced_ms = 0;
        self.utterance.clear();
        self.speech_ms = 0;
        self.silence_ms = 0;
        self.evidence = 0.0;
    }

    fn flush(&mut self) -> Option<VadEvent> {
        if matches!(self.phase, Phase::Speech | Phase::Hangover) {
            Some(self.close_utterance(false))
        } else {
            self.reset();
            None
        }
    }

    fn name(&self) -> &'static str {
        "energy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::features::{AudioAnalyzer, FeatureSource};

    struct Harness {
        vad: EnergyVad,
        analyzer: AudioAnalyzer,
        frame: usize,
    }

    impl Harness {
        fn new(cfg: VadConfig) -> Self {
            let rate = 16_000;
            let frame_ms = 20;
            Self {
                vad: EnergyVad::new(cfg, rate, frame_ms),
                analyzer: AudioAnalyzer::new(rate, FeatureSource::Input),
                frame: (rate * frame_ms / 1000) as usize,
            }
        }

        fn feed(&mut self, samples: &[f32]) -> Vec<VadEvent> {
            let mut events = Vec::new();
            for chunk in samples.chunks(self.frame) {
                let f = self.analyzer.process(chunk);
                if let Some(e) = self.vad.push(chunk, &f).event {
                    events.push(e);
                }
            }
            events
        }
    }

    fn silence(ms: usize) -> Vec<f32> {
        vec![0.0; 16 * ms]
    }

    /// Voiced-ish signal: low fundamental plus harmonics, which is what the
    /// voicing heuristic is tuned for.
    fn speech(ms: usize, amp: f32) -> Vec<f32> {
        let n = 16 * ms;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                amp * (0.6 * (2.0 * std::f32::consts::PI * 140.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 420.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 900.0 * t).sin())
            })
            .collect()
    }

    #[test]
    fn silence_never_triggers() {
        let mut h = Harness::new(VadConfig::default());
        let events = h.feed(&silence(5000));
        assert!(events.is_empty(), "{events:?}");
        assert!(!h.vad.is_speaking());
    }

    #[test]
    fn speech_produces_start_then_end() {
        let mut h = Harness::new(VadConfig::default());
        h.feed(&silence(600));
        let mut events = h.feed(&speech(1200, 0.4));
        events.extend(h.feed(&silence(1500)));

        assert!(
            matches!(events.first(), Some(VadEvent::SpeechStart { .. })),
            "{events:?}"
        );
        match events.last() {
            Some(VadEvent::SpeechEnd { duration_ms, .. }) => {
                assert!(*duration_ms > 800, "utterance only {duration_ms} ms");
            }
            other => panic!("expected SpeechEnd, got {other:?}"),
        }
    }

    #[test]
    fn preroll_is_included_in_the_utterance() {
        let cfg = VadConfig {
            preroll_ms: 300,
            ..Default::default()
        };
        let mut h = Harness::new(cfg);
        h.feed(&silence(800));
        h.feed(&speech(1000, 0.4));
        let events = h.feed(&silence(1200));
        let end = events
            .iter()
            .find_map(|e| match e {
                VadEvent::SpeechEnd { audio, .. } => Some(audio.len()),
                _ => None,
            })
            .expect("an utterance");
        // 1000 ms of speech is 16000 samples; pre-roll must add to that.
        assert!(end > 16_000, "no pre-roll: {end} samples");
    }

    #[test]
    fn a_short_pause_does_not_split_an_utterance() {
        let cfg = VadConfig {
            silence_ms: 700,
            ..Default::default()
        };
        let mut h = Harness::new(cfg);
        h.feed(&silence(500));
        let mut events = h.feed(&speech(700, 0.4));
        events.extend(h.feed(&silence(300))); // mid-sentence pause
        events.extend(h.feed(&speech(700, 0.4)));
        events.extend(h.feed(&silence(1500)));

        let starts = events
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechStart { .. }))
            .count();
        let ends = events
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechEnd { .. }))
            .count();
        assert_eq!((starts, ends), (1, 1), "{events:?}");
    }

    #[test]
    fn a_long_pause_does_split_utterances() {
        let mut h = Harness::new(VadConfig::default());
        h.feed(&silence(500));
        let mut events = h.feed(&speech(700, 0.4));
        events.extend(h.feed(&silence(1400)));
        events.extend(h.feed(&speech(700, 0.4)));
        events.extend(h.feed(&silence(1400)));
        let ends = events
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechEnd { .. }))
            .count();
        assert_eq!(ends, 2, "{events:?}");
    }

    #[test]
    fn blips_are_discarded() {
        let cfg = VadConfig {
            min_utterance_ms: 600,
            speech_ms: 40,
            ..Default::default()
        };
        let mut h = Harness::new(cfg);
        h.feed(&silence(500));
        let mut events = h.feed(&speech(200, 0.6));
        events.extend(h.feed(&silence(1200)));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VadEvent::Discarded { .. })),
            "{events:?}"
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, VadEvent::SpeechEnd { .. })));
    }

    #[test]
    fn a_stuck_microphone_is_cut_off_at_the_ceiling() {
        let cfg = VadConfig {
            max_utterance_ms: 1000,
            ..Default::default()
        };
        let mut h = Harness::new(cfg);
        h.feed(&silence(400));
        let events = h.feed(&speech(4000, 0.5));
        assert!(
            events.iter().any(|e| matches!(
                e,
                VadEvent::SpeechEnd {
                    truncated: true,
                    ..
                }
            )),
            "{events:?}"
        );
    }

    #[test]
    fn flush_closes_an_open_utterance() {
        let mut h = Harness::new(VadConfig::default());
        h.feed(&silence(400));
        h.feed(&speech(900, 0.4));
        assert!(h.vad.is_speaking());
        assert!(matches!(h.vad.flush(), Some(VadEvent::SpeechEnd { .. })));
        assert!(!h.vad.is_speaking());
        assert!(h.vad.flush().is_none());
    }

    #[test]
    fn disabled_vad_is_inert() {
        let cfg = VadConfig {
            enabled: false,
            ..Default::default()
        };
        let mut h = Harness::new(cfg);
        let events = h.feed(&speech(2000, 0.8));
        assert!(events.is_empty());
    }
}
