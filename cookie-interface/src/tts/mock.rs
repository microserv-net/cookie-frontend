//! A synthesiser that needs no model at all.
//!
//! It does not produce words — it produces *speech-shaped* audio: a pitched
//! source with two formants, an amplitude envelope driven by the syllables of
//! the real text, and pauses at punctuation. That matters more than it
//! sounds: the orb, the VAD ducking, the barge-in logic and the whole
//! streaming path are exercised end to end without downloading a gigabyte.

use std::time::Duration;

use crate::audio::AudioBuffer;
use crate::config::{TtsConfig, VoiceSpec};
use crate::error::Result;
use crate::util::{BoxFuture, Rng};

use super::{SpeechSynthesizer, SynthesisRequest, SynthesisStream, TtsCapabilities, VoiceInfo};

const SAMPLE_RATE: u32 = 24_000;
/// Chunk length; also the pacing quantum when `realtime` is on.
const CHUNK_MS: u64 = 60;

/// Deterministic synthetic voice.
#[derive(Debug, Clone)]
pub struct MockSynthesizer {
    /// Base pitch in Hz. The default sits in a female alto range.
    pitch_hz: f32,
    /// Emit chunks at wall-clock speed instead of as fast as possible.
    realtime: bool,
}

impl Default for MockSynthesizer {
    fn default() -> Self {
        Self {
            pitch_hz: 196.0,
            realtime: false,
        }
    }
}

impl MockSynthesizer {
    /// Build from configuration; `tts.options.realtime = true` paces output.
    pub fn from_config(cfg: &TtsConfig) -> Self {
        Self {
            pitch_hz: cfg
                .options
                .get("pitch_hz")
                .and_then(|v| v.as_float())
                .unwrap_or(196.0) as f32,
            realtime: cfg
                .options
                .get("realtime")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        }
    }

    /// Pace chunk delivery to wall-clock time (used by `--test`).
    pub fn realtime(mut self, on: bool) -> Self {
        self.realtime = on;
        self
    }

    /// Render `text` into one buffer. Exposed for tests and for the
    /// `--doctor` audio check.
    pub fn render(&self, text: &str, voice: &VoiceSpec) -> AudioBuffer {
        let rate = voice.rate.clamp(0.5, 2.0);
        // `VoiceSpec::pitch` is a semitone offset, not a multiplier.
        let pitch = self.pitch_hz * 2f32.powf(voice.pitch.clamp(-24.0, 24.0) / 12.0);
        // A stable seed per utterance keeps the waveform reproducible, which
        // is what lets the animation tests assert on audio features.
        let mut rng = Rng::new(fnv(text));
        let mut samples = Vec::new();
        let sr = SAMPLE_RATE as f32;
        let mut phase = 0.0f32;

        for token in tokenize(text) {
            match token {
                Token::Pause(ms) => {
                    let n = (sr * (ms as f32 / 1000.0) / rate) as usize;
                    samples.extend(std::iter::repeat(0.0).take(n));
                }
                Token::Syllable { len_ms, voiced } => {
                    let n = ((sr * (len_ms as f32 / 1000.0)) / rate).max(1.0) as usize;
                    let f1 = 500.0 + rng.f32() * 400.0;
                    let f2 = 1200.0 + rng.f32() * 900.0;
                    let target = 0.35 + rng.f32() * 0.25;
                    for i in 0..n {
                        let t = i as f32 / n as f32;
                        // Fast attack, gentle decay: consonant-ish onsets.
                        let env = if t < 0.12 {
                            t / 0.12
                        } else {
                            (1.0 - (t - 0.12) / 0.88).powf(0.6)
                        };
                        let s = if voiced {
                            phase += pitch / sr;
                            if phase >= 1.0 {
                                phase -= 1.0;
                            }
                            // Sawtooth-ish glottal source shaped by formants.
                            let src = 2.0 * phase - 1.0;
                            let a = (std::f32::consts::TAU * f1 * i as f32 / sr).sin();
                            let b = (std::f32::consts::TAU * f2 * i as f32 / sr).sin();
                            0.55 * src * (0.6 + 0.4 * a) + 0.18 * b
                        } else {
                            // Fricative: noise, no pitch.
                            (rng.f32() - 0.5) * 1.6
                        };
                        samples.push(s * env * target * voice.volume.clamp(0.0, 1.5));
                    }
                }
            }
        }
        AudioBuffer::mono(samples, SAMPLE_RATE)
    }
}

enum Token {
    Syllable { len_ms: u64, voiced: bool },
    Pause(u64),
}

/// Crude but stable syllabification: enough to make the envelope look like
/// speech rather than a buzzer.
fn tokenize(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        let letters: Vec<char> = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if letters.is_empty() {
            continue;
        }
        let vowels = letters
            .iter()
            .filter(|c| "aeiouyAEIOUY".contains(**c))
            .count()
            .max(1);
        for i in 0..vowels {
            out.push(Token::Syllable {
                len_ms: 120 + (letters.len().min(10) as u64) * 8,
                voiced: true,
            });
            if i + 1 < vowels {
                out.push(Token::Pause(10));
            }
        }
        if letters.iter().any(|c| "sfzxSFZX".contains(*c)) {
            out.push(Token::Syllable {
                len_ms: 60,
                voiced: false,
            });
        }
        let tail = word.chars().last().unwrap_or(' ');
        out.push(Token::Pause(match tail {
            '.' | '!' | '?' => 320,
            ',' | ';' | ':' => 180,
            _ => 70,
        }));
    }
    out
}

fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

impl SpeechSynthesizer for MockSynthesizer {
    fn name(&self) -> String {
        "mock".to_string()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities {
            streaming: true,
            rate: true,
            pitch: true,
            style: false,
            voice_listing: true,
            sample_rate: SAMPLE_RATE,
            extra_parameters: vec!["pitch_hz".into(), "realtime".into()],
        }
    }

    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>> {
        Box::pin(async move {
            let audio = self.render(&request.text, &request.voice);
            let chunk = (SAMPLE_RATE as u64 * CHUNK_MS / 1000) as usize;
            let (tx, stream) = SynthesisStream::channel(SAMPLE_RATE, 8);
            let realtime = self.realtime;
            tokio::spawn(async move {
                for part in audio.samples.chunks(chunk.max(1)) {
                    if realtime {
                        tokio::time::sleep(Duration::from_millis(CHUNK_MS)).await;
                    }
                    // A closed receiver means the utterance was interrupted.
                    if tx
                        .send(Ok(AudioBuffer::mono(part.to_vec(), SAMPLE_RATE)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            });
            Ok(stream)
        })
    }

    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async {
            Ok(vec![VoiceInfo {
                id: "mock-female-en-gb".into(),
                name: Some("Mock (British female)".into()),
                language: Some("en-GB".into()),
                gender: Some("female".into()),
            }])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_is_deterministic() {
        let s = MockSynthesizer::default();
        let v = VoiceSpec::default();
        assert_eq!(s.render("hello there", &v), s.render("hello there", &v));
    }

    #[test]
    fn longer_text_makes_longer_audio() {
        let s = MockSynthesizer::default();
        let v = VoiceSpec::default();
        let a = s.render("hello", &v).frames();
        let b = s.render("hello there, how are you today?", &v).frames();
        assert!(b > a * 2, "{a} vs {b}");
    }

    #[test]
    fn rate_shortens_audio() {
        let s = MockSynthesizer::default();
        let slow = s.render("hello there", &VoiceSpec::default()).frames();
        let fast = s
            .render(
                "hello there",
                &VoiceSpec {
                    rate: 2.0,
                    ..Default::default()
                },
            )
            .frames();
        assert!(fast < slow);
    }

    #[tokio::test]
    async fn stream_delivers_the_whole_utterance() {
        let s = MockSynthesizer::default();
        let req = SynthesisRequest::new("good evening", VoiceSpec::default());
        let expected = s.render(&req.text, &req.voice).frames();
        let stream = s.synthesize(req).await.unwrap();
        assert_eq!(stream.collect().await.unwrap().frames(), expected);
    }

    #[tokio::test]
    async fn dropping_the_stream_stops_synthesis() {
        let s = MockSynthesizer::default().realtime(true);
        let stream = s
            .synthesize(SynthesisRequest::new(
                "a fairly long sentence that would take a while to speak aloud",
                VoiceSpec::default(),
            ))
            .await
            .unwrap();
        drop(stream); // barge-in
        tokio::time::sleep(Duration::from_millis(150)).await;
        // Nothing to assert beyond "we did not hang or panic"; the spawned
        // task observes the closed channel and returns.
    }

    #[test]
    fn empty_text_produces_no_audio() {
        let s = MockSynthesizer::default();
        assert!(s.render("   ", &VoiceSpec::default()).is_empty());
    }
}
