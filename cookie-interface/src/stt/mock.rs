//! Deterministic in-process recogniser.
//!
//! It never loads a model and never touches the network, which is what makes
//! `cargo test` runnable on a machine with no microphone and no GPU. It is
//! also genuinely useful at runtime: `--test` falls back to it so a first-time
//! user can watch the whole pipeline light up before downloading anything.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::audio::AudioBuffer;
use crate::config::SttConfig;
use crate::error::Result;
use crate::util::{BoxFuture, Rng};

use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// Phrases handed out when no script is queued. Chosen so `--test` produces a
/// believable "name" answer.
const FALLBACK: &[&str] = &[
    "My name is Alex",
    "I'm Robin",
    "This is Sam",
    "Call me Jordan",
];

/// A recogniser that returns scripted text instead of running a model.
#[derive(Debug)]
pub struct MockRecognizer {
    script: Mutex<VecDeque<String>>,
    rng: Mutex<Rng>,
    /// Simulated model latency, so timing-sensitive code is exercised.
    latency_ms: u64,
    label: String,
}

impl Default for MockRecognizer {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl MockRecognizer {
    /// A recogniser that plays back `script` and then falls back to a stable
    /// rotation of canned phrases.
    pub fn new(script: Vec<String>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().collect()),
            rng: Mutex::new(Rng::new(0xC00C1E)),
            latency_ms: 40,
            label: "mock".to_string(),
        }
    }

    /// Build from configuration; `stt.options.script` may hold an array of
    /// strings to replay.
    pub fn from_config(cfg: &SttConfig) -> Self {
        let script = cfg
            .options
            .get("script")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let mut me = Self::new(script);
        me.latency_ms = cfg
            .options
            .get("latency_ms")
            .and_then(|v| v.as_integer())
            .unwrap_or(40)
            .clamp(0, 5_000) as u64;
        me
    }

    /// Queue another line for the next call. Used by `--test` and by tests.
    pub fn push(&self, line: impl Into<String>) {
        self.script.lock().unwrap().push_back(line.into());
    }

    fn next_text(&self, audio: &AudioBuffer) -> String {
        if let Some(next) = self.script.lock().unwrap().pop_front() {
            return next;
        }
        // Silence in, silence out: the engine relies on empty transcripts to
        // decide that an utterance was noise.
        if audio.rms() < 1e-4 {
            return String::new();
        }
        let idx = self.rng.lock().unwrap().below(FALLBACK.len());
        FALLBACK[idx].to_string()
    }
}

impl SpeechRecognizer for MockRecognizer {
    fn name(&self) -> String {
        format!("mock:{}", self.label)
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            confidence: true,
            native_partials: false,
            ..Default::default()
        }
    }

    fn transcribe<'a>(
        &'a self,
        audio: AudioBuffer,
        options: TranscribeOptions,
    ) -> BoxFuture<'a, Result<Transcript>> {
        Box::pin(async move {
            let started = Instant::now();
            if self.latency_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
            }
            let mut text = self.next_text(&audio);
            if options.interim {
                // An interim guess is deliberately a prefix of the final one:
                // that is what real streaming recognition looks like and it
                // stops UI code from assuming partials are stable.
                let words: Vec<&str> = text.split_whitespace().collect();
                let keep = words.len().saturating_sub(1).max(1);
                text = words[..keep.min(words.len())].join(" ");
            }
            Ok(Transcript {
                is_final: !options.interim,
                confidence: Some(if options.interim { 0.55 } else { 0.94 }),
                language: options.language.clone().or_else(|| Some("en".into())),
                audio_ms: audio.duration_ms(),
                latency_ms: started.elapsed().as_millis() as u64,
                text,
                segments: Vec::new(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(ms: u64) -> AudioBuffer {
        let n = (16_000 * ms / 1000) as usize;
        AudioBuffer::mono(
            (0..n).map(|i| ((i as f32) * 0.01).sin() * 0.3).collect(),
            16_000,
        )
    }

    #[tokio::test]
    async fn script_is_replayed_in_order() {
        let r = MockRecognizer::new(vec!["one".into(), "two".into()]);
        let a = r
            .transcribe(noisy(100), TranscribeOptions::default())
            .await
            .unwrap();
        let b = r
            .transcribe(noisy(100), TranscribeOptions::default())
            .await
            .unwrap();
        assert_eq!(a.text, "one");
        assert_eq!(b.text, "two");
        assert!(a.is_final);
    }

    #[tokio::test]
    async fn silence_transcribes_to_nothing() {
        let r = MockRecognizer::new(Vec::new());
        let t = r
            .transcribe(
                AudioBuffer::silent(1600, 16_000),
                TranscribeOptions::default(),
            )
            .await
            .unwrap();
        assert!(t.is_empty());
    }

    #[tokio::test]
    async fn interim_is_marked_non_final() {
        let r = MockRecognizer::new(vec!["my name is alex".into()]);
        let t = r
            .transcribe(noisy(100), TranscribeOptions::default().interim())
            .await
            .unwrap();
        assert!(!t.is_final);
        assert!("my name is alex".starts_with(&t.text));
    }

    #[test]
    fn fallback_is_deterministic() {
        let a = MockRecognizer::new(Vec::new());
        let b = MockRecognizer::new(Vec::new());
        let audio = noisy(200);
        assert_eq!(a.next_text(&audio), b.next_text(&audio));
    }
}
