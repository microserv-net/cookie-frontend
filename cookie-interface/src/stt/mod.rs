//! Speech recognition.
//!
//! The rest of the crate only ever sees [`SpeechRecognizer`]. Whether the
//! words come from a local `whisper-large-v3-turbo` behind a sidecar process,
//! from an OpenAI-compatible HTTP endpoint, or from the deterministic mock
//! used by the test-suite is a configuration detail.
//!
//! ## Why the trait looks like this
//!
//! Whisper-family models are *chunk* models, not true streaming models: they
//! transcribe a finished span of audio. Pretending otherwise in the trait
//! would force every provider to fake a streaming API. Instead the trait
//! exposes one honest operation — "turn this audio into text" — and the
//! engine produces partial results by re-transcribing the utterance-so-far at
//! an interval, which is exactly how streaming Whisper UIs work in practice.
//! A provider that dislikes that can advertise
//! [`SttCapabilities::native_partials`] and be fed incremental audio instead.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::audio::AudioBuffer;
use crate::config::{SttConfig, SttProviderKind};
use crate::error::Result;
use crate::events::TranscriptSegment;
use crate::util::BoxFuture;

#[cfg(feature = "http-providers")]
pub mod http;
pub mod local;
pub mod mock;
pub mod sidecar;

#[cfg(feature = "http-providers")]
pub use http::HttpRecognizer;
pub use local::LocalRecognizer;
pub use mock::MockRecognizer;
pub use sidecar::SidecarRecognizer;

/// What a recognizer produced for a span of audio.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    /// The recognised text, already trimmed.
    pub text: String,
    /// `false` for an interim guess that may still change.
    pub is_final: bool,
    /// Provider confidence in `0..=1`, when the provider reports one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Detected (or configured) BCP-47 language tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Word or phrase level timings, when the provider reports them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<TranscriptSegment>,
    /// Duration of the audio that produced this transcript.
    pub audio_ms: u64,
    /// Wall-clock time the provider took. Useful for `--doctor`.
    pub latency_ms: u64,
}

impl Transcript {
    /// A final transcript carrying nothing but text; used by simple providers.
    pub fn final_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into().trim().to_string(),
            is_final: true,
            ..Default::default()
        }
    }

    /// True when the provider returned nothing usable (silence, noise).
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Per-request recognition hints.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// BCP-47 language tag, or `None` to let the model detect it.
    pub language: Option<String>,
    /// Domain words / names the model should be biased towards.
    pub prompt: Option<String>,
    /// Ask for word timings. Providers that cannot do it simply omit them.
    pub want_timestamps: bool,
    /// `true` when the engine wants a quick interim answer rather than the
    /// best possible one; providers may trade accuracy for latency.
    pub interim: bool,
}

impl TranscribeOptions {
    /// Options derived from configuration.
    pub fn from_config(cfg: &SttConfig) -> Self {
        Self {
            language: if cfg.language.eq_ignore_ascii_case("auto") {
                None
            } else {
                Some(cfg.language.clone())
            },
            prompt: cfg
                .options
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            want_timestamps: cfg
                .options
                .get("timestamps")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            interim: false,
        }
    }

    /// The same options, marked as a low-latency interim pass.
    pub fn interim(mut self) -> Self {
        self.interim = true;
        self.want_timestamps = false;
        self
    }
}

/// What a recognizer can actually do.
///
/// The API surfaces this verbatim on `GET /v1/state` so a client never has to
/// guess whether asking for timestamps will do anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttCapabilities {
    /// Provider can return word/phrase timings.
    pub timestamps: bool,
    /// Provider reports a confidence score.
    pub confidence: bool,
    /// Provider detects the language on its own.
    pub language_detection: bool,
    /// Provider accepts an initial prompt / bias list.
    pub prompt: bool,
    /// Provider consumes audio incrementally and emits its own partials.
    /// When `false` the engine synthesises partials by re-running the model.
    pub native_partials: bool,
    /// Sample rate the provider wants, in Hz.
    pub sample_rate: u32,
}

impl Default for SttCapabilities {
    fn default() -> Self {
        Self {
            timestamps: false,
            confidence: false,
            language_detection: false,
            prompt: false,
            native_partials: false,
            sample_rate: 16_000,
        }
    }
}

/// Turns audio into text.
///
/// Implementations must be cheap to clone-by-`Arc` and safe to call from
/// several tasks at once; the engine may run an interim pass and a final pass
/// concurrently.
pub trait SpeechRecognizer: Send + Sync + 'static {
    /// Short stable identifier, e.g. `"http:whisper-large-v3-turbo"`.
    fn name(&self) -> String;

    /// What this provider honours.
    fn capabilities(&self) -> SttCapabilities;

    /// Load / warm / reach the model. Called once at startup, off the UI
    /// thread. Failing here is not fatal: the engine degrades to "deaf" and
    /// says so through a `provider.status` event.
    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Transcribe one span of audio.
    fn transcribe<'a>(
        &'a self,
        audio: AudioBuffer,
        options: TranscribeOptions,
    ) -> BoxFuture<'a, Result<Transcript>>;
}

impl fmt::Debug for dyn SpeechRecognizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeechRecognizer")
            .field("name", &self.name())
            .finish()
    }
}

/// Shared handle used throughout the engine.
pub type SharedRecognizer = Arc<dyn SpeechRecognizer>;

/// Build the recognizer described by configuration.
pub fn build(cfg: &SttConfig) -> Result<SharedRecognizer> {
    match cfg.provider {
        SttProviderKind::Local => Ok(Arc::new(LocalRecognizer::from_config(cfg)?)),
        SttProviderKind::Mock => Ok(Arc::new(MockRecognizer::from_config(cfg))),
        SttProviderKind::Sidecar => Ok(Arc::new(SidecarRecognizer::from_config(cfg)?)),
        #[cfg(feature = "http-providers")]
        SttProviderKind::Http => Ok(Arc::new(HttpRecognizer::from_config(cfg)?)),
        #[cfg(not(feature = "http-providers"))]
        SttProviderKind::Http => Err(crate::error::Error::ProviderNotCompiled {
            provider: "stt:http".into(),
            feature: "http-providers",
        }),
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn final_text_trims() {
        let t = Transcript::final_text("  hello there\n");
        assert_eq!(t.text, "hello there");
        assert!(t.is_final);
        assert!(!t.is_empty());
    }

    #[test]
    fn empty_transcript_detected() {
        assert!(Transcript::final_text("   ").is_empty());
    }

    #[test]
    fn auto_language_becomes_none() {
        let mut cfg = SttConfig::default();
        cfg.language = "auto".into();
        assert!(TranscribeOptions::from_config(&cfg).language.is_none());
        cfg.language = "en-GB".into();
        assert_eq!(
            TranscribeOptions::from_config(&cfg).language.as_deref(),
            Some("en-GB")
        );
    }

    #[test]
    fn interim_drops_timestamps() {
        let o = TranscribeOptions {
            want_timestamps: true,
            ..Default::default()
        }
        .interim();
        assert!(o.interim && !o.want_timestamps);
    }

    #[test]
    fn mock_provider_builds() {
        let cfg = SttConfig {
            provider: SttProviderKind::Mock,
            ..Default::default()
        };
        assert!(build(&cfg).is_ok());
    }
}
