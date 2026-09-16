//! Speech synthesis.
//!
//! Everything above this module talks to [`SpeechSynthesizer`] and never to a
//! model. Swapping Qwen3-TTS for Kokoro, for the operating system voice, or
//! for something that does not exist yet is a four-line configuration change.
//!
//! ## Streaming is the default, not an extra
//!
//! Synthesis returns a [`SynthesisStream`] rather than a finished buffer.
//! Providers that can only produce whole utterances simply push one chunk;
//! providers that stream push as they go and the orb starts moving hundreds of
//! milliseconds earlier. The engine feeds every chunk to the speaker *and* to
//! the audio analyser, which is what makes the visual reaction track the
//! actual voice rather than a guess.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::audio::AudioBuffer;
use crate::config::{TtsConfig, TtsProviderKind, VoiceSpec};
use crate::error::{Error, Result};
use crate::util::BoxFuture;

#[cfg(feature = "http-providers")]
pub mod http;
pub mod local;
pub mod mock;
pub mod sidecar;
pub mod system;

#[cfg(feature = "http-providers")]
pub use http::HttpSynthesizer;
pub use local::LocalSynthesizer;
pub use mock::MockSynthesizer;
pub use sidecar::SidecarSynthesizer;
pub use system::SystemSynthesizer;

/// One synthesis job.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Identifier that follows the utterance through events and the API.
    pub utterance_id: String,
    /// The text to speak. Already chunked by the engine for streaming input.
    pub text: String,
    /// Fully resolved voice (configuration defaults overlaid with per-request
    /// overrides).
    pub voice: VoiceSpec,
}

impl SynthesisRequest {
    /// Convenience for tests and the `--test` flow.
    pub fn new(text: impl Into<String>, voice: VoiceSpec) -> Self {
        Self {
            utterance_id: uuid::Uuid::new_v4().to_string(),
            text: text.into(),
            voice,
        }
    }
}

/// What a synthesiser actually honours.
///
/// This is reported verbatim by `GET /v1/state` so clients never send a
/// parameter into the void. Nothing here is aspirational: a provider that
/// ignores `pitch` must report `pitch: false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsCapabilities {
    /// Audio arrives in chunks while synthesis is still running.
    pub streaming: bool,
    /// `voice.rate` is applied.
    pub rate: bool,
    /// `voice.pitch` is applied.
    pub pitch: bool,
    /// `voice.style` / emotion is applied.
    pub style: bool,
    /// Provider can enumerate its voices.
    pub voice_listing: bool,
    /// Native output sample rate, in Hz.
    pub sample_rate: u32,
    /// Provider-specific keys accepted in `voice.extra`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_parameters: Vec<String>,
}

impl Default for TtsCapabilities {
    fn default() -> Self {
        Self {
            streaming: false,
            rate: false,
            pitch: false,
            style: false,
            voice_listing: false,
            sample_rate: 24_000,
            extra_parameters: Vec::new(),
        }
    }
}

/// A voice a provider can offer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
}

/// Audio produced by a synthesiser, chunk by chunk.
///
/// Dropping the stream cancels the underlying job — that is exactly what
/// barge-in does.
#[derive(Debug)]
pub struct SynthesisStream {
    sample_rate: u32,
    rx: mpsc::Receiver<Result<AudioBuffer>>,
}

impl SynthesisStream {
    /// Create a stream plus the sender a provider pushes into.
    ///
    /// The channel is bounded: a provider that outruns playback is made to
    /// wait rather than allowed to buffer an unbounded amount of audio.
    pub fn channel(sample_rate: u32, capacity: usize) -> (mpsc::Sender<Result<AudioBuffer>>, Self) {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        (tx, Self { sample_rate, rx })
    }

    /// A stream that yields one already-synthesised buffer.
    pub fn once(audio: AudioBuffer) -> Self {
        let (tx, rx) = mpsc::channel(1);
        let rate = audio.sample_rate;
        tx.try_send(Ok(audio)).expect("capacity 1");
        Self {
            sample_rate: rate,
            rx,
        }
    }

    /// Nominal sample rate of the chunks. Individual chunks carry their own
    /// rate too; the playback handle resamples per chunk regardless.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Next chunk, or `None` when synthesis is complete.
    pub async fn next_chunk(&mut self) -> Option<Result<AudioBuffer>> {
        self.rx.recv().await
    }

    /// Drain the whole stream into one buffer. Used when a client asked for a
    /// complete file rather than a live stream.
    pub async fn collect(mut self) -> Result<AudioBuffer> {
        let mut out: Option<AudioBuffer> = None;
        while let Some(chunk) = self.next_chunk().await {
            let chunk = chunk?;
            match &mut out {
                Some(buf) => {
                    let chunk = if chunk.sample_rate == buf.sample_rate {
                        chunk
                    } else {
                        chunk.resampled(buf.sample_rate)
                    };
                    buf.samples.extend_from_slice(&chunk.samples);
                }
                None => out = Some(chunk),
            }
        }
        Ok(out.unwrap_or_else(|| AudioBuffer::mono(Vec::new(), self.sample_rate)))
    }
}

/// Turns text into audio.
pub trait SpeechSynthesizer: Send + Sync + 'static {
    /// Short stable identifier, e.g. `"http:qwen3-tts-flash"`.
    fn name(&self) -> String;

    /// What this provider honours. Never optimistic.
    fn capabilities(&self) -> TtsCapabilities;

    /// Reach / warm the model. Non-fatal on failure: the engine reports a
    /// `provider.status` event and stays mute rather than exiting.
    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Begin synthesising. Returns as soon as the job is accepted, not when
    /// the audio is finished.
    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>>;

    /// Voices this provider can offer, when it can enumerate them.
    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

/// Shared handle used throughout the engine.
pub type SharedSynthesizer = Arc<dyn SpeechSynthesizer>;

/// Build the synthesiser described by configuration.
pub fn build(cfg: &TtsConfig) -> Result<SharedSynthesizer> {
    match cfg.provider {
        TtsProviderKind::Local => Ok(Arc::new(LocalSynthesizer::from_config(cfg)?)),
        TtsProviderKind::Mock => Ok(Arc::new(MockSynthesizer::from_config(cfg))),
        TtsProviderKind::System => Ok(Arc::new(SystemSynthesizer::from_config(cfg))),
        TtsProviderKind::Sidecar => Ok(Arc::new(SidecarSynthesizer::from_config(cfg)?)),
        #[cfg(feature = "http-providers")]
        TtsProviderKind::Http => Ok(Arc::new(HttpSynthesizer::from_config(cfg)?)),
        #[cfg(not(feature = "http-providers"))]
        TtsProviderKind::Http => Err(Error::ProviderNotCompiled {
            provider: "tts:http".into(),
            feature: "http-providers",
        }),
    }
}

/// Build the configured synthesiser, falling back to one that always works.
///
/// A missing model must not leave the user staring at a silent orb with no
/// explanation, so the fallback chain is: configured → operating system →
/// mock. The chosen provider is logged and announced through a
/// `provider.status` event.
pub fn build_with_fallback(cfg: &TtsConfig) -> (SharedSynthesizer, Option<Error>) {
    match build(cfg) {
        Ok(s) => (s, None),
        Err(e) => {
            tracing::warn!("falling back to the system voice: {e}");
            let fallback = SystemSynthesizer::from_config(cfg);
            if fallback.is_available() {
                (Arc::new(fallback), Some(e))
            } else {
                (Arc::new(MockSynthesizer::from_config(cfg)), Some(e))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn once_yields_exactly_one_chunk() {
        let mut s = SynthesisStream::once(AudioBuffer::silent(20, 24_000));
        assert!(s.next_chunk().await.is_some());
        assert!(s.next_chunk().await.is_none());
    }

    #[tokio::test]
    async fn collect_concatenates_and_resamples() {
        let (tx, stream) = SynthesisStream::channel(24_000, 4);
        tx.send(Ok(AudioBuffer::silent(10, 24_000))).await.unwrap();
        tx.send(Ok(AudioBuffer::silent(10, 16_000))).await.unwrap();
        drop(tx);
        let out = stream.collect().await.unwrap();
        assert_eq!(out.sample_rate, 24_000);
        // 10ms at 24k plus 10ms resampled from 16k to 24k, within a sample.
        assert!((out.frames() as i64 - 480).abs() <= 4, "{}", out.frames());
    }

    #[tokio::test]
    async fn errors_propagate_out_of_collect() {
        let (tx, stream) = SynthesisStream::channel(24_000, 2);
        tx.send(Err(Error::Tts("boom".into()))).await.unwrap();
        drop(tx);
        assert!(stream.collect().await.is_err());
    }

    #[test]
    fn mock_and_system_providers_build() {
        let mut cfg = TtsConfig::default();
        cfg.provider = TtsProviderKind::Mock;
        assert!(build(&cfg).is_ok());
        cfg.provider = TtsProviderKind::System;
        assert!(build(&cfg).is_ok());
    }

    #[test]
    fn fallback_never_returns_nothing() {
        let mut cfg = TtsConfig::default();
        cfg.provider = TtsProviderKind::Sidecar;
        cfg.sidecar_command.clear();
        let (provider, err) = build_with_fallback(&cfg);
        assert!(err.is_some());
        assert!(!provider.name().is_empty());
    }
}
