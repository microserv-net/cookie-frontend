//! Apple's on-device speech recogniser.
//!
//! On macOS this is simply the better tool, and the numbers are not close.
//! Whisper large-v3-turbo through onnxruntime costs eleven seconds for a
//! two-second sentence on an M-series laptop, because Whisper pads every
//! utterance to thirty seconds and the quantised graph cannot use the Neural
//! Engine. Apple's recogniser is already installed, already on-device,
//! already tuned for this exact hardware, and answers in a fraction of a
//! second.
//!
//! It also makes partials worth having. Re-running Whisper mid-sentence costs
//! more than the interim text is worth; re-running this one does not, so
//! [`SttCapabilities::cheap_partials`] is true and words appear while you are
//! still speaking.
//!
//! The recogniser itself is an Objective-C API, so it lives in a small Swift
//! helper (`native/CookieSpeech.swift`) that `--setup` compiles and this
//! drives over the JSON-lines protocol in [`crate::sidecar`]. Rust bindings
//! would mean unsafe blocks in a crate that forbids them, for no benefit: the
//! boundary is a pipe either way.
//!
//! Whisper remains configured and remains the provider everywhere else — this
//! is a macOS-only substitution, not a replacement.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};

use crate::audio::AudioBuffer;
use crate::config::SttConfig;
use crate::error::{Error, Result};
use crate::sidecar::SidecarProcess;
use crate::util::BoxFuture;

use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// Recogniser backed by the Swift helper.
#[derive(Debug)]
pub struct AppleRecognizer {
    process: SidecarProcess,
    helper: PathBuf,
    locale: String,
    timeout_ms: u64,
}

impl AppleRecognizer {
    /// Build from configuration. Nothing is spawned yet.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        if !cfg!(target_os = "macos") {
            return Err(Error::Config(
                "stt.provider = \"apple\" only works on macOS. \
                 Use \"local\" for Whisper."
                    .into(),
            ));
        }
        let helper = cfg
            .options
            .get("helper")
            .and_then(|value| value.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(default_helper_path);
        if !helper.exists() {
            return Err(Error::ModelUnavailable {
                name: "apple-on-device".into(),
                reason: format!(
                    "the speech helper is not built ({}). Run `cookie-interface --setup`.",
                    helper.display()
                ),
            });
        }
        let locale = if cfg.language.eq_ignore_ascii_case("auto") {
            "en-GB".to_string()
        } else {
            cfg.language.clone()
        };
        Ok(Self {
            process: SidecarProcess::new("stt", vec![helper.display().to_string()])?,
            helper,
            locale,
            // Apple's recogniser answers in well under a second; anything
            // near this means something is wrong rather than slow.
            timeout_ms: cfg.timeout_ms.clamp(5_000, 20_000),
        })
    }

    /// Where `--setup` puts the compiled helper.
    pub fn helper_path(&self) -> &std::path::Path {
        &self.helper
    }
}

/// The helper lives beside the models, so `--clear-cache` does not remove it
/// and a rebuild replaces it in place.
pub fn default_helper_path() -> PathBuf {
    crate::paths::Paths::discover()
        .map(|paths| paths.models_dir().join("cookie-speech"))
        .unwrap_or_else(|_| PathBuf::from("cookie-speech"))
}

impl SpeechRecognizer for AppleRecognizer {
    fn name(&self) -> String {
        format!("apple:{}", self.locale)
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            timestamps: false,
            confidence: false,
            language_detection: false,
            prompt: false,
            native_partials: false,
            // The whole point: it is quick enough to re-run mid-sentence, so
            // words appear while you are still speaking.
            cheap_partials: true,
            sample_rate: 16_000,
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            // This is what triggers the permission dialogue, so it happens at
            // startup rather than in the middle of the first thing anybody
            // says.
            let info = self.process.hello().await?;
            let on_device = info
                .get("on_device")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !on_device {
                // Worth saying out loud: without an on-device model the audio
                // would go to Apple's servers, which is not a thing to do by
                // accident.
                tracing::warn!(
                    locale = %self.locale,
                    "this locale has no on-device model installed, so recognition \
                     would not be local. Install it in System Settings, or use \
                     stt.provider = \"local\"."
                );
            }
            tracing::info!(locale = %self.locale, on_device, "apple recogniser ready");
            Ok(())
        })
    }

    fn transcribe<'a>(
        &'a self,
        audio: AudioBuffer,
        options: TranscribeOptions,
    ) -> BoxFuture<'a, Result<Transcript>> {
        Box::pin(async move {
            let started = Instant::now();
            let audio_ms = audio.duration_ms();
            let audio = if audio.sample_rate == 16_000 {
                audio
            } else {
                audio.resampled(16_000)
            };

            let request = json!({
                "op": "transcribe",
                "sample_rate": audio.sample_rate,
                "audio": crate::stt::sidecar::encode_f32(&audio.samples),
                "partial": options.interim,
            });
            let value = tokio::time::timeout(
                std::time::Duration::from_millis(self.timeout_ms),
                self.process.call(request),
            )
            .await
            .map_err(|_| {
                Error::Stt(format!(
                    "the recogniser did not answer within {}ms",
                    self.timeout_ms
                ))
            })??;

            Ok(Transcript {
                is_final: !options.interim,
                confidence: None,
                language: Some(self.locale.clone()),
                segments: Vec::new(),
                audio_ms,
                latency_ms: started.elapsed().as_millis() as u64,
                text: crate::parsing_text(&value),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SttProviderKind;

    fn config(helper: &std::path::Path) -> SttConfig {
        let mut cfg = SttConfig {
            provider: SttProviderKind::Apple,
            ..Default::default()
        };
        cfg.options.insert(
            "helper".into(),
            toml::Value::String(helper.display().to_string()),
        );
        cfg
    }

    #[test]
    fn a_missing_helper_says_what_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let error = AppleRecognizer::from_config(&config(&dir.path().join("nope"))).unwrap_err();
        if cfg!(target_os = "macos") {
            assert_eq!(error.code(), "model_unavailable");
            assert!(error.to_string().contains("--setup"), "{error}");
        } else {
            // Everywhere else the honest answer is that this provider does
            // not apply at all.
            assert_eq!(error.code(), "config_error");
            assert!(error.to_string().contains("macOS"), "{error}");
        }
    }

    #[test]
    fn capabilities_say_partials_are_worth_having() {
        let dir = tempfile::tempdir().unwrap();
        let helper = dir.path().join("cookie-speech");
        std::fs::write(&helper, b"").unwrap();
        let Ok(recognizer) = AppleRecognizer::from_config(&config(&helper)) else {
            // Not macOS; the construction test above covers that path.
            return;
        };
        let capabilities = recognizer.capabilities();
        assert!(
            capabilities.cheap_partials,
            "this is the provider partials exist for"
        );
        assert_eq!(capabilities.sample_rate, 16_000);
        assert!(recognizer.name().starts_with("apple:"));
    }

    #[test]
    fn the_helper_lives_with_the_models_not_the_cache() {
        // `--clear-cache` must not delete something that takes a compiler to
        // rebuild.
        let path = default_helper_path();
        assert!(path.ends_with("cookie-speech"), "{}", path.display());
    }
}
