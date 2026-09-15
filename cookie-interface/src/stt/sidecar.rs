//! Speech recognition through a local model subprocess.
//!
//! This is the recommended way to run `whisper-large-v3-turbo` locally:
//! whisper.cpp, faster-whisper or sherpa-onnx live in their own process and
//! speak the JSON-lines protocol from [`crate::sidecar`].

use base64::Engine as _;
use serde_json::{json, Value};

use crate::audio::AudioBuffer;
use crate::config::SttConfig;
use crate::error::{Error, Result};
use crate::events::TranscriptSegment;
use crate::sidecar::SidecarProcess;
use crate::util::BoxFuture;

use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// Recogniser backed by a subprocess.
#[derive(Debug)]
pub struct SidecarRecognizer {
    process: SidecarProcess,
    model: String,
    timeout_ms: u64,
    extra: Value,
}

impl SidecarRecognizer {
    /// Build from configuration. Does not spawn anything yet.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        Ok(Self {
            process: SidecarProcess::new("stt", cfg.sidecar_command.clone())?,
            model: cfg.model.clone(),
            timeout_ms: cfg.timeout_ms,
            extra: options_to_json(cfg),
        })
    }
}

/// Pass provider-specific TOML options through as JSON, untouched.
pub(crate) fn options_to_json(cfg: &SttConfig) -> Value {
    let mut map = serde_json::Map::new();
    for (k, v) in &cfg.options {
        if let Ok(v) = serde_json::to_value(v) {
            map.insert(k.clone(), v);
        }
    }
    Value::Object(map)
}

impl SpeechRecognizer for SidecarRecognizer {
    fn name(&self) -> String {
        format!("sidecar:{}", self.model)
    }

    fn capabilities(&self) -> SttCapabilities {
        // A sidecar may do more than this, but capabilities must describe
        // what we can *rely* on. Timestamps and confidence are read from the
        // response when present and simply omitted when not.
        SttCapabilities {
            timestamps: true,
            confidence: true,
            language_detection: true,
            prompt: true,
            native_partials: false,
            cheap_partials: false,
            sample_rate: 16_000,
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let info = self.process.hello().await?;
            tracing::info!(target: "cookie::stt", program = %self.process.program(), ?info, "stt sidecar ready");
            Ok(())
        })
    }

    fn transcribe<'a>(
        &'a self,
        audio: AudioBuffer,
        options: TranscribeOptions,
    ) -> BoxFuture<'a, Result<Transcript>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let audio_ms = audio.duration_ms();
            let request = json!({
                "op": "transcribe",
                "model": self.model,
                "sample_rate": audio.sample_rate,
                "encoding": "f32le",
                "audio": encode_f32(&audio.samples),
                "language": options.language,
                "prompt": options.prompt,
                "timestamps": options.want_timestamps,
                "interim": options.interim,
                "options": self.extra,
            });
            let fut = self.process.call(request);
            let value = tokio::time::timeout(
                std::time::Duration::from_millis(self.timeout_ms.max(1)),
                fut,
            )
            .await
            .map_err(|_| {
                Error::Stt(format!(
                    "sidecar did not answer within {}ms",
                    self.timeout_ms
                ))
            })??;

            Ok(parse_transcript(
                &value,
                audio_ms,
                started.elapsed().as_millis() as u64,
                &options,
            ))
        })
    }
}

/// Base64 of little-endian `f32` samples — the one encoding every language
/// can produce without a WAV library.
pub(crate) fn encode_f32(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Shared response shape for the sidecar and HTTP recognisers.
pub(crate) fn parse_transcript(
    value: &Value,
    audio_ms: u64,
    latency_ms: u64,
    options: &TranscribeOptions,
) -> Transcript {
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let segments =
        value
            .get("segments")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| {
                        Some(TranscriptSegment {
                            text: s.get("text")?.as_str()?.to_string(),
                            start_ms: s.get("start_ms").and_then(Value::as_f64).or_else(|| {
                                s.get("start").and_then(Value::as_f64).map(|v| v * 1000.0)
                            })? as u64,
                            end_ms: s.get("end_ms").and_then(Value::as_f64).or_else(|| {
                                s.get("end").and_then(Value::as_f64).map(|v| v * 1000.0)
                            })? as u64,
                            confidence: s
                                .get("confidence")
                                .and_then(Value::as_f64)
                                .map(|v| v as f32),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
    Transcript {
        is_final: !options.interim,
        confidence: value
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        language: value
            .get("language")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| options.language.clone()),
        segments,
        audio_ms,
        latency_ms,
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_samples_as_le_f32() {
        let encoded = encode_f32(&[1.0, -1.0]);
        let raw = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        assert_eq!(raw.len(), 8);
        assert_eq!(f32::from_le_bytes(raw[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(raw[4..8].try_into().unwrap()), -1.0);
    }

    #[test]
    fn parses_segments_in_both_unit_conventions() {
        let v = json!({
            "text": " hello ",
            "confidence": 0.8,
            "segments": [
                {"text": "hello", "start_ms": 10, "end_ms": 400},
                {"text": "there", "start": 0.5, "end": 0.9, "confidence": 0.7}
            ]
        });
        let t = parse_transcript(&v, 900, 12, &TranscribeOptions::default());
        assert_eq!(t.text, "hello");
        assert_eq!(t.segments.len(), 2);
        assert_eq!(t.segments[1].start_ms, 500);
        assert_eq!(t.segments[1].confidence, Some(0.7));
        assert!(t.is_final);
    }

    #[test]
    fn missing_text_yields_empty_transcript() {
        let t = parse_transcript(&json!({}), 0, 0, &TranscribeOptions::default());
        assert!(t.is_empty());
    }
}
