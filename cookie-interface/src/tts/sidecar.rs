//! Speech synthesis through a local model subprocess.
//!
//! This is how Qwen3-TTS, Kokoro or any Python model is used without putting
//! Python inside `cargo build`. The sidecar may stream: every `chunk` line it
//! writes is forwarded to the speaker and the orb immediately.

use base64::Engine as _;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::audio::AudioBuffer;
use crate::config::{TtsConfig, VoiceSpec};
use crate::error::{Error, Result};
use crate::sidecar::SidecarProcess;
use crate::util::BoxFuture;

use super::{SpeechSynthesizer, SynthesisRequest, SynthesisStream, TtsCapabilities, VoiceInfo};

/// Synthesiser backed by a subprocess.
#[derive(Debug)]
pub struct SidecarSynthesizer {
    process: std::sync::Arc<SidecarProcess>,
    model: String,
    default_rate: u32,
}

impl SidecarSynthesizer {
    /// Build from configuration. Nothing is spawned until first use.
    pub fn from_config(cfg: &TtsConfig) -> Result<Self> {
        Ok(Self {
            process: std::sync::Arc::new(SidecarProcess::new("tts", cfg.sidecar_command.clone())?),
            model: cfg.model.clone(),
            default_rate: cfg
                .options
                .get("sample_rate")
                .and_then(|v| v.as_integer())
                .unwrap_or(24_000)
                .clamp(8_000, 192_000) as u32,
        })
    }
}

/// Serialise a voice into the JSON the protocol specifies.
pub(crate) fn voice_json(voice: &VoiceSpec) -> Value {
    let mut extra = serde_json::Map::new();
    for (k, v) in &voice.extra {
        if let Ok(v) = serde_json::to_value(v) {
            extra.insert(k.clone(), v);
        }
    }
    json!({
        "id": voice.id,
        "language": voice.language,
        "gender": format!("{:?}", voice.gender).to_lowercase(),
        "style": voice.style,
        "rate": voice.rate,
        "pitch": voice.pitch,
        "volume": voice.volume,
        "extra": Value::Object(extra),
    })
}

/// Decode one `chunk`/`result` line into audio.
pub(crate) fn decode_chunk(value: &Value, default_rate: u32) -> Result<AudioBuffer> {
    let rate = value
        .get("sample_rate")
        .and_then(Value::as_u64)
        .unwrap_or(default_rate as u64) as u32;
    let b64 = value
        .get("audio")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Tts("sidecar chunk has no `audio` field".into()))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| Error::Tts(format!("sidecar audio was not valid base64: {e}")))?;
    match value
        .get("encoding")
        .and_then(Value::as_str)
        .unwrap_or("f32le")
    {
        "f32le" => Ok(AudioBuffer::mono(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            rate,
        )),
        "s16le" | "pcm" => Ok(crate::audio::wav::decode_pcm_s16le(&bytes, rate, 1)),
        "wav" => crate::audio::wav::decode_wav(&bytes),
        other => Err(Error::UnsupportedAudioFormat(format!(
            "sidecar sent `{other}`; use f32le, s16le or wav"
        ))),
    }
}

impl SpeechSynthesizer for SidecarSynthesizer {
    fn name(&self) -> String {
        format!("sidecar:{}", self.model)
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities {
            streaming: true,
            rate: true,
            pitch: true,
            style: true,
            voice_listing: true,
            sample_rate: self.default_rate,
            extra_parameters: vec!["anything the sidecar documents".into()],
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let info = self.process.hello().await?;
            tracing::info!(target: "cookie::tts", program = %self.process.program(), ?info, "tts sidecar ready");
            Ok(())
        })
    }

    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>> {
        Box::pin(async move {
            let (tx, stream) = SynthesisStream::channel(self.default_rate, 8);
            let (raw_tx, mut raw_rx) = mpsc::channel::<Value>(8);
            let process = self.process.clone();
            let default_rate = self.default_rate;
            let payload = json!({
                "op": "synthesize",
                "model": self.model,
                "text": request.text,
                "voice": voice_json(&request.voice),
                "utterance_id": request.utterance_id,
            });

            // One task drives the sidecar; another converts and forwards.
            // Splitting them keeps the protocol reader free of audio work.
            tokio::spawn(async move {
                if let Err(e) = process.call_streaming(payload, raw_tx).await {
                    tracing::warn!("tts sidecar: {e}");
                }
            });
            tokio::spawn(async move {
                while let Some(value) = raw_rx.recv().await {
                    let decoded = decode_chunk(&value, default_rate);
                    if tx.send(decoded).await.is_err() {
                        return; // interrupted
                    }
                }
            });
            Ok(stream)
        })
    }

    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async move {
            let value = self.process.call(json!({"op": "voices"})).await?;
            Ok(value
                .get("voices")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            Some(VoiceInfo {
                                id: v.get("id")?.as_str()?.to_string(),
                                name: v.get("name").and_then(Value::as_str).map(str::to_owned),
                                language: v
                                    .get("language")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                                gender: v.get("gender").and_then(Value::as_str).map(str::to_owned),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_f32_chunks() {
        let mut bytes = Vec::new();
        for s in [0.5f32, -0.5] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let v = json!({
            "sample_rate": 16000,
            "audio": base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        let audio = decode_chunk(&v, 24_000).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.samples, vec![0.5, -0.5]);
    }

    #[test]
    fn unknown_encoding_is_rejected_clearly() {
        let v = json!({"audio": "", "encoding": "mp3"});
        let err = decode_chunk(&v, 24_000).unwrap_err();
        assert_eq!(err.code(), "unsupported_audio_format");
    }

    #[test]
    fn missing_audio_field_is_an_error() {
        assert!(decode_chunk(&json!({}), 24_000).is_err());
    }

    #[test]
    fn voice_json_round_trips_the_important_fields() {
        let v = VoiceSpec::default();
        let j = voice_json(&v);
        assert_eq!(j["language"], v.language);
        assert_eq!(j["gender"], "female");
    }

    #[test]
    fn empty_command_is_rejected() {
        let mut cfg = TtsConfig::default();
        cfg.sidecar_command.clear();
        assert!(SidecarSynthesizer::from_config(&cfg).is_err());
    }
}
