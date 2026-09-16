//! Speech synthesis over HTTP.
//!
//! Two request dialects are supported, which between them cover almost
//! everything worth pointing this at:
//!
//! * [`HttpTtsDialect::OpenAiCompatible`] — `{model, input, voice,
//!   response_format, speed}`. Kokoro-FastAPI, openedai-speech, LocalAI and
//!   the hosted OpenAI endpoint all speak it.
//! * [`HttpTtsDialect::Qwen3`] — DashScope's `{model, input:{text, voice},
//!   parameters:{...}}` envelope, which is how Qwen3-TTS is exposed.
//!
//! When the server answers with raw PCM the audio is forwarded chunk by chunk
//! as it arrives, so the orb starts moving before synthesis has finished.
//! When it answers with a container (WAV) the body is decoded once at the end,
//! because a partial WAV is not decodable.

use std::time::Duration;

use base64::Engine as _;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

use crate::audio::{wav, AudioBuffer};
use crate::config::{HttpTtsDialect, TtsConfig, VoiceSpec};
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::{SpeechSynthesizer, SynthesisRequest, SynthesisStream, TtsCapabilities, VoiceInfo};

/// Synthesiser that talks to an HTTP model server.
#[derive(Debug)]
pub struct HttpSynthesizer {
    client: Client,
    endpoint: String,
    model: String,
    dialect: HttpTtsDialect,
    api_key: Option<String>,
    api_key_var: Option<String>,
    timeout: Duration,
    streaming: bool,
    sample_rate: u32,
    /// `response_format` asked of the server.
    format: String,
    extra: serde_json::Map<String, Value>,
}

impl HttpSynthesizer {
    /// Build from configuration.
    pub fn from_config(cfg: &TtsConfig) -> Result<Self> {
        let timeout = Duration::from_millis(cfg.timeout_ms.max(1));
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Network(format!("could not build HTTP client: {e}")))?;
        let format = cfg
            .options
            .get("response_format")
            .and_then(|v| v.as_str())
            .unwrap_or(if cfg.streaming { "pcm" } else { "wav" })
            .to_string();
        let sample_rate = cfg
            .options
            .get("sample_rate")
            .and_then(|v| v.as_integer())
            .unwrap_or(24_000)
            .clamp(8_000, 192_000) as u32;
        let mut extra = serde_json::Map::new();
        for (k, v) in &cfg.options {
            if matches!(k.as_str(), "response_format" | "sample_rate") {
                continue;
            }
            if let Ok(v) = serde_json::to_value(v) {
                extra.insert(k.clone(), v);
            }
        }
        if cfg.endpoint.starts_with("https://") && !cfg!(feature = "tls") {
            return Err(Error::Config(format!(
                "tts.endpoint is {} but this build has no TLS. \
                 Rebuild with `--features tls`, or use a local http:// endpoint.",
                cfg.endpoint
            )));
        }
        Ok(Self {
            client,
            endpoint: cfg.endpoint.clone(),
            model: cfg.model.clone(),
            dialect: cfg.dialect,
            api_key: cfg
                .api_key_env
                .as_deref()
                .and_then(|k| std::env::var(k).ok()),
            api_key_var: cfg.api_key_env.clone(),
            timeout,
            streaming: cfg.streaming,
            sample_rate,
            format,
            extra,
        })
    }

    /// Build the request body for the configured dialect.
    ///
    /// Only parameters the dialect actually defines are sent; nothing is
    /// invented, which is the same promise [`TtsCapabilities`] makes.
    pub(crate) fn body(&self, request: &SynthesisRequest) -> Value {
        let voice = resolve_voice_id(&request.voice, self.dialect);
        match self.dialect {
            HttpTtsDialect::OpenAiCompatible => {
                let mut body = json!({
                    "model": self.model,
                    "input": request.text,
                    "voice": voice,
                    "response_format": self.format,
                    "speed": request.voice.rate.clamp(0.25, 4.0),
                });
                merge(&mut body, &self.extra);
                merge_object(&mut body, &request.voice.extra);
                body
            }
            HttpTtsDialect::Qwen3 => {
                let mut parameters = json!({
                    "language_type": language_name(&request.voice.language),
                    "sample_rate": self.sample_rate,
                    "format": self.format,
                });
                if let Some(style) = &request.voice.style {
                    parameters["style"] = Value::String(style.clone());
                }
                merge(&mut parameters, &self.extra);
                merge_object(&mut parameters, &request.voice.extra);
                json!({
                    "model": self.model,
                    "input": { "text": request.text, "voice": voice },
                    "parameters": parameters,
                })
            }
        }
    }

    fn decode_body(&self, content_type: Option<&str>, bytes: &[u8]) -> Result<AudioBuffer> {
        // Some servers wrap audio in JSON (DashScope does when asked for
        // base64). Detect that before handing bytes to the audio decoder.
        if bytes.first() == Some(&b'{') {
            let v: Value = serde_json::from_slice(bytes)
                .map_err(|e| Error::Tts(format!("TTS response was not audio or JSON: {e}")))?;
            if let Some(b64) = find_base64_audio(&v) {
                let raw = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| Error::Tts(format!("TTS base64 audio was invalid: {e}")))?;
                return wav::decode_audio(&raw, None, self.sample_rate);
            }
            if let Some(url) = v.pointer("/output/audio/url").and_then(Value::as_str) {
                return Err(Error::Tts(format!(
                    "the server returned an audio URL ({url}) rather than audio; \
                     set `parameters.format` so it streams bytes instead"
                )));
            }
            return Err(Error::Tts(format!(
                "unexpected TTS JSON response: {}",
                String::from_utf8_lossy(&bytes[..bytes.len().min(300)])
            )));
        }
        wav::decode_audio(bytes, content_type, self.sample_rate)
    }
}

/// Pick the voice identifier to send.
///
/// `auto` means "you choose": for OpenAI-compatible servers we send a British
/// female default that Kokoro ships (`bf_emma`), and for Qwen3 the closest
/// documented persona. An explicit id always wins.
fn resolve_voice_id(voice: &VoiceSpec, dialect: HttpTtsDialect) -> String {
    if !voice.id.is_empty() && voice.id != "auto" {
        return voice.id.clone();
    }
    match dialect {
        HttpTtsDialect::OpenAiCompatible => "bf_emma".into(),
        HttpTtsDialect::Qwen3 => "Cherry".into(),
    }
}

fn language_name(tag: &str) -> &'static str {
    match tag.split('-').next().unwrap_or("en") {
        "zh" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ru" => "Russian",
        _ => "English",
    }
}

fn merge(target: &mut Value, extra: &serde_json::Map<String, Value>) {
    if let Some(obj) = target.as_object_mut() {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
}

fn merge_object(target: &mut Value, extra: &std::collections::BTreeMap<String, Value>) {
    if let Some(obj) = target.as_object_mut() {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
}

fn find_base64_audio(v: &Value) -> Option<&str> {
    for pointer in ["/output/audio/data", "/audio", "/data", "/output/audio"] {
        if let Some(s) = v.pointer(pointer).and_then(Value::as_str) {
            return Some(s);
        }
    }
    None
}

impl SpeechSynthesizer for HttpSynthesizer {
    fn name(&self) -> String {
        format!("http:{}", self.model)
    }

    fn capabilities(&self) -> TtsCapabilities {
        match self.dialect {
            HttpTtsDialect::OpenAiCompatible => TtsCapabilities {
                streaming: self.streaming && self.format == "pcm",
                rate: true,
                pitch: false, // not in the schema; we refuse to pretend
                style: false,
                voice_listing: true,
                sample_rate: self.sample_rate,
                extra_parameters: vec!["response_format".into(), "sample_rate".into()],
            },
            HttpTtsDialect::Qwen3 => TtsCapabilities {
                streaming: self.streaming && self.format == "pcm",
                rate: false,
                pitch: false,
                style: true,
                voice_listing: false,
                sample_rate: self.sample_rate,
                extra_parameters: vec![
                    "response_format".into(),
                    "sample_rate".into(),
                    "style".into(),
                ],
            },
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            if self.api_key.is_none() {
                if let Some(var) = &self.api_key_var {
                    if !self.endpoint.contains("127.0.0.1") && !self.endpoint.contains("localhost")
                    {
                        tracing::warn!("{var} is not set; {} may reject requests", self.endpoint);
                    }
                }
            }
            match tokio::time::timeout(
                Duration::from_millis(1500),
                self.client.get(&self.endpoint).send(),
            )
            .await
            {
                Ok(Err(e)) if e.is_connect() => Err(Error::ModelUnavailable {
                    name: self.model.clone(),
                    reason: format!("nothing is listening at {} ({e})", self.endpoint),
                }),
                _ => Ok(()),
            }
        })
    }

    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>> {
        Box::pin(async move {
            let body = self.body(&request);
            let mut req = self.client.post(&self.endpoint).timeout(self.timeout);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            if matches!(self.dialect, HttpTtsDialect::Qwen3) && self.streaming {
                // DashScope turns on incremental output with a header.
                req = req.header("X-DashScope-SSE", "enable");
            }
            let resp = req
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Network(format!("TTS request failed: {e}")))?;

            let status = resp.status();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(Error::Tts(format!(
                    "{} returned {status}: {}",
                    self.endpoint,
                    body.chars().take(400).collect::<String>()
                )));
            }

            let raw_pcm = content_type
                .as_deref()
                .map(|c| c.contains("pcm") || c.contains("octet-stream"))
                .unwrap_or(false)
                || self.format == "pcm";

            if self.streaming && raw_pcm {
                let (tx, stream) = SynthesisStream::channel(self.sample_rate, 8);
                let rate = self.sample_rate;
                tokio::spawn(async move {
                    let mut bytes = resp.bytes_stream();
                    // s16le frames are two bytes; a chunk boundary can split
                    // one, so carry the odd byte over.
                    let mut carry: Option<u8> = None;
                    while let Some(next) = bytes.next().await {
                        let chunk = match next {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(Error::Network(format!("TTS stream: {e}"))))
                                    .await;
                                return;
                            }
                        };
                        let mut buf: Vec<u8> = Vec::with_capacity(chunk.len() + 1);
                        if let Some(b) = carry.take() {
                            buf.push(b);
                        }
                        buf.extend_from_slice(&chunk);
                        if buf.len() % 2 == 1 {
                            carry = buf.pop();
                        }
                        if buf.is_empty() {
                            continue;
                        }
                        let audio = wav::decode_pcm_s16le(&buf, rate, 1);
                        if tx.send(Ok(audio)).await.is_err() {
                            return; // interrupted
                        }
                    }
                });
                return Ok(stream);
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| Error::Network(format!("reading TTS response: {e}")))?;
            let audio = self.decode_body(content_type.as_deref(), &bytes)?;
            if audio.is_empty() {
                return Err(Error::Tts("the TTS server returned no audio".into()));
            }
            Ok(SynthesisStream::once(audio))
        })
    }

    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async move {
            if !matches!(self.dialect, HttpTtsDialect::OpenAiCompatible) {
                return Ok(Vec::new());
            }
            // Kokoro-FastAPI and openedai-speech both expose this.
            let url = self.endpoint.replace("/audio/speech", "/audio/voices");
            let Ok(resp) = self.client.get(&url).send().await else {
                return Ok(Vec::new());
            };
            let Ok(value) = resp.json::<Value>().await else {
                return Ok(Vec::new());
            };
            Ok(value
                .get("voices")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| {
                            let id = v.as_str()?.to_string();
                            Some(VoiceInfo {
                                language: id.starts_with("bf_").then(|| "en-GB".to_string()),
                                gender: id
                                    .chars()
                                    .nth(1)
                                    .map(|c| if c == 'f' { "female" } else { "male" }.to_string()),
                                name: None,
                                id,
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
    use crate::config::TtsProviderKind;

    fn cfg(dialect: HttpTtsDialect) -> TtsConfig {
        TtsConfig {
            provider: TtsProviderKind::Http,
            dialect,
            endpoint: "http://127.0.0.1:8081/v1/audio/speech".into(),
            ..Default::default()
        }
    }

    #[test]
    fn openai_body_has_the_documented_shape() {
        let s = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::OpenAiCompatible)).unwrap();
        let body = s.body(&SynthesisRequest::new("hello", VoiceSpec::default()));
        assert_eq!(body["input"], "hello");
        assert_eq!(body["voice"], "bf_emma");
        assert!(body.get("pitch").is_none(), "must not invent parameters");
    }

    #[test]
    fn qwen3_body_nests_input_and_parameters() {
        let s = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::Qwen3)).unwrap();
        let voice = VoiceSpec {
            style: Some("warm".into()),
            ..Default::default()
        };
        let body = s.body(&SynthesisRequest::new("hello", voice));
        assert_eq!(body["input"]["text"], "hello");
        assert_eq!(body["parameters"]["language_type"], "English");
        assert_eq!(body["parameters"]["style"], "warm");
    }

    #[test]
    fn explicit_voice_id_is_passed_through() {
        let s = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::OpenAiCompatible)).unwrap();
        let voice = VoiceSpec {
            id: "bf_alice".into(),
            ..Default::default()
        };
        assert_eq!(
            s.body(&SynthesisRequest::new("x", voice))["voice"],
            "bf_alice"
        );
    }

    #[test]
    fn https_without_tls_feature_fails_early() {
        let mut c = cfg(HttpTtsDialect::Qwen3);
        c.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/tts".into();
        let built = HttpSynthesizer::from_config(&c);
        if cfg!(feature = "tls") {
            assert!(built.is_ok());
        } else {
            assert_eq!(built.unwrap_err().code(), "config_error");
        }
    }

    #[test]
    fn capabilities_track_the_dialect() {
        let openai = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::OpenAiCompatible)).unwrap();
        assert!(openai.capabilities().rate);
        assert!(!openai.capabilities().style);
        let qwen = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::Qwen3)).unwrap();
        assert!(qwen.capabilities().style);
        assert!(!qwen.capabilities().rate);
    }

    #[test]
    fn json_audio_payloads_are_decoded() {
        let s = HttpSynthesizer::from_config(&cfg(HttpTtsDialect::Qwen3)).unwrap();
        let pcm: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let payload = json!({
            "output": { "audio": { "data": base64::engine::general_purpose::STANDARD.encode(&pcm) } }
        })
        .to_string();
        let audio = s
            .decode_body(Some("application/json"), payload.as_bytes())
            .unwrap();
        assert_eq!(audio.frames(), 32);
    }
}
