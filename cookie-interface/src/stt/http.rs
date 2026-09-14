//! Speech recognition over HTTP.
//!
//! Targets the de-facto standard `POST /v1/audio/transcriptions` multipart
//! endpoint, which is spoken by whisper.cpp's `server`, faster-whisper-server,
//! Speaches, vLLM and the hosted APIs. Point it at `127.0.0.1` and no audio
//! leaves the machine.

use std::time::{Duration, Instant};

use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde_json::Value;

use crate::audio::{wav, AudioBuffer};
use crate::config::SttConfig;
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::sidecar::parse_transcript;
use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// Recogniser that posts WAV audio to an OpenAI-compatible endpoint.
#[derive(Debug)]
pub struct HttpRecognizer {
    client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl HttpRecognizer {
    /// Build from configuration. The API key is read from the environment
    /// variable named in the config, never from the config file itself.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        let timeout = Duration::from_millis(cfg.timeout_ms.max(1));
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Network(format!("could not build HTTP client: {e}")))?;
        Ok(Self {
            client,
            endpoint: cfg.endpoint.clone(),
            model: cfg.model.clone(),
            api_key: cfg
                .api_key_env
                .as_deref()
                .and_then(|k| std::env::var(k).ok()),
            timeout,
        })
    }

    fn form(&self, audio: &AudioBuffer, options: &TranscribeOptions) -> Result<Form> {
        let wav_bytes = wav::encode_mono_wav(&audio.samples, audio.sample_rate)?;
        let part = Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| Error::Other(e.to_string()))?;
        let mut form = Form::new()
            .text("model", self.model.clone())
            .text(
                "response_format",
                if options.want_timestamps {
                    "verbose_json"
                } else {
                    "json"
                },
            )
            .part("file", part);
        if let Some(lang) = &options.language {
            // The endpoint wants a bare ISO-639-1 code, not a full BCP-47 tag.
            form = form.text(
                "language",
                lang.split('-').next().unwrap_or("en").to_string(),
            );
        }
        if let Some(prompt) = &options.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if options.want_timestamps {
            form = form.text("timestamp_granularities[]", "segment");
        }
        Ok(form)
    }
}

impl SpeechRecognizer for HttpRecognizer {
    fn name(&self) -> String {
        format!("http:{}", self.model)
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            timestamps: true,
            confidence: false, // the OpenAI schema has no per-request score
            language_detection: true,
            prompt: true,
            native_partials: false,
            sample_rate: 16_000,
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            // A HEAD/GET on the endpoint root tells us whether anything is
            // listening without burning a transcription.
            let probe = self.endpoint.clone();
            match tokio::time::timeout(Duration::from_millis(1500), self.client.get(&probe).send())
                .await
            {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(e)) if e.is_connect() => Err(Error::ModelUnavailable {
                    name: self.model.clone(),
                    reason: format!("nothing is listening at {probe} ({e})"),
                }),
                // A 4xx/5xx from a live server is fine: it means something is
                // there, and GET is not the real verb.
                Ok(Err(_)) | Err(_) => Ok(()),
            }
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
            let form = self.form(&audio, &options)?;
            let mut req = self.client.post(&self.endpoint).timeout(self.timeout);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req
                .multipart(form)
                .send()
                .await
                .map_err(|e| Error::Network(format!("transcription request failed: {e}")))?;
            let status = resp.status();
            let body = resp
                .text()
                .await
                .map_err(|e| Error::Network(format!("reading transcription response: {e}")))?;
            if !status.is_success() {
                return Err(Error::Stt(format!(
                    "{} returned {status}: {}",
                    self.endpoint,
                    body.chars().take(400).collect::<String>()
                )));
            }
            let value: Value = serde_json::from_str(&body)
                .map_err(|e| Error::Stt(format!("transcription response was not JSON ({e})")))?;
            Ok(parse_transcript(
                &value,
                audio_ms,
                started.elapsed().as_millis() as u64,
                &options,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SttProviderKind;

    fn cfg() -> SttConfig {
        SttConfig {
            provider: SttProviderKind::Http,
            ..Default::default()
        }
    }

    #[test]
    fn builds_and_names_itself() {
        let r = HttpRecognizer::from_config(&cfg()).unwrap();
        assert_eq!(r.name(), "http:whisper-large-v3-turbo");
        assert!(r.capabilities().timestamps);
    }

    #[test]
    fn form_encodes_a_wav_and_language_code() {
        let r = HttpRecognizer::from_config(&cfg()).unwrap();
        let options = TranscribeOptions {
            language: Some("en-GB".into()),
            want_timestamps: true,
            ..Default::default()
        };
        // Building the form is the part we can assert without a server.
        assert!(r.form(&AudioBuffer::silent(160, 16_000), &options).is_ok());
    }

    #[tokio::test]
    async fn unreachable_endpoint_is_reported_as_model_unavailable() {
        let mut c = cfg();
        // Port 1 is reliably closed and refuses fast.
        c.endpoint = "http://127.0.0.1:1/v1/audio/transcriptions".into();
        let r = HttpRecognizer::from_config(&c).unwrap();
        let err = r.prepare().await.unwrap_err();
        assert_eq!(err.code(), "model_unavailable");
    }
}
