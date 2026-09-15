//! Kokoro, running locally, through sherpa-onnx.
//!
//! This is the voice the project is actually asking for: warm, unhurried,
//! British, female. `--setup` fetches Kokoro and records the paths; speaker 7
//! is `bf_emma`, and 8 (`bf_isabella`) is the other one worth trying.
//!
//! Synthesis is a whole utterance at a time, because that is what the tool
//! does — it writes a WAV and exits. The engine still streams it: the speech
//! task walks the buffer at wall-clock speed, so the orb reacts and barge-in
//! lands within a frame regardless.

use std::path::PathBuf;
use std::process::Stdio;

use crate::config::{TtsConfig, VoiceSpec};
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::{SpeechSynthesizer, SynthesisRequest, SynthesisStream, TtsCapabilities, VoiceInfo};

/// Kokoro via a local sherpa-onnx installation.
#[derive(Debug, Clone)]
pub struct LocalSynthesizer {
    runtime: PathBuf,
    model: PathBuf,
    voices: PathBuf,
    tokens: PathBuf,
    data_dir: Option<PathBuf>,
    threads: u32,
    timeout_ms: u64,
}

impl LocalSynthesizer {
    pub fn from_config(cfg: &TtsConfig) -> Result<Self> {
        let path = |key: &str| -> Result<PathBuf> {
            cfg.options
                .get(key)
                .and_then(|value| value.as_str())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "tts.options.{key} is not set. Run `cookie-interface --setup`."
                    ))
                })
        };
        let synthesizer = Self {
            runtime: path("runtime")?,
            model: path("model_path")?,
            voices: path("voices")?,
            tokens: path("tokens")?,
            data_dir: cfg
                .options
                .get("data_dir")
                .and_then(|value| value.as_str())
                .map(PathBuf::from),
            threads: cfg
                .options
                .get("threads")
                .and_then(|v| v.as_integer())
                .unwrap_or(2)
                .clamp(1, 32) as u32,
            timeout_ms: cfg.timeout_ms.max(5_000),
        };
        synthesizer.check_files()?;
        Ok(synthesizer)
    }

    fn check_files(&self) -> Result<()> {
        for (what, path) in [
            ("the speech runtime", &self.runtime),
            ("the voice model", &self.model),
            ("the voice data", &self.voices),
            ("the voice's tokens", &self.tokens),
        ] {
            if !path.exists() {
                return Err(Error::ModelUnavailable {
                    name: "kokoro".into(),
                    reason: format!("{what} is missing from {}", path.display()),
                });
            }
        }
        Ok(())
    }

    fn program(&self) -> PathBuf {
        let name = if cfg!(windows) {
            "sherpa-onnx-offline-tts.exe"
        } else {
            "sherpa-onnx-offline-tts"
        };
        self.runtime.join("bin").join(name)
    }

    /// Kokoro selects a speaker by index, and a voice id is that index.
    ///
    /// `auto` means the British female the project asks for. A name is
    /// accepted too, for the handful people actually use, because asking
    /// somebody to remember that 7 is Emma is a poor interface.
    fn speaker_id(voice: &VoiceSpec) -> u32 {
        let id = voice.id.trim();
        if let Ok(index) = id.parse::<u32>() {
            return index;
        }
        match id.to_lowercase().as_str() {
            "bf_emma" | "emma" | "auto" | "" => 7,
            "bf_isabella" | "isabella" => 8,
            "bm_george" | "george" => 9,
            "bm_lewis" | "lewis" => 10,
            "af_bella" | "bella" => 1,
            "af_sarah" | "sarah" => 3,
            "am_adam" | "adam" => 5,
            _ => 7,
        }
    }

    /// Kokoro takes a *length* scale: larger is slower, which is the inverse
    /// of the rate everything else in this crate speaks in.
    fn length_scale(voice: &VoiceSpec) -> f32 {
        (1.0 / voice.rate.clamp(0.5, 2.0)).clamp(0.5, 2.0)
    }
}

impl SpeechSynthesizer for LocalSynthesizer {
    fn name(&self) -> String {
        "local:kokoro".to_string()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities {
            // The tool writes a file and exits; the engine paces it, but the
            // model itself does not stream.
            streaming: false,
            rate: true,
            // Kokoro has no pitch control, and claiming one would be a knob
            // that does nothing.
            pitch: false,
            style: false,
            voice_listing: true,
            sample_rate: 24_000,
            extra_parameters: vec!["threads".into()],
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.check_files()?;
            let program = self.program();
            if !program.exists() {
                return Err(Error::ModelUnavailable {
                    name: "kokoro".into(),
                    reason: format!("{} is missing", program.display()),
                });
            }
            Ok(())
        })
    }

    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>> {
        Box::pin(async move {
            let wav = std::env::temp_dir().join(format!("cookie-tts-{}.wav", uuid::Uuid::new_v4()));

            let mut command = tokio::process::Command::new(self.program());
            command
                .arg(format!("--kokoro-model={}", self.model.display()))
                .arg(format!("--kokoro-voices={}", self.voices.display()))
                .arg(format!("--kokoro-tokens={}", self.tokens.display()))
                .arg(format!("--num-threads={}", self.threads))
                .arg(format!("--sid={}", Self::speaker_id(&request.voice)))
                .arg(format!(
                    "--kokoro-length-scale={:.3}",
                    Self::length_scale(&request.voice)
                ))
                .arg(format!("--output-filename={}", wav.display()));
            if let Some(data_dir) = &self.data_dir {
                command.arg(format!("--kokoro-data-dir={}", data_dir.display()));
            }
            command
                .arg(&request.text)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            crate::stt::local::with_library_path(&mut command, &self.runtime);

            let child = command.spawn().map_err(|e| Error::ModelUnavailable {
                name: "kokoro".into(),
                reason: format!("could not start the voice: {e}"),
            })?;
            let output = tokio::time::timeout(
                std::time::Duration::from_millis(self.timeout_ms),
                child.wait_with_output(),
            )
            .await;

            let output = match output {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => {
                    let _ = tokio::fs::remove_file(&wav).await;
                    return Err(Error::Tts(format!("the voice failed: {e}")));
                }
                Err(_) => {
                    let _ = tokio::fs::remove_file(&wav).await;
                    return Err(Error::Tts(format!(
                        "the voice took longer than {}ms and was stopped",
                        self.timeout_ms
                    )));
                }
            };
            if !output.status.success() {
                let _ = tokio::fs::remove_file(&wav).await;
                return Err(Error::Tts(format!(
                    "the voice exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                        .lines()
                        .last()
                        .unwrap_or_default()
                )));
            }

            let read = tokio::task::spawn_blocking({
                let wav = wav.clone();
                move || crate::audio::wav::read_wav(&wav)
            })
            .await
            .map_err(|e| Error::Tts(e.to_string()))?;
            let _ = tokio::fs::remove_file(&wav).await;
            let audio = read?;
            if audio.is_empty() {
                return Err(Error::Tts("the voice produced no audio".into()));
            }
            Ok(SynthesisStream::once(audio))
        })
    }

    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async {
            // Kokoro's en-v0_19 speaker order, which is fixed by the model
            // file rather than discoverable from it.
            Ok(vec![
                info("7", "Emma", "en-GB", "female"),
                info("8", "Isabella", "en-GB", "female"),
                info("9", "George", "en-GB", "male"),
                info("10", "Lewis", "en-GB", "male"),
                info("1", "Bella", "en-US", "female"),
                info("3", "Sarah", "en-US", "female"),
                info("5", "Adam", "en-US", "male"),
            ])
        })
    }
}

fn info(id: &str, name: &str, language: &str, gender: &str) -> VoiceInfo {
    VoiceInfo {
        id: id.to_string(),
        name: Some(name.to_string()),
        language: Some(language.to_string()),
        gender: Some(gender.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TtsProviderKind;

    fn config(dir: &std::path::Path) -> TtsConfig {
        let mut cfg = TtsConfig {
            provider: TtsProviderKind::Local,
            ..Default::default()
        };
        for (key, value) in [
            ("runtime", dir.join("runtime")),
            ("model_path", dir.join("model.int8.onnx")),
            ("voices", dir.join("voices.bin")),
            ("tokens", dir.join("tokens.txt")),
        ] {
            cfg.options
                .insert(key.into(), toml::Value::String(value.display().to_string()));
        }
        cfg
    }

    fn lay_out_files(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("runtime/bin")).unwrap();
        for name in ["model.int8.onnx", "voices.bin", "tokens.txt"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
    }

    #[test]
    fn the_default_voice_is_the_british_woman() {
        assert_eq!(LocalSynthesizer::speaker_id(&VoiceSpec::default()), 7);
        assert_eq!(
            LocalSynthesizer::speaker_id(&VoiceSpec {
                id: "bf_isabella".into(),
                ..Default::default()
            }),
            8
        );
        // A raw index still works, for anyone who knows what they want.
        assert_eq!(
            LocalSynthesizer::speaker_id(&VoiceSpec {
                id: "3".into(),
                ..Default::default()
            }),
            3
        );
        // And an unknown name lands on the house voice rather than failing.
        assert_eq!(
            LocalSynthesizer::speaker_id(&VoiceSpec {
                id: "nobody".into(),
                ..Default::default()
            }),
            7
        );
    }

    #[test]
    fn rate_is_inverted_because_kokoro_measures_length_not_speed() {
        let faster = VoiceSpec {
            rate: 2.0,
            ..Default::default()
        };
        let slower = VoiceSpec {
            rate: 0.5,
            ..Default::default()
        };
        assert!(LocalSynthesizer::length_scale(&faster) < 1.0);
        assert!(LocalSynthesizer::length_scale(&slower) > 1.0);
        assert_eq!(LocalSynthesizer::length_scale(&VoiceSpec::default()), 1.0);
    }

    #[test]
    fn missing_paths_say_what_to_run() {
        let error = LocalSynthesizer::from_config(&TtsConfig {
            provider: TtsProviderKind::Local,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("--setup"), "{error}");
    }

    #[test]
    fn a_missing_model_is_caught_before_the_first_utterance() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("runtime")).unwrap();
        let error = LocalSynthesizer::from_config(&config(dir.path())).unwrap_err();
        assert_eq!(error.code(), "model_unavailable");
    }

    #[test]
    fn capabilities_do_not_claim_a_pitch_control_kokoro_lacks() {
        let dir = tempfile::tempdir().unwrap();
        lay_out_files(dir.path());
        let capabilities = LocalSynthesizer::from_config(&config(dir.path()))
            .unwrap()
            .capabilities();
        assert!(capabilities.rate);
        assert!(!capabilities.pitch);
        assert!(!capabilities.streaming);
    }

    #[tokio::test]
    async fn the_offered_voices_are_led_by_the_british_women() {
        let dir = tempfile::tempdir().unwrap();
        lay_out_files(dir.path());
        let voices = LocalSynthesizer::from_config(&config(dir.path()))
            .unwrap()
            .voices()
            .await
            .unwrap();
        assert_eq!(voices[0].id, "7");
        assert_eq!(voices[0].language.as_deref(), Some("en-GB"));
        assert_eq!(voices[0].gender.as_deref(), Some("female"));
    }
}
