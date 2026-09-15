//! Whisper, running locally, through sherpa-onnx.
//!
//! `--setup` downloads a prebuilt sherpa-onnx and a converted
//! `whisper-large-v3-turbo`, and this drives the `sherpa-onnx-offline`
//! command with the paths it recorded.
//!
//! A process per utterance is a deliberate trade. Linking the C API would
//! mean this crate could not build without the shared libraries present, and
//! `cargo test` would need a 600 MB download; the alternative costs a few
//! tens of milliseconds of start-up against a model that takes hundreds. That
//! is a bad trade only when transcribing continuously, which a voice
//! assistant does not — it transcribes one utterance at a time, after the
//! person has stopped speaking.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;

use crate::audio::AudioBuffer;
use crate::config::SttConfig;
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// Whisper via a local sherpa-onnx installation.
#[derive(Debug, Clone)]
pub struct LocalRecognizer {
    runtime: PathBuf,
    encoder: PathBuf,
    decoder: PathBuf,
    tokens: PathBuf,
    model: String,
    language: Option<String>,
    threads: u32,
    timeout_ms: u64,
}

impl LocalRecognizer {
    /// Build from the paths `--setup` recorded.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        let path = |key: &str| -> Result<PathBuf> {
            cfg.options
                .get(key)
                .and_then(|value| value.as_str())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "stt.options.{key} is not set. Run `cookie-interface --setup`."
                    ))
                })
        };
        let recognizer = Self {
            runtime: path("runtime")?,
            encoder: path("encoder")?,
            decoder: path("decoder")?,
            tokens: path("tokens")?,
            model: cfg.model.clone(),
            language: (!cfg.language.eq_ignore_ascii_case("auto")).then(|| {
                cfg.language
                    .split('-')
                    .next()
                    .unwrap_or("en")
                    .to_lowercase()
            }),
            threads: cfg
                .options
                .get("threads")
                .and_then(|v| v.as_integer())
                .unwrap_or(4)
                .clamp(1, 32) as u32,
            timeout_ms: cfg.timeout_ms.max(5_000),
        };
        recognizer.check_files()?;
        Ok(recognizer)
    }

    /// Every path must exist before the first utterance, not during it.
    ///
    /// A missing model discovered mid-sentence is a silence the user cannot
    /// explain; discovered at startup it is a sentence telling them what to
    /// run.
    fn check_files(&self) -> Result<()> {
        for (what, path) in [
            ("the speech runtime", &self.runtime),
            ("the recogniser's encoder", &self.encoder),
            ("the recogniser's decoder", &self.decoder),
            ("the recogniser's tokens", &self.tokens),
        ] {
            if !path.exists() {
                return Err(Error::ModelUnavailable {
                    name: self.model.clone(),
                    reason: format!("{what} is missing from {}", path.display()),
                });
            }
        }
        Ok(())
    }

    fn program(&self) -> PathBuf {
        let name = if cfg!(windows) {
            "sherpa-onnx-offline.exe"
        } else {
            "sherpa-onnx-offline"
        };
        self.runtime.join("bin").join(name)
    }
}

/// Point the dynamic loader at the runtime's own libraries.
///
/// The prebuilt tools are linked against shared objects that sit beside them
/// and are not on any system path — which is the whole reason this is set
/// rather than left to chance.
pub(crate) fn with_library_path(command: &mut tokio::process::Command, runtime: &std::path::Path) {
    let lib = runtime.join("lib");
    let variable = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let existing = std::env::var_os(variable);
    let mut paths = vec![lib, runtime.join("bin")];
    if let Some(existing) = &existing {
        paths.extend(std::env::split_paths(existing));
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        command.env(variable, joined);
    }
}

impl SpeechRecognizer for LocalRecognizer {
    fn name(&self) -> String {
        format!("local:{}", self.model)
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            // sherpa-onnx prints the text and nothing else; anything more
            // would be invented.
            timestamps: false,
            confidence: false,
            language_detection: self.language.is_none(),
            prompt: false,
            native_partials: false,
            sample_rate: 16_000,
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.check_files()?;
            let program = self.program();
            if !program.exists() {
                return Err(Error::ModelUnavailable {
                    name: self.model.clone(),
                    reason: format!("{} is missing", program.display()),
                });
            }
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
            // Whisper wants 16 kHz mono; the pipeline is already there, but a
            // caller pushing audio over the API might not be.
            let audio = if audio.sample_rate == 16_000 {
                audio
            } else {
                audio.resampled(16_000)
            };

            let wav = std::env::temp_dir().join(format!("cookie-stt-{}.wav", uuid::Uuid::new_v4()));
            let samples = audio.samples.clone();
            let rate = audio.sample_rate;
            let written = tokio::task::spawn_blocking({
                let wav = wav.clone();
                move || crate::audio::wav::write_mono_wav(&wav, &samples, rate)
            })
            .await
            .map_err(|e| Error::Stt(e.to_string()))?;
            written?;

            let mut command = tokio::process::Command::new(self.program());
            command
                .arg(format!("--whisper-encoder={}", self.encoder.display()))
                .arg(format!("--whisper-decoder={}", self.decoder.display()))
                .arg(format!("--tokens={}", self.tokens.display()))
                .arg(format!("--num-threads={}", self.threads))
                .arg("--whisper-task=transcribe");
            if let Some(language) = options.language.as_deref().or(self.language.as_deref()) {
                command.arg(format!(
                    "--whisper-language={}",
                    language.split('-').next().unwrap_or("en")
                ));
            }
            command
                .arg(&wav)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            with_library_path(&mut command, &self.runtime);

            let child = command.spawn().map_err(|e| Error::ModelUnavailable {
                name: self.model.clone(),
                reason: format!("could not start the recogniser: {e}"),
            })?;
            let output = tokio::time::timeout(
                std::time::Duration::from_millis(self.timeout_ms),
                child.wait_with_output(),
            )
            .await;
            let _ = tokio::fs::remove_file(&wav).await;

            let output = match output {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => return Err(Error::Stt(format!("the recogniser failed: {e}"))),
                Err(_) => {
                    return Err(Error::Stt(format!(
                        "the recogniser took longer than {}ms and was stopped",
                        self.timeout_ms
                    )))
                }
            };
            if !output.status.success() {
                return Err(Error::Stt(format!(
                    "the recogniser exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                        .lines()
                        .last()
                        .unwrap_or_default()
                )));
            }

            let text = parse_output(&String::from_utf8_lossy(&output.stdout));
            Ok(Transcript {
                is_final: !options.interim,
                confidence: None,
                language: options.language.clone().or_else(|| self.language.clone()),
                segments: Vec::new(),
                audio_ms,
                latency_ms: started.elapsed().as_millis() as u64,
                text,
            })
        })
    }
}

/// Pull the transcript out of sherpa-onnx's output.
///
/// It prints a block of diagnostics and then the result; the text is the last
/// non-empty line that is not one of its own labels. Parsing by exclusion
/// rather than by pattern because the diagnostics change between releases and
/// the transcript does not.
pub(crate) fn parse_output(stdout: &str) -> String {
    const NOISE: &[&str] = &[
        "Creating recognizer",
        "Started",
        "Done!",
        "num threads",
        "decoding method",
        "Elapsed seconds",
        "Real time factor",
        "----",
        "/",
        "{",
    ];
    stdout
        .lines()
        .map(str::trim)
        .rev()
        .find(|line| {
            !line.is_empty()
                && !line.contains(".wav")
                && !NOISE.iter().any(|noise| line.starts_with(noise))
        })
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SttProviderKind;

    fn config(dir: &std::path::Path) -> SttConfig {
        let mut cfg = SttConfig {
            provider: SttProviderKind::Local,
            ..Default::default()
        };
        for (key, value) in [
            ("runtime", dir.join("runtime")),
            ("encoder", dir.join("encoder.onnx")),
            ("decoder", dir.join("decoder.onnx")),
            ("tokens", dir.join("tokens.txt")),
        ] {
            cfg.options
                .insert(key.into(), toml::Value::String(value.display().to_string()));
        }
        cfg
    }

    fn lay_out_files(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("runtime/bin")).unwrap();
        for name in ["encoder.onnx", "decoder.onnx", "tokens.txt"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
    }

    #[test]
    fn missing_paths_say_what_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let error = LocalRecognizer::from_config(&SttConfig {
            provider: SttProviderKind::Local,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("--setup"), "{error}");
        let _ = dir;
    }

    #[test]
    fn a_missing_model_file_is_caught_before_the_first_utterance() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("runtime")).unwrap();
        let error = LocalRecognizer::from_config(&config(dir.path())).unwrap_err();
        assert_eq!(error.code(), "model_unavailable");
        assert!(error.to_string().contains("encoder"), "{error}");
    }

    #[tokio::test]
    async fn prepare_notices_a_runtime_without_the_binary() {
        let dir = tempfile::tempdir().unwrap();
        lay_out_files(dir.path());
        let recognizer = LocalRecognizer::from_config(&config(dir.path())).unwrap();
        let error = recognizer.prepare().await.unwrap_err();
        assert_eq!(error.code(), "model_unavailable");
    }

    #[test]
    fn capabilities_claim_only_what_sherpa_actually_returns() {
        let dir = tempfile::tempdir().unwrap();
        lay_out_files(dir.path());
        let capabilities = LocalRecognizer::from_config(&config(dir.path()))
            .unwrap()
            .capabilities();
        assert!(!capabilities.timestamps, "sherpa-onnx prints text only");
        assert!(!capabilities.confidence);
        assert_eq!(capabilities.sample_rate, 16_000);
    }

    #[test]
    fn the_transcript_is_the_last_line_that_is_not_diagnostics() {
        let stdout = "Creating recognizer ...\n\
                      /tmp/cookie-stt-1.wav\n\
                      ----\n\
                      Elapsed seconds: 1.2\n\
                      Real time factor (RTF): 0.5\n\
                      My name is Robin\n";
        assert_eq!(parse_output(stdout), "My name is Robin");
    }

    #[test]
    fn silence_produces_an_empty_transcript_rather_than_a_label() {
        let stdout = "Creating recognizer ...\n/tmp/x.wav\n----\nDone!\n";
        assert!(parse_output(stdout).is_empty());
    }
}
