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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    /// `cpu`, `coreml`, `cuda`… Whatever the runtime was built with.
    provider: String,
    /// Set once the preferred provider has been shown not to work.
    ///
    /// Without this, every single utterance ran the recogniser twice — once
    /// to fail on CoreML and once to succeed on the CPU — which doubled the
    /// latency of a step that was already the slowest thing in the loop.
    /// A runtime built without a provider fails identically forever, so one
    /// failure is all the evidence there is going to be.
    fell_back: Arc<AtomicBool>,
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
                .unwrap_or_else(default_threads)
                .clamp(1, 32) as u32,
            // The first utterance includes loading 600 MB of weights, which
            // takes seconds even on fast hardware. A timeout tuned to steady
            // state kills the very first thing you say.
            timeout_ms: cfg.timeout_ms.max(120_000),
            provider: cfg
                .options
                .get("provider")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(default_provider),
            fell_back: Arc::new(AtomicBool::new(false)),
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

/// The fastest execution provider this platform is likely to have.
///
/// CoreML on Apple hardware, where it moves Whisper onto the Neural Engine
/// and is worth several times the CPU. It is a request rather than a
/// guarantee — a runtime built without it says so and we fall back — which is
/// why this is a default rather than an assumption.
fn default_provider() -> String {
    if cfg!(target_os = "macos") {
        "coreml".into()
    } else {
        "cpu".into()
    }
}

/// Half the cores, at least two.
///
/// All of them would win a benchmark and lose the application: the audio
/// callback needs a core to stay ahead of the microphone, and a recogniser
/// that starves it produces underruns, which sound worse than a transcript
/// arriving a moment later.
fn default_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(4);
    (cores / 2).max(2)
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
            cheap_partials: false,
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

            // Once the preferred provider has failed, stop asking: it will
            // fail the same way every time, and paying for that on every
            // utterance doubles the latency of the slowest step in the loop.
            let preferred = if self.fell_back.load(Ordering::Relaxed) {
                "cpu"
            } else {
                &self.provider
            };
            let mut attempt = self.run(&wav, &options, preferred).await;
            if attempt.is_err() && preferred != "cpu" {
                tracing::warn!(
                    provider = %self.provider,
                    "the recogniser refused that execution provider; using the CPU from now on"
                );
                self.fell_back.store(true, Ordering::Relaxed);
                attempt = self.run(&wav, &options, "cpu").await;
            }
            let _ = tokio::fs::remove_file(&wav).await;
            let text = attempt?;

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

impl LocalRecognizer {
    /// One invocation of the recogniser with a given execution provider.
    async fn run(
        &self,
        wav: &std::path::Path,
        options: &TranscribeOptions,
        provider: &str,
    ) -> Result<String> {
        let mut command = tokio::process::Command::new(self.program());
        command
            .arg(format!("--provider={provider}"))
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
            .arg(wav)
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

        Ok(parse_output(&String::from_utf8_lossy(&output.stdout)))
    }
}

/// Pull the transcript out of sherpa-onnx's output.
///
/// It prints its whole configuration, then the file name, then a block of
/// timings, and then the result as a JSON object:
///
/// ```text
/// {"lang": "", "text": "After early nightfall the yellow lamps …", "tokens": [...]}
/// ```
///
/// The first version of this discarded every line beginning with `{` as
/// noise, which threw away the only line that mattered and returned an empty
/// transcript for every utterance — recognition appeared to be silently doing
/// nothing. Parse the JSON; fall back to the last plain line for older builds
/// that printed bare text.
pub(crate) fn parse_output(stdout: &str) -> String {
    for line in stdout.lines().rev() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
                return text.trim().to_string();
            }
        }
    }

    // Older releases printed the text on its own line after the timings.
    const NOISE: &[&str] = &[
        "Creating recognizer",
        "recognizer created",
        "Started",
        "Done!",
        "num threads",
        "decoding method",
        "Elapsed seconds",
        "Real time factor",
        "Offline",
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
    fn the_transcript_is_read_out_of_the_json_sherpa_actually_prints() {
        // Captured verbatim from sherpa-onnx v1.13.8. The first version of
        // the parser discarded every line starting with `{` as noise, which
        // threw away this one and made recognition look silently broken.
        let stdout = concat!(
            "Creating recognizer ...\n",
            "recognizer created in 0.668 s\n",
            "Started\nDone!\n\n",
            "/tmp/cookie-stt-1.wav\n",
            "----\n",
            "num threads: 1\n",
            "decoding method: greedy_search\n",
            "Elapsed seconds: 1.095 s\n",
            "Real time factor (RTF): 1.095 / 6.625 = 0.165\n",
            r#"{"lang": "", "emotion": "", "text": "My name is Robin.", "tokens":[" My"]}"#,
            "\n"
        );
        assert_eq!(parse_output(stdout), "My name is Robin.");
    }

    #[test]
    fn a_bare_text_line_still_works_for_older_builds() {
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
        let stdout = concat!(
            "Creating recognizer ...\n/tmp/x.wav\n----\nDone!\n",
            r#"{"lang": "", "text": "", "tokens":[]}"#,
            "\n"
        );
        assert!(parse_output(stdout).is_empty());
    }
}
