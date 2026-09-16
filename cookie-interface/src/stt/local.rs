//! Whisper, running locally, through sherpa-onnx.
//!
//! `--setup` downloads a prebuilt sherpa-onnx and a converted
//! `whisper-large-v3-turbo`; this drives them.
//!
//! # Why a server and not a command
//!
//! The first version ran `sherpa-onnx-offline` once per utterance, which
//! meant loading 600 MB of weights once per utterance. Measured on a laptop
//! that was **thirteen to sixteen seconds** for a two-second sentence, nearly
//! all of it reading the model off disk and compiling it — the recognition
//! itself is a fraction of that.
//!
//! So the runtime's own `sherpa-onnx-offline-websocket-server` is started
//! once, holds the model in memory, and each utterance is a WebSocket message.
//! Measured against the same runtime on a single core, the same audio then
//! took 1.7 seconds, and the second and third attempts took the same as the
//! first — which is the property that matters.
//!
//! The one-shot command remains as a fallback for the case where the server
//! will not start, because a slow assistant beats a mute one.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::sync::Mutex;

use crate::audio::AudioBuffer;
use crate::config::SttConfig;
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::{SpeechRecognizer, SttCapabilities, TranscribeOptions, Transcript};

/// How long to wait for the server to load the model and start listening.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

/// Whisper via a local sherpa-onnx installation.
#[derive(Debug)]
pub struct LocalRecognizer {
    runtime: PathBuf,
    encoder: PathBuf,
    decoder: PathBuf,
    tokens: PathBuf,
    model: String,
    language: Option<String>,
    /// Kept for the one-shot fallback's own thread count; the resident server
    /// uses [`model_threads`] instead, which is everything but one core.
    #[allow(dead_code)]
    threads: u32,
    timeout_ms: u64,
    provider: String,
    /// Set once the preferred execution provider has been shown not to work.
    ///
    /// Without this, every utterance ran the recogniser twice — once to fail
    /// on CoreML, once to succeed on the CPU — doubling the latency of the
    /// slowest step in the loop. A runtime built without a provider fails
    /// identically forever, so one failure is all the evidence there is.
    fell_back: Arc<AtomicBool>,
    /// The resident server, once it is up. Zero means "not running".
    port: Arc<AtomicU16>,
    /// Held so the process is killed when the recogniser is dropped.
    server: Arc<Mutex<Option<tokio::process::Child>>>,
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
            timeout_ms: cfg.timeout_ms.max(120_000),
            provider: cfg
                .options
                .get("provider")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(default_provider),
            fell_back: Arc::new(AtomicBool::new(false)),
            port: Arc::new(AtomicU16::new(0)),
            server: Arc::new(Mutex::new(None)),
        };
        recognizer.check_files()?;
        Ok(recognizer)
    }

    /// Every path must exist before the first utterance, not during it.
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

    fn program(&self, name: &str) -> PathBuf {
        let name = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        self.runtime.join("bin").join(name)
    }

    fn provider_now(&self) -> String {
        if self.fell_back.load(Ordering::Relaxed) {
            "cpu".into()
        } else {
            self.provider.clone()
        }
    }

    /// The weights to hand this execution provider.
    ///
    /// `--setup` prefers the int8 files, which is right for the CPU — a
    /// quarter of the memory for a difference nobody hears. It is wrong for
    /// CoreML: the Neural Engine has no path for those quantised operators,
    /// so onnxruntime takes the graph back onto the CPU and the provider does
    /// nothing at all. That is the likeliest reason a two-second sentence
    /// still costs eleven seconds on hardware that should manage it in one.
    ///
    /// So when a non-CPU provider is in use and the float weights are sitting
    /// next to the quantised ones — they are, in the same archive — those are
    /// used instead.
    fn weights_for(&self, provider: &str) -> (PathBuf, PathBuf) {
        if provider == "cpu" {
            return (self.encoder.clone(), self.decoder.clone());
        }
        let unquantised = |path: &Path| -> PathBuf {
            let name = path.file_name().map(|n| n.to_string_lossy().to_string());
            match name {
                Some(name) if name.contains(".int8.") => {
                    let candidate = path.with_file_name(name.replace(".int8.", "."));
                    if candidate.exists() {
                        candidate
                    } else {
                        path.to_path_buf()
                    }
                }
                _ => path.to_path_buf(),
            }
        };
        (unquantised(&self.encoder), unquantised(&self.decoder))
    }

    /// Start the resident server if it is not already up, and return its port.
    async fn ensure_server(&self) -> Result<u16> {
        let existing = self.port.load(Ordering::Acquire);
        if existing != 0 {
            return Ok(existing);
        }
        let mut guard = self.server.lock().await;
        // Another task may have started it while we waited for the lock.
        let existing = self.port.load(Ordering::Acquire);
        if existing != 0 {
            return Ok(existing);
        }

        let port = free_port()?;
        let program = self.program("sherpa-onnx-offline-websocket-server");
        if !program.exists() {
            return Err(Error::ModelUnavailable {
                name: self.model.clone(),
                reason: format!("{} is missing", program.display()),
            });
        }

        let provider = self.provider_now();
        let (encoder, decoder) = self.weights_for(&provider);
        if encoder != self.encoder {
            tracing::info!(
                "using unquantised weights for {provider}: the Neural Engine \
                 cannot take the int8 graph and would hand it back to the CPU"
            );
        }
        let mut command = tokio::process::Command::new(&program);
        command
            .arg(format!("--port={port}"))
            // Decode workers, which is how many utterances can be in flight;
            // one is enough, and more of them compete with the threads that
            // do the actual work.
            .arg("--num-work-threads=1")
            .arg("--num-io-threads=1")
            // Threads *inside* the model. The recogniser is the slowest thing
            // in the loop and nothing else is running while it works, so it
            // gets everything but the core the audio callback needs.
            .arg(format!("--num-threads={}", model_threads()))
            .arg(format!("--provider={provider}"))
            .arg(format!("--whisper-encoder={}", encoder.display()))
            .arg(format!("--whisper-decoder={}", decoder.display()))
            .arg(format!("--tokens={}", self.tokens.display()))
            .arg("--whisper-task=transcribe");
        if let Some(language) = &self.language {
            command.arg(format!("--whisper-language={language}"));
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            // Kept, not discarded: this is where onnxruntime says whether it
            // actually took the execution provider it was handed, and
            // throwing it away is why "is CoreML working?" had no answer.
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        with_library_path(&mut command, &self.runtime);
        with_coreml_cache(&mut command);

        let child = command.spawn().map_err(|e| Error::ModelUnavailable {
            name: self.model.clone(),
            reason: format!("could not start the recogniser: {e}"),
        })?;
        let mut child = child;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let lower = line.to_lowercase();
                    // Anything about providers is worth surfacing at warn:
                    // it decides whether this runs in two seconds or twenty.
                    if lower.contains("coreml")
                        || lower.contains("provider")
                        || lower.contains("fallback")
                    {
                        tracing::warn!(target: "cookie::stt", "{line}");
                    } else {
                        tracing::debug!(target: "cookie::stt", "{line}");
                    }
                }
            });
        }
        *guard = Some(child);

        // The server binds its port only after the model is loaded, so a
        // successful connection is the signal that it is ready — more
        // reliable than parsing its log, which changes between releases.
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        while Instant::now() < deadline {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                self.port.store(port, Ordering::Release);
                tracing::info!(
                    port,
                    provider = %self.provider_now(),
                    "speech recogniser resident and ready"
                );
                return Ok(port);
            }
            if let Some(child) = guard.as_mut() {
                if let Ok(Some(status)) = child.try_wait() {
                    *guard = None;
                    return Err(Error::ModelUnavailable {
                        name: self.model.clone(),
                        reason: format!("the recogniser exited with {status} while loading"),
                    });
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        Err(Error::ModelUnavailable {
            name: self.model.clone(),
            reason: "the recogniser did not finish loading in time".into(),
        })
    }

    /// Send one utterance to the resident server.
    async fn transcribe_resident(&self, audio: &AudioBuffer) -> Result<String> {
        let port = self.ensure_server().await?;
        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
            .await
            .map_err(|e| Error::Stt(format!("could not reach the recogniser: {e}")))?;

        socket
            .send(tokio_tungstenite::tungstenite::Message::Binary(
                frame_utterance(&audio.samples, audio.sample_rate).into(),
            ))
            .await
            .map_err(|e| Error::Stt(format!("could not send audio: {e}")))?;

        let reply = tokio::time::timeout(Duration::from_millis(self.timeout_ms), socket.next())
            .await
            .map_err(|_| {
                Error::Stt(format!(
                    "the recogniser took longer than {}ms and was given up on",
                    self.timeout_ms
                ))
            })?;

        let text = match reply {
            Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => text.to_string(),
            Some(Ok(tokio_tungstenite::tungstenite::Message::Binary(bytes))) => {
                String::from_utf8_lossy(&bytes).to_string()
            }
            Some(Ok(_)) => String::new(),
            Some(Err(e)) => return Err(Error::Stt(format!("the recogniser errored: {e}"))),
            None => return Err(Error::Stt("the recogniser closed the connection".into())),
        };
        let _ = socket.close(None).await;
        Ok(parse_output(&text))
    }

    /// One invocation of the one-shot command, as a fallback.
    async fn transcribe_once(
        &self,
        audio: &AudioBuffer,
        options: &TranscribeOptions,
        provider: &str,
    ) -> Result<String> {
        let wav = std::env::temp_dir().join(format!("cookie-stt-{}.wav", uuid::Uuid::new_v4()));
        let samples = audio.samples.clone();
        let rate = audio.sample_rate;
        tokio::task::spawn_blocking({
            let wav = wav.clone();
            move || crate::audio::wav::write_mono_wav(&wav, &samples, rate)
        })
        .await
        .map_err(|e| Error::Stt(e.to_string()))??;

        let (encoder, decoder) = self.weights_for(provider);
        let mut command = tokio::process::Command::new(self.program("sherpa-onnx-offline"));
        command
            .arg(format!("--provider={provider}"))
            .arg(format!("--whisper-encoder={}", encoder.display()))
            .arg(format!("--whisper-decoder={}", decoder.display()))
            .arg(format!("--tokens={}", self.tokens.display()))
            .arg(format!("--num-threads={}", model_threads()))
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
        with_coreml_cache(&mut command);

        let child = command.spawn().map_err(|e| Error::ModelUnavailable {
            name: self.model.clone(),
            reason: format!("could not start the recogniser: {e}"),
        })?;
        let output = tokio::time::timeout(
            Duration::from_millis(self.timeout_ms),
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
                    .next_back()
                    .unwrap_or_default()
            )));
        }
        Ok(parse_output(&String::from_utf8_lossy(&output.stdout)))
    }
}

/// The message the offline WebSocket server expects.
///
/// Eight bytes of header — the sample rate, then the *byte* count of the
/// audio, both native-endian `i32` — followed by `f32` samples. Not a sample
/// count, which is what the first attempt sent and why the server reported an
/// utterance two thousand seconds long.
pub(crate) fn frame_utterance(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(samples) as i32;
    let mut message = Vec::with_capacity(8 + byte_len as usize);
    message.extend_from_slice(&(sample_rate as i32).to_ne_bytes());
    message.extend_from_slice(&byte_len.to_ne_bytes());
    for sample in samples {
        message.extend_from_slice(&sample.to_ne_bytes());
    }
    message
}

/// Half the cores, at least two.
fn default_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(4);
    (cores / 2).max(2)
}

/// Threads inside the model itself.
///
/// Everything but one core. The recogniser is the slowest step in the loop
/// and nothing else of consequence runs while it works — but the audio
/// callback still has to stay ahead of the microphone, and starving it
/// produces the underruns that sound worse than a slightly later transcript.
fn model_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(4);
    (cores - 1).max(2)
}

/// The fastest execution provider this platform is likely to have.
///
/// CoreML on Apple hardware, where it moves the encoder onto the Neural
/// Engine. It is a request rather than a guarantee: onnxruntime silently
/// falls back to the CPU for operators CoreML cannot take, and a runtime
/// built without the provider refuses outright.
fn default_provider() -> String {
    if cfg!(target_os = "macos") {
        "coreml".into()
    } else {
        "cpu".into()
    }
}

/// Whether a complaint is about the execution provider.
///
/// Distinguishing this from a real failure matters: falling back on a missing
/// model file would hide the actual problem behind a second, slower attempt
/// at the same impossible thing.
pub(crate) fn provider_was_refused(message: &str) -> bool {
    let message = message.to_lowercase();
    [
        "provider",
        "coreml",
        "execution",
        "not compiled",
        "unsupported",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

/// A port nobody is using, found by briefly binding one.
fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .map_err(|e| Error::Other(format!("could not find a free port: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| Error::Other(e.to_string()))?
        .port();
    drop(listener);
    Ok(port)
}

/// Where CoreML keeps its compiled models.
///
/// onnxruntime compiles the model for the Neural Engine on first use; without
/// somewhere to keep the result it recompiles on every run, which is slower
/// than never asking for CoreML at all. Inside the application's own cache
/// directory, so `--clear-cache` clears it.
fn with_coreml_cache(command: &mut tokio::process::Command) {
    let directory = crate::paths::Paths::discover()
        .map(|paths| paths.cache_dir().join("coreml"))
        .unwrap_or_else(|_| std::env::temp_dir().join("cookie-coreml"));
    let _ = std::fs::create_dir_all(&directory);
    command.env("ORT_COREML_CACHE_PATH", &directory);
}

/// Point the dynamic loader at the runtime's own libraries.
///
/// The prebuilt tools are linked against shared objects that sit beside them
/// and are not on any system path — which is the whole reason this is set
/// rather than left to chance.
pub(crate) fn with_library_path(command: &mut tokio::process::Command, runtime: &Path) {
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
            // Load the model now rather than during the first thing anybody
            // says. It takes seconds, and they should be spent at startup.
            match self.ensure_server().await {
                Ok(_) => Ok(()),
                Err(e)
                    if provider_was_refused(&e.to_string())
                        && !self.fell_back.load(Ordering::Relaxed) =>
                {
                    tracing::warn!(
                        provider = %self.provider,
                        "this runtime has no {} support; using the CPU from now on",
                        self.provider
                    );
                    self.fell_back.store(true, Ordering::Relaxed);
                    self.ensure_server().await.map(|_| ())
                }
                Err(e) => Err(e),
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
            // Whisper wants 16 kHz mono; the pipeline is already there, but a
            // caller pushing audio over the API might not be.
            let audio = if audio.sample_rate == 16_000 {
                audio
            } else {
                audio.resampled(16_000)
            };

            let text = match self.transcribe_resident(&audio).await {
                Ok(text) => text,
                Err(resident_error) => {
                    tracing::warn!(
                        "the resident recogniser failed ({resident_error}); \
                         falling back to one-shot, which is much slower"
                    );
                    let provider = self.provider_now();
                    match self.transcribe_once(&audio, &options, &provider).await {
                        Ok(text) => text,
                        Err(e) if provider != "cpu" && provider_was_refused(&e.to_string()) => {
                            self.fell_back.store(true, Ordering::Relaxed);
                            self.transcribe_once(&audio, &options, "cpu").await?
                        }
                        Err(e) => return Err(e),
                    }
                }
            };

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

/// Pull the transcript out of what sherpa-onnx returns.
///
/// Both the server and the command answer with a JSON object; the command
/// buries it under its own diagnostics.
///
/// ```text
/// {"lang": "", "text": "After early nightfall the yellow lamps …", "tokens": [...]}
/// ```
///
/// An early version discarded every line beginning with `{` as noise, which
/// threw away the only line that mattered and returned an empty transcript
/// for every utterance.
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

    fn config(dir: &Path) -> SttConfig {
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

    fn lay_out_files(dir: &Path) {
        std::fs::create_dir_all(dir.join("runtime/bin")).unwrap();
        for name in ["encoder.onnx", "decoder.onnx", "tokens.txt"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
    }

    #[test]
    fn the_utterance_frame_matches_what_the_server_reads() {
        // Captured from the server's source: eight bytes of header, the
        // sample rate then the *byte* count, both native-endian i32. Sending
        // a sample count instead had it report a two-thousand-second
        // utterance and refuse the connection.
        let framed = frame_utterance(&[1.0, -1.0], 16_000);
        assert_eq!(framed.len(), 8 + 8);
        assert_eq!(
            i32::from_ne_bytes(framed[0..4].try_into().unwrap()),
            16_000,
            "the first field is the sample rate"
        );
        assert_eq!(
            i32::from_ne_bytes(framed[4..8].try_into().unwrap()),
            8,
            "the second field is bytes, not samples"
        );
        assert_eq!(f32::from_ne_bytes(framed[8..12].try_into().unwrap()), 1.0);
    }

    #[test]
    fn missing_paths_say_what_to_run() {
        let error = LocalRecognizer::from_config(&SttConfig {
            provider: SttProviderKind::Local,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("--setup"), "{error}");
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
    async fn prepare_notices_a_runtime_without_the_server() {
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
        assert!(!capabilities.timestamps, "sherpa-onnx returns text only");
        assert!(!capabilities.confidence);
        assert!(
            !capabilities.cheap_partials,
            "a local model cannot absorb a partial every 400ms"
        );
        assert_eq!(capabilities.sample_rate, 16_000);
    }

    #[test]
    fn only_a_provider_complaint_triggers_the_fallback() {
        assert!(provider_was_refused(
            "this build does not support the CoreML execution provider"
        ));
        assert!(provider_was_refused("Unsupported provider: coreml"));
        assert!(!provider_was_refused("the recogniser's encoder is missing"));
        assert!(!provider_was_refused("No such file or directory"));
    }

    #[test]
    fn coreml_gets_the_float_weights_and_the_cpu_keeps_the_quantised_ones() {
        // The Neural Engine has no path for int8 operators, so handing it the
        // quantised graph makes onnxruntime give the whole thing back to the
        // CPU — the provider is accepted and then does nothing, which is the
        // hardest kind of "working" to notice.
        let dir = tempfile::tempdir().unwrap();
        lay_out_files(dir.path());
        for name in [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "encoder.onnx",
            "decoder.onnx",
        ] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        let mut cfg = config(dir.path());
        for (key, file) in [
            ("encoder", "encoder.int8.onnx"),
            ("decoder", "decoder.int8.onnx"),
        ] {
            cfg.options.insert(
                key.into(),
                toml::Value::String(dir.path().join(file).display().to_string()),
            );
        }
        let recognizer = LocalRecognizer::from_config(&cfg).unwrap();

        let (cpu_encoder, _) = recognizer.weights_for("cpu");
        assert!(cpu_encoder.to_string_lossy().contains("int8"));

        let (coreml_encoder, coreml_decoder) = recognizer.weights_for("coreml");
        assert!(
            !coreml_encoder.to_string_lossy().contains("int8"),
            "{coreml_encoder:?}"
        );
        assert!(!coreml_decoder.to_string_lossy().contains("int8"));
    }

    #[test]
    fn without_float_weights_the_quantised_ones_are_used_anyway() {
        // Better a provider that quietly falls back than a recogniser that
        // cannot start because a file is missing.
        let dir = tempfile::tempdir().unwrap();
        // Deliberately *not* lay_out_files: this is the case where only the
        // quantised weights were shipped.
        std::fs::create_dir_all(dir.path().join("runtime/bin")).unwrap();
        for name in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        let mut cfg = config(dir.path());
        cfg.options.insert(
            "decoder".into(),
            toml::Value::String(dir.path().join("decoder.int8.onnx").display().to_string()),
        );
        cfg.options.insert(
            "encoder".into(),
            toml::Value::String(dir.path().join("encoder.int8.onnx").display().to_string()),
        );
        let recognizer = LocalRecognizer::from_config(&cfg).unwrap();
        let (encoder, _) = recognizer.weights_for("coreml");
        assert!(encoder.to_string_lossy().contains("int8"));
    }

    #[test]
    fn free_ports_are_actually_free() {
        let port = free_port().unwrap();
        assert!(port > 1024);
        // Binding it again must work, or the probe left it occupied.
        assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn the_transcript_is_read_out_of_the_json_sherpa_actually_prints() {
        let stdout = concat!(
            "Creating recognizer ...\n",
            "recognizer created in 0.668 s\n",
            "/tmp/cookie-stt-1.wav\n----\n",
            "Elapsed seconds: 1.095 s\n",
            r#"{"lang": "", "emotion": "", "text": "My name is Robin.", "tokens":[" My"]}"#,
            "\n"
        );
        assert_eq!(parse_output(stdout), "My name is Robin.");
    }

    #[test]
    fn the_servers_bare_json_reply_parses_too() {
        let reply = r#"{"lang": "", "emotion": "", "event": "", "text": "Are you there?"}"#;
        assert_eq!(parse_output(reply), "Are you there?");
    }

    #[test]
    fn silence_produces_an_empty_transcript_rather_than_a_label() {
        let reply = r#"{"lang": "", "text": "", "tokens":[]}"#;
        assert!(parse_output(reply).is_empty());
    }
}
