//! The operating system's own synthesiser.
//!
//! Nothing to download, nothing to configure, available on a fresh machine —
//! which makes it the right default and the right fallback. Quality is well
//! below Qwen3-TTS, but macOS and Windows both ship convincing British female
//! voices (Serena / Hazel), which is exactly the voice this project wants.
//!
//! | Platform | Backend | Default voice |
//! |----------|---------|---------------|
//! | macOS    | `say`   | Serena (en-GB, female) |
//! | Windows  | SAPI via PowerShell | Hazel (en-GB, female) |
//! | Linux    | `espeak-ng` | `en-gb+f3` (female variant) |
//!
//! Linux users who want something better should point `tts.provider` at
//! `http` (Kokoro-FastAPI is a good match) or `sidecar`.

use std::path::PathBuf;
use std::process::Stdio;

use tokio::process::Command;

use crate::audio::{wav, AudioBuffer};
use crate::config::{TtsConfig, VoiceSpec};
use crate::error::{Error, Result};
use crate::util::BoxFuture;

use super::{SpeechSynthesizer, SynthesisRequest, SynthesisStream, TtsCapabilities, VoiceInfo};

/// Speech through the platform's built-in engine.
#[derive(Debug, Clone)]
pub struct SystemSynthesizer {
    /// Voice name override; `auto` picks the platform default below.
    voice_id: String,
    timeout_ms: u64,
}

impl SystemSynthesizer {
    /// Build from configuration.
    pub fn from_config(cfg: &TtsConfig) -> Self {
        Self {
            voice_id: cfg.voice.id.clone(),
            timeout_ms: cfg.timeout_ms.max(1_000),
        }
    }

    /// Whether the backing program exists on this machine.
    ///
    /// Used by `--doctor` and by the fallback chain, which must not fall back
    /// onto something that is also missing.
    pub fn is_available(&self) -> bool {
        which(self.program()).is_some()
    }

    fn program(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "say"
        } else if cfg!(target_os = "windows") {
            "powershell"
        } else {
            "espeak-ng"
        }
    }

    fn resolved_voice(&self, voice: &VoiceSpec) -> String {
        // Per-request id wins; then the configured id; then the platform
        // default. That ordering is what lets one config file name a voice
        // while a single utterance still overrides it.
        if !voice.id.is_empty() && voice.id != "auto" {
            return voice.id.clone();
        }
        if !self.voice_id.is_empty() && self.voice_id != "auto" {
            return self.voice_id.clone();
        }
        // British, female, warm — the project's stated house voice.
        if cfg!(target_os = "macos") {
            "Serena".into()
        } else if cfg!(target_os = "windows") {
            "Microsoft Hazel Desktop".into()
        } else {
            "en-gb+f3".into()
        }
    }

    fn temp_wav(&self) -> PathBuf {
        // `env::temp_dir` is the platform-correct location (TMPDIR, %TEMP%);
        // the file is removed as soon as it has been read.
        std::env::temp_dir().join(format!("cookie-tts-{}.wav", uuid::Uuid::new_v4()))
    }

    async fn run(&self, request: &SynthesisRequest, out: &PathBuf) -> Result<()> {
        let voice = self.resolved_voice(&request.voice);
        let text = request.text.clone();
        let mut cmd = Command::new(self.program());
        if cfg!(target_os = "macos") {
            // LEF32 keeps us in float; `say` writes a real WAV header.
            cmd.arg("-v")
                .arg(&voice)
                .arg("-o")
                .arg(out)
                .arg("--data-format=LEI16@22050")
                .arg("-r")
                .arg(
                    ((request.voice.rate.clamp(0.5, 2.0)) * 175.0)
                        .round()
                        .to_string(),
                )
                .arg("--")
                .arg(&text);
        } else if cfg!(target_os = "windows") {
            // SAPI rate is -10..10; map 0.5..2.0 onto roughly -5..5.
            let rate = ((request.voice.rate.clamp(0.5, 2.0) - 1.0) * 5.0).round() as i32;
            let script = format!(
                "Add-Type -AssemblyName System.Speech; \
                 $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                 try {{ $s.SelectVoice('{voice}') }} catch {{ }} \
                 $s.Rate = {rate}; \
                 $s.Volume = {volume}; \
                 $s.SetOutputToWaveFile('{path}'); \
                 $s.Speak([Console]::In.ReadToEnd()); \
                 $s.Dispose()",
                voice = voice.replace('\'', "''"),
                volume = (request.voice.volume.clamp(0.0, 1.0) * 100.0).round() as i32,
                path = out.display().to_string().replace('\'', "''"),
            );
            cmd.args(["-NoProfile", "-NonInteractive", "-Command", &script])
                .stdin(Stdio::piped());
        } else {
            cmd.arg("-v")
                .arg(&voice)
                .arg("-w")
                .arg(out)
                .arg("-s")
                .arg(
                    ((request.voice.rate.clamp(0.5, 2.0)) * 165.0)
                        .round()
                        .to_string(),
                )
                .arg("-p")
                // espeak-ng pitch is 0..99 around a midpoint of 50; the spec
                // carries semitones, so four points per semitone is about right.
                .arg(
                    (50.0 + request.voice.pitch * 4.0)
                        .round()
                        .clamp(0.0, 99.0)
                        .to_string(),
                )
                .arg("--")
                .arg(&text);
        }
        cmd.stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| Error::ModelUnavailable {
            name: self.program().to_string(),
            reason: format!("{e}. Install it, or set tts.provider to \"http\" or \"mock\"."),
        })?;
        if cfg!(target_os = "windows") {
            // The text goes over stdin so quoting can never break it.
            use tokio::io::AsyncWriteExt;
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes()).await;
                let _ = stdin.shutdown().await;
            }
        }
        let output = tokio::time::timeout(
            std::time::Duration::from_millis(self.timeout_ms),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| Error::Tts("system synthesiser timed out".into()))?
        .map_err(|e| Error::Tts(format!("system synthesiser failed: {e}")))?;

        if !output.status.success() {
            return Err(Error::Tts(format!(
                "{} exited with {}: {}",
                self.program(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(())
    }
}

/// Minimal cross-platform `which`.
fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(target_os = "windows") {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|s| s.to_lowercase())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let candidate = dir.join(format!("{program}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

impl SpeechSynthesizer for SystemSynthesizer {
    fn name(&self) -> String {
        format!("system:{}", self.program())
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities {
            // The platform engines write a file; there is no honest way to
            // stream from them, so we say so rather than pretending.
            streaming: false,
            rate: true,
            // espeak-ng has a pitch control; say and SAPI do not.
            pitch: !cfg!(any(target_os = "macos", target_os = "windows")),
            style: false,
            voice_listing: cfg!(target_os = "macos"),
            sample_rate: 22_050,
            extra_parameters: Vec::new(),
        }
    }

    fn prepare(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            if self.is_available() {
                Ok(())
            } else {
                Err(Error::ModelUnavailable {
                    name: self.program().to_string(),
                    reason: format!(
                        "`{}` is not on PATH. On Linux: `apt install espeak-ng`.",
                        self.program()
                    ),
                })
            }
        })
    }

    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> BoxFuture<'a, Result<SynthesisStream>> {
        Box::pin(async move {
            let path = self.temp_wav();
            let result = self.run(&request, &path).await;
            let audio = match result {
                Ok(()) => {
                    let read = tokio::task::spawn_blocking({
                        let path = path.clone();
                        move || wav::read_wav(&path)
                    })
                    .await
                    .map_err(|e| Error::Tts(e.to_string()))?;
                    let _ = tokio::fs::remove_file(&path).await;
                    read?
                }
                Err(e) => {
                    let _ = tokio::fs::remove_file(&path).await;
                    return Err(e);
                }
            };
            if audio.is_empty() {
                return Err(Error::Tts(
                    "the system synthesiser produced no audio".into(),
                ));
            }
            Ok(SynthesisStream::once(audio))
        })
    }

    fn voices(&self) -> BoxFuture<'_, Result<Vec<VoiceInfo>>> {
        Box::pin(async move {
            if !cfg!(target_os = "macos") {
                return Ok(Vec::new());
            }
            let out = Command::new("say").arg("-v").arg("?").output().await;
            let Ok(out) = out else { return Ok(Vec::new()) };
            Ok(String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|line| {
                    let mut parts = line.split_whitespace();
                    let id = parts.next()?.to_string();
                    let lang = parts.next()?.to_string();
                    Some(VoiceInfo {
                        name: Some(id.clone()),
                        language: Some(lang.replace('_', "-")),
                        gender: None,
                        id,
                    })
                })
                .collect())
        })
    }
}

/// Shared with `AudioBuffer` consumers in tests.
#[allow(unused)]
fn _assert_buffer_type(_: AudioBuffer) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_voice_is_british_female() {
        let s = SystemSynthesizer::from_config(&TtsConfig::default());
        let resolved = s.resolved_voice(&VoiceSpec::default());
        assert!(
            ["Serena", "Microsoft Hazel Desktop", "en-gb+f3"].contains(&resolved.as_str()),
            "{resolved}"
        );
    }

    #[test]
    fn explicit_voice_id_wins() {
        let s = SystemSynthesizer::from_config(&TtsConfig::default());
        let v = VoiceSpec {
            id: "Daniel".into(),
            ..Default::default()
        };
        assert_eq!(s.resolved_voice(&v), "Daniel");
    }

    #[test]
    fn which_finds_something_that_exists() {
        let probe = if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        };
        assert!(which(probe).is_some());
        assert!(which("definitely-not-here-8a1b").is_none());
    }

    #[tokio::test]
    async fn prepare_reports_a_missing_engine_clearly() {
        let s = SystemSynthesizer::from_config(&TtsConfig::default());
        match s.prepare().await {
            Ok(()) => assert!(s.is_available()),
            Err(e) => assert_eq!(e.code(), "model_unavailable"),
        }
    }

    #[test]
    fn capabilities_never_claim_streaming() {
        let s = SystemSynthesizer::from_config(&TtsConfig::default());
        assert!(!s.capabilities().streaming);
    }
}
