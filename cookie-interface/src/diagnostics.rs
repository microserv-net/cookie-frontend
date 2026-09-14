//! "Cookie, are you alright?"
//!
//! One routine answers that question, and it is the same routine behind
//! `--doctor`, `GET /v1/diagnostics` and the spoken intent. There is exactly
//! one definition of "alright" in this program, which is the only way the
//! spoken answer can be trusted to match the command-line one.
//!
//! Every check is *non-destructive* and time-boxed: nothing here opens a
//! stream that stays open, downloads a model, or writes user data. A machine
//! with no microphone must still be able to run diagnostics and be told, in
//! plain English, that it has no microphone.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::paths::Paths;

/// How a single capability came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Health {
    /// Working.
    Ok,
    /// Working, but not the way the user asked for — a fallback is in use.
    Degraded,
    /// Not working.
    Failed,
    /// Deliberately switched off; not a problem.
    Disabled,
    /// Could not be determined without doing something intrusive.
    Unknown,
}

impl Health {
    fn rank(self) -> u8 {
        match self {
            Health::Failed => 3,
            Health::Degraded => 2,
            Health::Unknown => 1,
            Health::Ok | Health::Disabled => 0,
        }
    }
}

/// One line of the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    /// Machine-readable name, e.g. `audio.input`.
    pub name: String,
    /// Human-readable capability, e.g. "microphone".
    pub capability: String,
    pub health: Health,
    /// One sentence, written to be *read aloud* as well as printed.
    pub detail: String,
    /// What to do about it, when there is something to do.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    pub elapsed_ms: u64,
}

impl Check {
    fn new(
        name: &str,
        capability: &str,
        health: Health,
        detail: impl Into<String>,
        started: Instant,
    ) -> Self {
        Self {
            name: name.into(),
            capability: capability.into(),
            health,
            detail: detail.into(),
            hint: None,
            elapsed_ms: started.elapsed().as_millis() as u64,
        }
    }

    fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

/// The whole picture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub version: String,
    pub overall: Health,
    pub checks: Vec<Check>,
    pub elapsed_ms: u64,
}

impl DiagnosticReport {
    /// Checks that are actually wrong, worst first.
    pub fn problems(&self) -> Vec<&Check> {
        let mut bad: Vec<&Check> = self
            .checks
            .iter()
            .filter(|c| matches!(c.health, Health::Failed | Health::Degraded))
            .collect();
        bad.sort_by_key(|c| std::cmp::Reverse(c.health.rank()));
        bad
    }

    /// The answer Cookie says out loud.
    ///
    /// Written as speech, not as a status dump: nobody wants to hear
    /// "audio.input: OK" read to them. Good news is one sentence; bad news
    /// leads with what is broken and what to do about it.
    pub fn spoken_summary(&self) -> String {
        let problems = self.problems();
        if problems.is_empty() {
            let mut parts = vec!["Yes, I'm fine.".to_string()];
            let working: Vec<&str> = self
                .checks
                .iter()
                .filter(|c| c.health == Health::Ok)
                .map(|c| c.capability.as_str())
                .collect();
            if !working.is_empty() {
                parts.push(format!("{} — all working.", join_naturally(&working)));
            }
            return parts.join(" ");
        }

        let failed: Vec<&str> = problems
            .iter()
            .filter(|c| c.health == Health::Failed)
            .map(|c| c.capability.as_str())
            .collect();
        let degraded: Vec<&str> = problems
            .iter()
            .filter(|c| c.health == Health::Degraded)
            .map(|c| c.capability.as_str())
            .collect();

        let mut sentences = Vec::new();
        if !failed.is_empty() {
            sentences.push(format!(
                "Not entirely. {} {} not working.",
                capitalise(&join_naturally(&failed)),
                if failed.len() == 1 { "is" } else { "are" }
            ));
        } else {
            sentences.push("Mostly.".to_string());
        }
        if !degraded.is_empty() {
            sentences.push(format!(
                "{} {} running on a fallback.",
                capitalise(&join_naturally(&degraded)),
                if degraded.len() == 1 { "is" } else { "are" }
            ));
        }
        // One concrete detail beats a list; the full report is on the API.
        // Each fragment is punctuated so the speech synthesiser pauses
        // between them instead of running them into one breathless sentence.
        if let Some(first) = problems.first() {
            sentences.push(sentence(&first.detail));
            if let Some(hint) = &first.hint {
                sentences.push(sentence(&capitalise(hint)));
            }
        }
        sentences.join(" ")
    }

    /// The same information as lines for a terminal.
    pub fn to_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for check in &self.checks {
            let mark = match check.health {
                Health::Ok => "ok      ",
                Health::Degraded => "degraded",
                Health::Failed => "FAILED  ",
                Health::Disabled => "off     ",
                Health::Unknown => "unknown ",
            };
            lines.push(format!(
                "  [{mark}] {:<22} {}",
                check.capability, check.detail
            ));
            if let Some(hint) = &check.hint {
                lines.push(format!("           └─ {hint}"));
            }
        }
        lines
    }
}

fn join_naturally(items: &[&str]) -> String {
    match items {
        [] => String::new(),
        [one] => one.to_string(),
        [a, b] => format!("{a} and {b}"),
        _ => {
            let (last, rest) = items.split_last().unwrap();
            format!("{}, and {last}", rest.join(", "))
        }
    }
}

/// Make sure a fragment ends like a sentence.
fn sentence(text: &str) -> String {
    let trimmed = text.trim_end();
    if trimmed.ends_with(['.', '!', '?']) {
        trimmed.to_string()
    } else {
        format!("{trimmed}.")
    }
}

fn capitalise(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// What the engine can tell diagnostics about the running system.
///
/// Passed in rather than discovered so that `--doctor` (no engine) and the
/// spoken answer (live engine) share one implementation.
#[derive(Debug, Clone, Default)]
pub struct LiveStatus {
    pub stt_provider: Option<String>,
    pub tts_provider: Option<String>,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub hardware_audio: bool,
    /// Provider readiness reported by `provider.status` events so far.
    pub stt_ready: Option<bool>,
    pub tts_ready: Option<bool>,
    /// Whether the renderer got a GPU surface.
    pub renderer_ok: Option<bool>,
    /// Backend reachability, when a backend is configured.
    pub backend_ok: Option<bool>,
    pub backend_detail: Option<String>,
    /// Number of backend tasks currently in flight.
    pub active_tasks: usize,
}

/// Run every check.
///
/// `probe_devices` enumerates real audio hardware, which takes a moment and
/// is skipped in tests.
pub async fn run(
    config: &Config,
    paths: &Paths,
    live: &LiveStatus,
    probe_devices: bool,
) -> DiagnosticReport {
    let overall_start = Instant::now();
    let mut checks = Vec::new();

    checks.push(check_config(config, paths));
    checks.push(check_storage(paths));
    checks.push(check_input(config, live, probe_devices));
    checks.push(check_output(config, live, probe_devices));
    checks.push(check_stt(config, live));
    checks.push(check_tts(config, live));
    checks.push(check_renderer(config, live));
    checks.push(check_api(config).await);
    checks.push(check_backend(config, live));

    // `Unknown` is not a fault. It means a check could not be settled without
    // doing something intrusive — no microphone has been opened yet, the
    // backend has not been called. Letting it set the overall verdict would
    // make a perfectly healthy machine answer "I'm not sure", which is worse
    // than useless: the individual check still says so.
    let overall = if checks.iter().any(|c| c.health == Health::Failed) {
        Health::Failed
    } else if checks.iter().any(|c| c.health == Health::Degraded) {
        Health::Degraded
    } else {
        Health::Ok
    };

    DiagnosticReport {
        version: crate::VERSION.to_string(),
        overall,
        checks,
        elapsed_ms: overall_start.elapsed().as_millis() as u64,
    }
}

fn check_config(config: &Config, paths: &Paths) -> Check {
    let started = Instant::now();
    match config.validate() {
        Ok(()) => Check::new(
            "config",
            "configuration",
            Health::Ok,
            format!("loaded from {}", paths.config_file().display()),
            started,
        ),
        Err(e) => Check::new(
            "config",
            "configuration",
            Health::Failed,
            e.to_string(),
            started,
        )
        .with_hint("run `cookie-interface --config` to inspect the file"),
    }
}

fn check_storage(paths: &Paths) -> Check {
    let started = Instant::now();
    match paths.ensure() {
        Ok(()) => Check::new(
            "storage",
            "storage",
            Health::Ok,
            format!("data lives in {}", paths.data_dir().display()),
            started,
        ),
        Err(e) => Check::new("storage", "storage", Health::Failed, e.to_string(), started)
            .with_hint("check the permissions on your user data directory"),
    }
}

fn check_input(_config: &Config, live: &LiveStatus, probe: bool) -> Check {
    let started = Instant::now();
    if !live.hardware_audio && live.input_device.is_some() {
        return Check::new(
            "audio.input",
            "my hearing",
            Health::Degraded,
            "I'm using a simulated microphone, so I can't actually hear the room",
            started,
        )
        .with_hint("build with the `audio-io` feature for real capture");
    }
    #[cfg(feature = "audio-io")]
    if probe {
        match crate::audio::CpalInput::list() {
            Ok(devices) if devices.is_empty() => {
                return Check::new(
                    "audio.input",
                    "my hearing",
                    Health::Failed,
                    "there's no microphone on this machine",
                    started,
                )
                .with_hint("plug one in, or drive me through POST /v1/audio instead");
            }
            Ok(devices) => {
                let name = live
                    .input_device
                    .clone()
                    .unwrap_or_else(|| devices[0].clone());
                return Check::new(
                    "audio.input",
                    "my hearing",
                    Health::Ok,
                    format!("listening through {name}"),
                    started,
                );
            }
            Err(e) => {
                return Check::new(
                    "audio.input",
                    "my hearing",
                    Health::Failed,
                    format!("the audio system wouldn't answer: {e}"),
                    started,
                )
            }
        }
    }
    let _ = probe;
    Check::new(
        "audio.input",
        "my hearing",
        if live.hardware_audio {
            Health::Unknown
        } else {
            Health::Degraded
        },
        live.input_device
            .clone()
            .map(|d| format!("configured for {d}"))
            .unwrap_or_else(|| "no capture device has been opened yet".into()),
        started,
    )
}

fn check_output(_config: &Config, live: &LiveStatus, probe: bool) -> Check {
    let started = Instant::now();
    #[cfg(feature = "audio-io")]
    if probe && live.hardware_audio {
        match crate::audio::CpalOutput::list() {
            Ok(devices) if devices.is_empty() => {
                return Check::new(
                    "audio.output",
                    "my voice",
                    Health::Failed,
                    "there's nothing to play sound through",
                    started,
                )
                .with_hint("connect speakers or headphones");
            }
            Ok(devices) => {
                let name = live
                    .output_device
                    .clone()
                    .unwrap_or_else(|| devices[0].clone());
                return Check::new(
                    "audio.output",
                    "my voice",
                    Health::Ok,
                    format!("speaking through {name}"),
                    started,
                );
            }
            Err(e) => {
                return Check::new(
                    "audio.output",
                    "my voice",
                    Health::Failed,
                    format!("the audio system wouldn't answer: {e}"),
                    started,
                )
            }
        }
    }
    let _ = probe;
    Check::new(
        "audio.output",
        "my voice",
        if live.hardware_audio {
            Health::Unknown
        } else {
            Health::Degraded
        },
        "audio output is simulated in this build",
        started,
    )
}

fn check_stt(config: &Config, live: &LiveStatus) -> Check {
    let started = Instant::now();
    let name = live
        .stt_provider
        .clone()
        .unwrap_or_else(|| format!("{:?}", config.stt.provider).to_lowercase());
    match live.stt_ready {
        Some(true) => Check::new(
            "stt",
            "speech recognition",
            Health::Ok,
            format!("{name} is loaded"),
            started,
        ),
        Some(false) => Check::new(
            "stt",
            "speech recognition",
            Health::Failed,
            format!("{name} wouldn't load"),
            started,
        )
        .with_hint("run `cookie-interface --setup`, or check stt.endpoint"),
        None if matches!(config.stt.provider, crate::config::SttProviderKind::Mock) => Check::new(
            "stt",
            "speech recognition",
            Health::Degraded,
            "I'm using the stand-in recogniser, so I'm not really transcribing",
            started,
        )
        .with_hint("set stt.provider to \"http\" or \"sidecar\" for a real model"),
        None => Check::new(
            "stt",
            "speech recognition",
            Health::Unknown,
            format!("{name} hasn't been exercised yet"),
            started,
        ),
    }
}

fn check_tts(config: &Config, live: &LiveStatus) -> Check {
    let started = Instant::now();
    let name = live
        .tts_provider
        .clone()
        .unwrap_or_else(|| format!("{:?}", config.tts.provider).to_lowercase());
    match live.tts_ready {
        Some(true) => Check::new(
            "tts",
            "my voice model",
            Health::Ok,
            format!("{name} is ready"),
            started,
        ),
        Some(false) => Check::new(
            "tts",
            "my voice model",
            Health::Failed,
            format!("{name} wouldn't load"),
            started,
        )
        .with_hint("check tts.endpoint, or fall back to tts.provider = \"system\""),
        None if matches!(config.tts.provider, crate::config::TtsProviderKind::Mock) => Check::new(
            "tts",
            "my voice model",
            Health::Degraded,
            "I'm speaking with the placeholder voice",
            started,
        ),
        None => Check::new(
            "tts",
            "my voice model",
            Health::Unknown,
            format!("{name} hasn't spoken yet"),
            started,
        ),
    }
}

fn check_renderer(config: &Config, live: &LiveStatus) -> Check {
    let started = Instant::now();
    if !config.ui.enabled {
        return Check::new(
            "renderer",
            "my face",
            Health::Disabled,
            "the orb is switched off in configuration",
            started,
        );
    }
    match live.renderer_ok {
        Some(true) => Check::new(
            "renderer",
            "my face",
            Health::Ok,
            format!("drawing in {:?} mode", config.ui.presentation),
            started,
        ),
        Some(false) => Check::new(
            "renderer",
            "my face",
            Health::Degraded,
            "I couldn't get a graphics surface, so I'm running without the orb",
            started,
        )
        .with_hint("update your graphics drivers; everything else still works"),
        None => Check::new(
            "renderer",
            "my face",
            Health::Unknown,
            "the window hasn't opened yet",
            started,
        ),
    }
}

async fn check_api(config: &Config) -> Check {
    let started = Instant::now();
    let addr = std::net::SocketAddr::new(config.api.bind, config.api.port);
    // Binding is the only honest test of "is the port free", and it is
    // instantly reversible.
    match tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpListener::bind(addr),
    )
    .await
    {
        Ok(Ok(listener)) => {
            drop(listener);
            Check::new(
                "api",
                "my local API",
                Health::Ok,
                format!("port {} is available", config.api.port),
                started,
            )
        }
        Ok(Err(_)) => Check::new(
            "api",
            "my local API",
            // In a running instance this is *expected*: we are the process
            // holding the port.
            Health::Ok,
            format!("port {} is in use, most likely by me", config.api.port),
            started,
        ),
        Err(_) => Check::new(
            "api",
            "my local API",
            Health::Unknown,
            "the network stack didn't answer in time",
            started,
        ),
    }
}

fn check_backend(config: &Config, live: &LiveStatus) -> Check {
    let started = Instant::now();
    if !config.backend.enabled {
        return Check::new(
            "backend",
            "my connection to the backend",
            Health::Disabled,
            "no backend is configured; I'm running as a voice interface only",
            started,
        );
    }
    let url = config.backend.chat_url();
    match live.backend_ok {
        Some(true) => Check::new(
            "backend",
            "my connection to the backend",
            Health::Ok,
            format!("connected to {url}"),
            started,
        ),
        Some(false) => Check::new(
            "backend",
            "my connection to the backend",
            Health::Failed,
            live.backend_detail
                .clone()
                .unwrap_or_else(|| format!("I can't reach {url}")),
            started,
        )
        .with_hint("check the machine is on, and that backend.base_url is right"),
        None => Check::new(
            "backend",
            "my connection to the backend",
            Health::Unknown,
            format!("{url} hasn't been contacted yet"),
            started,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> (Paths, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Paths::rooted(dir.path()), dir)
    }

    #[tokio::test]
    async fn a_healthy_system_says_so_in_one_breath() {
        let (paths, _dir) = paths();
        let mut config = Config::default();
        config.ui.enabled = false;
        let live = LiveStatus {
            stt_ready: Some(true),
            tts_ready: Some(true),
            hardware_audio: true,
            stt_provider: Some("http:whisper-large-v3-turbo".into()),
            tts_provider: Some("http:qwen3-tts-flash".into()),
            input_device: Some("Default".into()),
            output_device: Some("Default".into()),
            ..Default::default()
        };
        let report = run(&config, &paths, &live, false).await;
        let spoken = report.spoken_summary();
        assert!(spoken.starts_with("Yes, I'm fine."), "{spoken}");
        assert!(
            !spoken.contains("audio.input"),
            "spoke an identifier: {spoken}"
        );
    }

    #[tokio::test]
    async fn a_broken_model_is_named_in_plain_english() {
        let (paths, _dir) = paths();
        let config = Config::default();
        let live = LiveStatus {
            stt_ready: Some(false),
            tts_ready: Some(true),
            hardware_audio: true,
            ..Default::default()
        };
        let report = run(&config, &paths, &live, false).await;
        assert_eq!(report.overall, Health::Failed);
        let spoken = report.spoken_summary();
        assert!(
            spoken.to_lowercase().contains("speech recognition"),
            "{spoken}"
        );
        assert!(spoken.starts_with("Not entirely."), "{spoken}");
    }

    #[tokio::test]
    async fn a_disabled_backend_is_not_a_fault() {
        let (paths, _dir) = paths();
        let mut config = Config::default();
        config.ui.enabled = false;
        config.backend.enabled = false;
        let live = LiveStatus {
            stt_ready: Some(true),
            tts_ready: Some(true),
            hardware_audio: true,
            ..Default::default()
        };
        let report = run(&config, &paths, &live, false).await;
        assert!(report.problems().is_empty());
        assert_eq!(report.overall, Health::Ok);
    }

    #[tokio::test]
    async fn an_unreachable_backend_is_reported_with_a_hint() {
        let (paths, _dir) = paths();
        let mut config = Config::default();
        config.backend.enabled = true;
        config.ui.enabled = false;
        let live = LiveStatus {
            stt_ready: Some(true),
            tts_ready: Some(true),
            hardware_audio: true,
            backend_ok: Some(false),
            backend_detail: Some("I can't reach the Cookie backend".into()),
            ..Default::default()
        };
        let report = run(&config, &paths, &live, false).await;
        let problems = report.problems();
        assert_eq!(problems.len(), 1);
        assert!(problems[0].hint.is_some());
        assert!(report.spoken_summary().to_lowercase().contains("backend"));
    }

    #[test]
    fn lists_are_joined_the_way_people_speak() {
        assert_eq!(join_naturally(&["a"]), "a");
        assert_eq!(join_naturally(&["a", "b"]), "a and b");
        assert_eq!(join_naturally(&["a", "b", "c"]), "a, b, and c");
    }

    #[tokio::test]
    async fn every_check_is_present_and_named() {
        let (paths, _dir) = paths();
        let report = run(&Config::default(), &paths, &LiveStatus::default(), false).await;
        let names: Vec<&str> = report.checks.iter().map(|c| c.name.as_str()).collect();
        for expected in [
            "config",
            "storage",
            "audio.input",
            "audio.output",
            "stt",
            "tts",
            "renderer",
            "api",
            "backend",
        ] {
            assert!(names.contains(&expected), "missing {expected}");
        }
        assert!(!report.to_lines().is_empty());
    }
}
