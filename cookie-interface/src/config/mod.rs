//! Persistent configuration.
//!
//! Design rules:
//!
//! * every field has a default, so an empty (or partial, or older) config file
//!   still loads — unknown keys are ignored rather than rejected, which is what
//!   lets a newer build read an older file and vice versa;
//! * writes are atomic (temp file + rename) because a torn config file after a
//!   power cut is a support nightmare;
//! * nothing here knows about a *specific* STT/TTS model. Provider-specific
//!   knobs live in a free-form `options` map that each provider validates, so
//!   adding a provider never changes this file's schema.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::paths::Paths;
use crate::retention::RetentionPolicy;

pub mod voice;

pub use voice::{Gender, VoicePatch, VoiceSpec};

/// Bumped only for changes that need migration logic. Readers must tolerate a
/// *higher* version by ignoring unknown fields (serde does this for us).
pub const CONFIG_VERSION: u32 = 2;

/// Bring an older configuration file up to date.
///
/// A default that changes only helps people who have never run the program.
/// Anybody who has is holding a file written against the old one — which is
/// how an installation ended up asking for a wake word that had since been
/// switched off, and showing an orb whose visibility rule had since been
/// rewritten. The behaviour people see comes from their file, so the file has
/// to move.
///
/// Only settings the user is unlikely to have deliberately chosen are
/// touched, and only when they still hold the old default. Anything edited by
/// hand is left exactly as it is.
pub fn migrate(config: &mut Config) -> Vec<String> {
    let mut changes = Vec::new();
    if config.version >= CONFIG_VERSION {
        return changes;
    }

    // v1 → v2. The wake word shipped on, then went off; the orb's visibility
    // keyed off "the microphone is open", which is true from startup; and the
    // VAD had no absolute floor, so a quiet room registered as speech.
    if config.wake.enabled {
        config.wake.enabled = false;
        changes.push("the wake word is off until recognition is quick enough for it".into());
    }
    if config.ui.visibility.when_listening {
        config.ui.visibility.when_listening = false;
        changes.push("the orb no longer appears merely because the microphone is open".into());
    }
    if config.vad.floor_db > -20.0 {
        config.vad.floor_db = VadConfig::default().floor_db;
        changes.push("the voice detector has an absolute silence floor now".into());
    }
    if config.ui.width > 60 || config.ui.height > 60 {
        let defaults = UiConfig::default();
        config.ui.width = defaults.width;
        config.ui.height = defaults.height;
        changes.push("the orb is the size of a cursor now".into());
    }

    config.version = CONFIG_VERSION;
    changes
}

pub const DEFAULT_PORT: u16 = 8787;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub api: ApiConfig,
    /// Where the Cookie *backend* lives, if there is one. See
    /// [`BackendConfig`]; leaving it disabled is the normal state for a
    /// front-end driven entirely over the HTTP API.
    pub backend: BackendConfig,
    /// What Cookie may do on this machine, and when she asks first.
    pub tools: ToolsConfig,
    /// When Cookie is being spoken to, as opposed to merely listening.
    pub wake: WakeConfig,
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub retention: RetentionConfig,
    pub ui: UiConfig,
    pub logging: LoggingConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            api: ApiConfig::default(),
            backend: BackendConfig::default(),
            tools: ToolsConfig::default(),
            wake: WakeConfig::default(),
            audio: AudioConfig::default(),
            vad: VadConfig::default(),
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
            retention: RetentionConfig::default(),
            ui: UiConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Config {
    /// Load from `paths.config_file()`, falling back to defaults when the file
    /// does not exist yet. The caller decides whether to persist the defaults
    /// (first run does; `--test` does not, so a dry run leaves no residue).
    pub fn load(paths: &Paths) -> Result<Self> {
        let file = paths.config_file();
        match std::fs::read_to_string(&file) {
            Ok(text) => {
                let cfg: Config = toml::from_str(&text)?;
                cfg.validate()?;
                Ok(cfg)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(Error::io(&file, e)),
        }
    }

    /// Load, and write defaults back if there was no file. Returns whether this
    /// looked like a first run.
    ///
    /// An existing file is migrated if it predates the current version, and
    /// written back — see [`migrate`]. The alternative is a changed default
    /// that only takes effect for people who have never run the program,
    /// which is precisely the wrong half of the audience.
    pub fn load_or_init(paths: &Paths) -> Result<(Self, bool)> {
        let existed = paths.config_file().exists();
        let mut cfg = Self::load(paths)?;
        if !existed {
            cfg.save(paths)?;
            return Ok((cfg, true));
        }
        let changes = migrate(&mut cfg);
        if !changes.is_empty() {
            for change in &changes {
                println!("  updated your configuration: {change}");
            }
            cfg.save(paths)?;
        }
        Ok((cfg, false))
    }

    /// Atomic write: serialise to `config.toml.new`, fsync, rename over the
    /// target. A crash mid-write leaves the previous file intact.
    pub fn save(&self, paths: &Paths) -> Result<()> {
        let dir = paths.config_dir();
        std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        let target = paths.config_file();
        let tmp = target.with_extension("toml.new");
        let text = toml::to_string_pretty(self)?;

        use std::io::Write;
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| Error::io(&tmp, e))?;
            f.write_all(HEADER.as_bytes())
                .and_then(|_| f.write_all(text.as_bytes()))
                .and_then(|_| f.sync_all())
                .map_err(|e| Error::io(&tmp, e))?;
        }
        std::fs::rename(&tmp, &target).map_err(|e| Error::io(&target, e))?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.audio.sample_rate < 8_000 || self.audio.sample_rate > 192_000 {
            return Err(Error::Config(format!(
                "audio.sample_rate {} is outside 8000..=192000",
                self.audio.sample_rate
            )));
        }
        if self.audio.frame_ms == 0 || self.audio.frame_ms > 100 {
            return Err(Error::Config(
                "audio.frame_ms must be between 1 and 100".into(),
            ));
        }
        if !(0.25..=4.0).contains(&self.tts.voice.rate) {
            return Err(Error::Config("tts.voice.rate must be in 0.25..=4.0".into()));
        }
        if !(-12.0..=12.0).contains(&self.tts.voice.pitch) {
            return Err(Error::Config(
                "tts.voice.pitch must be in -12.0..=12.0 semitones".into(),
            ));
        }
        if self.vad.silence_ms < 100 {
            return Err(Error::Config(
                "vad.silence_ms below 100 will cut words in half".into(),
            ));
        }
        if self.backend.enabled {
            let url = self.backend.base_url.trim();
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(Error::Config(format!(
                    "backend.base_url must start with http:// or https:// (got {url:?})"
                )));
            }
            if url.starts_with("https://") && !cfg!(feature = "tls") {
                return Err(Error::Config(
                    "backend.base_url is https but this build has no TLS; \
                     rebuild with `--features tls` or use http:// on the LAN"
                        .into(),
                ));
            }
        }
        if self.api.max_body_bytes < 1024 {
            return Err(Error::Config("api.max_body_bytes is unusably small".into()));
        }
        if self.ui.target_fps == 0 || self.ui.target_fps > 480 {
            return Err(Error::Config("ui.target_fps must be in 1..=480".into()));
        }
        Ok(())
    }

    /// Path the config would be written to; handy in error messages.
    pub fn location(paths: &Paths) -> std::path::PathBuf {
        paths.config_file()
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn from_toml_str(text: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn read_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        Self::from_toml_str(&text)
    }
}

const HEADER: &str = "\
# cookie-interface configuration.
#
# This file is rewritten by the application (for example when you change the
# audio retention policy from the API), so comments you add below may be lost.
# Unknown keys are ignored, which keeps old and new builds compatible.
# Full reference: docs/configuration.md
\n";

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    /// Loopback by default. Binding to 0.0.0.0 puts a live microphone feed on
    /// the LAN, so it is never the default and requires `allow_remote`.
    pub bind: IpAddr,
    pub port: u16,
    /// Must be set to true *and* `bind` changed for a non-loopback bind to be
    /// accepted. Two locks on the same door, on purpose.
    pub allow_remote: bool,
    /// Optional shared secret. When set, every request needs
    /// `Authorization: Bearer <token>`. Required if `allow_remote` is on.
    pub auth_token: Option<String>,
    pub max_body_bytes: usize,
    /// Origins allowed for browser clients. Empty means no CORS headers.
    pub cors_allow_origins: Vec<String>,
    /// Cap on simultaneous SSE/WebSocket subscribers.
    pub max_subscribers: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: DEFAULT_PORT,
            allow_remote: false,
            auth_token: None,
            max_body_bytes: 1 << 20, // 1 MiB of text is already absurd for speech
            cors_allow_origins: Vec::new(),
            max_subscribers: 32,
        }
    }
}

impl ApiConfig {
    pub fn is_loopback(&self) -> bool {
        match self.bind {
            IpAddr::V4(v4) => v4.is_loopback(),
            IpAddr::V6(v6) => v6.is_loopback(),
        }
    }
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Substring match against device names; `None` uses the system default.
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    /// Internal pipeline rate. 16 kHz mono is what every Whisper-family model
    /// wants; device rates are resampled to this on the capture side.
    pub sample_rate: u32,
    /// Analysis/VAD frame size. 20 ms is the usual compromise between latency
    /// and stable energy estimates.
    pub frame_ms: u32,
    pub input_gain: f32,
    pub output_gain: f32,
    /// Drop the first N ms after a device opens; many microphones emit a click.
    pub warmup_ms: u32,
    /// Suppress capture while Cookie is speaking (poor man's echo control).
    /// Turn it off when you have a headset and want true barge-in.
    pub duck_input_while_speaking: bool,
    /// Open the microphone as soon as the application starts and keep it open.
    /// On by default: an assistant you have to click before speaking to is not
    /// an assistant. Turn it off for push-to-talk via `POST /v1/listen`.
    pub listen_on_start: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: None,
            output_device: None,
            sample_rate: 16_000,
            frame_ms: 20,
            input_gain: 1.0,
            output_gain: 1.0,
            warmup_ms: 120,
            duck_input_while_speaking: true,
            listen_on_start: true,
        }
    }
}

impl AudioConfig {
    pub fn frame_samples(&self) -> usize {
        (self.sample_rate as usize * self.frame_ms as usize) / 1000
    }
}

// ---------------------------------------------------------------------------
// VAD
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    pub enabled: bool,
    /// Speech must exceed the adaptive noise floor by this much (dB).
    pub threshold_db: f32,
    /// Absolute level below which nothing counts as speech, in dBFS.
    ///
    /// The threshold above is *relative* to an adaptive noise floor, and in a
    /// quiet room that floor drops to around -60 dBFS — at which point a
    /// fluctuation of a few decibels clears it and the fan becomes an
    /// utterance. That is what produced "speech detected at -48 dB" with
    /// nobody in the room. Speech into a laptop microphone sits between -30
    /// and -12 dBFS; anything below this is the room, whatever its margin
    /// over the floor.
    pub floor_db: f32,
    /// Consecutive speech-ish milliseconds before we declare speech started.
    pub speech_ms: u32,
    /// Trailing silence that ends an utterance.
    pub silence_ms: u32,
    /// Audio kept from *before* the trigger, so the first phoneme survives.
    pub preroll_ms: u32,
    /// Hard ceiling; protects against a stuck-open microphone.
    pub max_utterance_ms: u32,
    /// Ignore blips shorter than this (door slams, key clicks).
    pub min_utterance_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Above the adaptive noise floor. Nine was too eager: a quiet room
            // on a laptop microphone drifts by that much, and the result was
            // "speech detected" at -45 dB with nobody talking.
            threshold_db: 11.0,
            floor_db: -42.0,
            // Long enough that a keyboard clack or a chair creak cannot open
            // an utterance on its own.
            speech_ms: 200,
            silence_ms: 700,
            preroll_ms: 300,
            max_utterance_ms: 30_000,
            // Judged on voiced audio only. Anything shorter than this is a
            // noise, not a sentence.
            min_utterance_ms: 350,
        }
    }
}

// ---------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SttProviderKind {
    /// Whisper running on this machine through sherpa-onnx, fetched by
    /// `--setup`. What you want unless you have a model server already.
    Local,
    /// Deterministic in-process fake. No audio leaves the machine, no model
    /// needed; used by tests and by `--test` when nothing else is configured.
    Mock,
    /// OpenAI-compatible `/audio/transcriptions` endpoint. Works with
    /// whisper.cpp's server, faster-whisper-server, vLLM, or a hosted API.
    Http,
    /// A local process speaking the newline-delimited JSON protocol in
    /// docs/models.md. This is how sherpa-onnx / whisper.cpp binaries are
    /// plugged in without dragging a C++ build into `cargo build`.
    Sidecar,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub provider: SttProviderKind,
    /// Model identifier passed through to the provider, e.g.
    /// `whisper-large-v3-turbo`.
    pub model: String,
    /// BCP-47 tag, or `auto` to let the model decide.
    pub language: String,
    /// Ask the provider for partial hypotheses while speech is in flight.
    pub partials: bool,
    /// How often partials are requested, in milliseconds.
    pub partial_interval_ms: u32,
    pub endpoint: String,
    pub api_key_env: Option<String>,
    pub timeout_ms: u64,
    /// Command + args for `Sidecar`.
    pub sidecar_command: Vec<String>,
    /// Provider-specific extras, passed through untouched.
    pub options: BTreeMap<String, toml::Value>,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            provider: SttProviderKind::Mock,
            model: "whisper-large-v3-turbo".into(),
            language: "en".into(),
            partials: true,
            partial_interval_ms: 400,
            endpoint: "http://127.0.0.1:8080/v1/audio/transcriptions".into(),
            api_key_env: None,
            timeout_ms: 30_000,
            sidecar_command: Vec::new(),
            options: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TtsProviderKind {
    /// Kokoro running on this machine through sherpa-onnx, fetched by
    /// `--setup`. The British female voice the project is built around.
    Local,
    /// Deterministic synthetic tone-speech. Not a voice; it exists so the
    /// pipeline, the API and the orb can be exercised with no model at all.
    Mock,
    /// The operating system's own synthesiser (`say`, SAPI, `espeak-ng`).
    /// Always available, no download, decent British female voices on macOS
    /// and Windows.
    System,
    /// HTTP provider: Qwen3-TTS, Kokoro-FastAPI, or any OpenAI-compatible
    /// `/audio/speech` endpoint. Supports chunked streaming where the server
    /// does.
    Http,
    /// Local process speaking the JSON-lines protocol (docs/models.md).
    Sidecar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HttpTtsDialect {
    /// `POST {endpoint}` with `{model, input, voice, response_format}` —
    /// OpenAI `/v1/audio/speech`, Kokoro-FastAPI and friends.
    OpenAiCompatible,
    /// Alibaba DashScope style `{model, input:{text, voice}, parameters:{}}`,
    /// which is how Qwen3-TTS is exposed.
    Qwen3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub provider: TtsProviderKind,
    pub model: String,
    pub voice: VoiceSpec,
    pub dialect: HttpTtsDialect,
    pub endpoint: String,
    /// Name of the environment variable holding the key. The key itself is
    /// never written to the config file.
    pub api_key_env: Option<String>,
    pub timeout_ms: u64,
    /// Request chunked audio and start playing before synthesis finishes.
    pub streaming: bool,
    pub sidecar_command: Vec<String>,
    /// Keep a copy of generated audio on disk (governed by `retention`).
    /// Off means audio only ever exists in memory.
    pub persist_audio: bool,
    pub options: BTreeMap<String, toml::Value>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            provider: TtsProviderKind::System,
            model: "qwen3-tts-flash".into(),
            voice: VoiceSpec::default(),
            dialect: HttpTtsDialect::Qwen3,
            endpoint: "http://127.0.0.1:8081/v1/audio/speech".into(),
            api_key_env: Some("COOKIE_TTS_API_KEY".into()),
            timeout_ms: 30_000,
            streaming: true,
            sidecar_command: Vec::new(),
            persist_audio: true,
            options: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    pub policy: RetentionPolicy,
    /// Background sweep cadence. A sweep also always runs at startup.
    pub sweep_interval_minutes: u64,
    /// Belt and braces: even under `forever`, never keep more than this many
    /// files. 0 disables the cap.
    pub max_files: usize,
    /// Same idea for total bytes. 0 disables.
    pub max_bytes: u64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            policy: RetentionPolicy::Hours(24),
            sweep_interval_minutes: 30,
            max_files: 500,
            max_bytes: 512 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PresentationMode {
    /// Borderless, transparent, always-on-top orb. The default: it is the
    /// presentation that actually feels like a presence rather than an app.
    FloatingOrb,
    /// Ordinary decorated window. Fallback when the compositor cannot do
    /// transparency, and the right choice for a second monitor.
    Window,
    /// Small transparent orb that rides alongside the mouse pointer. The
    /// default: Cookie appears next to whatever you are already looking at
    /// instead of owning a window somewhere else on the screen.
    CursorCompanion,
    /// Small orb pinned to a screen corner.
    Docked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DockCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub enabled: bool,
    pub presentation: PresentationMode,
    pub always_on_top: bool,
    /// Transparent background. Automatically disabled with a warning if the
    /// platform rejects it.
    pub transparent: bool,
    /// Let clicks pass through to whatever is behind the orb. Useful for the
    /// companion modes; makes the orb non-interactive.
    pub click_through: bool,
    pub width: u32,
    pub height: u32,
    pub dock_corner: DockCorner,
    /// Distance from the cursor in `cursor-companion` mode, in logical pixels.
    pub cursor_offset: [f32; 2],
    /// How the orb decides when to be on screen at all.
    pub visibility: VisibilityConfig,
    /// Seconds of smoothing applied to cursor following. 0.0 means the orb is
    /// pinned to the pointer with no lag at all, which is the default: a
    /// trailing orb reads as sluggish software rather than as a companion.
    pub cursor_follow_lag: f32,
    pub target_fps: u32,
    pub vsync: bool,
    /// Drop to this frame rate when the window is not focused. Saves a laptop
    /// battery without making the orb look frozen.
    pub unfocused_fps: u32,
    pub theme: ThemeConfig,
    pub animation: AnimationConfig,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            presentation: PresentationMode::CursorCompanion,
            always_on_top: true,
            transparent: true,
            click_through: true,
            // Measured against the pointer, because that is what it sits
            // beside: the orb itself is about twenty points across, a little
            // smaller than the arrow. The window is wider than the orb
            // because the glow needs somewhere to fall off.
            // The body is about five points across — a bead, not a bubble.
            // The window is much wider than that because the glow has to fade
            // to nothing *inside* it: anything clipped at the window edge is
            // clipped along a straight line, and that straight line is the
            // boundary that keeps reappearing.
            width: 28,
            height: 28,
            dock_corner: DockCorner::BottomRight,
            // To the right of the arrow and level with it: the pointer's hot
            // spot is its top-left corner, so anything below reads as
            // detached, and anything to the left sits under the hand.
            // Measured from the window's centre to the pointer's hot spot.
            cursor_offset: [14.0, 2.0],
            visibility: VisibilityConfig::default(),
            cursor_follow_lag: 0.0,
            target_fps: 60,
            vsync: true,
            unfocused_fps: 30,
            theme: ThemeConfig::default(),
            animation: AnimationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThemeConfig {
    /// Base hue in degrees. 28 is the warm brown the assistant is named
    /// after; the reference image's violet is 276 if you prefer it.
    pub hue: f32,
    /// Secondary hue used for the inner core and rim scatter. Sitting it a
    /// little above the base hue gives the body an amber centre.
    pub hue_secondary: f32,
    pub saturation: f32,
    /// Overall emission multiplier.
    pub intensity: f32,
    /// Background alpha when transparency is unavailable.
    pub background_alpha: f32,
    /// Bloom/glow strength.
    pub glow: f32,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            hue: 28.0,
            hue_secondary: 42.0,
            saturation: 0.92,
            // Well below one. Above it the shader saturates and the brown
            // washes out to white, which is what made it look like a lamp
            // rather than a small warm thing.
            intensity: 0.78,
            background_alpha: 0.0,
            // Enough to look lit from within, not enough to draw a halo.
            glow: 0.7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnimationConfig {
    /// Fixed seed for reproducible behaviour selection. `None` seeds from the
    /// clock and logs the seed it chose.
    pub seed: Option<u64>,
    /// 0 = always pick the calmest behaviour, 1 = full variety.
    pub variation: f32,
    /// Scales every audio-reactive displacement.
    pub reactivity: f32,
    /// Honour "reduce motion": slower, smaller, no flicker.
    pub reduce_motion: bool,
    /// Seconds a behaviour runs before the scheduler considers a change.
    pub min_behavior_seconds: f32,
    pub max_behavior_seconds: f32,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            seed: None,
            variation: 0.85,
            reactivity: 1.0,
            reduce_motion: false,
            min_behavior_seconds: 6.0,
            max_behavior_seconds: 22.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// `error`, `warn`, `info`, `debug`, `trace`, or any `RUST_LOG` filter.
    pub level: String,
    pub to_file: bool,
    /// Log recognised text. Off by default: transcripts are the most sensitive
    /// thing this process handles.
    pub log_transcripts: bool,
    pub json: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            to_file: false,
            log_transcripts: false,
            json: false,
        }
    }
}

/// When the orb is on screen.
///
/// Cookie is not a desktop widget. An assistant that glows next to your
/// pointer all day stops being a presence and becomes clutter, so by default
/// the orb is *absent* until there is a reason for it: you called her, she is
/// working, she is speaking, something went wrong, or she is watching the
/// screen in order to offer a suggestion. Each of those is a separate switch
/// because people disagree about which ones are welcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VisibilityConfig {
    /// Show while idle. Off by default — this is the setting that decides
    /// whether Cookie is a presence or a permanent fixture.
    pub when_idle: bool,
    /// Show merely because the microphone is open.
    ///
    /// Off, and this is the setting that was wrong: the microphone is open
    /// from the moment the application starts, so "show while listening"
    /// meant "show always", which is exactly what the orb must not do.
    pub when_listening: bool,
    /// Show while you are actually speaking.
    ///
    /// This is the one that matters. The orb appears when you start talking
    /// and leaves when you stop — it is a response to you, not a fixture.
    pub when_hearing_speech: bool,
    /// Show while a request is being worked on, locally or by the backend.
    pub when_working: bool,
    /// Show while she is speaking.
    pub when_speaking: bool,
    /// Show while an error is being reported.
    pub when_error: bool,
    /// Show while passively observing (screen capture for suggestions). This
    /// one is deliberately not optional in spirit: observation that cannot be
    /// seen is surveillance, so turning it off also turns observation off.
    pub when_observing: bool,
    /// Seconds the orb lingers after the reason for it disappears, so a quick
    /// exchange does not flicker it in and out.
    pub linger_seconds: f32,
    /// Seconds the appearance and disappearance fade take.
    pub fade_seconds: f32,
}

impl Default for VisibilityConfig {
    fn default() -> Self {
        Self {
            when_idle: false,
            when_listening: false,
            when_hearing_speech: true,
            when_working: true,
            when_speaking: true,
            when_error: true,
            when_observing: true,
            linger_seconds: 1.2,
            fade_seconds: 0.35,
        }
    }
}

impl VisibilityConfig {
    /// Whether a given voice state is a reason to be on screen.
    /// Whether a state on its own is a reason to be on screen.
    ///
    /// Note what is absent: hearing speech is not a state, it is an event,
    /// and the renderer tracks it separately. A state machine that is
    /// "Listening" for eight hours says nothing about whether anybody is
    /// talking.
    pub fn wants(&self, state: crate::state::VoiceState) -> bool {
        use crate::state::VoiceState::*;
        match state {
            Idle => self.when_idle,
            Listening => self.when_listening,
            Processing => self.when_working,
            Speaking => self.when_speaking,
            Interrupted => self.when_speaking || self.when_idle,
            Error => self.when_error,
        }
    }
}

/// The wake word.
///
/// The microphone is open all the time; this is what decides whether anything
/// heard is meant for Cookie. Until she is called, transcripts are checked for
/// her name and discarded — nothing is emitted, nothing reaches the backend,
/// and the orb does not appear.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WakeConfig {
    /// Require the name before acting on anything.
    ///
    /// Off for now. The wake word depends on recognising one short word
    /// reliably in a room, and until recognition itself is comfortably fast
    /// and accurate it gates everything behind its worst case — which is what
    /// made the whole thing feel broken. The gate is built and tested; this
    /// flag turns it on.
    pub enabled: bool,
    /// What to call her.
    pub word: String,
    /// How long she stays attentive after being called, in seconds.
    ///
    /// Long enough for a follow-up sentence without repeating the name, short
    /// enough that forgetting to dismiss her is harmless — which matters,
    /// because forgetting is the normal case.
    pub attention_secs: u64,
    /// Say something short when she starts listening.
    ///
    /// On, because the alternative is being answered by silence and having no
    /// way to tell whether the name was heard, the request was heard, or
    /// nothing was.
    pub acknowledge: bool,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            word: "cookie".into(),
            attention_secs: 20,
            acknowledge: true,
        }
    }
}

/// Local tool execution.
///
/// Off by default. The backend being able to run commands on your laptop is
/// a decision you should make deliberately, not one you discover you made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    /// Allow the backend to ask this machine to do things at all.
    pub enabled: bool,
    /// Tools that are never offered, by name — e.g. `["shell.run"]` for a
    /// machine where you want Cookie to look but not touch.
    pub blocked: Vec<String>,
    /// Anything at or below this risk runs without asking:
    /// `safe`, `normal`, `dangerous`, `critical`.
    pub auto_approve_up_to: crate::tools::Risk,
    /// Refuse anything above this outright, whatever is said.
    pub refuse_above: crate::tools::Risk,
    /// Always confirm irreversible operations, even after "stop asking".
    /// Turning this off is possible and inadvisable.
    pub always_confirm_critical: bool,
    /// How long to wait for a spoken answer to a confirmation before giving
    /// up and telling the backend the request was declined. Silence is not
    /// consent, so this expires into a refusal rather than an approval.
    pub confirmation_timeout_secs: u64,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            blocked: Vec::new(),
            auto_approve_up_to: crate::tools::Risk::Safe,
            refuse_above: crate::tools::Risk::Critical,
            always_confirm_critical: true,
            confirmation_timeout_secs: 45,
        }
    }
}

impl ToolsConfig {
    /// The policy these settings describe.
    pub fn policy(&self) -> crate::tools::PermissionPolicy {
        crate::tools::PermissionPolicy {
            auto_approve_up_to: self.auto_approve_up_to,
            refuse_above: self.refuse_above,
            always_confirm_critical: self.always_confirm_critical,
            blocked: self.blocked.iter().cloned().collect(),
            suppressed: None,
        }
    }
}

/// Connection to the Cookie backend (the mind).
///
/// `cookie-interface` is deliberately usable with no backend at all: another
/// program can drive it entirely through the local HTTP API. But the common
/// deployment is "the interface runs on the user's machine, the backend runs
/// somewhere else" — a home server, a container, a box on the LAN — so the
/// interface also knows how to *call out*.
///
/// When `enabled`, every final transcript is posted to
/// `{base_url}{chat_path}` and whatever the backend streams back is spoken as
/// it arrives. The backend never needs to know where the interface is, which
/// means it works unchanged behind NAT, over Tailscale, or on localhost.
///
/// ```toml
/// [backend]
/// enabled = true
/// base_url = "http://192.168.1.42:8080/api"
/// chat_path = "/v1/chat"
/// ```
///
/// The wire protocol is documented in `docs/backend.md`. It is intentionally
/// small — one POST that streams newline-delimited JSON back — so that the
/// future Cookie backend can implement it in any language.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    /// Off by default: an interface that silently phones home would be a
    /// surprising default for a microphone application.
    pub enabled: bool,
    /// Root of the backend API, e.g. `http://192.168.1.42:8080/api`.
    /// A trailing slash is tolerated.
    pub base_url: String,
    /// Path appended to `base_url` for a turn of conversation.
    pub chat_path: String,
    /// Optional health probe path, used by `--doctor`.
    pub health_path: String,
    /// Path used to cancel backend work.
    pub cancel_path: String,
    /// Environment variable holding a bearer token, if the backend wants one.
    /// The token itself is never stored in the config file.
    pub auth_token_env: Option<String>,
    /// How long to wait for the backend to *start* answering.
    pub connect_timeout_ms: u64,
    /// How long a single turn may take in total.
    pub request_timeout_ms: u64,
    /// Forward interim transcripts as well as final ones, so a backend can
    /// start thinking before the user stops talking.
    pub send_partials: bool,
    /// Speak the backend's reply as it streams, rather than waiting for the
    /// full text. Off only makes sense for debugging.
    pub speak_streaming_reply: bool,
    /// Opaque session identifier sent with every turn. Generated on first run
    /// when empty; the backend decides what, if anything, it means.
    pub session_id: String,
    /// Extra headers, e.g. `{ "X-Cookie-Client" = "kitchen" }`.
    pub headers: BTreeMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://127.0.0.1:8080/api".into(),
            chat_path: "/v1/chat".into(),
            health_path: "/v1/health".into(),
            cancel_path: "/v1/cancel".into(),
            auth_token_env: Some("COOKIE_BACKEND_TOKEN".into()),
            connect_timeout_ms: 5_000,
            request_timeout_ms: 120_000,
            send_partials: false,
            speak_streaming_reply: true,
            session_id: String::new(),
            headers: BTreeMap::new(),
        }
    }
}

impl BackendConfig {
    /// Absolute URL for a conversational turn.
    pub fn chat_url(&self) -> String {
        join_url(&self.base_url, &self.chat_path)
    }

    /// Absolute URL for the health probe.
    pub fn health_url(&self) -> String {
        join_url(&self.base_url, &self.health_path)
    }

    /// Bearer token from the environment, if configured and present.
    pub fn auth_token(&self) -> Option<String> {
        self.auth_token_env
            .as_deref()
            .and_then(|k| std::env::var(k).ok())
            .filter(|v| !v.is_empty())
    }
}

/// Join a base URL and a path. Public so the backend client can build the
/// less common endpoints without duplicating the rule.
pub fn join_public_url(base: &str, path: &str) -> String {
    join_url(base, path)
}

/// Join a base URL and a path without producing `//` or losing a segment.
fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn default_bind_is_loopback() {
        let c = Config::default();
        assert!(
            c.api.is_loopback(),
            "must not expose the microphone by default"
        );
        assert!(!c.api.allow_remote);
        assert_eq!(c.api.port, DEFAULT_PORT);
    }

    #[test]
    fn roundtrips_through_toml() {
        let cfg = Config::default();
        let text = cfg.to_toml().unwrap();
        let back = Config::from_toml_str(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn partial_file_fills_in_defaults() {
        let text = r#"
            version = 1
            [api]
            port = 9999
        "#;
        let cfg = Config::from_toml_str(text).unwrap();
        assert_eq!(cfg.api.port, 9999);
        assert_eq!(cfg.audio.sample_rate, 16_000);
        assert_eq!(cfg.tts.voice.language, "en-GB");
    }

    #[test]
    fn unknown_keys_are_ignored_for_forward_compatibility() {
        let text = r#"
            version = 99
            future_section_from_a_newer_build = true
            [api]
            port = 1234
            some_future_key = "hello"
        "#;
        let cfg = Config::from_toml_str(text).unwrap();
        assert_eq!(cfg.api.port, 1234);
    }

    #[test]
    fn validation_rejects_nonsense() {
        let mut c = Config::default();
        c.audio.sample_rate = 10;
        assert!(c.validate().is_err());

        let mut c = Config::default();
        c.tts.voice.rate = 99.0;
        assert!(c.validate().is_err());

        let mut c = Config::default();
        c.vad.silence_ms = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn frame_samples_matches_rate_and_frame_ms() {
        let a = AudioConfig::default();
        assert_eq!(a.frame_samples(), 320); // 20 ms @ 16 kHz
    }

    #[test]
    fn an_old_file_is_brought_up_to_date() {
        // The situation this exists for: a file written when the wake word
        // shipped on and the orb appeared whenever the microphone was open.
        // The user sees their file's behaviour, not the code's defaults.
        let mut old = Config {
            version: 1,
            ..Default::default()
        };
        old.wake.enabled = true;
        old.ui.visibility.when_listening = true;
        old.vad.floor_db = 0.0;
        old.ui.width = 160;
        old.ui.height = 160;

        let changes = migrate(&mut old);
        assert_eq!(changes.len(), 4, "{changes:?}");
        assert!(!old.wake.enabled);
        assert!(!old.ui.visibility.when_listening);
        assert!(old.vad.floor_db < -20.0);
        assert!(old.ui.width <= 60);
        assert_eq!(old.version, CONFIG_VERSION);
    }

    #[test]
    fn migration_leaves_a_current_file_alone() {
        let mut current = Config::default();
        current.wake.enabled = true; // deliberately switched on
        assert!(migrate(&mut current).is_empty());
        assert!(current.wake.enabled, "a deliberate choice must survive");
    }

    #[test]
    fn backend_urls_join_cleanly() {
        let mut b = BackendConfig::default();
        b.base_url = "http://192.168.1.42:8080/api/".into();
        assert_eq!(b.chat_url(), "http://192.168.1.42:8080/api/v1/chat");
        b.chat_path = "v1/turn".into();
        assert_eq!(b.chat_url(), "http://192.168.1.42:8080/api/v1/turn");
    }

    #[test]
    fn backend_url_scheme_is_validated() {
        let mut c = Config::default();
        c.backend.enabled = true;
        c.backend.base_url = "192.168.1.42:8080".into();
        assert!(c.validate().is_err());
        c.backend.base_url = "http://192.168.1.42:8080".into();
        assert!(c.validate().is_ok());
    }
}
