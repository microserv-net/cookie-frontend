//! Crate-wide error type.
//!
//! Errors carry enough context to be actionable in a log line *and* to be
//! rendered to an API client as structured JSON (see `api::schema::ApiError`),
//! because "something went wrong" is useless to someone whose microphone is
//! muted.

use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("could not determine a per-user {kind} directory on this platform")]
    NoPlatformDir { kind: &'static str },

    #[error("i/o error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("audio device error: {0}")]
    AudioDevice(String),

    #[error("audio format {0} is not supported by this build")]
    UnsupportedAudioFormat(String),

    #[error("speech recognition failed: {0}")]
    Stt(String),

    #[error("speech synthesis failed: {0}")]
    Tts(String),

    #[error("model {name} is unavailable: {reason}")]
    ModelUnavailable { name: String, reason: String },

    #[error("model {name} is corrupted: {reason}")]
    ModelCorrupted { name: String, reason: String },

    #[error(
        "provider {provider} is not compiled into this build (rebuild with --features {feature})"
    )]
    ProviderNotCompiled {
        provider: String,
        feature: &'static str,
    },

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("invalid state transition: {from} -> {to}")]
    InvalidTransition { from: String, to: String },

    #[error("renderer error: {0}")]
    Renderer(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("the operation was cancelled")]
    Cancelled,

    #[error("port {port} is already in use")]
    PortInUse { port: u16 },

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }

    /// Stable, machine-readable discriminator for API clients. New variants
    /// may be added; clients must treat unknown codes as generic failures.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Config(_) => "config_error",
            Error::NoPlatformDir { .. } => "no_platform_dir",
            Error::Io { .. } => "io_error",
            Error::AudioDevice(_) => "audio_device_error",
            Error::UnsupportedAudioFormat(_) => "unsupported_audio_format",
            Error::Stt(_) => "stt_error",
            Error::Tts(_) => "tts_error",
            Error::ModelUnavailable { .. } => "model_unavailable",
            Error::ModelCorrupted { .. } => "model_corrupted",
            Error::ProviderNotCompiled { .. } => "provider_not_compiled",
            Error::BadRequest(_) => "bad_request",
            Error::InvalidTransition { .. } => "invalid_transition",
            Error::Renderer(_) => "renderer_error",
            Error::Network(_) => "network_error",
            Error::Cancelled => "cancelled",
            Error::PortInUse { .. } => "port_in_use",
            Error::Other(_) => "internal_error",
        }
    }

    /// A hint the user can actually act on, where we have one.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Error::AudioDevice(_) => Some(
                "check that an input/output device is connected and not exclusively \
                 held by another application; `cookie-interface --doctor` lists devices",
            ),
            Error::ModelUnavailable { .. } => {
                Some("run `cookie-interface --setup` to download the configured models")
            }
            Error::ModelCorrupted { .. } => {
                Some("run `cookie-interface --setup --force` to re-download the model")
            }
            Error::ProviderNotCompiled { .. } => {
                Some("see docs/installation.md for the feature flags of each provider")
            }
            Error::PortInUse { .. } => {
                Some("pass a different --port, or stop the process already bound to it")
            }
            _ => None,
        }
    }
}

impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<toml::ser::Error> for Error {
    fn from(e: toml::ser::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::BadRequest(e.to_string())
    }
}
