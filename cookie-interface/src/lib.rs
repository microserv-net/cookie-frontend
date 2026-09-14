//! # cookie-interface
//!
//! The front-facing voice interface for the Cookie assistant: microphone,
//! voice activity detection, speech recognition, speech synthesis, an
//! audio-reactive procedural orb, and a local streaming HTTP API that ties
//! them together.
//!
//! **What this crate is not:** it contains no language model, no agent, no
//! memory, no reasoning and no personality. It gives Cookie ears, a voice and
//! a face; the mind lives in a separate process (the Cookie *backend*) and
//! talks to this one over HTTP.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod api;
pub mod audio;
pub mod backend;
pub mod config;
pub mod diagnostics;
pub mod engine;
pub mod error;
pub mod events;
pub mod intent;
pub mod paths;
pub mod retention;
pub mod sidecar;
pub mod state;
pub mod stt;
pub mod tasks;
pub mod tts;
pub mod util;

pub use config::Config;
pub use error::{Error, Result};
pub use events::{Command, EventBus, VoiceEvent};
pub use paths::Paths;
pub use state::VoiceState;

/// Crate version, surfaced by `--version`, `/v1/health` and the `ready` event.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version of the HTTP API and event stream. Within `v1`, fields and
/// event types may be *added* but never removed or repurposed.
pub const API_VERSION: &str = "v1";
