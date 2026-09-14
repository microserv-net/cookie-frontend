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

pub mod error;
pub mod paths;
pub mod state;
pub mod util;

pub use error::{Error, Result};
pub use paths::Paths;
pub use state::VoiceState;

/// Crate version, surfaced by `--version`, `/v1/health` and the `ready` event.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version of the HTTP API and event stream. Within `v1`, fields and
/// event types may be *added* but never removed or repurposed.
pub const API_VERSION: &str = "v1";
