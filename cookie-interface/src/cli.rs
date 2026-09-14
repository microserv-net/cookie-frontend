//! Command line.
//!
//! One binary, a handful of switches, no subcommand tree. Running
//! `cookie-interface` with no arguments should be the right thing to do, and
//! everything else exists because there is a real moment when you need it:
//! setting up, checking on it, moving the port, or reproducing a bug.

use std::net::IpAddr;

use clap::Parser;

/// Cookie's voice interface: microphone, recognition, speech and the orb.
#[derive(Debug, Parser)]
#[command(
    name = "cookie-interface",
    version,
    about = "The face, ears and voice of the Cookie assistant",
    long_about = None,
)]
pub struct Cli {
    /// Port for the local HTTP API.
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,

    /// Address to bind the API to. Loopback unless you really mean it;
    /// binding elsewhere also requires `api.allow_remote` and an auth token.
    #[arg(long, value_name = "IP")]
    pub bind: Option<IpAddr>,

    /// Address of the Cookie backend, e.g. `http://192.168.1.42:8080/api`.
    /// Overrides `[backend].base_url` and switches the backend on.
    #[arg(long, value_name = "URL")]
    pub backend: Option<String>,

    /// Run a full end-to-end check: ask your name, listen, and greet you.
    #[arg(long)]
    pub test: bool,

    /// Prepare directories, write a default configuration, and verify that
    /// the configured models can actually be reached.
    #[arg(long)]
    pub setup: bool,

    /// Report on every capability and exit. The same checks Cookie runs when
    /// you ask her whether she is alright.
    #[arg(long)]
    pub doctor: bool,

    /// Print the configuration file path and contents, then exit.
    #[arg(long)]
    pub config: bool,

    /// Set a configuration value and save it, e.g.
    /// `--set tts.voice.rate=0.95`. Repeatable.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,

    /// Run without the orb. The API, speech and recognition all still work.
    #[arg(long)]
    pub no_ui: bool,

    /// Fix the animation seed so a visual bug can be reproduced exactly.
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// Change the generated-audio retention policy and save it:
    /// `immediate`, `1h`, `24h`, `7d`, `30d`, `forever`.
    #[arg(long, value_name = "POLICY")]
    pub retention: Option<String>,

    /// Delete every piece of generated audio now, then continue.
    #[arg(long)]
    pub clear_audio: bool,

    /// Empty the cache directory (downloaded models are *not* touched).
    #[arg(long)]
    pub clear_cache: bool,

    /// Log level: error, warn, info, debug, trace. `RUST_LOG` wins if set.
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,

    /// Print every configuration and storage path, then exit.
    #[arg(long)]
    pub paths: bool,
}

impl Cli {
    /// True when the invocation is a one-shot command rather than a session.
    pub fn is_oneshot(&self) -> bool {
        self.doctor || self.config || self.paths || (self.setup && !self.test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn bare_invocation_starts_a_session() {
        let cli = Cli::parse_from(["cookie-interface"]);
        assert!(!cli.is_oneshot());
        assert!(cli.port.is_none());
        assert!(!cli.test);
    }

    #[test]
    fn port_and_bind_parse() {
        let cli = Cli::parse_from(["cookie-interface", "--port", "8787", "--bind", "0.0.0.0"]);
        assert_eq!(cli.port, Some(8787));
        assert_eq!(cli.bind.unwrap().to_string(), "0.0.0.0");
    }

    #[test]
    fn setting_values_is_repeatable() {
        let cli = Cli::parse_from([
            "cookie-interface",
            "--set",
            "tts.voice.rate=0.95",
            "--set",
            "ui.theme.hue=28",
        ]);
        assert_eq!(cli.set.len(), 2);
    }

    #[test]
    fn doctor_is_a_oneshot() {
        assert!(Cli::parse_from(["cookie-interface", "--doctor"]).is_oneshot());
        assert!(Cli::parse_from(["cookie-interface", "--paths"]).is_oneshot());
        // `--setup --test` is a session: set up, then run the live check.
        assert!(!Cli::parse_from(["cookie-interface", "--setup", "--test"]).is_oneshot());
    }
}
