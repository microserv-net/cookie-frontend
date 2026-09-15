//! The binary.
//!
//! Structure worth noting: the async runtime is created explicitly rather
//! than with `#[tokio::main]`, because the window and the GPU have to own the
//! **main thread** on macOS and Windows. So the runtime runs everything else
//! in the background and the orb runs here, which is the opposite of the
//! usual arrangement and the only one that works on all three platforms.

use std::sync::Arc;

use clap::Parser;
use cookie_interface::api::ApiState;
use cookie_interface::cli::Cli;
use cookie_interface::config::Config;
use cookie_interface::engine::{Devices, Engine};
use cookie_interface::error::{Error, Result};
use cookie_interface::paths::Paths;
use cookie_interface::retention::{RetentionManager, RetentionPolicy};
use cookie_interface::{api, diagnostics, setup, testmode};

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_logging(&cli.log_level);

    match real_main(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cookie-interface: {e}");
            if let Some(hint) = e.hint() {
                eprintln!("  hint: {hint}");
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_logging(level: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    // `RUST_LOG` always wins; `--log-level` is the convenience.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("cookie_interface={level},warn")));
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .try_init();
}

fn real_main(cli: Cli) -> Result<()> {
    let paths = Arc::new(Paths::discover()?);
    // `load_or_init` tells us whether it had to create the file, which is the
    // one moment worth mentioning the config location unprompted.
    let (mut config, created) = Config::load_or_init(&paths)?;
    if created {
        println!(
            "wrote a default configuration to {}",
            paths.config_file().display()
        );
    }

    // Command-line overrides are applied before anything reads the config.
    apply_overrides(&mut config, &cli)?;
    config.validate()?;

    if !cli.set.is_empty() || cli.retention.is_some() {
        config.save(&paths)?;
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("cookie")
        .build()
        .map_err(|e| Error::Other(format!("could not start the async runtime: {e}")))?;

    // --- one-shot commands -------------------------------------------------

    if cli.paths {
        println!("{}", paths.describe());
        return Ok(());
    }

    if cli.config {
        println!("# {}", paths.config_file().display());
        println!("{}", config.to_toml()?);
        return Ok(());
    }

    let mut config = Arc::new(config);

    if cli.setup {
        let depth = if cli.config_only {
            setup::Depth::ConfigOnly
        } else {
            setup::Depth::Models
        };
        println!("Setting up cookie-interface…");
        if depth == setup::Depth::Models {
            if let Some(platform) = setup::Platform::detect() {
                println!(
                    "  about {:.0} MB to download, once, into {}",
                    setup::total_bytes(platform) as f64 / 1_000_000.0,
                    paths.models_dir().display()
                );
            }
        }
        let report = runtime.block_on(setup::run(&config, &paths, depth))?;
        for line in report.to_lines(&paths) {
            println!("{line}");
        }
        println!("\n{}", paths.describe());
        if !cli.test {
            return Ok(());
        }
        // `--setup --test` continues into the live check, and it must use the
        // configuration that was just written rather than the one loaded
        // before anything was downloaded.
        config = Arc::new(Config::load(&paths)?);
    }

    if cli.doctor {
        let report = runtime.block_on(async {
            let status = diagnostics::LiveStatus {
                hardware_audio: cfg!(feature = "audio-io"),
                ..Default::default()
            };
            diagnostics::run(&config, &paths, &status, true).await
        });
        println!("cookie-interface {}\n", cookie_interface::VERSION);
        for line in report.to_lines() {
            println!("{line}");
        }
        println!("\n  {}", report.spoken_summary());
        return match report.overall {
            diagnostics::Health::Failed => Err(Error::Other(
                "some capabilities are not working; see above".into(),
            )),
            _ => Ok(()),
        };
    }

    // --- housekeeping ------------------------------------------------------

    let retention = Arc::new(RetentionManager::open(&paths, config.retention.clone())?);

    if cli.clear_cache {
        let cache = paths.cache_dir().to_path_buf();
        if cache.exists() {
            std::fs::remove_dir_all(&cache).map_err(|e| Error::io(&cache, e))?;
        }
        paths.ensure()?;
        println!("cache cleared: {}", cache.display());
    }

    if cli.clear_audio {
        let report = retention.purge_all()?;
        println!(
            "generated audio cleared: {} files, {:.1} MiB",
            report.deleted,
            report.freed_bytes as f64 / 1_048_576.0
        );
    }

    // The retention policy is enforced *before* normal operation, every time.
    // A session that crashed mid-write, or a policy shortened while the
    // application was closed, is dealt with here rather than whenever the
    // first periodic sweep happens to fall.
    match retention.sweep() {
        Ok(report) if report.deleted > 0 || report.failed > 0 || report.adopted > 0 => {
            tracing::info!(
                deleted = report.deleted,
                adopted = report.adopted,
                failed = report.failed,
                freed_bytes = report.freed_bytes,
                "startup retention sweep"
            );
        }
        Ok(_) => {}
        Err(e) => tracing::warn!("startup retention sweep failed: {e}"),
    }

    // --- the session -------------------------------------------------------

    let devices = Devices::from_config(&config);
    let engine = runtime.block_on(Engine::start(
        config.clone(),
        paths.clone(),
        devices,
        retention.clone(),
    ))?;

    if cli.test {
        let outcome = runtime.block_on(testmode::run(&engine));
        runtime.block_on(engine.shutdown());
        return match outcome {
            Ok(outcome) => {
                println!(
                    "\n  heard: {}\n  name:  {}\n  audio: {}",
                    outcome.heard.unwrap_or_else(|| "—".into()),
                    outcome.name.unwrap_or_else(|| "—".into()),
                    if outcome.spoke { "ok" } else { "problem" }
                );
                Ok(())
            }
            Err(e) => Err(e),
        };
    }

    // The API server runs for as long as the process does.
    let api_state = ApiState {
        engine: engine.clone(),
        config: config.clone(),
        started: std::time::Instant::now(),
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = runtime.spawn(async move {
        if let Err(e) = api::serve(api_state, async {
            let _ = shutdown_rx.await;
        })
        .await
        {
            tracing::error!("http api: {e}");
            eprintln!("cookie-interface: {e}");
        }
    });

    println!(
        "cookie-interface {} — API on http://{}:{}",
        cookie_interface::VERSION,
        config.api.bind,
        config.api.port
    );
    if config.backend.enabled {
        println!("  backend: {}", config.backend.chat_url());
    }

    let ui = config.ui.enabled && !cli.no_ui;
    if ui {
        #[cfg(feature = "ui")]
        {
            // Enter the runtime so renderer tasks spawned from window events
            // have a reactor, then hand the main thread to winit.
            let _guard = runtime.enter();
            cookie_interface::renderer::run(engine.clone(), config.clone())?;
        }
        #[cfg(not(feature = "ui"))]
        {
            tracing::warn!("this build has no renderer; running headless");
            wait_for_signal(&runtime);
        }
    } else {
        wait_for_signal(&runtime);
    }

    let _ = shutdown_tx.send(());
    runtime.block_on(engine.shutdown());
    runtime.block_on(async {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
    });
    Ok(())
}

fn wait_for_signal(runtime: &tokio::runtime::Runtime) {
    runtime.block_on(async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => println!("\nstopping"),
            Err(e) => tracing::error!("could not listen for ctrl-c: {e}"),
        }
    });
}

/// Apply `--port`, `--set` and friends on top of the loaded configuration.
fn apply_overrides(config: &mut Config, cli: &Cli) -> Result<()> {
    if let Some(port) = cli.port {
        config.api.port = port;
    }
    if let Some(bind) = cli.bind {
        config.api.bind = bind;
    }
    if let Some(backend) = &cli.backend {
        config.backend.base_url = backend.clone();
        config.backend.enabled = true;
    }
    if let Some(seed) = cli.seed {
        config.ui.animation.seed = Some(seed);
    }
    if cli.no_ui {
        config.ui.enabled = false;
    }
    if let Some(policy) = &cli.retention {
        config.retention.policy = policy.parse::<RetentionPolicy>().map_err(Error::Config)?;
    }
    for assignment in &cli.set {
        apply_assignment(config, assignment)?;
    }
    Ok(())
}

/// `--set a.b.c=value`, applied through the serialised form so every
/// configuration field is reachable without a hand-written match arm.
fn apply_assignment(config: &mut Config, assignment: &str) -> Result<()> {
    let (key, value) = assignment
        .split_once('=')
        .ok_or_else(|| Error::Config(format!("--set expects key=value, got {assignment:?}")))?;
    let mut document: toml::Value = toml::Value::try_from(&*config)?;

    let parsed: toml::Value = value
        .parse::<toml::Value>()
        .or_else(|_| format!("\"{value}\"").parse::<toml::Value>())
        .map_err(|e| Error::Config(format!("could not read the value {value:?}: {e}")))?;

    let parts: Vec<&str> = key.split('.').collect();
    let mut cursor = &mut document;
    for part in &parts[..parts.len() - 1] {
        cursor = cursor
            .get_mut(*part)
            .ok_or_else(|| Error::Config(format!("no such setting: {key}")))?;
    }
    let last = parts[parts.len() - 1];
    let table = cursor
        .as_table_mut()
        .ok_or_else(|| Error::Config(format!("no such setting: {key}")))?;
    if !table.contains_key(last) {
        return Err(Error::Config(format!("no such setting: {key}")));
    }
    table.insert(last.to_string(), parsed);

    *config = document
        .try_into()
        .map_err(|e| Error::Config(format!("{key} did not take that value: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_reaches_nested_settings() {
        let mut config = Config::default();
        apply_assignment(&mut config, "tts.voice.rate=0.95").unwrap();
        assert_eq!(config.tts.voice.rate, 0.95);
        apply_assignment(&mut config, "ui.theme.hue=300").unwrap();
        assert_eq!(config.ui.theme.hue, 300.0);
        apply_assignment(&mut config, "backend.enabled=true").unwrap();
        assert!(config.backend.enabled);
    }

    #[test]
    fn set_accepts_bare_strings() {
        let mut config = Config::default();
        apply_assignment(&mut config, "backend.base_url=http://10.0.0.5:8080/api").unwrap();
        assert_eq!(config.backend.base_url, "http://10.0.0.5:8080/api");
    }

    #[test]
    fn unknown_settings_are_refused_rather_than_ignored() {
        let mut config = Config::default();
        assert!(apply_assignment(&mut config, "tts.voice.vibe=cool").is_err());
        assert!(apply_assignment(&mut config, "nonsense=1").is_err());
        assert!(apply_assignment(&mut config, "no-equals-sign").is_err());
    }

    #[test]
    fn overrides_do_what_the_flags_say() {
        let cli = Cli::parse_from([
            "cookie-interface",
            "--port",
            "9001",
            "--no-ui",
            "--seed",
            "42",
            "--backend",
            "http://192.168.1.42:8080/api",
            "--retention",
            "7d",
        ]);
        let mut config = Config::default();
        apply_overrides(&mut config, &cli).unwrap();
        assert_eq!(config.api.port, 9001);
        assert!(!config.ui.enabled);
        assert_eq!(config.ui.animation.seed, Some(42));
        assert!(config.backend.enabled);
        assert_eq!(config.retention.policy.to_string(), "7d");
    }

    #[test]
    fn a_bad_retention_policy_is_rejected() {
        let cli = Cli::parse_from(["cookie-interface", "--retention", "whenever"]);
        let mut config = Config::default();
        assert!(apply_overrides(&mut config, &cli).is_err());
    }
}
