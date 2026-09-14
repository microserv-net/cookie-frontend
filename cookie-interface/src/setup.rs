//! First run.
//!
//! What `--setup` deliberately does **not** do is download gigabytes during
//! `cargo build`. Build-time downloads make builds non-reproducible, break on
//! aeroplanes and corporate proxies, and turn a compile error into a network
//! error. So the build produces a binary that runs immediately against the
//! operating system's own voice, and models are fetched here — once,
//! explicitly, with a progress line and a checksum.
//!
//! A model is described in configuration rather than hard-coded, because the
//! whole point of the provider abstraction is that the model is replaceable:
//!
//! ```toml
//! [stt.options]
//! model_url = "https://example.invalid/whisper-large-v3-turbo.onnx"
//! model_sha256 = "9f2c…"
//! ```

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::paths::Paths;

/// What setup did, for printing and for tests.
#[derive(Debug, Default, PartialEq)]
pub struct SetupReport {
    pub created_directories: Vec<PathBuf>,
    pub wrote_config: bool,
    pub downloaded: Vec<String>,
    pub skipped: Vec<String>,
    pub warnings: Vec<String>,
}

/// Prepare this machine.
///
/// Idempotent: running it twice is harmless, and running it after an
/// interrupted attempt resumes rather than starts over.
pub async fn run(config: &Config, paths: &Paths) -> Result<SetupReport> {
    let mut report = SetupReport::default();

    for dir in paths.all_directories() {
        if !dir.exists() {
            std::fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
            report.created_directories.push(dir);
        }
    }

    let config_file = paths.config_file();
    if !config_file.exists() {
        config.save(paths)?;
        report.wrote_config = true;
    }

    for (kind, options) in [("stt", &config.stt.options), ("tts", &config.tts.options)] {
        let Some(url) = options.get("model_url").and_then(|v| v.as_str()) else {
            continue;
        };
        let expected = options
            .get("model_sha256")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let name = url.rsplit('/').next().unwrap_or("model.bin").to_string();
        let target = paths.model_dir(kind).join(&name);
        match fetch_model(url, &target, expected.as_deref()).await {
            Ok(true) => report.downloaded.push(name),
            Ok(false) => report.skipped.push(name),
            Err(e) => report.warnings.push(format!("{kind} model: {e}")),
        }
    }

    Ok(report)
}

/// Download `url` to `target` unless it is already there and intact.
///
/// Returns `true` if something was actually fetched. The download goes to a
/// `.part` file and is renamed only after the checksum matches, so an
/// interrupted setup never leaves a half-model that looks complete.
pub async fn fetch_model(url: &str, target: &Path, expected_sha256: Option<&str>) -> Result<bool> {
    if target.exists() {
        match expected_sha256 {
            Some(expected) => {
                let actual = sha256_file(target)?;
                if actual.eq_ignore_ascii_case(expected) {
                    return Ok(false);
                }
                tracing::warn!(
                    "{} does not match its checksum; downloading it again",
                    target.display()
                );
                std::fs::remove_file(target).map_err(|e| Error::io(target, e))?;
            }
            None => return Ok(false),
        }
    }

    #[cfg(not(feature = "http-providers"))]
    {
        let _ = url;
        Err(Error::ProviderNotCompiled {
            provider: "model download".into(),
            feature: "http-providers",
        })
    }

    #[cfg(feature = "http-providers")]
    {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let part = target.with_extension("part");
        println!("  downloading {url}");
        println!("          to {}", target.display());

        let response = reqwest::get(url)
            .await
            .map_err(|e| Error::Network(format!("could not start the download: {e}")))?;
        if !response.status().is_success() {
            return Err(Error::Network(format!(
                "the server answered {} for {url}",
                response.status()
            )));
        }
        let total = response.content_length();
        let mut file = tokio::fs::File::create(&part)
            .await
            .map_err(|e| Error::io(&part, e))?;
        let mut stream = response.bytes_stream();
        let mut written: u64 = 0;
        let mut last_report = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Error::Network(format!("download interrupted: {e}")))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| Error::io(&part, e))?;
            written += chunk.len() as u64;
            if last_report.elapsed() >= std::time::Duration::from_secs(2) {
                last_report = std::time::Instant::now();
                match total {
                    Some(total) if total > 0 => println!(
                        "          {:.0}% ({:.1} MiB of {:.1} MiB)",
                        written as f64 / total as f64 * 100.0,
                        written as f64 / 1_048_576.0,
                        total as f64 / 1_048_576.0
                    ),
                    _ => println!("          {:.1} MiB", written as f64 / 1_048_576.0),
                }
            }
        }
        file.flush().await.map_err(|e| Error::io(&part, e))?;
        file.sync_all().await.map_err(|e| Error::io(&part, e))?;
        drop(file);

        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&part)?;
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = std::fs::remove_file(&part);
                return Err(Error::ModelCorrupted {
                    name: target.display().to_string(),
                    reason: format!("checksum was {actual}, expected {expected}"),
                });
            }
        }
        std::fs::rename(&part, target).map_err(|e| Error::io(target, e))?;
        Ok(true)
    }
}

/// Hex SHA-256 of a file, streamed rather than read into memory.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| Error::io(path, e))?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

impl SetupReport {
    /// Lines to print after setup.
    pub fn to_lines(&self, paths: &Paths) -> Vec<String> {
        let mut lines = Vec::new();
        if self.wrote_config {
            lines.push(format!("  wrote {}", paths.config_file().display()));
        } else {
            lines.push(format!("  kept  {}", paths.config_file().display()));
        }
        for dir in &self.created_directories {
            lines.push(format!("  made  {}", dir.display()));
        }
        for name in &self.downloaded {
            lines.push(format!("  got   {name}"));
        }
        for name in &self.skipped {
            lines.push(format!("  have  {name}"));
        }
        for warning in &self.warnings {
            lines.push(format!("  note  {warning}"));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn setup_creates_directories_and_a_config() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::rooted(dir.path());
        let config = Config::default();

        let report = run(&config, &paths).await.unwrap();
        assert!(report.wrote_config);
        assert!(paths.config_file().exists());
        assert!(paths.audio_dir().exists());
        assert!(!report.to_lines(&paths).is_empty());
    }

    #[tokio::test]
    async fn setup_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::rooted(dir.path());
        let config = Config::default();

        run(&config, &paths).await.unwrap();
        let second = run(&config, &paths).await.unwrap();
        assert!(!second.wrote_config);
        assert!(second.created_directories.is_empty());
    }

    #[test]
    fn checksums_are_stable_hex() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("x.bin");
        std::fs::write(&file, b"cookie").unwrap();
        let digest = sha256_file(&file).unwrap();
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, sha256_file(&file).unwrap());
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn an_existing_unverified_file_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, b"pretend model").unwrap();
        // No checksum configured: we must not re-download on every start.
        let fetched = fetch_model("http://127.0.0.1:1/model.onnx", &file, None)
            .await
            .unwrap();
        assert!(!fetched);
    }

    #[tokio::test]
    async fn a_corrupt_model_is_reported_not_used() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, b"corrupted").unwrap();
        // Wrong checksum plus an unreachable server: the error must be about
        // the download, and the bad file must have been removed first.
        let result = fetch_model(
            "http://127.0.0.1:1/model.onnx",
            &file,
            Some("00".repeat(32).as_str()),
        )
        .await;
        assert!(result.is_err());
        assert!(!file.exists());
    }
}
