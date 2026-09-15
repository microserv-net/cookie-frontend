//! First run: fetching the things that make Cookie actually hear and speak.
//!
//! What `--setup` deliberately does *not* do is download during `cargo
//! build`. Build-time downloads make builds non-reproducible, break on
//! aeroplanes and behind corporate proxies, and turn a compile error into a
//! network error. So the binary you build works immediately with the
//! operating system's own voice, and the real models arrive here — once,
//! explicitly, with a progress line and a checksum.
//!
//! # What gets fetched
//!
//! | | | |
//! |---|---|---|
//! | runtime | sherpa-onnx, prebuilt for this platform | 20–45 MB |
//! | hearing | `whisper-large-v3-turbo` | 564 MB |
//! | voice | Kokoro, including British female voices | 103 MB |
//!
//! The runtime is prebuilt on purpose. Building sherpa-onnx from source needs
//! CMake and a C++ toolchain on three operating systems, which is exactly the
//! kind of prerequisite that turns "try this" into "spend an afternoon".
//!
//! # Why a subprocess rather than linking
//!
//! The models run in sherpa-onnx's own command-line tools, driven by
//! [`crate::stt::local`] and [`crate::tts::local`]. Linking the C API would
//! mean this crate could not be built without the shared libraries present,
//! and `cargo test` would need a 600 MB download. A process per utterance
//! costs a few tens of milliseconds of start-up against a model that takes
//! hundreds — a bad trade only if you are transcribing continuously, which a
//! voice assistant is not.

mod assets;

use std::path::{Path, PathBuf};

use crate::config::{Config, SttProviderKind, TtsProviderKind};
use crate::error::{Error, Result};
use crate::paths::Paths;

pub use assets::{total_bytes, Asset, Platform};

/// What setup did, for printing and for tests.
#[derive(Debug, Default, PartialEq)]
pub struct SetupReport {
    pub created_directories: Vec<PathBuf>,
    pub wrote_config: bool,
    pub downloaded: Vec<String>,
    pub skipped: Vec<String>,
    pub warnings: Vec<String>,
    /// Where the runtime and models ended up, once discovered.
    pub runtime: Option<PathBuf>,
    pub recogniser: Option<PathBuf>,
    pub voice: Option<PathBuf>,
}

impl SetupReport {
    pub fn to_lines(&self, paths: &Paths) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(if self.wrote_config {
            format!("  wrote {}", paths.config_file().display())
        } else {
            format!("  kept  {}", paths.config_file().display())
        });
        for dir in &self.created_directories {
            lines.push(format!("  made  {}", dir.display()));
        }
        for name in &self.downloaded {
            lines.push(format!("  got   {name}"));
        }
        for name in &self.skipped {
            lines.push(format!("  have  {name}"));
        }
        if let Some(path) = &self.runtime {
            lines.push(format!("  runtime  {}", path.display()));
        }
        if let Some(path) = &self.recogniser {
            lines.push(format!("  hearing  {}", path.display()));
        }
        if let Some(path) = &self.voice {
            lines.push(format!("  voice    {}", path.display()));
        }
        for warning in &self.warnings {
            lines.push(format!("  note  {warning}"));
        }
        lines
    }
}

/// How much to fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// Directories and configuration only. Used by tests and by anyone who
    /// wants to point at their own model server instead.
    ConfigOnly,
    /// Everything: runtime, recogniser and voice.
    Models,
}

/// Prepare this machine.
///
/// Idempotent: running it twice is harmless, and running it after an
/// interrupted attempt resumes rather than starting over.
pub async fn run(config: &Config, paths: &Paths, depth: Depth) -> Result<SetupReport> {
    let mut report = SetupReport::default();

    for dir in paths.all_directories() {
        if !dir.exists() {
            std::fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
            report.created_directories.push(dir);
        }
    }

    if depth == Depth::ConfigOnly {
        write_config_if_missing(config, paths, &mut report)?;
        return Ok(report);
    }

    let platform = Platform::detect().ok_or_else(|| {
        Error::Other(format!(
            "there is no prebuilt speech runtime for {} on {}. \
             Point stt.endpoint and tts.endpoint at a model server instead.",
            std::env::consts::ARCH,
            std::env::consts::OS
        ))
    })?;

    println!("  platform {platform}");
    let mut updated = config.clone();

    for asset in assets::required(platform) {
        let target = paths.models_dir().join(asset.directory_name());
        if target.exists() && asset.is_complete(&target) {
            report.skipped.push(asset.name.to_string());
            continue;
        }
        fetch_and_extract(&asset, paths, &mut report).await?;
    }

    // Everything is located by *searching* the extracted tree rather than by
    // hard-coded filenames. Upstream renames files between releases, and a
    // setup that breaks on a rename is a setup that breaks.
    let models = paths.models_dir();
    report.runtime = assets::find_runtime(&models);
    report.recogniser = assets::find_whisper(&models).map(|w| w.encoder.clone());
    report.voice = assets::find_kokoro(&models).map(|k| k.model.clone());

    match (&report.runtime, assets::find_whisper(&models)) {
        (Some(runtime), Some(whisper)) => {
            updated.stt.provider = SttProviderKind::Local;
            updated.stt.model = "whisper-large-v3-turbo".into();
            updated.stt.options.insert(
                "runtime".into(),
                toml::Value::String(runtime.display().to_string()),
            );
            updated.stt.options.insert(
                "encoder".into(),
                toml::Value::String(whisper.encoder.display().to_string()),
            );
            updated.stt.options.insert(
                "decoder".into(),
                toml::Value::String(whisper.decoder.display().to_string()),
            );
            updated.stt.options.insert(
                "tokens".into(),
                toml::Value::String(whisper.tokens.display().to_string()),
            );
        }
        _ => report
            .warnings
            .push("no recogniser was found after extraction; hearing is unchanged".into()),
    }

    match (&report.runtime, assets::find_kokoro(&models)) {
        (Some(runtime), Some(kokoro)) => {
            updated.tts.provider = TtsProviderKind::Local;
            updated.tts.model = "kokoro".into();
            updated.tts.voice.id = assets::BRITISH_FEMALE_VOICE.to_string();
            updated.tts.options.insert(
                "runtime".into(),
                toml::Value::String(runtime.display().to_string()),
            );
            updated.tts.options.insert(
                "model_path".into(),
                toml::Value::String(kokoro.model.display().to_string()),
            );
            updated.tts.options.insert(
                "voices".into(),
                toml::Value::String(kokoro.voices.display().to_string()),
            );
            updated.tts.options.insert(
                "tokens".into(),
                toml::Value::String(kokoro.tokens.display().to_string()),
            );
            if let Some(data_dir) = &kokoro.data_dir {
                updated.tts.options.insert(
                    "data_dir".into(),
                    toml::Value::String(data_dir.display().to_string()),
                );
            }
        }
        _ => report
            .warnings
            .push("no voice was found after extraction; speech is unchanged".into()),
    }

    // Writing the configuration is the last thing, so an interrupted setup
    // never leaves a config pointing at files that are not there.
    updated.save(paths)?;
    report.wrote_config = true;
    Ok(report)
}

fn write_config_if_missing(config: &Config, paths: &Paths, report: &mut SetupReport) -> Result<()> {
    if !paths.config_file().exists() {
        config.save(paths)?;
        report.wrote_config = true;
    }
    Ok(())
}

async fn fetch_and_extract(asset: &Asset, paths: &Paths, report: &mut SetupReport) -> Result<()> {
    let archive = paths.cache_dir().join(asset.file_name());
    let downloaded = fetch(asset, &archive).await?;
    if downloaded {
        report.downloaded.push(asset.name.to_string());
    } else {
        report.skipped.push(format!("{} (cached)", asset.name));
    }

    println!("  extracting {}", asset.name);
    extract(&archive, &paths.models_dir())?;
    // The archive is large and the extracted tree is what matters, so the
    // download is not kept: re-fetching is one command, and half a gigabyte
    // of cache nobody knows about is not.
    let _ = std::fs::remove_file(&archive);
    Ok(())
}

/// Download `asset` to `target` unless it is already there and intact.
async fn fetch(asset: &Asset, target: &Path) -> Result<bool> {
    if target.exists() {
        if let Some(expected) = asset.sha256 {
            if crate::setup::sha256_file(target)?.eq_ignore_ascii_case(expected) {
                return Ok(false);
            }
            let _ = std::fs::remove_file(target);
        } else {
            return Ok(false);
        }
    }

    #[cfg(not(feature = "http-providers"))]
    {
        let _ = asset;
        return Err(Error::ProviderNotCompiled {
            provider: "model download".into(),
            feature: "http-providers",
        });
    }

    #[cfg(feature = "http-providers")]
    {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let part = target.with_extension("part");
        println!("  downloading {} ({})", asset.name, asset.human_size());

        let response = reqwest::get(asset.url).await.map_err(|e| {
            // Every asset is served over https, so a build without TLS fails
            // here and nowhere else — and "error sending request" gives no
            // hint at all about why.
            if asset.url.starts_with("https://") && !cfg!(feature = "tls") {
                Error::Config(
                    "this build has no TLS, so it cannot download anything over https. \
                     Rebuild with `cargo build --release` (TLS is on by default), or \
                     unpack the models yourself — see docs/models.md."
                        .into(),
                )
            } else {
                Error::Network(format!(
                    "could not reach {}: {e}. Check the network and try again; \
                     --setup resumes rather than starting over.",
                    asset.url
                ))
            }
        })?;
        if !response.status().is_success() {
            return Err(Error::Network(format!(
                "the server answered {} for {}",
                response.status(),
                asset.url
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
                        "          {:.0}%  {:.0} of {:.0} MiB",
                        written as f64 / total as f64 * 100.0,
                        written as f64 / 1_048_576.0,
                        total as f64 / 1_048_576.0
                    ),
                    _ => println!("          {:.0} MiB", written as f64 / 1_048_576.0),
                }
            }
        }
        file.flush().await.map_err(|e| Error::io(&part, e))?;
        file.sync_all().await.map_err(|e| Error::io(&part, e))?;
        drop(file);

        if let Some(expected) = asset.sha256 {
            let actual = sha256_file(&part)?;
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = std::fs::remove_file(&part);
                return Err(Error::ModelCorrupted {
                    name: asset.name.to_string(),
                    reason: format!("checksum was {actual}, expected {expected}"),
                });
            }
        }
        // Renamed only after the checksum matches, so an interrupted setup
        // never leaves a half-model that looks complete.
        std::fs::rename(&part, target).map_err(|e| Error::io(target, e))?;
        Ok(true)
    }
}

/// Extract a `.tar.bz2` into `into`.
///
/// Shelling out to `tar` rather than linking a decompressor: it is present on
/// macOS, on every Linux, and on Windows 10 and later (as bsdtar, which reads
/// bzip2). One less dependency, and the one that would have been added exists
/// to do exactly this.
fn extract(archive: &Path, into: &Path) -> Result<()> {
    std::fs::create_dir_all(into).map_err(|e| Error::io(into, e))?;
    let output = std::process::Command::new("tar")
        .arg("-xjf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .output()
        .map_err(|e| {
            Error::Other(format!(
                "could not run `tar` to unpack {}: {e}. \
                 Install tar, or unpack it into {} yourself.",
                archive.display(),
                into.display()
            ))
        })?;
    if !output.status.success() {
        return Err(Error::Other(format!(
            "unpacking {} failed: {}",
            archive.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn config_only_setup_creates_directories_and_a_config() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::rooted(dir.path());
        let report = run(&Config::default(), &paths, Depth::ConfigOnly)
            .await
            .unwrap();
        assert!(report.wrote_config);
        assert!(paths.config_file().exists());
        assert!(paths.audio_dir().exists());
        assert!(!report.to_lines(&paths).is_empty());
    }

    #[tokio::test]
    async fn setup_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::rooted(dir.path());
        run(&Config::default(), &paths, Depth::ConfigOnly)
            .await
            .unwrap();
        let second = run(&Config::default(), &paths, Depth::ConfigOnly)
            .await
            .unwrap();
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
    }

    #[test]
    fn extraction_explains_itself_when_tar_is_missing_or_the_archive_is_bad() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("not-really.tar.bz2");
        std::fs::write(&archive, b"this is not an archive").unwrap();
        let error = extract(&archive, &dir.path().join("out")).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("unpacking") || message.contains("could not run"),
            "{message}"
        );
    }
}
