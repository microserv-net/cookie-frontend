//! What to download, and how to find it again afterwards.
//!
//! Pinned by version and verified by checksum. Upstream moves files between
//! releases, so nothing downstream hard-codes a filename: everything is
//! located by *searching* the extracted tree for the shape of file it needs.
//! A setup that breaks when a model is renamed is a setup that breaks.

use std::fmt;
use std::path::{Path, PathBuf};

/// The sherpa-onnx release these assets come from.
///
/// Read only by the test that asserts every runtime URL points at it — a
/// version bump that updates one URL and forgets another would otherwise ship
/// a mismatched runtime, and that fails at the first utterance rather than at
/// build time.
#[cfg_attr(not(test), allow(dead_code))]
///
/// Public because the asset URLs are asserted against it: a version bump that
/// updates one URL and not the others would otherwise ship a mismatched
/// runtime, which fails at the first utterance rather than at build time.
pub const RUNTIME_VERSION: &str = "v1.13.8";

/// Kokoro's British female voice.
///
/// Kokoro identifies speakers by index. In `kokoro-en-v0_19` the order is
/// af, af_bella, af_nicole, af_sarah, af_sky, am_adam, am_michael, bf_emma,
/// bf_isabella, bm_george, bm_lewis — so 7 is `bf_emma`, which is the warm,
/// unhurried British voice the project asks for. 8 (`bf_isabella`) is the
/// other one worth trying.
pub const BRITISH_FEMALE_VOICE: &str = "7";

/// One downloadable archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: &'static str,
    pub url: &'static str,
    /// `None` where upstream published the file before release digests
    /// existed. Absence is recorded rather than faked.
    pub sha256: Option<&'static str>,
    pub bytes: u64,
    /// The directory the archive unpacks into.
    pub unpacks_to: &'static str,
    /// A file that must exist afterwards for the extraction to count.
    pub proof: &'static str,
}

impl Asset {
    pub fn file_name(&self) -> String {
        self.url.rsplit('/').next().unwrap_or(self.name).to_string()
    }

    pub fn directory_name(&self) -> &str {
        self.unpacks_to
    }

    pub fn human_size(&self) -> String {
        format!("{:.0} MiB", self.bytes as f64 / 1_048_576.0)
    }

    /// Whether a previous run already unpacked this.
    pub fn is_complete(&self, directory: &Path) -> bool {
        directory.join(self.proof).exists()
    }
}

/// Which prebuilt runtime this machine needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    MacOs,
    Linux,
    Windows,
}

impl Platform {
    pub fn detect() -> Option<Self> {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            // Universal, so one archive covers both Apple Silicon and Intel;
            // the extra megabytes are cheaper than getting this wrong on a
            // machine somebody cannot test on.
            ("macos", _) => Some(Platform::MacOs),
            ("linux", "x86_64") => Some(Platform::Linux),
            ("windows", "x86_64") => Some(Platform::Windows),
            _ => None,
        }
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Platform::MacOs => "macOS (universal)",
            Platform::Linux => "Linux x86-64",
            Platform::Windows => "Windows x86-64",
        };
        f.write_str(name)
    }
}

const MACOS_RUNTIME: Asset = Asset {
    name: "speech runtime (macOS)",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-osx-universal2-shared.tar.bz2",
    sha256: Some("2249f97f10df7d828af1b0d10e3d1eeaa2198b60b9a999a7ed853c150181ccfd"),
    bytes: 43_600_000,
    unpacks_to: "sherpa-onnx-v1.13.8-osx-universal2-shared",
    proof: "bin/sherpa-onnx-offline-tts",
};

const LINUX_RUNTIME: Asset = Asset {
    name: "speech runtime (Linux)",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-linux-x64-shared.tar.bz2",
    sha256: Some("c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4"),
    bytes: 28_200_000,
    unpacks_to: "sherpa-onnx-v1.13.8-linux-x64-shared",
    proof: "bin/sherpa-onnx-offline-tts",
};

const WINDOWS_RUNTIME: Asset = Asset {
    name: "speech runtime (Windows)",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-win-x64-shared-MD-Release.tar.bz2",
    sha256: Some("3e971a04b2e0ba4dfa53d381a006367ce8c9f5f09b4ae00043e9845c2baded22"),
    bytes: 20_500_000,
    unpacks_to: "sherpa-onnx-v1.13.8-win-x64-shared-MD-Release",
    proof: "bin/sherpa-onnx-offline-tts.exe",
};

/// `whisper-large-v3-turbo`, converted for sherpa-onnx.
const WHISPER_TURBO: Asset = Asset {
    name: "whisper-large-v3-turbo",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
    // Published before GitHub recorded release digests. Rather than invent
    // one, the absence is recorded and the extraction proof does the work.
    sha256: None,
    bytes: 563_800_000,
    unpacks_to: "sherpa-onnx-whisper-turbo",
    proof: "turbo-tokens.txt",
};

/// Kokoro, int8. The full-precision build is three times the size for a
/// difference nobody hears through a laptop speaker.
const KOKORO: Asset = Asset {
    name: "kokoro (British female voice)",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-en-v0_19.tar.bz2",
    sha256: Some("c9f0dd393615805b0bab050c340834d5e684e732aec91c0e860cd30e982c08bd"),
    bytes: 103_200_000,
    unpacks_to: "kokoro-int8-en-v0_19",
    proof: "voices.bin",
};

/// Everything `--setup` fetches, in the order it fetches it.
///
/// Runtime first: it is the smallest, and a failure there means the models
/// would be unusable anyway.
pub fn required(platform: Platform) -> Vec<Asset> {
    let runtime = match platform {
        Platform::MacOs => MACOS_RUNTIME,
        Platform::Linux => LINUX_RUNTIME,
        Platform::Windows => WINDOWS_RUNTIME,
    };
    vec![runtime, KOKORO, WHISPER_TURBO]
}

/// Total download for a fresh machine.
pub fn total_bytes(platform: Platform) -> u64 {
    required(platform).iter().map(|asset| asset.bytes).sum()
}

/// The extracted sherpa-onnx tree, whichever platform's it is.
pub fn find_runtime(models: &Path) -> Option<PathBuf> {
    children(models).into_iter().find(|path| {
        path.join("bin").is_dir()
            && (path.join("bin/sherpa-onnx-offline-tts").exists()
                || path.join("bin/sherpa-onnx-offline-tts.exe").exists())
    })
}

/// Where the Whisper files ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperFiles {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokens: PathBuf,
}

pub fn find_whisper(models: &Path) -> Option<WhisperFiles> {
    // Every extracted directory is examined, not only the first: the models
    // directory also holds the runtime and the voice, and which of them sorts
    // first is not something to depend on.
    children(models).into_iter().find_map(|directory| {
        let files = entries(&directory);
        // int8 where it exists: a quarter of the memory for a difference that
        // does not survive a laptop microphone.
        let encoder = pick(&files, &["encoder.int8.onnx", "encoder.onnx"])?;
        let decoder = pick(&files, &["decoder.int8.onnx", "decoder.onnx"])?;
        let tokens = files.iter().find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with("tokens.txt"))
        })?;
        Some(WhisperFiles {
            encoder,
            decoder,
            tokens: tokens.clone(),
        })
    })
}

/// Where the Kokoro files ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KokoroFiles {
    pub model: PathBuf,
    pub voices: PathBuf,
    pub tokens: PathBuf,
    /// espeak-ng's pronunciation data, if the archive shipped it.
    pub data_dir: Option<PathBuf>,
}

pub fn find_kokoro(models: &Path) -> Option<KokoroFiles> {
    children(models)
        .into_iter()
        .filter(|directory| directory.join("voices.bin").exists())
        .find_map(|directory| {
            let model = pick(&entries(&directory), &["model.int8.onnx", "model.onnx"])?;
            let data_dir = directory.join("espeak-ng-data");
            Some(KokoroFiles {
                model,
                voices: directory.join("voices.bin"),
                tokens: directory.join("tokens.txt"),
                data_dir: data_dir.is_dir().then_some(data_dir),
            })
        })
}

fn children(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut directories: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    directories.sort();
    directories
}

fn entries(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    entries.flatten().map(|entry| entry.path()).collect()
}

/// First file whose name ends with one of `suffixes`, in order of preference.
fn pick(files: &[PathBuf], suffixes: &[&str]) -> Option<PathBuf> {
    for suffix in suffixes {
        if let Some(found) = files.iter().find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(suffix))
        }) {
            return Some(found.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_fetches_a_runtime_and_both_models() {
        for platform in [Platform::MacOs, Platform::Linux, Platform::Windows] {
            let assets = required(platform);
            assert_eq!(assets.len(), 3, "{platform}");
            assert!(assets[0].name.contains("runtime"));
            // Runtime first: it is the smallest, and without it the models
            // are unusable anyway.
            assert!(assets[0].bytes < assets[2].bytes);
            assert!(total_bytes(platform) > 600_000_000);
        }
    }

    #[test]
    fn asset_urls_are_pinned_to_the_recorded_version() {
        for platform in [Platform::MacOs, Platform::Linux, Platform::Windows] {
            let runtime = &required(platform)[0];
            assert!(runtime.url.contains(RUNTIME_VERSION), "{}", runtime.url);
            assert!(runtime.sha256.is_some(), "the runtime must be verifiable");
            assert!(runtime.url.starts_with("https://"));
        }
    }

    #[test]
    fn file_names_come_from_the_urls_rather_than_being_repeated() {
        assert_eq!(KOKORO.file_name(), "kokoro-int8-en-v0_19.tar.bz2");
        assert_eq!(
            WHISPER_TURBO.file_name(),
            "sherpa-onnx-whisper-turbo.tar.bz2"
        );
    }

    #[test]
    fn a_half_extracted_model_does_not_count_as_present() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join(KOKORO.unpacks_to);
        std::fs::create_dir_all(&target).unwrap();
        assert!(
            !KOKORO.is_complete(&target),
            "an empty directory is not a model"
        );
        std::fs::write(target.join("voices.bin"), b"x").unwrap();
        assert!(KOKORO.is_complete(&target));
    }

    #[test]
    fn the_runtime_is_found_by_shape_not_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("sherpa-onnx-v9.9.9-some-future-name");
        std::fs::create_dir_all(runtime.join("bin")).unwrap();
        std::fs::write(runtime.join("bin/sherpa-onnx-offline-tts"), b"").unwrap();
        assert_eq!(find_runtime(dir.path()), Some(runtime));
    }

    #[test]
    fn whisper_prefers_the_quantised_weights() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("sherpa-onnx-whisper-turbo");
        std::fs::create_dir_all(&model).unwrap();
        for name in [
            "turbo-encoder.onnx",
            "turbo-encoder.int8.onnx",
            "turbo-decoder.onnx",
            "turbo-decoder.int8.onnx",
            "turbo-tokens.txt",
        ] {
            std::fs::write(model.join(name), b"").unwrap();
        }
        let found = find_whisper(dir.path()).unwrap();
        assert!(found.encoder.to_string_lossy().contains("int8"));
        assert!(found.decoder.to_string_lossy().contains("int8"));
        assert!(found.tokens.ends_with("turbo-tokens.txt"));
    }

    #[test]
    fn kokoro_is_found_with_its_pronunciation_data() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("kokoro-int8-en-v0_19");
        std::fs::create_dir_all(model.join("espeak-ng-data")).unwrap();
        for name in ["model.int8.onnx", "voices.bin", "tokens.txt"] {
            std::fs::write(model.join(name), b"").unwrap();
        }
        let found = find_kokoro(dir.path()).unwrap();
        assert!(found.model.to_string_lossy().ends_with("model.int8.onnx"));
        assert!(found.data_dir.is_some());
    }

    #[test]
    fn nothing_extracted_means_nothing_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_runtime(dir.path()).is_none());
        assert!(find_whisper(dir.path()).is_none());
        assert!(find_kokoro(dir.path()).is_none());
    }

    #[test]
    fn this_machine_is_recognised_or_honestly_refused() {
        // Whatever the CI runner is, `detect` must either name it or return
        // None — never guess an archive that will not run.
        match Platform::detect() {
            Some(platform) => assert!(!required(platform).is_empty()),
            None => assert!(matches!(
                std::env::consts::ARCH,
                "aarch64" | "arm" | "riscv64" | "powerpc64" | "s390x"
            )),
        }
    }
}
