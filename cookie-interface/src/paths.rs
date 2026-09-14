//! Per-user, per-platform storage locations.
//!
//! Nothing in this crate is allowed to hard-code `~/.cookie-interface`. Every
//! path a user's machine ever sees comes from here, which also means the
//! retention manager has a single, well-defined root it is permitted to delete
//! inside of (see `retention`).
//!
//! Resolved locations (qualifier `dev.cookie`, organisation `Cookie`,
//! application `cookie-interface`):
//!
//! | purpose | Linux | macOS | Windows |
//! |---|---|---|---|
//! | config | `~/.config/cookie-interface` | `~/Library/Application Support/dev.cookie.cookie-interface` | `%APPDATA%\Cookie\cookie-interface\config` |
//! | data | `~/.local/share/cookie-interface` | `~/Library/Application Support/dev.cookie.cookie-interface` | `%APPDATA%\Cookie\cookie-interface\data` |
//! | cache | `~/.cache/cookie-interface` | `~/Library/Caches/dev.cookie.cookie-interface` | `%LOCALAPPDATA%\Cookie\cookie-interface\cache` |
//!
//! Models live under *cache* (large, re-downloadable), generated speech under
//! *data/audio* (user content, governed by the retention policy), logs under
//! *data/logs*.

use std::path::{Path, PathBuf};

use directories::ProjectDirs;

use crate::error::{Error, Result};

pub const QUALIFIER: &str = "dev";
pub const ORGANISATION: &str = "Cookie";
pub const APPLICATION: &str = "cookie-interface";

/// Resolved storage roots. Cheap to clone, safe to pass around.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    config_dir: PathBuf,
    data_dir: PathBuf,
    cache_dir: PathBuf,
}

impl Paths {
    /// Resolve the real per-user directories for this platform.
    pub fn discover() -> Result<Self> {
        // An explicit override makes the whole application testable and lets
        // power users run several isolated profiles. Documented in
        // docs/configuration.md.
        if let Some(root) = std::env::var_os("COOKIE_INTERFACE_HOME") {
            return Ok(Self::rooted(PathBuf::from(root)));
        }

        let dirs = ProjectDirs::from(QUALIFIER, ORGANISATION, APPLICATION)
            .ok_or(Error::NoPlatformDir { kind: "config" })?;

        Ok(Self {
            config_dir: dirs.config_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
            cache_dir: dirs.cache_dir().to_path_buf(),
        })
    }

    /// All three roots under one directory. Used by `COOKIE_INTERFACE_HOME`,
    /// by the test-suite, and by portable installations.
    pub fn rooted(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    /// Generated speech. The retention manager owns everything in here.
    pub fn audio_dir(&self) -> PathBuf {
        self.data_dir.join("audio")
    }

    /// Crash-safe record of what is in `audio_dir` and when it expires.
    pub fn audio_ledger(&self) -> PathBuf {
        self.audio_dir().join("ledger.jsonl")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }

    /// Downloaded model weights. Under cache because they are large and can
    /// always be fetched again; never holds user content.
    pub fn models_dir(&self) -> PathBuf {
        self.cache_dir.join("models")
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.models_dir().join(sanitise_component(id))
    }

    /// Partial downloads, scratch files. Safe to wipe at any time.
    pub fn tmp_dir(&self) -> PathBuf {
        self.cache_dir.join("tmp")
    }

    /// True if `path` is inside one of our managed roots. The retention
    /// manager refuses to unlink anything for which this returns false.
    pub fn is_managed(&self, path: &Path) -> bool {
        let candidates = [&self.config_dir, &self.data_dir, &self.cache_dir];
        // Compare lexically after normalising `.` / `..`; we cannot canonicalise
        // because the file may already be gone.
        let norm = normalise(path);
        candidates.iter().any(|root| {
            let root = normalise(root);
            norm.starts_with(&root) && norm != root
        })
    }

    /// Create every directory the application expects to exist.
    /// Every directory this application will ever write to.
    ///
    /// Also the whitelist the retention manager checks against: nothing
    /// outside this list is ever deleted.
    pub fn all_directories(&self) -> Vec<PathBuf> {
        vec![
            self.config_dir().to_path_buf(),
            self.data_dir().to_path_buf(),
            self.cache_dir().to_path_buf(),
            self.audio_dir(),
            self.logs_dir(),
            self.models_dir(),
            self.tmp_dir(),
        ]
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            self.config_dir.clone(),
            self.data_dir.clone(),
            self.cache_dir.clone(),
            self.audio_dir(),
            self.logs_dir(),
            self.models_dir(),
            self.tmp_dir(),
        ] {
            std::fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
        }
        Ok(())
    }

    /// Human-readable summary, used by `--config` and `--doctor`.
    pub fn describe(&self) -> String {
        format!(
            "config : {}\ndata   : {}\ncache  : {}\naudio  : {}\nmodels : {}\nlogs   : {}",
            self.config_dir.display(),
            self.data_dir.display(),
            self.cache_dir.display(),
            self.audio_dir().display(),
            self.models_dir().display(),
            self.logs_dir().display(),
        )
    }
}

/// Strip anything that could escape a directory or upset a filesystem.
///
/// Used for every path component that comes from configuration or an API
/// request, so `"../../etc/passwd"` becomes `"......etc.passwd"` rather than a
/// traversal.
pub fn sanitise_component(raw: &str) -> String {
    let mut out: String = raw
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '-',
        })
        .collect();
    // `..` is the only dangerous sequence a dot can form, and a filename has
    // no legitimate use for it. Removing it before collapsing separators means
    // "../../etc/passwd" becomes "etc-passwd" rather than "-..-..-etc-passwd",
    // which would be harmless but unreadable in the ledger.
    while out.contains("..") {
        out = out.replace("..", "");
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    while out.starts_with(['.', '-']) {
        out.remove(0);
    }
    while out.ends_with(['.', '-']) {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("unnamed");
    }
    out.truncate(96);
    out
}

/// Lexical normalisation: resolve `.` and `..` without touching the disk.
fn normalise(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rooted_layout_is_separated_by_purpose() {
        let p = Paths::rooted("/tmp/ci-test");
        assert!(p.config_file().starts_with("/tmp/ci-test/config"));
        assert!(p.audio_dir().starts_with("/tmp/ci-test/data"));
        assert!(p.models_dir().starts_with("/tmp/ci-test/cache"));
        assert_ne!(p.audio_dir(), p.models_dir());
    }

    #[test]
    fn managed_paths_are_recognised() {
        let p = Paths::rooted("/tmp/ci-test");
        assert!(p.is_managed(&p.audio_dir().join("a.wav")));
        assert!(!p.is_managed(Path::new("/etc/passwd")));
        assert!(!p.is_managed(Path::new("/tmp/ci-test/data")), "root itself");
    }

    #[test]
    fn traversal_cannot_escape_managed_root() {
        let p = Paths::rooted("/tmp/ci-test");
        let escape = p.audio_dir().join("../../../../etc/passwd");
        assert!(!p.is_managed(&escape));
    }

    #[test]
    fn component_sanitisation_blocks_traversal() {
        assert_eq!(sanitise_component("../../etc/passwd"), "etc-passwd");
        assert_eq!(sanitise_component(""), "unnamed");
        assert_eq!(
            sanitise_component("whisper-large-v3-turbo"),
            "whisper-large-v3-turbo"
        );
        assert!(!sanitise_component("a/b\\c").contains('/'));
    }
}
