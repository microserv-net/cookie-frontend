//! Generated-audio retention.
//!
//! Synthesised speech is user content: it is a recording of what an assistant
//! said, often in reply to something private. The policy must therefore be
//! *enforced*, not displayed — which is a stronger requirement than it sounds,
//! because the interesting cases are all failure cases:
//!
//! * the process was killed between writing a file and recording it;
//! * the process was off for a week and 4 000 files are now overdue;
//! * a file in the ledger was deleted by hand;
//! * a file on disk was never in the ledger at all;
//! * the disk is read-only and deletion fails.
//!
//! The design that survives all five: an append-only JSONL ledger written
//! *before* the audio file exists, a full reconciliation sweep at every
//! startup before normal operation begins, and expiry computed from the
//! *current* policy at sweep time rather than from a stored deadline (so
//! shortening the policy retroactively cleans up, which is what a user
//! changing it to "1 hour" plainly means).
//!
//! Safety rule, enforced in one place: nothing is ever unlinked unless it sits
//! directly inside the managed audio directory.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::config::RetentionConfig;
use crate::error::{Error, Result};
use crate::paths::{sanitise_component, Paths};
use crate::util::unix_now;

/// How long generated audio is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Delete as soon as playback finishes. Nothing durable ever exists.
    Immediate,
    Hours(u32),
    Days(u32),
    /// Keep until the user deletes it (still subject to `max_files`/`max_bytes`).
    Forever,
}

impl RetentionPolicy {
    /// Presets offered by `--config` and the API.
    pub const PRESETS: &'static [RetentionPolicy] = &[
        RetentionPolicy::Immediate,
        RetentionPolicy::Hours(1),
        RetentionPolicy::Hours(24),
        RetentionPolicy::Days(7),
        RetentionPolicy::Days(30),
        RetentionPolicy::Forever,
    ];

    pub fn seconds(&self) -> Option<u64> {
        match self {
            RetentionPolicy::Immediate => Some(0),
            RetentionPolicy::Hours(h) => Some(*h as u64 * 3600),
            RetentionPolicy::Days(d) => Some(*d as u64 * 86_400),
            RetentionPolicy::Forever => None,
        }
    }

    /// Whether a file created at `created_at` is expired at `now`.
    pub fn is_expired(&self, created_at: u64, now: u64) -> bool {
        match self.seconds() {
            None => false,
            Some(0) => true,
            Some(s) => now.saturating_sub(created_at) >= s,
        }
    }

    /// Should audio be written to disk at all under this policy?
    pub fn persists(&self) -> bool {
        !matches!(self, RetentionPolicy::Immediate)
    }

    pub fn as_str(&self) -> String {
        match self {
            RetentionPolicy::Immediate => "immediate".into(),
            RetentionPolicy::Hours(h) => format!("{h}h"),
            RetentionPolicy::Days(d) => format!("{d}d"),
            RetentionPolicy::Forever => "forever".into(),
        }
    }
}

impl std::fmt::Display for RetentionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_str())
    }
}

impl std::str::FromStr for RetentionPolicy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "immediate" | "never" | "0" | "0s" => return Ok(RetentionPolicy::Immediate),
            "forever" | "indefinitely" | "keep" => return Ok(RetentionPolicy::Forever),
            _ => {}
        }
        let (num, unit) = s.split_at(s.len().saturating_sub(1));
        let value: u32 = num
            .parse()
            .map_err(|_| format!("unrecognised retention policy {s:?}"))?;
        match unit {
            "h" => Ok(RetentionPolicy::Hours(value)),
            "d" => Ok(RetentionPolicy::Days(value)),
            "w" => Ok(RetentionPolicy::Days(value.saturating_mul(7))),
            _ => Err(format!(
                "unrecognised retention unit in {s:?} (use h, d, w, immediate or forever)"
            )),
        }
    }
}

impl Serialize for RetentionPolicy {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.as_str())
    }
}

impl<'de> Deserialize<'de> for RetentionPolicy {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

/// Lifecycle of one generated file, as recorded in the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordState {
    /// The row was written; the audio file may or may not exist yet. Any
    /// `Pending` row found at startup is the fingerprint of a crash.
    Pending,
    Complete,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioRecord {
    pub id: String,
    /// File name only, never a path: the directory is always `audio_dir`.
    pub file: String,
    pub created_at: u64,
    pub state: RecordState,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<String>,
}

/// What a sweep did. Returned to the caller and emitted as an event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepReport {
    pub scanned: usize,
    pub deleted: usize,
    pub freed_bytes: u64,
    /// Files that could not be deleted (permissions, locks). Retried next sweep.
    pub failed: usize,
    /// Files on disk that had no ledger row and were adopted into it.
    pub adopted: usize,
    /// Rows whose file had already vanished.
    pub missing: usize,
    /// Rows left over from an interrupted session.
    pub recovered_pending: usize,
}

impl SweepReport {
    pub fn is_clean(&self) -> bool {
        self.failed == 0
    }
}

/// Owns the audio directory and its ledger.
///
/// All methods are synchronous and do blocking file I/O. Callers on an async
/// runtime must use `tokio::task::spawn_blocking` — the engine does.
pub struct RetentionManager {
    paths: Paths,
    dir: PathBuf,
    inner: Mutex<Inner>,
}

struct Inner {
    config: RetentionConfig,
    records: BTreeMap<String, AudioRecord>,
    /// Appends since the last compaction; used to decide when to rewrite.
    appends: usize,
}

impl RetentionManager {
    /// Open (and create) the audio directory and load the ledger.
    pub fn open(paths: &Paths, config: RetentionConfig) -> Result<Self> {
        let dir = paths.audio_dir();
        std::fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
        let records = load_ledger(&paths.audio_ledger())?;
        Ok(Self {
            paths: paths.clone(),
            dir,
            inner: Mutex::new(Inner {
                config,
                records,
                appends: 0,
            }),
        })
    }

    pub fn directory(&self) -> &Path {
        &self.dir
    }

    pub fn policy(&self) -> RetentionPolicy {
        self.inner.lock().expect("retention mutex").config.policy
    }

    pub fn config(&self) -> RetentionConfig {
        self.inner.lock().expect("retention mutex").config.clone()
    }

    /// Change the policy. The caller persists the config; the *next* sweep
    /// enforces it, and `sweep()` is called immediately by the API handler so
    /// "set to immediate" visibly empties the directory.
    pub fn set_policy(&self, policy: RetentionPolicy) {
        self.inner.lock().expect("retention mutex").config.policy = policy;
    }

    pub fn record_count(&self) -> usize {
        self.inner.lock().expect("retention mutex").records.len()
    }

    /// Reserve a path for a new utterance and write the `Pending` row *first*.
    ///
    /// Order matters: a row with no file is recoverable (the sweep drops it),
    /// whereas a file with no row is an orphan that only heuristics can date.
    pub fn begin(&self, utterance_id: &str) -> Result<PendingAudio> {
        let id = format!("{}-{}", unix_now(), sanitise_component(utterance_id));
        let file = format!("{id}.wav");
        let record = AudioRecord {
            id: id.clone(),
            file: file.clone(),
            created_at: unix_now(),
            state: RecordState::Pending,
            bytes: 0,
            duration_ms: 0,
            utterance_id: Some(utterance_id.to_string()),
        };
        self.append(&record)?;
        Ok(PendingAudio {
            path: self.dir.join(&file),
            record,
        })
    }

    /// Mark a reserved file as written.
    pub fn commit(&self, mut pending: PendingAudio, bytes: u64, duration_ms: u64) -> Result<()> {
        pending.record.state = RecordState::Complete;
        pending.record.bytes = bytes;
        pending.record.duration_ms = duration_ms;
        self.append(&pending.record)
    }

    /// Abandon a reserved file (synthesis failed or was interrupted).
    pub fn abort(&self, pending: PendingAudio) {
        let _ = std::fs::remove_file(&pending.path);
        let mut record = pending.record;
        record.state = RecordState::Deleted;
        let _ = self.append(&record);
    }

    fn append(&self, record: &AudioRecord) -> Result<()> {
        {
            let mut inner = self.inner.lock().expect("retention mutex");
            inner.records.insert(record.id.clone(), record.clone());
            inner.appends += 1;
        }
        let path = self.paths.audio_ledger();
        let line = serde_json::to_string(record).map_err(|e| Error::Other(e.to_string()))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        writeln!(f, "{line}").map_err(|e| Error::io(&path, e))?;
        // Durability: the ledger is the only thing that makes an orphan
        // recoverable, so it is worth the fsync.
        let _ = f.sync_data();
        Ok(())
    }

    /// Reconcile disk against the ledger and enforce the policy.
    ///
    /// Safe to call at any time; the engine runs it once at startup *before*
    /// accepting requests, and then periodically.
    pub fn sweep(&self) -> Result<SweepReport> {
        let now = unix_now();
        let (policy, max_files, max_bytes) = {
            let inner = self.inner.lock().expect("retention mutex");
            (
                inner.config.policy,
                inner.config.max_files,
                inner.config.max_bytes,
            )
        };
        let mut report = SweepReport::default();

        // --- 1. what is actually on disk --------------------------------
        let mut on_disk: BTreeMap<String, (PathBuf, u64, u64)> = BTreeMap::new();
        match std::fs::read_dir(&self.dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name == "ledger.jsonl" || name.ends_with(".tmp") {
                        continue;
                    }
                    let meta = match entry.metadata() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let created = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(now);
                    on_disk.insert(name, (path, meta.len(), created));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::io(&self.dir, e)),
        }

        let mut records = {
            let inner = self.inner.lock().expect("retention mutex");
            inner.records.clone()
        };

        // --- 2. resolve rows left behind by a crash ----------------------
        for record in records.values_mut() {
            if record.state == RecordState::Pending {
                report.recovered_pending += 1;
                if on_disk.contains_key(&record.file) {
                    // The file made it; adopt it as if it had been committed.
                    record.state = RecordState::Complete;
                } else {
                    record.state = RecordState::Deleted;
                }
            }
        }

        // --- 3. adopt orphans -------------------------------------------
        let known: std::collections::HashSet<String> =
            records.values().map(|r| r.file.clone()).collect();
        for (name, (_, size, created)) in &on_disk {
            if known.contains(name) {
                continue;
            }
            report.adopted += 1;
            let id = name.trim_end_matches(".wav").to_string();
            records.insert(
                id.clone(),
                AudioRecord {
                    id,
                    file: name.clone(),
                    created_at: *created,
                    state: RecordState::Complete,
                    bytes: *size,
                    duration_ms: 0,
                    utterance_id: None,
                },
            );
        }

        // --- 4. expire --------------------------------------------------
        let mut live: Vec<AudioRecord> = Vec::new();
        for mut record in records.into_values() {
            if record.state == RecordState::Deleted {
                // Keep the row only while its file might still exist.
                if on_disk.contains_key(&record.file) {
                    let path = self.dir.join(&record.file);
                    match self.unlink(&path) {
                        Ok(freed) => {
                            report.deleted += 1;
                            report.freed_bytes += freed;
                        }
                        Err(_) => {
                            report.failed += 1;
                            live.push(record);
                        }
                    }
                }
                continue;
            }

            report.scanned += 1;
            let exists = on_disk.contains_key(&record.file);
            if !exists {
                report.missing += 1;
                continue; // row dropped; tolerate files deleted behind our back
            }
            if policy.is_expired(record.created_at, now) {
                let path = self.dir.join(&record.file);
                match self.unlink(&path) {
                    Ok(freed) => {
                        report.deleted += 1;
                        report.freed_bytes += freed;
                    }
                    Err(_) => {
                        report.failed += 1;
                        record.state = RecordState::Complete;
                        live.push(record);
                    }
                }
            } else {
                live.push(record);
            }
        }

        // --- 5. caps ----------------------------------------------------
        live.sort_by_key(|r| r.created_at);
        if max_files > 0 && live.len() > max_files {
            let excess = live.len() - max_files;
            for record in live.drain(..excess).collect::<Vec<_>>() {
                let path = self.dir.join(&record.file);
                match self.unlink(&path) {
                    Ok(freed) => {
                        report.deleted += 1;
                        report.freed_bytes += freed;
                    }
                    Err(_) => {
                        report.failed += 1;
                        live.push(record);
                    }
                }
            }
            live.sort_by_key(|r| r.created_at);
        }
        if max_bytes > 0 {
            let mut total: u64 = live.iter().map(|r| r.bytes).sum();
            let mut idx = 0;
            while total > max_bytes && idx < live.len() {
                let record = live[idx].clone();
                let path = self.dir.join(&record.file);
                match self.unlink(&path) {
                    Ok(freed) => {
                        total = total.saturating_sub(record.bytes.max(freed));
                        report.deleted += 1;
                        report.freed_bytes += freed;
                        live.remove(idx);
                    }
                    Err(_) => {
                        report.failed += 1;
                        idx += 1;
                    }
                }
            }
        }

        // --- 6. compact the ledger --------------------------------------
        let compacted: BTreeMap<String, AudioRecord> =
            live.into_iter().map(|r| (r.id.clone(), r)).collect();
        self.write_ledger(&compacted)?;
        {
            let mut inner = self.inner.lock().expect("retention mutex");
            inner.records = compacted;
            inner.appends = 0;
        }
        Ok(report)
    }

    /// Delete everything we manage, regardless of age (`--clear-audio`).
    pub fn purge_all(&self) -> Result<SweepReport> {
        let saved = self.policy();
        self.set_policy(RetentionPolicy::Immediate);
        let report = self.sweep();
        self.set_policy(saved);
        report
    }

    /// The one place a file is unlinked. Refuses anything outside the managed
    /// audio directory, so a corrupt ledger cannot turn into `rm -rf`.
    fn unlink(&self, path: &Path) -> Result<u64> {
        if path.parent() != Some(self.dir.as_path()) || !self.paths.is_managed(path) {
            tracing::error!(path = %path.display(), "refusing to delete a path outside the managed audio directory");
            return Err(Error::Other("path outside managed directory".into()));
        }
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        match std::fs::remove_file(path) {
            Ok(()) => Ok(size),
            // Already gone is success, not failure.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(Error::io(path, e)),
        }
    }

    fn write_ledger(&self, records: &BTreeMap<String, AudioRecord>) -> Result<()> {
        let path = self.paths.audio_ledger();
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| Error::io(&tmp, e))?;
            for record in records.values() {
                let line =
                    serde_json::to_string(record).map_err(|e| Error::Other(e.to_string()))?;
                writeln!(f, "{line}").map_err(|e| Error::io(&tmp, e))?;
            }
            f.sync_all().map_err(|e| Error::io(&tmp, e))?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| Error::io(&path, e))?;
        Ok(())
    }

    /// Records currently tracked, newest first. Used by `--doctor`.
    pub fn list(&self) -> Vec<AudioRecord> {
        let inner = self.inner.lock().expect("retention mutex");
        let mut out: Vec<AudioRecord> = inner.records.values().cloned().collect();
        out.sort_by_key(|a| std::cmp::Reverse(a.created_at));
        out
    }
}

/// A reserved audio file. Must be `commit`ed or `abort`ed.
#[derive(Debug, Clone)]
pub struct PendingAudio {
    pub path: PathBuf,
    record: AudioRecord,
}

impl PendingAudio {
    pub fn id(&self) -> &str {
        &self.record.id
    }
}

/// Read the append-only ledger. Later rows win; unparseable rows are skipped
/// (a torn final line after a crash must not brick startup).
fn load_ledger(path: &Path) -> Result<BTreeMap<String, AudioRecord>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(Error::io(path, e)),
    };
    let mut out = BTreeMap::new();
    let mut skipped = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<AudioRecord>(line) {
            Ok(record) => {
                out.insert(record.id.clone(), record);
            }
            Err(_) => skipped += 1,
        }
    }
    if skipped > 0 {
        tracing::warn!(skipped, "skipped unreadable rows in the audio ledger");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(dir: &Path, policy: RetentionPolicy) -> RetentionManager {
        let paths = Paths::rooted(dir);
        paths.ensure().unwrap();
        RetentionManager::open(
            &paths,
            RetentionConfig {
                policy,
                sweep_interval_minutes: 30,
                max_files: 0,
                max_bytes: 0,
            },
        )
        .unwrap()
    }

    fn write_audio(m: &RetentionManager, id: &str) -> PathBuf {
        let pending = m.begin(id).unwrap();
        std::fs::write(&pending.path, b"RIFFfake").unwrap();
        let path = pending.path.clone();
        m.commit(pending, 8, 100).unwrap();
        path
    }

    #[test]
    fn policy_parsing_and_display_roundtrip() {
        for p in RetentionPolicy::PRESETS {
            let s = p.as_str();
            assert_eq!(s.parse::<RetentionPolicy>().unwrap(), *p, "{s}");
        }
        assert_eq!(
            "7d".parse::<RetentionPolicy>().unwrap(),
            RetentionPolicy::Days(7)
        );
        assert_eq!(
            "2w".parse::<RetentionPolicy>().unwrap(),
            RetentionPolicy::Days(14)
        );
        assert!("banana".parse::<RetentionPolicy>().is_err());
    }

    #[test]
    fn policy_serialises_as_a_short_string() {
        let json = serde_json::to_string(&RetentionPolicy::Hours(24)).unwrap();
        assert_eq!(json, "\"24h\"");
        let back: RetentionPolicy = serde_json::from_str("\"forever\"").unwrap();
        assert_eq!(back, RetentionPolicy::Forever);
    }

    #[test]
    fn expiry_maths() {
        let now = 1_000_000u64;
        assert!(RetentionPolicy::Immediate.is_expired(now, now));
        assert!(!RetentionPolicy::Forever.is_expired(0, now));
        assert!(RetentionPolicy::Hours(1).is_expired(now - 3601, now));
        assert!(!RetentionPolicy::Hours(1).is_expired(now - 3599, now));
    }

    #[test]
    fn files_within_the_window_survive_a_sweep() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        let path = write_audio(&m, "u1");
        let report = m.sweep().unwrap();
        assert_eq!(report.deleted, 0);
        assert!(path.exists());
    }

    #[test]
    fn immediate_policy_deletes_everything() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Immediate);
        let path = write_audio(&m, "u1");
        let report = m.sweep().unwrap();
        assert_eq!(report.deleted, 1);
        assert!(!path.exists());
        assert_eq!(m.record_count(), 0);
    }

    #[test]
    fn expired_files_are_deleted_and_fresh_ones_kept() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Hours(1));
        let old = write_audio(&m, "old");
        let fresh = write_audio(&m, "fresh");

        // Backdate the old row by rewriting the ledger through the manager's
        // own API surface: appending a newer row for the same id wins.
        {
            let mut record = m
                .list()
                .into_iter()
                .find(|r| r.utterance_id.as_deref() == Some("old"))
                .unwrap();
            record.created_at = unix_now() - 7200;
            m.append(&record).unwrap();
        }

        let report = m.sweep().unwrap();
        assert_eq!(report.deleted, 1, "{report:?}");
        assert!(!old.exists());
        assert!(fresh.exists());
    }

    #[test]
    fn a_pending_row_from_a_crash_is_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        // Simulate a crash: reserve, write the file, never commit.
        let pending = m.begin("crashed").unwrap();
        std::fs::write(&pending.path, b"RIFFfake").unwrap();
        let path = pending.path.clone();
        drop(pending);

        // Restart.
        let m2 = manager(dir.path(), RetentionPolicy::Days(7));
        let report = m2.sweep().unwrap();
        assert_eq!(report.recovered_pending, 1, "{report:?}");
        assert!(path.exists(), "a completed file must be adopted, not lost");
        assert_eq!(m2.record_count(), 1);
    }

    #[test]
    fn a_pending_row_with_no_file_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        let pending = m.begin("never-written").unwrap();
        drop(pending);

        let m2 = manager(dir.path(), RetentionPolicy::Days(7));
        let report = m2.sweep().unwrap();
        assert_eq!(report.recovered_pending, 1);
        assert_eq!(m2.record_count(), 0);
    }

    #[test]
    fn orphan_files_are_adopted_then_governed() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        let orphan = m.directory().join("stray.wav");
        std::fs::write(&orphan, b"RIFFfake").unwrap();

        let report = m.sweep().unwrap();
        assert_eq!(report.adopted, 1);
        assert!(orphan.exists(), "adoption must not mean immediate deletion");
        assert_eq!(m.record_count(), 1);

        // Under an immediate policy the adopted orphan does get removed.
        m.set_policy(RetentionPolicy::Immediate);
        let report = m.sweep().unwrap();
        assert_eq!(report.deleted, 1);
        assert!(!orphan.exists());
    }

    #[test]
    fn missing_files_are_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        let path = write_audio(&m, "u1");
        std::fs::remove_file(&path).unwrap();
        let report = m.sweep().unwrap();
        assert_eq!(report.missing, 1);
        assert_eq!(report.failed, 0);
        assert_eq!(m.record_count(), 0);
    }

    #[test]
    fn a_shortened_policy_applies_retroactively() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Forever);
        let path = write_audio(&m, "u1");
        m.sweep().unwrap();
        assert!(path.exists());

        m.set_policy(RetentionPolicy::Immediate);
        m.sweep().unwrap();
        assert!(!path.exists(), "policy change must clean up existing audio");
    }

    #[test]
    fn max_files_cap_removes_the_oldest() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::rooted(dir.path());
        paths.ensure().unwrap();
        let m = RetentionManager::open(
            &paths,
            RetentionConfig {
                policy: RetentionPolicy::Forever,
                sweep_interval_minutes: 30,
                max_files: 2,
                max_bytes: 0,
            },
        )
        .unwrap();
        for i in 0..5 {
            let pending = m.begin(&format!("u{i}")).unwrap();
            std::fs::write(&pending.path, b"RIFFfake").unwrap();
            let mut record = pending.record.clone();
            record.created_at = 1_000 + i as u64;
            record.state = RecordState::Complete;
            record.bytes = 8;
            m.append(&record).unwrap();
        }
        let report = m.sweep().unwrap();
        assert_eq!(m.record_count(), 2, "{report:?}");
        assert_eq!(report.deleted, 3);
    }

    #[test]
    fn ledger_survives_a_torn_line() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(7));
        write_audio(&m, "u1");
        let ledger = Paths::rooted(dir.path()).audio_ledger();
        let mut text = std::fs::read_to_string(&ledger).unwrap();
        text.push_str("{\"id\":\"broken\",\"fi");
        std::fs::write(&ledger, text).unwrap();

        let m2 = manager(dir.path(), RetentionPolicy::Days(7));
        assert_eq!(m2.record_count(), 1, "good rows must still load");
    }

    #[test]
    fn refuses_to_delete_outside_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Immediate);
        let outside = dir.path().join("precious.txt");
        std::fs::write(&outside, b"do not delete").unwrap();
        assert!(m.unlink(&outside).is_err());
        assert!(outside.exists());
    }

    #[test]
    fn purge_all_empties_then_restores_the_policy() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path(), RetentionPolicy::Days(30));
        let path = write_audio(&m, "u1");
        let report = m.purge_all().unwrap();
        assert_eq!(report.deleted, 1);
        assert!(!path.exists());
        assert_eq!(m.policy(), RetentionPolicy::Days(30));
    }
}
