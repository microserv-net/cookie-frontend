//! Files.
//!
//! Every path the backend sends is resolved here — `~` expanded, relative
//! paths made absolute — before anything touches the disk, so that what gets
//! described in a confirmation is the same thing that gets acted on.
//!
//! Reads are truncated. A model that asks for a file does not want three
//! megabytes of it, and sending three megabytes back would cost more context
//! than the answer is worth.

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use super::{require_arg, resolve_path, string_arg, LocalTool, Risk, SimpleTool, ToolOutcome};

/// Largest slice of a file returned in one read.
const MAX_READ_BYTES: usize = 64 * 1024;
/// Most matches returned by a search.
const MAX_MATCHES: usize = 60;
/// How deep a search descends before giving up.
const MAX_DEPTH: usize = 8;

pub(super) fn tools() -> Vec<Arc<dyn LocalTool>> {
    vec![
        Arc::new(SimpleTool {
            name: "filesystem.search",
            risk: |_| Risk::Safe,
            describe: |args| {
                let pattern = string_arg(args, "pattern").unwrap_or_else(|| "files".into());
                match string_arg(args, "root") {
                    Some(root) => format!("I'm about to look for {pattern} under {root}"),
                    None => format!("I'm about to look for {pattern} in your home folder"),
                }
            },
            run: |id, args| Box::pin(search(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "filesystem.read",
            risk: |_| Risk::Safe,
            describe: |args| {
                let path = string_arg(args, "path").unwrap_or_else(|| "a file".into());
                format!("I'm about to read {path}")
            },
            run: |id, args| Box::pin(read(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "filesystem.write",
            // Writing over something that exists is a different proposition
            // from creating something new, and the risk reflects that.
            risk: |args| match string_arg(args, "path") {
                Some(path) if resolve_path(&path).exists() => Risk::Normal,
                _ => Risk::Normal,
            },
            describe: |args| {
                let path = string_arg(args, "path").unwrap_or_else(|| "a file".into());
                if resolve_path(&path).exists() {
                    format!("I'm about to overwrite {path}")
                } else {
                    format!("I'm about to create {path}")
                }
            },
            run: |id, args| Box::pin(write(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "filesystem.delete",
            risk: |args| match string_arg(args, "path") {
                Some(path) if resolve_path(&path).is_dir() => Risk::Dangerous,
                _ => Risk::Dangerous,
            },
            describe: |args| {
                let path = string_arg(args, "path").unwrap_or_else(|| "a file".into());
                let resolved = resolve_path(&path);
                if resolved.is_dir() {
                    format!("I'm about to delete the folder {path} and everything in it")
                } else {
                    format!("I'm about to delete {path}")
                }
            },
            run: |id, args| Box::pin(delete(id, args)),
        }),
    ]
}

async fn search(id: String, args: Value) -> ToolOutcome {
    let pattern = match require_arg(&args, "pattern") {
        Ok(p) => p,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    let root = string_arg(&args, "root")
        .map(|r| resolve_path(&r))
        .or_else(super::home_dir)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    if !root.exists() {
        return ToolOutcome::failed(&id, format!("{} does not exist", root.display()));
    }

    let needle = pattern.trim_matches('*').to_lowercase();
    let matches = tokio::task::spawn_blocking(move || {
        let mut found = Vec::new();
        walk(&root, 0, &needle, &mut found);
        found
    })
    .await
    .unwrap_or_default();

    if matches.is_empty() {
        return ToolOutcome {
            id: id.clone(),
            ok: false,
            summary: format!("nothing matching {pattern}"),
            evidence: format!("0 matches for {pattern}"),
            data: Some(json!([])),
            error: None,
        };
    }
    let shown: Vec<String> = matches.iter().take(5).cloned().collect();
    ToolOutcome::ok(
        &id,
        format!("{} matches, including {}", matches.len(), shown.join(", ")),
        format!("{} files matched {pattern}", matches.len()),
    )
    .with_data(json!(matches))
}

/// Depth-limited walk that skips the directories nobody means.
fn walk(dir: &Path, depth: usize, needle: &str, found: &mut Vec<String>) {
    if depth > MAX_DEPTH || found.len() >= MAX_MATCHES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return; // unreadable directories are common and not worth failing on
    };
    for entry in entries.flatten() {
        if found.len() >= MAX_MATCHES {
            return;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_lowercase();
        // Hidden directories, build output and dependency trees are almost
        // never what somebody means by "find my project", and descending them
        // makes a search take minutes.
        if path.is_dir() {
            if name.starts_with('.')
                || matches!(
                    name.as_str(),
                    "node_modules" | "target" | "venv" | "__pycache__"
                )
            {
                continue;
            }
            walk(&path, depth + 1, needle, found);
        } else if needle.is_empty() || name.contains(needle) {
            found.push(path.to_string_lossy().to_string());
        }
    }
}

async fn read(id: String, args: Value) -> ToolOutcome {
    let path = match require_arg(&args, "path") {
        Ok(p) => resolve_path(&p),
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    let from = args.get("from_line").and_then(Value::as_u64).unwrap_or(0) as usize;
    let to = args
        .get("to_line")
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let contents = match tokio::fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(e) => {
            return ToolOutcome::failed(&id, format!("could not read {}: {e}", path.display()))
        }
    };
    let total = contents.lines().count();
    let slice: String = if from > 0 || to.is_some() {
        contents
            .lines()
            .skip(from.saturating_sub(1))
            .take(
                to.map(|t| t.saturating_sub(from.saturating_sub(1)))
                    .unwrap_or(usize::MAX),
            )
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        contents
    };
    let truncated = slice.len() > MAX_READ_BYTES;
    let slice: String = slice.chars().take(MAX_READ_BYTES).collect();

    ToolOutcome::ok(
        &id,
        format!(
            "{} ({total} lines{})",
            path.display(),
            if truncated { ", truncated" } else { "" }
        ),
        format!("read {} bytes from {}", slice.len(), path.display()),
    )
    .with_data(json!(slice))
}

async fn write(id: String, args: Value) -> ToolOutcome {
    let path = match require_arg(&args, "path") {
        Ok(p) => resolve_path(&p),
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    let content = match args.get("content").and_then(Value::as_str) {
        Some(c) => c.to_string(),
        None => return ToolOutcome::failed(&id, "that needs content"),
    };
    let existed = path.exists();
    if let Some(parent) = path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return ToolOutcome::failed(&id, format!("could not create {}: {e}", parent.display()));
        }
    }
    match tokio::fs::write(&path, content.as_bytes()).await {
        Ok(()) => ToolOutcome::ok(
            &id,
            format!(
                "{} {}",
                if existed { "updated" } else { "created" },
                path.display()
            ),
            format!("wrote {} bytes to {}", content.len(), path.display()),
        ),
        Err(e) => ToolOutcome::failed(&id, format!("could not write {}: {e}", path.display())),
    }
}

async fn delete(id: String, args: Value) -> ToolOutcome {
    let path = match require_arg(&args, "path") {
        Ok(p) => resolve_path(&p),
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    if !path.exists() {
        return ToolOutcome::failed(&id, format!("{} is not there", path.display()));
    }
    // Refusing outright rather than asking: no plausible request needs this,
    // and a confirmation dialogue is not a safeguard against a model that has
    // lost the plot.
    if is_protected(&path) {
        return ToolOutcome::failed(
            &id,
            format!(
                "I won't delete {} — it's not something I should touch",
                path.display()
            ),
        );
    }
    let result = if path.is_dir() {
        tokio::fs::remove_dir_all(&path).await
    } else {
        tokio::fs::remove_file(&path).await
    };
    match result {
        Ok(()) => ToolOutcome::ok(
            &id,
            format!("deleted {}", path.display()),
            format!("{} no longer exists", path.display()),
        ),
        Err(e) => ToolOutcome::failed(&id, format!("could not delete {}: {e}", path.display())),
    }
}

/// Paths no request should ever be allowed to remove.
fn is_protected(path: &Path) -> bool {
    let Ok(canonical) = path.canonicalize() else {
        return false;
    };
    if canonical.parent().is_none() {
        return true; // a filesystem root
    }
    if let Some(home) = super::home_dir() {
        if canonical == home {
            return true;
        }
    }
    let protected = [
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/proc",
        "/sys",
        "/usr",
        "/var",
        "C:\\Windows",
        "C:\\Program Files",
    ];
    protected.iter().any(|p| canonical == Path::new(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{PermissionPolicy, ToolHost, ToolRequest};

    fn host() -> ToolHost {
        ToolHost::new(PermissionPolicy::default())
    }

    fn request(name: &str, arguments: Value) -> ToolRequest {
        ToolRequest {
            id: "c1".into(),
            name: name.into(),
            version: 1,
            arguments,
            risk: None,
        }
    }

    #[tokio::test]
    async fn reading_a_file_returns_its_contents_and_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.txt");
        std::fs::write(&file, "one\ntwo\nthree").unwrap();

        let outcome = host()
            .run(&request(
                "filesystem.read",
                json!({"path": file.to_str().unwrap()}),
            ))
            .await;
        assert!(outcome.ok);
        assert!(outcome.summary.contains("3 lines"));
        assert!(outcome.evidence.contains("bytes"));
        assert_eq!(outcome.data.unwrap().as_str().unwrap(), "one\ntwo\nthree");
    }

    #[tokio::test]
    async fn reading_a_line_range_returns_only_those_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.txt");
        std::fs::write(&file, "one\ntwo\nthree\nfour").unwrap();

        let outcome = host()
            .run(&request(
                "filesystem.read",
                json!({"path": file.to_str().unwrap(), "from_line": 2, "to_line": 3}),
            ))
            .await;
        assert_eq!(outcome.data.unwrap().as_str().unwrap(), "two\nthree");
    }

    #[tokio::test]
    async fn writing_reports_whether_it_created_or_overwrote() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("new").join("thing.txt");
        let args = json!({"path": file.to_str().unwrap(), "content": "hello"});

        let created = host().run(&request("filesystem.write", args.clone())).await;
        assert!(created.ok && created.summary.starts_with("created"));
        let updated = host().run(&request("filesystem.write", args)).await;
        assert!(updated.summary.starts_with("updated"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
    }

    #[tokio::test]
    async fn searching_skips_the_directories_nobody_means() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
        std::fs::write(dir.path().join("node_modules/wanted.rs"), "").unwrap();
        std::fs::write(dir.path().join("wanted.rs"), "").unwrap();

        let outcome = host()
            .run(&request(
                "filesystem.search",
                json!({"pattern": "*.rs", "root": dir.path().to_str().unwrap()}),
            ))
            .await;
        let found = outcome.data.unwrap();
        assert_eq!(found.as_array().unwrap().len(), 1, "{found}");
    }

    #[tokio::test]
    async fn deleting_something_that_matters_is_refused_not_confirmed() {
        let home = super::super::home_dir().unwrap();
        let outcome = host()
            .run(&request(
                "filesystem.delete",
                json!({"path": home.to_str().unwrap()}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("won't delete"));
    }

    #[tokio::test]
    async fn deleting_an_ordinary_file_works_and_says_what_changed() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("old.log");
        std::fs::write(&file, "x").unwrap();
        let outcome = host()
            .run(&request(
                "filesystem.delete",
                json!({"path": file.to_str().unwrap()}),
            ))
            .await;
        assert!(outcome.ok);
        assert!(!file.exists());
        assert!(outcome.evidence.contains("no longer exists"));
    }

    #[test]
    fn descriptions_say_what_will_happen_in_plain_words() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("config.toml");
        std::fs::write(&existing, "").unwrap();
        let host = host();

        let overwrite = host
            .consider(&request(
                "filesystem.write",
                json!({"path": existing.to_str().unwrap(), "content": "x"}),
            ))
            .unwrap();
        match overwrite {
            crate::tools::Decision::Confirm { question, .. } => {
                assert!(question.starts_with("I'm about to overwrite"), "{question}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_missing_argument_is_explained_rather_than_panicking() {
        let outcome = host().run(&request("filesystem.read", json!({}))).await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("needs a path"));
    }
}
