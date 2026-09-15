//! Git.
//!
//! Declared in the backend's catalogue since the tool system landed and
//! unimplemented until now, which meant a plan that reached for `git.status`
//! got "this machine has no tool called git.status" — a confusing way to find
//! out that a feature does not exist.
//!
//! Everything here runs `git` with structured arguments, never a command
//! string. The risk classifications are the interesting part: reading is
//! free, committing asks, and pushing with force asks in the register
//! reserved for things that cannot be undone.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::process::Command;

use super::{resolve_path, string_arg, LocalTool, Risk, SimpleTool, ToolOutcome};

/// Long enough for a push over a slow connection, short enough that a
/// credential prompt nobody can see does not hang the turn forever.
const TIMEOUT_SECONDS: u64 = 120;
const MAX_OUTPUT_CHARS: usize = 8_000;

pub(super) fn tools() -> Vec<Arc<dyn LocalTool>> {
    vec![
        Arc::new(SimpleTool {
            name: "git.status",
            risk: |_| Risk::Safe,
            describe: |args| format!("I'm about to check the state of {}", repo_label(args)),
            run: |id, args| Box::pin(status(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "git.diff",
            risk: |_| Risk::Safe,
            describe: |args| format!("I'm about to look at the changes in {}", repo_label(args)),
            run: |id, args| Box::pin(diff(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "git.commit",
            risk: |_| Risk::Normal,
            describe: |args| {
                let message = string_arg(args, "message").unwrap_or_default();
                if message.is_empty() {
                    "I'm about to commit the changes".to_string()
                } else {
                    format!(
                        "I'm about to commit the changes as \"{}\"",
                        shorten(&message)
                    )
                }
            },
            run: |id, args| Box::pin(commit(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "git.push",
            // Force-pushing rewrites somebody else's history as well as your
            // own, and no amount of confirmation wording makes that ordinary.
            risk: |args| {
                if args.get("force").and_then(Value::as_bool).unwrap_or(false) {
                    Risk::Critical
                } else {
                    Risk::Normal
                }
            },
            describe: |args| {
                if args.get("force").and_then(Value::as_bool).unwrap_or(false) {
                    "I'm about to force push, which overwrites what is on the remote".to_string()
                } else {
                    "I'm about to push the current branch".to_string()
                }
            },
            run: |id, args| Box::pin(push(id, args)),
        }),
    ]
}

fn repo_label(args: &Value) -> String {
    string_arg(args, "repository").unwrap_or_else(|| "this repository".into())
}

fn shorten(text: &str) -> String {
    if text.chars().count() <= 60 {
        return text.to_string();
    }
    format!("{}…", text.chars().take(60).collect::<String>())
}

/// Where to run. Defaults to the working directory, which is the project the
/// user started Cookie in.
fn repository(args: &Value) -> PathBuf {
    string_arg(args, "repository")
        .map(|raw| resolve_path(&raw))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

async fn run_git(
    id: &str,
    args: &Value,
    arguments: &[&str],
) -> std::result::Result<String, Box<ToolOutcome>> {
    let directory = repository(args);
    if !directory.join(".git").exists() && !directory.join("..").join(".git").exists() {
        return Err(Box::new(ToolOutcome::failed(
            id,
            format!("{} is not inside a git repository", directory.display()),
        )));
    }

    let mut command = Command::new("git");
    command
        .args(arguments)
        .current_dir(&directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Git will happily wait forever on a credential prompt nobody can
        // see; this makes it fail instead, which the model can act on.
        .env("GIT_TERMINAL_PROMPT", "0")
        .kill_on_drop(true);

    let child = command.spawn().map_err(|e| {
        Box::new(ToolOutcome::failed(
            id,
            format!("could not run git: {e}. Is it installed?"),
        ))
    })?;
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(TIMEOUT_SECONDS),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| {
        Box::new(ToolOutcome::failed(
            id,
            "git did not finish in time and was stopped",
        ))
    })?
    .map_err(|e| Box::new(ToolOutcome::failed(id, format!("git failed: {e}"))))?;

    let stdout = truncate(&String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        let stderr = truncate(&String::from_utf8_lossy(&output.stderr));
        return Err(Box::new(ToolOutcome {
            id: id.to_string(),
            ok: false,
            summary: format!("git {} failed", arguments.first().unwrap_or(&"")),
            evidence: format!(
                "exit code {}: {}",
                output.status.code().unwrap_or(-1),
                shorten_output(&stderr)
            ),
            data: Some(json!({"stdout": stdout, "stderr": stderr})),
            error: None,
        }));
    }
    Ok(stdout)
}

fn truncate(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_OUTPUT_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(MAX_OUTPUT_CHARS).collect::<String>() + "…"
}

fn shorten_output(text: &str) -> String {
    text.lines().next_back().unwrap_or("").trim().to_string()
}

async fn status(id: String, args: Value) -> ToolOutcome {
    // Porcelain v1 is the stable machine-readable form; the human one changes
    // between versions and is translated.
    let output = match run_git(&id, &args, &["status", "--porcelain=v1", "--branch"]).await {
        Ok(output) => output,
        Err(outcome) => return *outcome,
    };
    let mut lines = output.lines();
    let branch = lines
        .next()
        .unwrap_or("")
        .trim_start_matches("## ")
        .to_string();
    let changes: Vec<&str> = lines.collect();

    let summary = if changes.is_empty() {
        format!("on {branch}, nothing to commit")
    } else {
        format!("on {branch}, {} file(s) changed", changes.len())
    };
    ToolOutcome::ok(
        &id,
        summary,
        format!("{} change(s) on {branch}", changes.len()),
    )
    .with_data(json!({"branch": branch, "changes": changes}))
}

async fn diff(id: String, args: Value) -> ToolOutcome {
    let mut arguments = vec!["diff", "--stat", "--patch"];
    let path = string_arg(&args, "path");
    if let Some(path) = &path {
        arguments.push("--");
        arguments.push(path);
    }
    let output = match run_git(&id, &args, &arguments).await {
        Ok(output) => output,
        Err(outcome) => return *outcome,
    };
    if output.is_empty() {
        return ToolOutcome {
            id: id.clone(),
            ok: true,
            summary: "no uncommitted changes".into(),
            evidence: "the diff is empty".into(),
            data: Some(json!("")),
            error: None,
        };
    }
    let lines = output.lines().count();
    ToolOutcome::ok(
        &id,
        format!("a diff of {lines} lines"),
        format!("{lines} lines of diff"),
    )
    .with_data(json!(output))
}

async fn commit(id: String, args: Value) -> ToolOutcome {
    let Some(message) = string_arg(&args, "message") else {
        return ToolOutcome::failed(&id, "a commit needs a message");
    };
    if let Err(outcome) = run_git(&id, &args, &["add", "-A"]).await {
        return *outcome;
    }
    let output = match run_git(&id, &args, &["commit", "-m", &message]).await {
        Ok(output) => output,
        Err(outcome) => return *outcome,
    };
    // The short hash is the evidence: it is checkable, unlike "committed".
    let hash = match run_git(&id, &args, &["rev-parse", "--short", "HEAD"]).await {
        Ok(hash) => hash,
        Err(outcome) => return *outcome,
    };
    ToolOutcome::ok(
        &id,
        format!("committed as {hash}"),
        format!("commit {hash}: {}", output.lines().next().unwrap_or("")),
    )
    .with_data(json!({"commit": hash}))
}

async fn push(id: String, args: Value) -> ToolOutcome {
    let force = args.get("force").and_then(Value::as_bool).unwrap_or(false);
    let mut arguments = vec!["push"];
    if force {
        // `--force-with-lease` rather than `--force`: it refuses when the
        // remote has moved since you last looked, which is the case where
        // force-pushing destroys somebody else's work.
        arguments.push("--force-with-lease");
    }
    match run_git(&id, &args, &arguments).await {
        Ok(output) => ToolOutcome::ok(
            &id,
            if force { "force pushed" } else { "pushed" },
            format!(
                "git push succeeded: {}",
                output.lines().next_back().unwrap_or("")
            ),
        ),
        Err(outcome) => *outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Decision, PermissionPolicy, ToolHost, ToolRequest};

    fn request(name: &str, arguments: Value) -> ToolRequest {
        ToolRequest {
            id: "c1".into(),
            name: name.into(),
            version: 1,
            arguments,
            risk: None,
        }
    }

    async fn repository() -> Option<tempfile::TempDir> {
        let dir = tempfile::tempdir().ok()?;
        let ok = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .ok()?
            .success();
        if !ok {
            return None;
        }
        for (key, value) in [
            ("user.email", "cookie@example.invalid"),
            ("user.name", "Cookie"),
        ] {
            let _ = std::process::Command::new("git")
                .args(["config", key, value])
                .current_dir(dir.path())
                .status();
        }
        Some(dir)
    }

    #[test]
    fn reading_is_free_and_writing_is_not() {
        let host = ToolHost::new(PermissionPolicy::default());
        assert_eq!(
            host.consider(&request("git.status", json!({}))).unwrap(),
            Decision::Allow
        );
        assert!(matches!(
            host.consider(&request("git.commit", json!({"message": "x"})))
                .unwrap(),
            Decision::Confirm { .. }
        ));
    }

    #[test]
    fn a_force_push_is_asked_about_in_a_different_register() {
        let host = ToolHost::new(PermissionPolicy::default());
        match host
            .consider(&request("git.push", json!({"force": true})))
            .unwrap()
        {
            Decision::Confirm { question, risk } => {
                assert_eq!(risk, Risk::Critical);
                assert!(question.contains("can't be undone"), "{question}");
                assert!(
                    question.contains("overwrites what is on the remote"),
                    "{question}"
                );
            }
            other => panic!("{other:?}"),
        }
        // And an ordinary push is merely normal.
        assert_eq!(
            host.risk_of(&request("git.push", json!({}))),
            Some(Risk::Normal)
        );
    }

    #[tokio::test]
    async fn status_reports_the_branch_and_the_changes() {
        let Some(dir) = repository().await else {
            eprintln!("no git on this machine, skipping");
            return;
        };
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();

        let host = ToolHost::new(PermissionPolicy::default());
        let outcome = host
            .run(&request(
                "git.status",
                json!({"repository": dir.path().to_str().unwrap()}),
            ))
            .await;
        assert!(outcome.ok, "{outcome:?}");
        assert!(outcome.summary.contains("1 file(s) changed"), "{outcome:?}");
    }

    #[tokio::test]
    async fn committing_returns_a_hash_as_evidence() {
        let Some(dir) = repository().await else {
            return;
        };
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let repository = json!({"repository": dir.path().to_str().unwrap(),
                                "message": "add a"});

        let host = ToolHost::new(PermissionPolicy::default());
        let outcome = host.run(&request("git.commit", repository)).await;
        assert!(outcome.ok, "{outcome:?}");
        // "committed" is a claim; a hash is checkable, which is what the
        // backend's validator needs.
        assert!(outcome.evidence.starts_with("commit "), "{outcome:?}");
    }

    #[tokio::test]
    async fn somewhere_that_is_not_a_repository_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(PermissionPolicy::default());
        let outcome = host
            .run(&request(
                "git.status",
                json!({"repository": dir.path().to_str().unwrap()}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("not inside a git repository"));
    }
}
