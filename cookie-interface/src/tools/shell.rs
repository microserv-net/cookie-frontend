//! Commands.
//!
//! `shell.run` takes a program and a list of arguments — never a string to be
//! interpreted by a shell. That is the difference between the backend asking
//! for `git status` and the backend being able to ask for anything at all
//! with a semicolon in it, and it is why this file spawns the program
//! directly rather than going through `sh -c`.
//!
//! Risk is judged from the program, not from the sentence that produced it:
//! `ls` is safe, `git commit` is normal, `rm` is dangerous, `sudo` and a
//! force push are critical. The user gets asked accordingly, in words that
//! describe the consequence rather than the command line.

use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::process::Command;

use super::{require_arg, resolve_path, string_arg, LocalTool, Risk, SimpleTool, ToolOutcome};

/// How long a command may run before it is killed.
const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
/// How much output is kept. Beyond this the interesting part is the tail.
const MAX_OUTPUT_CHARS: usize = 8_000;

pub(super) fn tools() -> Vec<Arc<dyn LocalTool>> {
    vec![
        Arc::new(SimpleTool {
            name: "shell.which",
            risk: |_| Risk::Safe,
            describe: |args| {
                let command = string_arg(args, "command").unwrap_or_else(|| "a command".into());
                format!("I'm about to check whether {command} is installed")
            },
            run: |id, args| Box::pin(which(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "shell.run",
            risk: |args| classify(&string_arg(args, "command").unwrap_or_default(), args),
            describe: |args| describe_command(args),
            run: |id, args| Box::pin(run(id, args)),
        }),
    ]
}

/// What a program can do, judged by name.
///
/// A deny-list would be the wrong shape — there are too many ways to destroy
/// a machine to enumerate — so this is a graded classification with an
/// unknown-is-normal default. Anything genuinely destructive that is not
/// listed still gets `Normal`, which still asks.
pub(crate) fn classify(program: &str, args: &Value) -> Risk {
    let name = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .to_lowercase();
    let arguments: Vec<String> = args
        .get("arguments")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .collect()
        })
        .unwrap_or_default();
    let joined = arguments.join(" ");

    match name.as_str() {
        // Reading the world.
        "ls" | "dir" | "pwd" | "cat" | "head" | "tail" | "find" | "grep" | "rg" | "which"
        | "where" | "whoami" | "date" | "df" | "du" | "ps" | "echo" | "wc" | "file" | "stat" => {
            Risk::Safe
        }
        // Irreversible or privileged, whatever the arguments.
        "sudo" | "doas" | "su" | "mkfs" | "fdisk" | "diskutil" | "dd" | "shutdown" | "reboot" => {
            Risk::Critical
        }
        "rm" | "rmdir" | "del" | "kill" | "pkill" | "chmod" | "chown" => Risk::Dangerous,
        "apt" | "apt-get" | "brew" | "pip" | "pip3" | "npm" | "cargo" if is_install(&joined) => {
            Risk::Dangerous
        }
        "git" => {
            if joined.contains("push") && (joined.contains("--force") || joined.contains("-f")) {
                Risk::Critical
            } else if joined.starts_with("status")
                || joined.starts_with("log")
                || joined.starts_with("diff")
                || joined.starts_with("show")
                || joined.starts_with("branch")
            {
                Risk::Safe
            } else {
                Risk::Normal
            }
        }
        // Building and testing read a lot and write only to their own output
        // directories, and asking every time would make the assistant useless
        // for the thing it is most useful for.
        "cargo" | "make" | "npm" | "pytest" | "go" | "mvn" | "gradle" if is_build(&joined) => {
            Risk::Safe
        }
        _ => Risk::Normal,
    }
}

fn is_install(arguments: &str) -> bool {
    ["install", "add", "remove", "uninstall", "upgrade"]
        .iter()
        .any(|verb| arguments.split_whitespace().any(|word| word == *verb))
}

fn is_build(arguments: &str) -> bool {
    ["build", "test", "check", "clippy", "fmt", "run", "lint"]
        .iter()
        .any(|verb| arguments.split_whitespace().any(|word| word == *verb))
}

/// The sentence shown when asking permission.
fn describe_command(args: &Value) -> String {
    let command = string_arg(args, "command").unwrap_or_else(|| "a command".into());
    let arguments: Vec<String> = args
        .get("arguments")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let joined = arguments.join(" ");
    let name = command.rsplit(['/', '\\']).next().unwrap_or(&command);

    // Known shapes get a sentence about the consequence. Everything else
    // falls back to naming the program, which is still more use than pasting
    // an argv.
    match name {
        "rm" | "del" => "I'm about to delete some files".to_string(),
        "git" if joined.starts_with("commit") => "I'm about to commit the changes".to_string(),
        "git" if joined.starts_with("push") => "I'm about to push the branch".to_string(),
        "git" if joined.starts_with("checkout") || joined.starts_with("switch") => {
            "I'm about to switch branches".to_string()
        }
        "cargo" | "npm" | "make" | "go" if is_install(&joined) => {
            "I'm about to install some packages".to_string()
        }
        "cargo" | "npm" | "make" | "go" | "pytest" => {
            "I'm about to build and test the project".to_string()
        }
        _ if joined.is_empty() => format!("I'm about to run {name}"),
        _ => format!("I'm about to run {name} {}", shorten(&joined, 60)),
    }
}

fn shorten(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    format!("{}…", text.chars().take(limit).collect::<String>())
}

async fn which(id: String, args: Value) -> ToolOutcome {
    let command = match require_arg(&args, "command") {
        Ok(c) => c,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    match crate::tools::system::locate(&command) {
        Some(path) => ToolOutcome::ok(
            &id,
            format!("{command} is at {}", path.display()),
            format!("{command} found at {}", path.display()),
        )
        .with_data(json!(path.to_string_lossy())),
        None => ToolOutcome {
            id: id.clone(),
            ok: false,
            summary: format!("{command} is not installed"),
            evidence: format!("{command} is not on PATH"),
            data: None,
            error: None,
        },
    }
}

async fn run(id: String, args: Value) -> ToolOutcome {
    let program = match require_arg(&args, "command") {
        Ok(c) => c,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    let arguments: Vec<String> = args
        .get("arguments")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let timeout = args
        .get("timeout_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
        .clamp(1, 900);

    let mut command = Command::new(&program);
    command
        .args(&arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = string_arg(&args, "cwd") {
        command.current_dir(resolve_path(&cwd));
    }

    let child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            return ToolOutcome::failed(
                &id,
                format!("could not run {program}: {e}. Is it installed?"),
            )
        }
    };

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(timeout),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => return ToolOutcome::failed(&id, format!("{program} failed: {e}")),
        Err(_) => {
            return ToolOutcome::failed(
                &id,
                format!("{program} was still running after {timeout} seconds, so I stopped it"),
            )
        }
    };

    let code = output.status.code().unwrap_or(-1);
    let stdout = tail(&String::from_utf8_lossy(&output.stdout));
    let stderr = tail(&String::from_utf8_lossy(&output.stderr));
    let succeeded = output.status.success();

    // The evidence is the exit code and the tail of the output, because that
    // is what the backend's validator can actually judge — "it worked" is an
    // opinion, "exit code 0, 42 passed" is not.
    let evidence = format!(
        "exit code {code}{}{}",
        if stdout.is_empty() {
            String::new()
        } else {
            format!("; output: {}", shorten(&stdout, 300))
        },
        if stderr.is_empty() {
            String::new()
        } else {
            format!("; errors: {}", shorten(&stderr, 300))
        }
    );

    ToolOutcome {
        id: id.clone(),
        ok: succeeded,
        summary: if succeeded {
            format!("{program} finished cleanly")
        } else {
            format!("{program} exited with code {code}")
        },
        evidence,
        data: Some(json!({"exit_code": code, "stdout": stdout, "stderr": stderr})),
        error: None,
    }
}

/// Keep the end of long output: errors are at the bottom, not the top.
fn tail(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_OUTPUT_CHARS {
        return trimmed.to_string();
    }
    let kept: String = trimmed
        .chars()
        .skip(trimmed.chars().count() - MAX_OUTPUT_CHARS)
        .collect();
    format!("…{kept}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Decision, PermissionPolicy, ToolHost, ToolRequest};

    fn request(arguments: Value) -> ToolRequest {
        ToolRequest {
            id: "c1".into(),
            name: "shell.run".into(),
            version: 1,
            arguments,
            risk: None,
        }
    }

    #[test]
    fn reading_commands_are_safe_and_destructive_ones_are_not() {
        assert_eq!(classify("ls", &json!({})), Risk::Safe);
        assert_eq!(classify("/usr/bin/grep", &json!({})), Risk::Safe);
        assert_eq!(classify("rm", &json!({})), Risk::Dangerous);
        assert_eq!(classify("sudo", &json!({})), Risk::Critical);
        // Unknown means normal, which still asks.
        assert_eq!(classify("some-tool-nobody-knows", &json!({})), Risk::Normal);
    }

    #[test]
    fn git_is_judged_by_what_it_is_being_asked_to_do() {
        assert_eq!(
            classify("git", &json!({"arguments": ["status"]})),
            Risk::Safe
        );
        assert_eq!(
            classify("git", &json!({"arguments": ["commit", "-m", "x"]})),
            Risk::Normal
        );
        assert_eq!(
            classify("git", &json!({"arguments": ["push", "--force"]})),
            Risk::Critical
        );
    }

    #[test]
    fn building_and_testing_do_not_interrupt_you_but_installing_does() {
        assert_eq!(
            classify("cargo", &json!({"arguments": ["test"]})),
            Risk::Safe
        );
        assert_eq!(
            classify("npm", &json!({"arguments": ["run", "build"]})),
            Risk::Safe
        );
        assert_eq!(
            classify("pip", &json!({"arguments": ["install", "x"]})),
            Risk::Dangerous
        );
    }

    #[test]
    fn confirmations_describe_the_consequence_not_the_command_line() {
        let host = ToolHost::new(PermissionPolicy::default());
        let decision = host
            .consider(&request(
                json!({"command": "rm", "arguments": ["-rf", "target"]}),
            ))
            .unwrap();
        match decision {
            Decision::Confirm { question, .. } => {
                assert!(
                    question.starts_with("I'm about to delete some files"),
                    "{question}"
                );
                assert!(!question.contains("-rf"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn output_and_exit_code_come_back_as_evidence() {
        let host = ToolHost::new(PermissionPolicy::default());
        let (program, arguments) = if cfg!(windows) {
            ("cmd", json!(["/C", "echo hello"]))
        } else {
            ("echo", json!(["hello"]))
        };
        let outcome = host
            .run(&request(
                json!({"command": program, "arguments": arguments}),
            ))
            .await;
        assert!(outcome.ok, "{outcome:?}");
        assert!(outcome.evidence.contains("exit code 0"));
        assert!(outcome.data.unwrap()["stdout"]
            .as_str()
            .unwrap()
            .contains("hello"));
    }

    #[tokio::test]
    async fn a_failing_command_is_reported_rather_than_hidden() {
        let host = ToolHost::new(PermissionPolicy::default());
        let (program, arguments) = if cfg!(windows) {
            ("cmd", json!(["/C", "exit 3"]))
        } else {
            ("sh", json!(["-c", "exit 3"]))
        };
        let outcome = host
            .run(&request(
                json!({"command": program, "arguments": arguments}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.evidence.contains("exit code 3"));
    }

    #[tokio::test]
    async fn a_missing_program_says_so_usefully() {
        let host = ToolHost::new(PermissionPolicy::default());
        let outcome = host
            .run(&request(
                json!({"command": "definitely-not-installed-9f3a"}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("Is it installed?"));
    }

    #[tokio::test]
    async fn a_command_that_will_not_finish_is_stopped() {
        let host = ToolHost::new(PermissionPolicy::default());
        let (program, arguments) = if cfg!(windows) {
            // Not `timeout`: it refuses to run when stdin is not a console,
            // which is exactly how we spawn it, so it exits immediately and
            // the test measures nothing.
            ("cmd", json!(["/C", "ping -n 31 127.0.0.1"]))
        } else {
            ("sleep", json!(["30"]))
        };
        let outcome = host
            .run(&request(
                json!({"command": program, "arguments": arguments, "timeout_seconds": 1}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("so I stopped it"));
    }
}
