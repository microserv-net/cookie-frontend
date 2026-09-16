//! Applications, browsers, and finding programs.
//!
//! Everything here is platform-specific in implementation and identical in
//! interface, which is the point: the backend asks to open an application by
//! name and never learns whether that meant `open -a`, `start`, or
//! `xdg-open`.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::process::Command;

use super::{require_arg, string_arg, LocalTool, Risk, SimpleTool, ToolOutcome};

pub(super) fn tools() -> Vec<Arc<dyn LocalTool>> {
    vec![
        Arc::new(SimpleTool {
            name: "app.open",
            risk: |_| Risk::Normal,
            describe: |args| {
                let name = string_arg(args, "name").unwrap_or_else(|| "an application".into());
                format!("I'm about to open {name}")
            },
            run: |id, args| Box::pin(open_app(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "app.close",
            risk: |_| Risk::Normal,
            describe: |args| {
                let name = string_arg(args, "name").unwrap_or_else(|| "an application".into());
                format!("I'm about to close {name}")
            },
            run: |id, args| Box::pin(close_app(id, args)),
        }),
        Arc::new(SimpleTool {
            name: "browser.open",
            risk: |_| Risk::Safe,
            describe: |args| {
                let url = string_arg(args, "url").unwrap_or_else(|| "a page".into());
                format!("I'm about to open {url}")
            },
            run: |id, args| Box::pin(open_url(id, args)),
        }),
    ]
}

/// Cross-platform `which`.
pub(crate) fn locate(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|s| s.to_lowercase())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        for extension in &extensions {
            let candidate = dir.join(format!("{program}{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

async fn open_app(id: String, args: Value) -> ToolOutcome {
    let name = match require_arg(&args, "name") {
        Ok(n) => n,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("open");
        c.arg("-a").arg(&name);
        c
    } else if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.arg("/C").arg("start").arg("").arg(&name);
        c
    } else {
        // On Linux the name is usually the binary; fall back to the desktop
        // entry if it is not on PATH.
        match locate(&name) {
            Some(path) => Command::new(path),
            None => {
                let mut c = Command::new("gtk-launch");
                c.arg(&name);
                c
            }
        }
    };
    command.stdout(Stdio::null()).stderr(Stdio::piped());

    match command.spawn() {
        Ok(_) => ToolOutcome::ok(
            &id,
            format!("opened {name}"),
            format!("{name} was launched"),
        ),
        Err(e) => ToolOutcome::failed(&id, format!("could not open {name}: {e}")),
    }
}

async fn close_app(id: String, args: Value) -> ToolOutcome {
    let name = match require_arg(&args, "name") {
        Ok(n) => n,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    // Asking the application to quit, not killing it: an editor with unsaved
    // work should get the chance to say so.
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("osascript");
        c.arg("-e")
            .arg(format!("tell application \"{name}\" to quit"));
        c
    } else if cfg!(target_os = "windows") {
        let mut c = Command::new("taskkill");
        c.arg("/IM").arg(format!("{name}.exe"));
        c
    } else {
        let mut c = Command::new("pkill");
        c.arg("-TERM").arg("-x").arg(&name);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::piped());

    match command.status().await {
        Ok(status) if status.success() => ToolOutcome::ok(
            &id,
            format!("closed {name}"),
            format!("{name} was asked to quit"),
        ),
        Ok(_) => ToolOutcome {
            id: id.clone(),
            ok: false,
            summary: format!("{name} does not seem to be running"),
            evidence: format!("nothing matching {name} was running"),
            data: None,
            error: None,
        },
        Err(e) => ToolOutcome::failed(&id, format!("could not close {name}: {e}")),
    }
}

async fn open_url(id: String, args: Value) -> ToolOutcome {
    let url = match require_arg(&args, "url") {
        Ok(u) => u,
        Err(e) => return ToolOutcome::failed(&id, e),
    };
    // Anything but http(s) here means the model has been persuaded to open a
    // `file://` or a custom scheme, and neither is a browsing request.
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolOutcome::failed(&id, format!("{url} is not a web address"));
    }
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("open");
        c.arg(&url);
        c
    } else if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.arg("/C").arg("start").arg("").arg(&url);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(&url);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::null());

    match command.spawn() {
        Ok(_) => ToolOutcome::ok(&id, format!("opened {url}"), format!("{url} was opened"))
            .with_data(json!(url)),
        Err(e) => ToolOutcome::failed(&id, format!("could not open {url}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{PermissionPolicy, ToolHost, ToolRequest};

    fn request(name: &str, arguments: Value) -> ToolRequest {
        ToolRequest {
            id: "c1".into(),
            name: name.into(),
            version: 1,
            arguments,
            risk: None,
        }
    }

    #[test]
    fn locate_finds_something_that_exists() {
        let probe = if cfg!(windows) { "cmd" } else { "sh" };
        assert!(locate(probe).is_some());
        assert!(locate("definitely-not-here-8a1b").is_none());
    }

    #[tokio::test]
    async fn non_web_addresses_are_refused() {
        let host = ToolHost::new(PermissionPolicy::default());
        let outcome = host
            .run(&request(
                "browser.open",
                json!({"url": "file:///etc/passwd"}),
            ))
            .await;
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("not a web address"));
    }

    #[test]
    fn opening_a_page_does_not_interrupt_you_but_opening_an_app_does() {
        let host = ToolHost::new(PermissionPolicy::default());
        assert_eq!(
            host.consider(&request(
                "browser.open",
                json!({"url": "https://example.com"})
            ))
            .unwrap(),
            crate::tools::Decision::Allow
        );
        assert!(matches!(
            host.consider(&request("app.open", json!({"name": "Visual Studio Code"})))
                .unwrap(),
            crate::tools::Decision::Confirm { .. }
        ));
    }
}
