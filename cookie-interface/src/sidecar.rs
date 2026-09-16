//! A tiny newline-delimited-JSON transport for model subprocesses.
//!
//! ## Why a sidecar at all
//!
//! The best STT/TTS runtimes are C++ (whisper.cpp, sherpa-onnx) or Python
//! (Qwen3-TTS, Kokoro). Linking either into this crate would mean `cargo
//! build` needs CMake, a C++ toolchain, or a Python environment — on three
//! operating systems, for a project whose whole point is being easy to run.
//! So the model runs in its own process and speaks a protocol that is four
//! lines long. You can write a conforming sidecar in twenty lines of Python.
//!
//! ## Protocol
//!
//! One JSON object per line, in both directions. A request always carries an
//! `id`; every response for that request echoes it.
//!
//! ```text
//! → {"id":"1","op":"transcribe","sample_rate":16000,"audio":"<base64 f32le>"}
//! ← {"id":"1","type":"result","text":"hello there","confidence":0.93}
//!
//! → {"id":"2","op":"synthesize","text":"Good evening.","voice":{...}}
//! ← {"id":"2","type":"chunk","sample_rate":24000,"audio":"<base64 f32le>"}
//! ← {"id":"2","type":"chunk","sample_rate":24000,"audio":"<base64 f32le>"}
//! ← {"id":"2","type":"end"}
//! ```
//!
//! `{"type":"error","message":"..."}` ends any exchange. Anything written to
//! the sidecar's stderr is forwarded to the log at `warn` level, which is how
//! model loading progress and Python tracebacks reach the user.
//!
//! The full schema, including a reference Python implementation, is in
//! `docs/models.md`.

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex};

use crate::error::{Error, Result};

struct Inner {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

/// A long-lived model subprocess.
///
/// The process is spawned lazily on first use and re-spawned automatically
/// after a crash, so a model that dies mid-session costs one failed request
/// rather than the whole application.
#[derive(Debug)]
pub struct SidecarProcess {
    command: Vec<String>,
    label: String,
    inner: Mutex<Option<Inner>>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Inner")
    }
}

impl SidecarProcess {
    /// `command` is `[program, args...]` exactly as configured.
    pub fn new(label: impl Into<String>, command: Vec<String>) -> Result<Self> {
        if command.is_empty() {
            return Err(Error::Config(format!(
                "{} sidecar is selected but no command is configured; \
                 set `sidecar_command = [\"python\", \"my_model.py\"]`",
                label.into()
            )));
        }
        Ok(Self {
            label: label.into(),
            command,
            inner: Mutex::new(None),
            next_id: AtomicU64::new(1),
        })
    }

    /// The configured program name, for logs and `--doctor`.
    pub fn program(&self) -> &str {
        &self.command[0]
    }

    fn spawn(&self) -> Result<Inner> {
        let mut cmd = Command::new(&self.command[0]);
        cmd.args(&self.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| Error::ModelUnavailable {
            name: self.command[0].clone(),
            reason: format!("could not start {} sidecar: {e}", self.label),
        })?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        if let Some(stderr) = child.stderr.take() {
            let label = self.label.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(target: "cookie::sidecar", sidecar = %label, "{line}");
                }
            });
        }
        Ok(Inner {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
        })
    }

    /// Send a request and collect every response line for it, up to and
    /// including the terminator.
    ///
    /// `on_line` is called for each `chunk`/`result` object. It returns
    /// `false` to abandon the exchange early (used for barge-in).
    async fn exchange<F>(&self, mut request: Value, mut on_line: F) -> Result<()>
    where
        F: FnMut(Value) -> bool,
    {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        request["id"] = Value::String(id.clone());
        let payload = format!("{request}\n");

        let mut guard = self.inner.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn()?);
        }

        // One retry: a sidecar that was killed while idle should not surface
        // as a user-visible failure.
        for attempt in 0..2 {
            let inner = match guard.as_mut() {
                Some(i) => i,
                None => {
                    *guard = Some(self.spawn()?);
                    guard.as_mut().expect("just spawned")
                }
            };
            match Self::run_once(inner, &payload, &id, &mut on_line).await {
                Ok(()) => return Ok(()),
                Err(e) if attempt == 0 && e.is_transport() => {
                    tracing::warn!(
                        sidecar = %self.label,
                        "sidecar transport failed ({e}); restarting the process"
                    );
                    if let Some(mut old) = guard.take() {
                        let _ = old.child.start_kill();
                    }
                }
                Err(e) => {
                    if e.is_transport() {
                        if let Some(mut old) = guard.take() {
                            let _ = old.child.start_kill();
                        }
                    }
                    return Err(e);
                }
            }
        }
        unreachable!("loop returns on both branches")
    }

    async fn run_once<F>(inner: &mut Inner, payload: &str, id: &str, on_line: &mut F) -> Result<()>
    where
        F: FnMut(Value) -> bool,
    {
        inner
            .stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| Error::Network(format!("sidecar stdin: {e}")))?;
        inner
            .stdin
            .flush()
            .await
            .map_err(|e| Error::Network(format!("sidecar stdin: {e}")))?;

        loop {
            let line = inner
                .lines
                .next_line()
                .await
                .map_err(|e| Error::Network(format!("sidecar stdout: {e}")))?
                .ok_or_else(|| Error::Network("sidecar closed its output".into()))?;
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    // A sidecar that prints stray text is common (model
                    // libraries love printing banners). Log and keep reading.
                    tracing::debug!("ignoring non-JSON sidecar line ({e}): {line}");
                    continue;
                }
            };
            if value
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|v| v != id)
            {
                continue;
            }
            match value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("result")
            {
                "error" => {
                    let msg = value
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unspecified sidecar error");
                    return Err(Error::Other(msg.to_string()));
                }
                "end" => return Ok(()),
                "result" => {
                    on_line(value);
                    return Ok(());
                }
                _ => {
                    if !on_line(value) {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Send a request and return its single `result` object.
    pub async fn call(&self, request: Value) -> Result<Value> {
        let mut out = None;
        self.exchange(request, |v| {
            out = Some(v);
            true
        })
        .await?;
        out.ok_or_else(|| Error::Other("sidecar produced no result".into()))
    }

    /// Send a request and forward every `chunk` object to `tx` as it arrives.
    ///
    /// Returns when the sidecar sends `end`, when it errors, or when the
    /// receiver is dropped (which is how speech is interrupted).
    pub async fn call_streaming(&self, request: Value, tx: mpsc::Sender<Value>) -> Result<()> {
        self.exchange(request, |v| tx.try_send(v).is_ok() || !tx.is_closed())
            .await
    }

    /// Ask the sidecar to describe itself. Used by `prepare()` and
    /// `--doctor`; a sidecar that does not implement it simply errors and we
    /// treat that as "alive but taciturn".
    pub async fn hello(&self) -> Result<Value> {
        match self.call(json!({"op": "hello"})).await {
            Ok(v) => Ok(v),
            Err(Error::Other(msg)) => Ok(json!({"note": msg})),
            Err(e) => Err(e),
        }
    }
}

impl Error {
    /// Transport-level failures justify restarting a sidecar; model-level
    /// ones do not.
    fn is_transport(&self) -> bool {
        matches!(self, Error::Network(_) | Error::Io { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_command_is_rejected() {
        assert!(SidecarProcess::new("stt", Vec::new()).is_err());
    }

    #[tokio::test]
    async fn missing_program_reports_model_unavailable() {
        let p =
            SidecarProcess::new("stt", vec!["definitely-not-a-real-program-9f3a".into()]).unwrap();
        let err = p.call(json!({"op":"hello"})).await.unwrap_err();
        assert_eq!(err.code(), "model_unavailable");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn echo_sidecar_round_trips() {
        // A one-line sidecar: read a request, answer with a result.
        let script = r#"
import sys, json
for line in sys.stdin:
    req = json.loads(line)
    print(json.dumps({"id": req["id"], "type": "result", "text": "hello there"}), flush=True)
"#;
        let python = ["python3", "python"]
            .iter()
            .find(|p| std::process::Command::new(p).arg("-V").output().is_ok())
            .copied();
        let Some(python) = python else {
            return; // no interpreter in this environment; nothing to assert
        };
        let p =
            SidecarProcess::new("stt", vec![python.into(), "-c".into(), script.into()]).unwrap();
        let out = p.call(json!({"op":"transcribe"})).await.unwrap();
        assert_eq!(out["text"], "hello there");
    }
}
