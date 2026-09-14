//! Talking to the Cookie **backend**.
//!
//! `cookie-interface` is the face; the mind is a different program, usually on
//! a different machine — a home server, a container, a box on the LAN reached
//! as `http://192.168.1.42:8080/api`. This module is the one place the
//! interface dials out.
//!
//! ## The contract, in full
//!
//! One request per conversational turn:
//!
//! ```text
//! POST {base_url}{chat_path}
//! Content-Type: application/json
//! Accept: application/x-ndjson, application/json
//!
//! {
//!   "protocol": "cookie-interface/1",
//!   "session_id": "6b1c…",
//!   "utterance_id": "9f2a…",
//!   "text": "what's the weather like",
//!   "final": true,
//!   "client": { "name": "cookie-interface", "version": "0.1.0" },
//!   "interface": { "speech": true, "listening": true, "visual": "orb" }
//! }
//! ```
//!
//! The backend answers in whichever of three shapes suits it. All three are
//! accepted, because the backend does not exist yet and pinning it to one
//! would be a guess:
//!
//! * **newline-delimited JSON** (preferred) — `{"type":"delta","text":"…"}`
//!   repeatedly, then `{"type":"end"}`. Each delta is spoken as soon as a
//!   sentence boundary is reached, so the reply starts before the backend has
//!   finished thinking.
//! * **server-sent events** — the same objects behind `data:` lines.
//! * **a plain JSON object** — `{"reply":"…"}` (or `text`/`message`), spoken
//!   in one go.
//!
//! ## Room to grow
//!
//! Any object type the interface does not recognise is ignored rather than
//! treated as an error, and `{"type":"ui", …}` is **reserved**: it is where
//! future Cookie visual instructions will arrive (colour, mood, gesture,
//! overlay, attention target). A backend written today against this protocol
//! will keep working when that lands; see `docs/backend.md` and the same note
//! on the HTTP API in [`crate::api`].

use serde::Serialize;
use serde_json::Value;

use crate::config::BackendConfig;
use crate::error::Result;
use crate::events::Command;

/// Wire protocol identifier sent with every turn.
pub const BACKEND_PROTOCOL: &str = "cookie-interface/1";

/// What the interface tells the backend about itself.
///
/// Additive by design: a future version will describe visual capabilities
/// here, and a backend that ignores unknown keys keeps working.
#[derive(Debug, Clone, Serialize)]
pub struct InterfaceDescriptor {
    pub speech: bool,
    pub listening: bool,
    pub visual: &'static str,
    pub version: &'static str,
}

impl Default for InterfaceDescriptor {
    fn default() -> Self {
        Self {
            speech: true,
            listening: true,
            visual: "orb",
            version: crate::VERSION,
        }
    }
}

/// Scheduling hints attached to a turn.
///
/// See [`crate::tasks`] for why the interface is the right place to decide
/// these. A backend that ignores them still works correctly; it just makes a
/// constrained machine feel worse.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TurnOptions {
    /// `interactive`, `normal` or `background`.
    pub priority: crate::tasks::Priority,
    /// Ask the backend to suspend heavier work at its next natural checkpoint
    /// — after a model call returns, before a model swap, between plan steps —
    /// answer this, and then resume. Never a request to abandon anything.
    pub preempt: bool,
}

impl Default for TurnOptions {
    fn default() -> Self {
        Self {
            priority: crate::tasks::Priority::Normal,
            preempt: false,
        }
    }
}

/// One turn of conversation sent to the backend.
#[derive(Debug, Clone, Serialize)]
pub struct TurnRequest {
    pub protocol: &'static str,
    pub session_id: String,
    pub utterance_id: String,
    pub text: String,
    pub r#final: bool,
    pub interface: InterfaceDescriptor,
    /// Scheduling hints; see [`TurnOptions`].
    pub scheduling: TurnOptions,
    /// Tasks the interface believes are still in flight. Lets a backend that
    /// restarted notice the disagreement and clean up.
    pub active_tasks: Vec<String>,
}

#[cfg(feature = "http-providers")]
pub use imp::BackendClient;

#[cfg(not(feature = "http-providers"))]
pub use stub::BackendClient;

#[cfg(feature = "http-providers")]
mod imp {
    use std::time::Duration;

    use futures_util::StreamExt;
    use reqwest::Client;
    use tokio::sync::mpsc;

    use super::*;
    use crate::error::Error;
    use crate::events::SpeakRequest;

    /// HTTP client for the Cookie backend.
    #[derive(Debug)]
    pub struct BackendClient {
        client: Client,
        config: BackendConfig,
        session_id: String,
    }

    impl BackendClient {
        /// `None` when no backend is configured — the ordinary case for a
        /// front-end driven over the local API.
        pub fn from_config(config: &BackendConfig) -> Option<Self> {
            if !config.enabled {
                return None;
            }
            let client = Client::builder()
                .connect_timeout(Duration::from_millis(config.connect_timeout_ms.max(100)))
                .timeout(Duration::from_millis(config.request_timeout_ms.max(1_000)))
                .build()
                .map_err(|e| tracing::error!("backend client: {e}"))
                .ok()?;
            let session_id = if config.session_id.is_empty() {
                uuid::Uuid::new_v4().to_string()
            } else {
                config.session_id.clone()
            };
            tracing::info!(url = %config.chat_url(), "cookie backend configured");
            Some(Self {
                client,
                config: config.clone(),
                session_id,
            })
        }

        /// The session identifier sent with every turn.
        pub fn session_id(&self) -> &str {
            &self.session_id
        }

        /// Probe the backend's health endpoint. Used by `--doctor`.
        pub async fn health(&self) -> Result<Value> {
            let resp = self
                .request(self.config.health_url())
                .send()
                .await
                .map_err(|e| Error::Network(format!("backend health: {e}")))?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(Error::Network(format!(
                    "backend health returned {status}: {}",
                    body.chars().take(200).collect::<String>()
                )));
            }
            Ok(serde_json::from_str(&body).unwrap_or(Value::String(body)))
        }

        fn request(&self, url: String) -> reqwest::RequestBuilder {
            let mut req = self.client.get(&url);
            if let Some(token) = self.config.auth_token() {
                req = req.bearer_auth(token);
            }
            for (k, v) in &self.config.headers {
                req = req.header(k, v);
            }
            req
        }

        /// Send one utterance and speak whatever comes back.
        ///
        /// Commands are pushed into the engine's own queue, so a backend reply
        /// travels exactly the same path as a reply from the local API — the
        /// engine cannot tell them apart, and neither can the orb.
        #[allow(clippy::too_many_arguments)]
        pub async fn turn(
            &self,
            utterance_id: &str,
            text: &str,
            options: TurnOptions,
            active_tasks: Vec<String>,
            commands: mpsc::Sender<Command>,
            tasks: std::sync::Arc<crate::tasks::TaskRegistry>,
            bus: std::sync::Arc<crate::events::EventBus>,
        ) -> Result<()> {
            let body = TurnRequest {
                protocol: BACKEND_PROTOCOL,
                session_id: self.session_id.clone(),
                utterance_id: utterance_id.to_string(),
                text: text.to_string(),
                r#final: true,
                interface: InterfaceDescriptor::default(),
                scheduling: options,
                active_tasks,
            };
            let mut req = self
                .client
                .post(self.config.chat_url())
                .header(
                    "Accept",
                    "application/x-ndjson, text/event-stream, application/json",
                )
                .json(&body);
            if let Some(token) = self.config.auth_token() {
                req = req.bearer_auth(token);
            }
            for (k, v) in &self.config.headers {
                req = req.header(k, v);
            }

            let resp = req
                .send()
                .await
                .map_err(|e| Error::Network(format!("backend request failed: {e}")))?;
            let status = resp.status();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_ascii_lowercase();

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(Error::Network(format!(
                    "backend returned {status}: {}",
                    body.chars().take(300).collect::<String>()
                )));
            }

            let streaming = self.config.speak_streaming_reply
                && (content_type.contains("ndjson")
                    || content_type.contains("event-stream")
                    || content_type.contains("jsonl"));

            if !streaming {
                let value: Value = resp
                    .json()
                    .await
                    .map_err(|e| Error::Network(format!("backend reply was not JSON: {e}")))?;
                if let Some(reply) = reply_text(&value) {
                    if !reply.trim().is_empty() {
                        let _ = commands
                            .send(Command::Speak(Box::new(SpeakRequest {
                                utterance_id: format!("{utterance_id}-reply"),
                                text: reply,
                                voice: Default::default(),
                                interrupt: true,
                                echo_text: false,
                                persist: true,
                            })))
                            .await;
                    }
                }
                return Ok(());
            }

            self.stream_reply(utterance_id, resp, commands, tasks, bus)
                .await
        }

        /// Ask the backend to abandon work.
        ///
        /// `task_id` of `None` means everything. Cancelling is always
        /// explicit — nothing in the speech path calls this on its own.
        pub async fn cancel(&self, task_id: Option<&str>) -> Result<()> {
            let url =
                crate::config::join_public_url(&self.config.base_url, &self.config.cancel_path);
            let mut req = self.client.post(&url).json(&serde_json::json!({
                "protocol": BACKEND_PROTOCOL,
                "session_id": self.session_id,
                "task_id": task_id,
            }));
            if let Some(token) = self.config.auth_token() {
                req = req.bearer_auth(token);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| Error::Network(format!("cancel request failed: {e}")))?;
            if !resp.status().is_success() {
                return Err(Error::Network(format!(
                    "backend refused the cancellation: {}",
                    resp.status()
                )));
            }
            Ok(())
        }

        async fn stream_reply(
            &self,
            utterance_id: &str,
            resp: reqwest::Response,
            commands: mpsc::Sender<Command>,
            tasks: std::sync::Arc<crate::tasks::TaskRegistry>,
            bus: std::sync::Arc<crate::events::EventBus>,
        ) -> Result<()> {
            let reply_id = format!("{utterance_id}-reply");
            let mut opened = false;
            let mut bytes = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(next) = bytes.next().await {
                let chunk = next.map_err(|e| Error::Network(format!("backend stream: {e}")))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(newline) = buffer.find('\n') {
                    let line: String = buffer.drain(..=newline).collect();
                    let line = line.trim();
                    let Some(value) = parse_line(line) else {
                        continue;
                    };
                    match value.get("type").and_then(Value::as_str).unwrap_or("delta") {
                        "delta" | "text" | "token" => {
                            let Some(text) = value
                                .get("text")
                                .or_else(|| value.get("delta"))
                                .and_then(Value::as_str)
                            else {
                                continue;
                            };
                            if !opened {
                                opened = true;
                                let _ = commands
                                    .send(Command::SpeakStreamOpen(Box::new(SpeakRequest {
                                        utterance_id: reply_id.clone(),
                                        text: String::new(),
                                        voice: Default::default(),
                                        interrupt: true,
                                        echo_text: false,
                                        persist: true,
                                    })))
                                    .await;
                            }
                            let _ = commands
                                .send(Command::SpeakStreamDelta {
                                    utterance_id: reply_id.clone(),
                                    text: text.to_string(),
                                })
                                .await;
                        }
                        "interrupt" => {
                            let _ = commands
                                .send(Command::Interrupt {
                                    utterance_id: None,
                                    source: "backend".into(),
                                })
                                .await;
                        }
                        "listen" => {
                            let _ = commands
                                .send(Command::StartListening {
                                    continuous: value
                                        .get("continuous")
                                        .and_then(Value::as_bool)
                                        .unwrap_or(false),
                                    source: "backend".into(),
                                })
                                .await;
                        }
                        "error" => {
                            let message = value
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("the backend reported an error");
                            if opened {
                                let _ = commands
                                    .send(Command::SpeakStreamEnd {
                                        utterance_id: reply_id.clone(),
                                    })
                                    .await;
                            }
                            return Err(Error::Other(message.to_string()));
                        }
                        // Progress on backend work: recorded so the orb stays
                        // up while something is happening, so "what are you
                        // doing?" has an answer, and so a suspended heavy task
                        // is visibly suspended rather than apparently stuck.
                        "task" | "task.update" | "progress" => {
                            let Some(task_id) = value
                                .get("id")
                                .or_else(|| value.get("task_id"))
                                .and_then(Value::as_str)
                            else {
                                continue;
                            };
                            let state = match value
                                .get("state")
                                .and_then(Value::as_str)
                                .unwrap_or("running")
                            {
                                "queued" => crate::tasks::TaskState::Queued,
                                "suspended" | "paused" | "yielded" => {
                                    crate::tasks::TaskState::Suspended
                                }
                                "completed" | "done" | "finished" => {
                                    crate::tasks::TaskState::Completed
                                }
                                "failed" | "error" => crate::tasks::TaskState::Failed,
                                "cancelled" | "canceled" => crate::tasks::TaskState::Cancelled,
                                _ => crate::tasks::TaskState::Running,
                            };
                            let weight =
                                value
                                    .get("weight")
                                    .and_then(Value::as_str)
                                    .map(|w| match w {
                                        "heavy" => crate::tasks::Weight::Heavy,
                                        "light" => crate::tasks::Weight::Light,
                                        _ => crate::tasks::Weight::Normal,
                                    });
                            let task = tasks.upsert(
                                task_id,
                                value
                                    .get("title")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                                state,
                                weight,
                                value
                                    .get("detail")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                            );
                            bus.emit(crate::events::VoiceEvent::Task {
                                id: task.id,
                                state: task.state.as_str().to_string(),
                                title: task.title,
                                detail: task.detail,
                                weight: format!("{:?}", task.weight).to_lowercase(),
                            });
                        }
                        "end" | "done" => break,
                        // RESERVED: `ui` and anything else is where future
                        // Cookie visual instructions will arrive. Ignoring
                        // them is what keeps this client forward-compatible.
                        other => {
                            tracing::debug!(kind = other, "ignoring unsupported backend message");
                        }
                    }
                }
            }

            if opened {
                let _ = commands
                    .send(Command::SpeakStreamEnd {
                        utterance_id: reply_id,
                    })
                    .await;
            }
            Ok(())
        }
    }

    /// Parse one NDJSON or SSE line. `None` for blanks, comments and
    /// SSE bookkeeping fields.
    pub(super) fn parse_line(line: &str) -> Option<Value> {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            return None;
        }
        let payload = if let Some(rest) = line.strip_prefix("data:") {
            rest.trim()
        } else if line.contains(':') && !line.starts_with('{') {
            return None; // `event:`/`id:` lines
        } else {
            line
        };
        if payload == "[DONE]" {
            return Some(serde_json::json!({"type": "end"}));
        }
        serde_json::from_str(payload).ok()
    }

    /// Pull the reply text out of a non-streaming answer.
    pub(super) fn reply_text(value: &Value) -> Option<String> {
        for key in ["reply", "text", "message", "content", "output"] {
            if let Some(s) = value.get(key).and_then(Value::as_str) {
                return Some(s.to_string());
            }
        }
        value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }
}

#[cfg(not(feature = "http-providers"))]
mod stub {
    use super::*;
    use tokio::sync::mpsc;

    /// Placeholder used when the crate is built without HTTP support.
    #[derive(Debug)]
    pub struct BackendClient;

    impl BackendClient {
        /// Always `None`: without the `http-providers` feature there is no
        /// way to reach a backend, and pretending otherwise would hide the
        /// reason speech never gets answered.
        pub fn from_config(config: &BackendConfig) -> Option<Self> {
            if config.enabled {
                tracing::warn!(
                    "backend.enabled is set but this build lacks the `http-providers` feature"
                );
            }
            None
        }

        pub async fn turn(
            &self,
            _utterance_id: &str,
            _text: &str,
            _commands: mpsc::Sender<Command>,
        ) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "http-providers"))]
mod tests {
    use super::imp::{parse_line, reply_text};
    use super::*;
    use serde_json::json;

    #[test]
    fn disabled_backend_builds_nothing() {
        assert!(BackendClient::from_config(&BackendConfig::default()).is_none());
    }

    #[test]
    fn enabled_backend_builds() {
        let config = BackendConfig {
            enabled: true,
            ..Default::default()
        };
        let client = BackendClient::from_config(&config).expect("client");
        assert!(!client.session_id().is_empty());
    }

    #[test]
    fn ndjson_and_sse_lines_both_parse() {
        assert_eq!(
            parse_line(r#"{"type":"delta","text":"hi"}"#).unwrap()["text"],
            "hi"
        );
        assert_eq!(
            parse_line(r#"data: {"type":"delta","text":"hi"}"#).unwrap()["text"],
            "hi"
        );
        assert_eq!(parse_line("data: [DONE]").unwrap()["type"], "end");
        assert!(parse_line("event: message").is_none());
        assert!(parse_line("").is_none());
        assert!(parse_line(": keep-alive").is_none());
    }

    #[test]
    fn reply_text_handles_the_usual_shapes() {
        assert_eq!(reply_text(&json!({"reply": "a"})).as_deref(), Some("a"));
        assert_eq!(reply_text(&json!({"text": "b"})).as_deref(), Some("b"));
        assert_eq!(
            reply_text(&json!({"choices":[{"message":{"content":"c"}}]})).as_deref(),
            Some("c")
        );
        assert!(reply_text(&json!({"nothing": 1})).is_none());
    }

    #[test]
    fn turn_request_serialises_with_the_documented_keys() {
        let body = TurnRequest {
            protocol: BACKEND_PROTOCOL,
            session_id: "s".into(),
            utterance_id: "u".into(),
            text: "hello".into(),
            r#final: true,
            interface: InterfaceDescriptor::default(),
            scheduling: TurnOptions {
                priority: crate::tasks::Priority::Interactive,
                preempt: true,
            },
            active_tasks: vec!["task-1".into()],
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["protocol"], "cookie-interface/1");
        assert_eq!(json["final"], true);
        assert_eq!(json["interface"]["visual"], "orb");
        // The scheduling hint is part of the contract, not an optimisation:
        // a backend on one-model-at-a-time hardware needs it.
        assert_eq!(json["scheduling"]["priority"], "interactive");
        assert_eq!(json["scheduling"]["preempt"], true);
        assert_eq!(json["active_tasks"][0], "task-1");
    }
}
