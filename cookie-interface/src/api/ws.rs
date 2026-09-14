//! `GET /v1/stream` — one WebSocket that carries both directions.
//!
//! Events flow out continuously; commands flow in as JSON objects with a
//! `type` field. It is the same vocabulary as the REST endpoints, so a client
//! that already speaks the API only has to learn the envelope:
//!
//! ```json
//! {"type":"speak","text":"Good evening."}
//! {"type":"speak.open","utterance_id":"u1"}
//! {"type":"speak.delta","utterance_id":"u1","text":"one sentence at a time. "}
//! {"type":"speak.end","utterance_id":"u1"}
//! {"type":"interrupt"}
//! {"type":"listen","continuous":true}
//! {"type":"audio","sample_rate":16000,"pcm":"<base64 s16le>"}
//! ```
//!
//! Binary frames are accepted as raw 16-bit PCM at the pipeline sample rate,
//! which is the cheapest way to push a live microphone in from another
//! process.
//!
//! Unknown message types are answered with an error object rather than a
//! disconnect: this is the surface future Cookie visual commands will extend.

use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};

use crate::events::{Command, SpeakRequest};

use super::{authorize, ApiError, ApiState};

/// Upgrade handler.
pub(super) async fn handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> std::result::Result<Response, ApiError> {
    authorize(&state, &headers)?;
    if state.engine.bus().subscriber_count() >= state.config.api.max_subscribers {
        return Err(ApiError {
            error: "too many subscribers".into(),
            code: "bad_request".into(),
            hint: Some("raise api.max_subscribers".into()),
        });
    }
    Ok(upgrade.on_upgrade(move |socket| run(socket, state)))
}

async fn run(socket: WebSocket, state: ApiState) {
    let (mut sink, mut stream) = socket.split();
    let mut events = state.engine.bus().subscribe();

    // Outbound: every event, as JSON text frames.
    let outbound = tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(envelope) => {
                    let Ok(text) = serde_json::to_string(&envelope) else {
                        continue;
                    };
                    if sink
                        .send(Message::Text(Utf8Bytes::from(text)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    let notice = json!({
                        "type": "warning",
                        "message": format!("dropped {n} events; this client is too slow")
                    });
                    let _ = sink
                        .send(Message::Text(Utf8Bytes::from(notice.to_string())))
                        .await;
                }
                Err(_) => return,
            }
        }
    });

    // Inbound: commands.
    while let Some(Ok(message)) = stream.next().await {
        match message {
            Message::Text(text) => {
                if let Some(command) = parse(&state, &text) {
                    if state.engine.send(command).await.is_err() {
                        break;
                    }
                }
            }
            Message::Binary(bytes) => {
                if bytes.len() % 2 != 0 {
                    continue;
                }
                let audio =
                    crate::audio::wav::decode_pcm_s16le(&bytes, state.config.audio.sample_rate, 1);
                if state
                    .engine
                    .send(Command::PushAudio {
                        samples: audio.samples,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    outbound.abort();
}

/// Translate one inbound frame into an engine command.
pub(super) fn parse(state: &ApiState, text: &str) -> Option<Command> {
    let value: Value = serde_json::from_str(text).ok()?;
    let kind = value.get("type").and_then(Value::as_str)?;
    let voice = match value.get("voice") {
        Some(patch) => {
            let patch = serde_json::from_value(patch.clone()).ok()?;
            state.config.tts.voice.overlaid(&patch)
        }
        None => state.config.tts.voice.clone(),
    };
    let id = || {
        value
            .get("utterance_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
    };
    match kind {
        "speak" => {
            let text = value.get("text").and_then(Value::as_str)?.to_string();
            Some(Command::Speak(Box::new(SpeakRequest {
                utterance_id: id(),
                text,
                voice,
                interrupt: value
                    .get("interrupt")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                echo_text: value
                    .get("echo_text")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                persist: value
                    .get("persist")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            })))
        }
        "speak.open" => Some(Command::SpeakStreamOpen(Box::new(SpeakRequest {
            utterance_id: id(),
            text: value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            voice,
            interrupt: value
                .get("interrupt")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            echo_text: value
                .get("echo_text")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            persist: value
                .get("persist")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        }))),
        "speak.delta" => Some(Command::SpeakStreamDelta {
            utterance_id: id(),
            text: value.get("text").and_then(Value::as_str)?.to_string(),
        }),
        "speak.end" => Some(Command::SpeakStreamEnd { utterance_id: id() }),
        "interrupt" => Some(Command::Interrupt {
            utterance_id: value
                .get("utterance_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            source: "websocket".into(),
        }),
        "listen" => Some(Command::StartListening {
            continuous: value
                .get("continuous")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            source: "websocket".into(),
        }),
        "listen.stop" => Some(Command::StopListening {
            source: "websocket".into(),
        }),
        // Cancelling work and interrupting speech are different frames on
        // purpose; see the note on `Command::Interrupt`.
        "cancel" => Some(Command::CancelTask {
            task_id: value
                .get("task_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            source: "websocket".into(),
        }),
        "diagnostics" => Some(Command::RunDiagnostics {
            speak: value.get("speak").and_then(Value::as_bool).unwrap_or(false),
        }),
        "audio" => {
            let encoded = value.get("pcm").and_then(Value::as_str)?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .ok()?;
            let rate = value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(state.config.audio.sample_rate as u64) as u32;
            let audio = crate::audio::wav::decode_pcm_s16le(&bytes, rate, 1);
            let audio = if rate == state.config.audio.sample_rate {
                audio
            } else {
                audio.resampled(state.config.audio.sample_rate)
            };
            Some(Command::PushAudio {
                samples: audio.samples,
            })
        }
        other => {
            // RESERVED: future `ui.*` commands land here.
            tracing::debug!(kind = other, "ignoring unsupported websocket message");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::{Devices, Engine};
    use crate::paths::Paths;
    use crate::retention::RetentionManager;
    use std::sync::Arc;
    use std::time::Instant;

    async fn state() -> (ApiState, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Arc::new(Paths::rooted(dir.path()));
        paths.ensure().unwrap();
        let mut config = Config::default();
        config.stt.provider = crate::config::SttProviderKind::Mock;
        config.tts.provider = crate::config::TtsProviderKind::Mock;
        config.audio.listen_on_start = false;
        let config = Arc::new(config);
        let retention = Arc::new(RetentionManager::open(&paths, config.retention.clone()).unwrap());
        let engine = Engine::start(
            config.clone(),
            paths,
            Devices::mock(config.audio.sample_rate),
            retention,
        )
        .await
        .unwrap();
        (
            ApiState {
                engine,
                config,
                started: Instant::now(),
            },
            dir,
        )
    }

    #[tokio::test]
    async fn speak_frames_become_speak_commands() {
        let (state, _dir) = state().await;
        let command = parse(&state, r#"{"type":"speak","text":"hello"}"#).unwrap();
        match command {
            Command::Speak(request) => assert_eq!(request.text, "hello"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_frames_map_to_the_streaming_commands() {
        let (state, _dir) = state().await;
        assert!(matches!(
            parse(&state, r#"{"type":"speak.open","utterance_id":"u1"}"#),
            Some(Command::SpeakStreamOpen(_))
        ));
        assert!(matches!(
            parse(
                &state,
                r#"{"type":"speak.delta","utterance_id":"u1","text":"x"}"#
            ),
            Some(Command::SpeakStreamDelta { .. })
        ));
        assert!(matches!(
            parse(&state, r#"{"type":"speak.end","utterance_id":"u1"}"#),
            Some(Command::SpeakStreamEnd { .. })
        ));
    }

    #[tokio::test]
    async fn base64_audio_is_decoded_and_resampled() {
        let (state, _dir) = state().await;
        let pcm: Vec<u8> = (0..640).map(|i| (i % 256) as u8).collect();
        let frame = json!({
            "type": "audio",
            "sample_rate": 8000,
            "pcm": base64::engine::general_purpose::STANDARD.encode(&pcm),
        })
        .to_string();
        match parse(&state, &frame) {
            Some(Command::PushAudio { samples }) => {
                // 320 source frames at 8 kHz become ~640 at 16 kHz.
                assert!(samples.len() > 500, "{}", samples.len());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_and_malformed_frames_are_ignored() {
        let (state, _dir) = state().await;
        assert!(parse(&state, r#"{"type":"ui.flash"}"#).is_none());
        assert!(parse(&state, "not json").is_none());
        assert!(parse(&state, r#"{"text":"no type"}"#).is_none());
    }

    #[tokio::test]
    async fn voice_overrides_are_layered_onto_the_configured_voice() {
        let (state, _dir) = state().await;
        let frame = r#"{"type":"speak","text":"hi","voice":{"rate":1.25}}"#;
        match parse(&state, frame) {
            Some(Command::Speak(request)) => assert_eq!(request.voice.rate, 1.25),
            other => panic!("unexpected {other:?}"),
        }
    }
}
