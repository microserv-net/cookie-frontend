//! The local HTTP API.
//!
//! This is how anything else — the Cookie backend, a script, a browser tab,
//! `curl` — gives Cookie a voice and hears what was said to her.
//!
//! ## Streaming runs both ways
//!
//! * **Out:** `GET /v1/events` (SSE) and `GET /v1/stream` (WebSocket) carry
//!   partial transcripts as they are recognised, final transcripts, state
//!   changes and speech progress. `Last-Event-ID` resumes an SSE stream.
//! * **In:** `POST /v1/speak/stream` accepts a *streaming request body* of
//!   newline-delimited JSON, so a language model can pipe tokens straight in
//!   and Cookie starts speaking at the first sentence boundary instead of
//!   waiting for the last token. The same thing is possible over the
//!   WebSocket, which additionally lets one connection both send and receive.
//! * **Audio in:** `POST /v1/audio` takes raw 16-bit PCM for callers that
//!   already have microphone audio of their own.
//!
//! ## Versioning and forward compatibility
//!
//! Everything lives under `/v1`. Within `v1` the rules are:
//!
//! * fields may be **added** to any request or response,
//! * new event `type`s may appear at any time,
//! * nothing is removed or repurposed.
//!
//! Clients must therefore ignore unknown fields and unknown event types.
//!
//! **Reserved for the future:** as Cookie gains visual capabilities beyond
//! the orb, this protocol is where they will be exposed — a `visual` object
//! on `/v1/speak`, `ui.*` event types on the event stream, and a `capabilities`
//! block on `/v1/state` describing what the running interface can render.
//! None of that exists yet; the extension points are called out here and in
//! `docs/api.md` so that clients written today keep working when it does.

mod ws;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;

use crate::config::{Config, VoicePatch, VoiceSpec};
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::events::{Command, EventEnvelope, SpeakRequest};
use crate::retention::RetentionPolicy;

/// Shared state behind every handler.
#[derive(Clone)]
pub struct ApiState {
    pub engine: Engine,
    pub config: Arc<Config>,
    pub started: Instant,
}

impl std::fmt::Debug for ApiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiState").finish()
    }
}

/// Build the router. Exposed so tests can drive it without a socket.
pub fn router(state: ApiState) -> Router {
    let limit = state.config.api.max_body_bytes;
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/state", get(state_handler))
        .route("/v1/config", get(config_handler))
        .route("/v1/speak", post(speak))
        .route("/v1/speak/stream", post(speak_stream))
        .route("/v1/interrupt", post(interrupt))
        .route("/v1/listen", post(listen))
        .route("/v1/listen/stop", post(listen_stop))
        .route("/v1/audio", post(push_audio))
        .route("/v1/events", get(events))
        .route("/v1/transcripts", get(transcripts))
        .route("/v1/stream", get(ws::handler))
        .route("/v1/diagnostics", get(diagnostics))
        .route("/v1/tasks", get(tasks_handler))
        .route("/v1/cancel", post(cancel))
        .route("/v1/retention", get(retention_get).post(retention_set))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

/// Bind and serve until `shutdown` resolves.
pub async fn serve(
    state: ApiState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let api = &state.config.api;
    if !api.bind.is_loopback() && !api.allow_remote {
        return Err(Error::Config(format!(
            "api.bind is {} but api.allow_remote is false. \
             Exposing a live microphone to the network must be deliberate: \
             set allow_remote = true and an auth_token.",
            api.bind
        )));
    }
    if !api.bind.is_loopback() && api.auth_token.is_none() {
        return Err(Error::Config(
            "a non-loopback bind requires api.auth_token to be set".into(),
        ));
    }
    let addr = SocketAddr::new(api.bind, api.port);
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            Error::PortInUse { port: api.port }
        } else {
            Error::io(std::path::Path::new(&addr.to_string()), e)
        }
    })?;
    tracing::info!(%addr, "http api listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| Error::Other(format!("http server stopped: {e}")))
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// JSON error body. Every failing request produces exactly this shape.
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.code.as_str() {
            "bad_request" => StatusCode::BAD_REQUEST,
            "unauthorized" => StatusCode::UNAUTHORIZED,
            "config_error" => StatusCode::INTERNAL_SERVER_ERROR,
            "model_unavailable" | "provider_not_compiled" => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self {
            error: e.to_string(),
            code: e.code().to_string(),
            hint: e.hint().map(str::to_owned),
        }
    }
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            error: message.into(),
            code: "bad_request".into(),
            hint: None,
        }
    }

    fn unauthorized() -> Self {
        Self {
            error: "missing or invalid bearer token".into(),
            code: "unauthorized".into(),
            hint: Some("send `Authorization: Bearer <api.auth_token>`".into()),
        }
    }
}

/// Check the bearer token when one is configured.
fn authorize(state: &ApiState, headers: &HeaderMap) -> std::result::Result<(), ApiError> {
    let Some(expected) = state.config.api.auth_token.as_deref() else {
        return Ok(());
    };
    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    match provided {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(ApiError::unauthorized()),
    }
}

/// Comparison that does not leak the token through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async fn health(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "version": crate::VERSION,
        "api_version": crate::API_VERSION,
        "uptime_seconds": state.started.elapsed().as_secs(),
    }))
}

async fn state_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let features = state.engine.bus().current_features();
    let info = state.engine.info();
    Ok(Json(json!({
        "state": state.engine.state(),
        "since_seq": state.engine.bus().last_seq(),
        "providers": {
            "stt": { "name": info.stt_provider, "capabilities": info.stt },
            "tts": { "name": info.tts_provider, "capabilities": info.tts },
        },
        "audio": {
            "input_device": info.input_device,
            "output_device": info.output_device,
            "hardware": info.hardware_audio,
            "level_db": features.level_db,
        },
        "backend": info.backend_url,
        "voice": state.config.tts.voice,
        // Reserved: a `capabilities` block describing renderable visual
        // features will be added here. Clients must ignore unknown keys.
    })))
}

async fn config_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    // Secrets live in the environment, never in the config; even so, the
    // token is redacted here so a screenshot of this endpoint is harmless.
    let mut value = serde_json::to_value(&*state.config).unwrap_or(json!({}));
    if let Some(api) = value.get_mut("api").and_then(|v| v.as_object_mut()) {
        if api.get("auth_token").is_some_and(|v| !v.is_null()) {
            api.insert("auth_token".into(), json!("<redacted>"));
        }
    }
    Ok(Json(value))
}

// ---------------------------------------------------------------------------
// Speech out
// ---------------------------------------------------------------------------

/// Body of `POST /v1/speak`.
///
/// Only parameters the active provider honours have any effect; check
/// `providers.tts.capabilities` on `/v1/state` before sending `pitch` or
/// `style`. Unknown fields are rejected so a typo is not silently ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakBody {
    pub text: String,
    /// Sparse voice overrides layered on top of the configured voice.
    #[serde(default)]
    pub voice: Option<VoicePatch>,
    /// Stop whatever is currently being said first. Defaults to true.
    #[serde(default = "default_true")]
    pub interrupt: bool,
    /// Include chunk text in `speak.chunk` events.
    #[serde(default)]
    pub echo_text: bool,
    /// Keep the rendered audio on disk, subject to the retention policy.
    #[serde(default = "default_true")]
    pub persist: bool,
    /// Caller-supplied id; one is generated when absent.
    #[serde(default)]
    pub utterance_id: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Maximum characters accepted in one utterance.
const MAX_SPEAK_CHARS: usize = 8_000;

fn build_request(
    state: &ApiState,
    text: String,
    voice: Option<VoicePatch>,
    interrupt: bool,
    echo_text: bool,
    persist: bool,
    utterance_id: Option<String>,
) -> std::result::Result<SpeakRequest, ApiError> {
    if text.chars().count() > MAX_SPEAK_CHARS {
        return Err(ApiError::bad_request(format!(
            "text is longer than {MAX_SPEAK_CHARS} characters"
        )));
    }
    let voice: VoiceSpec = match voice {
        Some(patch) => state.config.tts.voice.overlaid(&patch),
        None => state.config.tts.voice.clone(),
    };
    voice.validate().map_err(ApiError::bad_request)?;
    Ok(SpeakRequest {
        utterance_id: utterance_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        text,
        voice,
        interrupt,
        echo_text,
        persist,
    })
}

async fn speak(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<SpeakBody>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    if body.text.trim().is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    let request = build_request(
        &state,
        body.text,
        body.voice,
        body.interrupt,
        body.echo_text,
        body.persist,
        body.utterance_id,
    )?;
    let id = request.utterance_id.clone();
    state
        .engine
        .send(Command::Speak(Box::new(request)))
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({
        "utterance_id": id,
        "accepted": true,
        // Follow progress on /v1/events, filtering on this id.
        "events": format!("/{}/events", crate::API_VERSION),
    })))
}

/// One line of a streamed speak request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeakStreamLine {
    /// Text to append. Spoken once a sentence boundary is reached.
    #[serde(default)]
    text: Option<String>,
    /// Set on the last line (or simply close the body).
    #[serde(default)]
    end: bool,
    /// Only honoured on the first line.
    #[serde(default)]
    voice: Option<VoicePatch>,
    #[serde(default)]
    utterance_id: Option<String>,
    #[serde(default)]
    echo_text: Option<bool>,
    #[serde(default)]
    interrupt: Option<bool>,
    #[serde(default)]
    persist: Option<bool>,
}

/// `POST /v1/speak/stream` — newline-delimited JSON in, newline-delimited
/// JSON out.
///
/// ```text
/// {"text":"Good evening. ","voice":{"rate":0.95}}
/// {"text":"The kettle has boiled."}
/// {"end":true}
/// ```
///
/// The request body is consumed as it arrives, which is the whole point: a
/// language model can pipe tokens in and Cookie speaks the first sentence
/// while the rest is still being generated.
async fn speak_stream(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> std::result::Result<Response, ApiError> {
    authorize(&state, &headers)?;

    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, std::io::Error>>(16);
    let engine = state.engine.clone();
    let config = state.config.clone();
    let limit = state.config.api.max_body_bytes;

    tokio::spawn(async move {
        let mut stream = body.into_data_stream();
        let mut buffer = String::new();
        let mut opened: Option<String> = None;
        let mut total = 0usize;

        macro_rules! reply {
            ($value:expr) => {{
                let line = format!("{}\n", $value);
                if tx.send(Ok(line)).await.is_err() {
                    return;
                }
            }};
        }

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    reply!(json!({"type": "error", "message": e.to_string()}));
                    break;
                }
            };
            total += chunk.len();
            if total > limit {
                reply!(json!({"type": "error", "message": "request body too large"}));
                break;
            }
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline) = buffer.find('\n') {
                let line: String = buffer.drain(..=newline).collect();
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let parsed: SpeakStreamLine = match serde_json::from_str(&line) {
                    Ok(p) => p,
                    Err(e) => {
                        reply!(json!({"type": "error", "message": format!("bad line: {e}")}));
                        continue;
                    }
                };
                let id = match &opened {
                    Some(id) => id.clone(),
                    None => {
                        let voice = match parsed.voice.clone() {
                            Some(patch) => config.tts.voice.overlaid(&patch),
                            None => config.tts.voice.clone(),
                        };
                        if let Err(e) = voice.validate() {
                            reply!(json!({"type": "error", "message": e.to_string()}));
                            return;
                        }
                        let id = parsed
                            .utterance_id
                            .clone()
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                        let request = SpeakRequest {
                            utterance_id: id.clone(),
                            text: String::new(),
                            voice,
                            interrupt: parsed.interrupt.unwrap_or(true),
                            echo_text: parsed.echo_text.unwrap_or(false),
                            persist: parsed.persist.unwrap_or(true),
                        };
                        if engine
                            .send(Command::SpeakStreamOpen(Box::new(request)))
                            .await
                            .is_err()
                        {
                            reply!(json!({"type": "error", "message": "engine unavailable"}));
                            return;
                        }
                        opened = Some(id.clone());
                        reply!(json!({"type": "started", "utterance_id": id}));
                        id
                    }
                };
                if let Some(text) = parsed.text.filter(|t| !t.is_empty()) {
                    let accepted = text.chars().count();
                    if engine
                        .send(Command::SpeakStreamDelta {
                            utterance_id: id.clone(),
                            text,
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                    reply!(json!({"type": "accepted", "characters": accepted}));
                }
                if parsed.end {
                    let _ = engine
                        .send(Command::SpeakStreamEnd {
                            utterance_id: id.clone(),
                        })
                        .await;
                    reply!(json!({"type": "end", "utterance_id": id}));
                    return;
                }
            }
        }

        // Body finished without an explicit `end`: close the utterance so the
        // tail is still spoken. A dropped connection must not leave Cookie
        // holding half a sentence.
        if let Some(id) = opened {
            let _ = engine
                .send(Command::SpeakStreamEnd {
                    utterance_id: id.clone(),
                })
                .await;
            reply!(json!({"type": "end", "utterance_id": id}));
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(axum::body::Body::from_stream(stream))
        .expect("valid response"))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InterruptBody {
    #[serde(default)]
    utterance_id: Option<String>,
}

async fn interrupt(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Option<Json<InterruptBody>>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let utterance_id = body.and_then(|Json(b)| b.utterance_id);
    state
        .engine
        .send(Command::Interrupt {
            utterance_id,
            source: "api".into(),
        })
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"accepted": true})))
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListenBody {
    /// Keep listening after the first utterance. Defaults to true.
    #[serde(default = "default_true")]
    continuous: bool,
}

async fn listen(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Option<Json<ListenBody>>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let continuous = body.map(|Json(b)| b.continuous).unwrap_or(true);
    state
        .engine
        .send(Command::StartListening {
            continuous,
            source: "api".into(),
        })
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"listening": true, "continuous": continuous})))
}

async fn listen_stop(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    state
        .engine
        .send(Command::StopListening {
            source: "api".into(),
        })
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"listening": false})))
}

#[derive(Debug, Deserialize)]
struct AudioQuery {
    /// Sample rate of the submitted PCM. Defaults to the pipeline rate.
    #[serde(default)]
    sample_rate: Option<u32>,
}

/// `POST /v1/audio` — raw 16-bit little-endian mono PCM for callers that have
/// their own microphone. The audio bypasses the VAD: the caller decides where
/// the utterance begins and ends.
async fn push_audio(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<AudioQuery>,
    body: axum::body::Bytes,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    if body.len() % 2 != 0 {
        return Err(ApiError::bad_request(
            "body length must be even (16-bit PCM)",
        ));
    }
    if body.is_empty() {
        return Err(ApiError::bad_request("no audio submitted"));
    }
    let rate = query.sample_rate.unwrap_or(state.config.audio.sample_rate);
    let audio = crate::audio::wav::decode_pcm_s16le(&body, rate, 1);
    let audio = if rate == state.config.audio.sample_rate {
        audio
    } else {
        audio.resampled(state.config.audio.sample_rate)
    };
    let ms = audio.duration_ms();
    state
        .engine
        .send(Command::PushAudio {
            samples: audio.samples,
        })
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"accepted": true, "duration_ms": ms})))
}

// ---------------------------------------------------------------------------
// Event streams
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct EventQuery {
    /// Comma-separated event types, e.g. `transcript.final,state`.
    #[serde(default)]
    types: Option<String>,
}

fn event_stream(
    state: &ApiState,
    filter: Option<Vec<String>>,
    after: Option<u64>,
) -> impl Stream<Item = std::result::Result<Event, std::convert::Infallible>> {
    let rx = state.engine.bus().subscribe();
    BroadcastStream::new(rx).filter_map(move |item| {
        let filter = filter.clone();
        async move {
            let envelope: EventEnvelope = item.ok()?;
            if let Some(after) = after {
                if envelope.seq <= after {
                    return None;
                }
            }
            if let Some(types) = &filter {
                if !types.iter().any(|t| t == envelope.event.kind()) {
                    return None;
                }
            }
            let data = serde_json::to_string(&envelope).ok()?;
            Some(Ok(Event::default()
                .id(envelope.seq.to_string())
                .event(envelope.event.kind())
                .data(data)))
        }
    })
}

/// `GET /v1/events` — server-sent events for everything.
///
/// Reconnecting clients send `Last-Event-ID`; events already delivered are
/// skipped. Anything still in the broadcast buffer is replayed, which covers
/// a brief drop but not a long absence — this is a live stream, not a log.
async fn events(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<EventQuery>,
) -> std::result::Result<
    Sse<impl Stream<Item = std::result::Result<Event, std::convert::Infallible>>>,
    ApiError,
> {
    authorize(&state, &headers)?;
    if state.engine.bus().subscriber_count() >= state.config.api.max_subscribers {
        return Err(ApiError {
            error: "too many event subscribers".into(),
            code: "bad_request".into(),
            hint: Some("raise api.max_subscribers".into()),
        });
    }
    let after = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let filter = query
        .types
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
    Ok(Sse::new(event_stream(&state, filter, after)).keep_alive(KeepAlive::default()))
}

/// `GET /v1/transcripts` — the same stream narrowed to what the user said.
///
/// Convenience for the common backend case: one connection, only speech.
async fn transcripts(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<
    Sse<impl Stream<Item = std::result::Result<Event, std::convert::Infallible>>>,
    ApiError,
> {
    authorize(&state, &headers)?;
    let filter = Some(vec![
        "transcript.partial".to_string(),
        "transcript.final".to_string(),
        "speech.detected".to_string(),
        "speech.ended".to_string(),
    ]);
    let after = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    Ok(Sse::new(event_stream(&state, filter, after)).keep_alive(KeepAlive::default()))
}

// ---------------------------------------------------------------------------
// Diagnostics and tasks
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct DiagnosticsQuery {
    /// Also say the answer out loud. Off by default: a monitoring script
    /// polling this endpoint should not make the assistant talk to an empty
    /// room.
    #[serde(default)]
    speak: bool,
    /// Enumerate real audio devices. Costs a few hundred milliseconds.
    #[serde(default = "default_true")]
    probe_devices: bool,
}

/// `GET /v1/diagnostics` — the same checks behind `--doctor` and behind
/// "Cookie, are you alright?".
async fn diagnostics(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<DiagnosticsQuery>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let report = state.engine.diagnose(query.probe_devices).await;
    if query.speak {
        let _ = state
            .engine
            .send(Command::RunDiagnostics { speak: true })
            .await;
    }
    Ok(Json(json!({
        "overall": report.overall,
        "summary": report.spoken_summary(),
        "checks": report.checks,
        "elapsed_ms": report.elapsed_ms,
    })))
}

/// `GET /v1/tasks` — what the backend is working on, as the interface sees it.
async fn tasks_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let tasks = state.engine.tasks();
    Ok(Json(json!({
        "active": tasks.active(),
        "summary": tasks.spoken_summary(),
        "heavy_in_flight": tasks.heavy_in_flight(),
    })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelBody {
    /// Omit to cancel everything in flight.
    #[serde(default)]
    task_id: Option<String>,
}

/// `POST /v1/cancel` — abandon backend work.
///
/// Deliberately separate from `/v1/interrupt`: interrupting speech must never
/// throw away work, and throwing away work must always be asked for.
async fn cancel(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Option<Json<CancelBody>>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let task_id = body.and_then(|Json(b)| b.task_id);
    let cancelled = match &task_id {
        Some(id) => vec![id.clone()],
        None => state.engine.tasks().active_ids(),
    };
    state
        .engine
        .send(Command::CancelTask {
            task_id,
            source: "api".into(),
        })
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"cancelled": cancelled})))
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

async fn retention_get(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let retention = state.engine.retention();
    Ok(Json(json!({
        "policy": retention.policy(),
        "options": RetentionPolicy::PRESETS,
        "directory": retention.directory(),
        "files": retention.record_count(),
    })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetentionBody {
    /// One of `immediate`, `1h`, `24h`, `7d`, `30d`, `forever`.
    policy: String,
    /// Run a sweep immediately with the new policy.
    #[serde(default = "default_true")]
    sweep: bool,
}

/// Changing the policy takes effect immediately *and* retroactively: files
/// already on disk are judged against the new policy, so shortening it
/// deletes old audio rather than only affecting future recordings.
async fn retention_set(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<RetentionBody>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    authorize(&state, &headers)?;
    let policy: RetentionPolicy = body
        .policy
        .parse()
        .map_err(|e: String| ApiError::bad_request(e))?;
    let retention = state.engine.retention().clone();
    retention.set_policy(policy);
    let report = if body.sweep {
        let retention = retention.clone();
        tokio::task::spawn_blocking(move || retention.sweep())
            .await
            .map_err(|e| ApiError::from(Error::Other(e.to_string())))?
            .map_err(ApiError::from)?
    } else {
        Default::default()
    };
    Ok(Json(json!({
        "policy": policy,
        "swept": { "deleted": report.deleted, "freed_bytes": report.freed_bytes },
        // The caller changed the running policy; persisting it to the config
        // file is the CLI's job (`--config retention.policy=7d`).
        "persisted": false,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SttProviderKind, TtsProviderKind};
    use crate::engine::Devices;
    use crate::paths::Paths;
    use crate::retention::RetentionManager;
    use axum_test::TestServer;

    async fn server(mut config: Config) -> (TestServer, tempfile::TempDir) {
        config.stt.provider = SttProviderKind::Mock;
        config.tts.provider = TtsProviderKind::Mock;
        config.audio.listen_on_start = false;
        let dir = tempfile::tempdir().unwrap();
        let paths = Arc::new(Paths::rooted(dir.path()));
        paths.ensure().unwrap();
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
        let state = ApiState {
            engine,
            config,
            started: Instant::now(),
        };
        (TestServer::new(router(state)).unwrap(), dir)
    }

    #[tokio::test]
    async fn health_reports_the_version() {
        let (server, _dir) = server(Config::default()).await;
        let response = server.get("/v1/health").await;
        response.assert_status_ok();
        assert_eq!(
            response.json::<serde_json::Value>()["version"],
            crate::VERSION
        );
    }

    #[tokio::test]
    async fn speak_accepts_text_and_returns_an_id() {
        let (server, _dir) = server(Config::default()).await;
        let response = server
            .post("/v1/speak")
            .json(&json!({"text": "Good evening."}))
            .await;
        response.assert_status_ok();
        let body: serde_json::Value = response.json();
        assert!(body["utterance_id"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[tokio::test]
    async fn empty_text_is_rejected() {
        let (server, _dir) = server(Config::default()).await;
        let response = server.post("/v1/speak").json(&json!({"text": "  "})).await;
        response.assert_status(StatusCode::BAD_REQUEST);
        assert_eq!(response.json::<serde_json::Value>()["code"], "bad_request");
    }

    #[tokio::test]
    async fn unknown_fields_are_rejected_rather_than_ignored() {
        let (server, _dir) = server(Config::default()).await;
        let response = server
            .post("/v1/speak")
            .json(&json!({"text": "hi", "speeed": 2}))
            .await;
        assert!(!response.status_code().is_success());
    }

    #[tokio::test]
    async fn auth_token_is_enforced() {
        let mut config = Config::default();
        config.api.auth_token = Some("s3cret".into());
        let (server, _dir) = server(config).await;

        server
            .get("/v1/state")
            .await
            .assert_status(StatusCode::UNAUTHORIZED);
        server
            .get("/v1/state")
            .add_header("authorization", "Bearer s3cret")
            .await
            .assert_status_ok();
        // Health stays open so a supervisor can probe it without a secret.
        server.get("/v1/health").await.assert_status_ok();
    }

    #[tokio::test]
    async fn state_reports_provider_capabilities() {
        let (server, _dir) = server(Config::default()).await;
        let body: serde_json::Value = server.get("/v1/state").await.json();
        assert!(body["providers"]["tts"]["capabilities"]["streaming"].is_boolean());
        assert_eq!(body["state"], "idle");
    }

    #[tokio::test]
    async fn config_endpoint_redacts_the_token() {
        let mut config = Config::default();
        config.api.auth_token = Some("s3cret".into());
        let (server, _dir) = server(config).await;
        let body: serde_json::Value = server
            .get("/v1/config")
            .add_header("authorization", "Bearer s3cret")
            .await
            .json();
        assert_eq!(body["api"]["auth_token"], "<redacted>");
    }

    #[tokio::test]
    async fn streaming_speak_accepts_ndjson() {
        let (server, _dir) = server(Config::default()).await;
        let body = "{\"text\":\"Good evening. \"}\n{\"text\":\"All is well.\"}\n{\"end\":true}\n";
        let response = server
            .post("/v1/speak/stream")
            .text(body)
            .content_type("application/x-ndjson")
            .await;
        response.assert_status_ok();
        let text = response.text();
        assert!(text.contains("\"started\""), "{text}");
        assert!(text.contains("\"end\""), "{text}");
    }

    #[tokio::test]
    async fn pcm_upload_is_length_checked() {
        let (server, _dir) = server(Config::default()).await;
        let response = server
            .post("/v1/audio")
            .bytes(vec![0u8, 1, 2].into())
            .content_type("application/octet-stream")
            .await;
        response.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn retention_policy_can_be_read_and_changed() {
        let (server, _dir) = server(Config::default()).await;
        server.get("/v1/retention").await.assert_status_ok();
        let response = server
            .post("/v1/retention")
            .json(&json!({"policy": "1h"}))
            .await;
        response.assert_status_ok();
        assert_eq!(response.json::<serde_json::Value>()["policy"], "1h");

        server
            .post("/v1/retention")
            .json(&json!({"policy": "yesterday"}))
            .await
            .assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn diagnostics_answers_in_plain_english() {
        let (server, _dir) = server(Config::default()).await;
        let body: serde_json::Value = server
            .get("/v1/diagnostics?probe_devices=false")
            .await
            .json();
        assert!(body["checks"].as_array().is_some_and(|c| c.len() >= 8));
        let summary = body["summary"].as_str().unwrap();
        assert!(!summary.contains("audio.input"), "{summary}");
    }

    #[tokio::test]
    async fn tasks_and_cancel_are_separate_from_speech() {
        let (server, _dir) = server(Config::default()).await;
        let body: serde_json::Value = server.get("/v1/tasks").await.json();
        assert_eq!(body["active"].as_array().unwrap().len(), 0);
        assert_eq!(body["heavy_in_flight"], false);
        // Cancelling with nothing running is a no-op, not an error.
        server
            .post("/v1/cancel")
            .json(&json!({}))
            .await
            .assert_status_ok();
    }

    #[tokio::test]
    async fn interrupt_and_listen_are_accepted() {
        let (server, _dir) = server(Config::default()).await;
        server
            .post("/v1/interrupt")
            .json(&json!({}))
            .await
            .assert_status_ok();
        server
            .post("/v1/listen")
            .json(&json!({}))
            .await
            .assert_status_ok();
        server.post("/v1/listen/stop").await.assert_status_ok();
    }

    #[test]
    fn constant_time_eq_behaves_like_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
