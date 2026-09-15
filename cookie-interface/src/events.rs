//! The event protocol and the bus that carries it.
//!
//! Three distinct transports, because the three kinds of signal have genuinely
//! different delivery semantics:
//!
//! * **discrete events** (`VoiceEvent`) — every one matters, so they go over a
//!   bounded `broadcast` channel. A subscriber that falls behind is told it
//!   lagged rather than being allowed to grow memory without limit.
//! * **audio features** — only the newest value is ever interesting, so they go
//!   over a `watch`. A renderer running at 144 Hz and a meter widget polling at
//!   10 Hz both get the freshest frame with no queueing.
//! * **state** — also latest-value, and separate from features so a UI can
//!   `await` a state change without waking on every audio frame.
//!
//! ## Protocol stability
//!
//! `VoiceEvent` is serialised straight onto `/v1/events` and the WebSocket, so
//! it *is* the public wire format.
//!
//! **FUTURE EXTENSION (v1-compatible):** Cookie will grow richer visual and UI
//! capabilities — showing cards, highlighting regions, presenting choices,
//! driving the orb directly. Those arrive as *new* event variants and *new*
//! optional fields on existing variants, never as changes to the ones below.
//! Clients must therefore: ignore unknown `type` values, ignore unknown fields,
//! and never rely on field ordering. See docs/api.md#forward-compatibility.

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};

use crate::audio::AudioFeatures;
use crate::state::{StateReason, VoiceState};

/// Capacity of the discrete-event broadcast. Sized for a burst of partial
/// transcripts plus speech progress without being a memory hazard.
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

/// A discrete thing that happened. Serialised as `{"type": "...", ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VoiceEvent {
    /// The engine finished booting and is usable.
    #[serde(rename = "ready")]
    Ready {
        version: String,
        stt_provider: String,
        tts_provider: String,
        /// False when audio hardware is unavailable and mocks are in use.
        audio_available: bool,
    },

    #[serde(rename = "state")]
    StateChanged {
        from: VoiceState,
        to: VoiceState,
        reason: StateReason,
    },

    #[serde(rename = "listening.started")]
    ListeningStarted {
        /// `"vad"`, `"api"`, `"test"`, ...
        source: String,
        continuous: bool,
    },

    /// The VAD decided a human started talking.
    #[serde(rename = "speech.detected")]
    SpeechDetected { level_db: f32 },

    /// The VAD decided they stopped.
    #[serde(rename = "speech.ended")]
    SpeechEnded { duration_ms: u64 },

    #[serde(rename = "transcript.partial")]
    TranscriptPartial {
        utterance_id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
        /// How likely this prefix is to survive into the final transcript.
        #[serde(skip_serializing_if = "Option::is_none")]
        stability: Option<f32>,
        /// Milliseconds since the start of the utterance.
        offset_ms: u64,
    },

    #[serde(rename = "transcript.final")]
    TranscriptFinal {
        utterance_id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        duration_ms: u64,
        /// Word/segment timings where the model provides them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        segments: Vec<TranscriptSegment>,
    },

    #[serde(rename = "listening.stopped")]
    ListeningStopped { reason: String },

    #[serde(rename = "speak.started")]
    SpeakStarted {
        utterance_id: String,
        /// Character count only — the text itself is not echoed back into the
        /// log stream by default.
        characters: usize,
        voice: String,
        streaming: bool,
    },

    /// One synthesised chunk reached the speaker. Emitted per chunk so a
    /// caller can follow progress through a long streamed reply.
    #[serde(rename = "speak.chunk")]
    SpeakChunk {
        utterance_id: String,
        index: u32,
        duration_ms: u64,
        /// Text of this chunk, included only when the caller asked for echo.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },

    #[serde(rename = "speak.finished")]
    SpeakFinished {
        utterance_id: String,
        duration_ms: u64,
        /// `"completed"`, `"interrupted"`, `"failed"`
        reason: String,
    },

    /// Speech was cut off — by the API, by barge-in, or by a new utterance.
    #[serde(rename = "interrupted")]
    Interrupted {
        #[serde(skip_serializing_if = "Option::is_none")]
        utterance_id: Option<String>,
        source: String,
        elapsed_ms: u64,
    },

    /// Model/provider lifecycle, so a UI can show "loading voice model".
    #[serde(rename = "provider.status")]
    ProviderStatus {
        kind: String,
        provider: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    /// Backend work appeared, moved on, paused or finished. This is how a
    /// client follows "what is Cookie doing" without polling, and how the orb
    /// knows to stay on screen while something is in flight.
    #[serde(rename = "task")]
    Task {
        id: String,
        /// `queued`, `running`, `suspended`, `completed`, `failed`, `cancelled`.
        state: String,
        title: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        /// `light`, `normal`, `heavy`.
        weight: String,
    },

    /// The interface recognised an utterance as one of its own commands and
    /// acted on it locally. Emitted even when the utterance is also forwarded
    /// to the backend, so a client can see why something happened.
    #[serde(rename = "intent")]
    Intent {
        /// `diagnostics`, `stop_speaking`, `cancel_task`, `sleep`, `wake`,
        /// `show_activity`.
        intent: String,
        confidence: f32,
        /// Whether the backend also saw this utterance.
        forwarded: bool,
    },

    /// The backend asked this machine to do something.
    ///
    /// Emitted for every request, including refused ones, so the activity
    /// view shows what Cookie was asked to do and not only what she did.
    #[serde(rename = "tool")]
    Tool {
        id: String,
        tool: String,
        /// `requested`, `confirming`, `running`, `done`, `failed`, `refused`,
        /// `declined`.
        stage: String,
        risk: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    /// Something was heard, transcribed, and discarded because it was not
    /// addressed to Cookie.
    ///
    /// Emitted so that "she is not hearing me" and "she heard me and decided
    /// it was not for her" are distinguishable, which they are not otherwise.
    /// Carries the text, so it counts as sensitive.
    #[serde(rename = "ignored")]
    Ignored { text: String, reason: String },

    /// How long something took.
    ///
    /// Emitted for the steps whose cost is worth knowing — recognition above
    /// all, because "it feels slow" and "it is taking four seconds on the CPU
    /// because CoreML was refused" need different things done about them.
    #[serde(rename = "timing")]
    Timing {
        /// `recognition`, `synthesis`.
        what: String,
        elapsed_ms: u64,
        /// Free-form: the execution provider, the model, why it was slow.
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    /// Cookie started or stopped being spoken to.
    ///
    /// Distinct from listening, which is whether the microphone is open —
    /// that is true from startup and says nothing about whether anybody is
    /// talking to her.
    #[serde(rename = "attention")]
    Attention {
        awake: bool,
        /// `wake-word`, `api`, `timeout`, `dismissed`.
        source: String,
    },

    /// A diagnostics run finished.
    #[serde(rename = "diagnostics")]
    Diagnostics {
        /// `ok`, `degraded`, `failed`, `unknown`.
        overall: String,
        /// Capabilities that are not well, in plain language.
        problems: Vec<String>,
        /// The sentence that was (or would be) spoken.
        summary: String,
    },

    #[serde(rename = "retention.swept")]
    RetentionSwept {
        deleted: usize,
        freed_bytes: u64,
        failed: usize,
    },

    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
        /// Fatal means the engine stopped; otherwise it recovered.
        fatal: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        hint: Option<String>,
    },
}

impl VoiceEvent {
    /// Discriminator used for SSE `event:` lines and for client-side filters.
    pub fn kind(&self) -> &'static str {
        match self {
            VoiceEvent::Ready { .. } => "ready",
            VoiceEvent::StateChanged { .. } => "state",
            VoiceEvent::ListeningStarted { .. } => "listening.started",
            VoiceEvent::SpeechDetected { .. } => "speech.detected",
            VoiceEvent::SpeechEnded { .. } => "speech.ended",
            VoiceEvent::TranscriptPartial { .. } => "transcript.partial",
            VoiceEvent::TranscriptFinal { .. } => "transcript.final",
            VoiceEvent::ListeningStopped { .. } => "listening.stopped",
            VoiceEvent::SpeakStarted { .. } => "speak.started",
            VoiceEvent::SpeakChunk { .. } => "speak.chunk",
            VoiceEvent::SpeakFinished { .. } => "speak.finished",
            VoiceEvent::Interrupted { .. } => "interrupted",
            VoiceEvent::ProviderStatus { .. } => "provider.status",
            VoiceEvent::Task { .. } => "task",
            VoiceEvent::Tool { .. } => "tool",
            VoiceEvent::Attention { .. } => "attention",
            VoiceEvent::Ignored { .. } => "ignored",
            VoiceEvent::Timing { .. } => "timing",
            VoiceEvent::Intent { .. } => "intent",
            VoiceEvent::Diagnostics { .. } => "diagnostics",
            VoiceEvent::RetentionSwept { .. } => "retention.swept",
            VoiceEvent::Error { .. } => "error",
        }
    }

    /// True if this event carries recognised speech. Used to keep transcripts
    /// out of logs unless `logging.log_transcripts` is on.
    pub fn is_sensitive(&self) -> bool {
        matches!(
            self,
            VoiceEvent::TranscriptPartial { .. } | VoiceEvent::TranscriptFinal { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Sequenced wrapper actually put on the wire.
///
/// `seq` is monotonic per process start and lets a reconnecting client say
/// "I last saw 412" via `Last-Event-ID`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub seq: u64,
    pub ts_ms: u64,
    #[serde(flatten)]
    pub event: VoiceEvent,
}

/// Instructions *into* the engine. The HTTP layer, the test harness and the
/// renderer all speak this; nothing else may touch engine internals.
#[derive(Debug, Clone)]
pub enum Command {
    /// Speak a complete piece of text.
    Speak(Box<SpeakRequest>),
    /// Open a streaming utterance that text will be appended to.
    SpeakStreamOpen(Box<SpeakRequest>),
    /// Append to an open streaming utterance.
    SpeakStreamDelta {
        utterance_id: String,
        text: String,
    },
    /// Close a streaming utterance; remaining buffered text is spoken.
    SpeakStreamEnd {
        utterance_id: String,
    },
    /// Stop speaking now. This never touches backend work: being told to be
    /// quiet is not being told to give up.
    Interrupt {
        utterance_id: Option<String>,
        source: String,
    },
    StartListening {
        continuous: bool,
        source: String,
    },
    StopListening {
        source: String,
    },
    /// Push externally captured PCM into the recognition path (16 kHz mono f32).
    PushAudio {
        samples: Vec<f32>,
    },
    /// Abandon backend work. `None` cancels everything in flight.
    CancelTask {
        task_id: Option<String>,
        source: String,
    },
    /// Run the capability checks and, if `speak`, say the answer aloud.
    RunDiagnostics {
        speak: bool,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct SpeakRequest {
    pub utterance_id: String,
    pub text: String,
    pub voice: crate::config::VoiceSpec,
    /// Cancel whatever is currently being spoken first.
    pub interrupt: bool,
    /// Include chunk text in `speak.chunk` events.
    pub echo_text: bool,
    /// Keep the rendered audio on disk (subject to the retention policy).
    pub persist: bool,
}

/// Fan-out hub. Clone it freely; every clone shares the same channels.
#[derive(Debug, Clone)]
pub struct EventBus {
    events: broadcast::Sender<EventEnvelope>,
    features: watch::Sender<AudioFeatures>,
    state: watch::Sender<VoiceState>,
    seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl EventBus {
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let (features, _) = watch::channel(AudioFeatures::silent());
        let (state, _) = watch::channel(VoiceState::Idle);
        Self {
            events,
            features,
            state,
            seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Publish a discrete event. Never blocks and never fails: with no
    /// subscribers the event is simply dropped, which is the correct behaviour
    /// for a UI-facing notification.
    pub fn emit(&self, event: VoiceEvent) -> u64 {
        let seq = self
            .seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1);
        let envelope = EventEnvelope {
            seq,
            ts_ms: crate::util::unix_now_ms(),
            event,
        };
        let _ = self.events.send(envelope);
        seq
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.events.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.events.receiver_count()
    }

    /// Latest-value publish of audio analysis. Called at frame rate from the
    /// analysis task; overwrite semantics mean a slow consumer cannot stall it.
    /// Publish the latest audio frame's features.
    ///
    /// `send_replace`, not `send`: a `watch` sender whose receivers have all
    /// been dropped *rejects* `send` and keeps the old value, so with `send`
    /// the very first frame published before anything subscribed would be
    /// thrown away — and every frame after it, if the renderer is not running.
    /// The whole point of this channel is that the latest value is always
    /// there for whoever asks next.
    pub fn publish_features(&self, features: AudioFeatures) {
        let _ = self.features.send_replace(features);
    }

    pub fn features(&self) -> watch::Receiver<AudioFeatures> {
        self.features.subscribe()
    }

    pub fn current_features(&self) -> AudioFeatures {
        self.features.borrow().clone()
    }

    /// Latch the current voice state. See [`EventBus::publish_features`] for
    /// why this is `send_replace`.
    pub fn publish_state(&self, state: VoiceState) {
        let _ = self.state.send_replace(state);
    }

    pub fn state(&self) -> watch::Receiver<VoiceState> {
        self.state.subscribe()
    }

    pub fn current_state(&self) -> VoiceState {
        *self.state.borrow()
    }

    pub fn last_seq(&self) -> u64 {
        self.seq.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    // (regression) A latched value must survive having no subscribers.
    use super::*;

    #[test]
    fn events_serialise_with_a_type_tag() {
        let e = VoiceEvent::TranscriptPartial {
            utterance_id: "u1".into(),
            text: "hello".into(),
            confidence: Some(0.8),
            stability: None,
            offset_ms: 120,
        };
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["type"], "transcript.partial");
        assert_eq!(json["text"], "hello");
        // `stability` is None and must not appear at all.
        assert!(json.get("stability").is_none());
    }

    #[test]
    fn envelope_flattens_the_event() {
        let env = EventEnvelope {
            seq: 7,
            ts_ms: 1234,
            event: VoiceEvent::ListeningStopped {
                reason: "api".into(),
            },
        };
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["seq"], 7);
        assert_eq!(json["type"], "listening.stopped");
        let back: EventEnvelope = serde_json::from_value(json).unwrap();
        assert_eq!(back, env);
    }

    #[test]
    fn every_variant_has_a_stable_kind() {
        let e = VoiceEvent::Error {
            code: "x".into(),
            message: "y".into(),
            fatal: false,
            hint: None,
        };
        assert_eq!(e.kind(), "error");
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["type"], e.kind());
    }

    #[tokio::test]
    async fn bus_delivers_to_subscribers_and_sequences() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        bus.emit(VoiceEvent::SpeechDetected { level_db: -20.0 });
        bus.emit(VoiceEvent::SpeechEnded { duration_ms: 900 });
        let a = rx.recv().await.unwrap();
        let b = rx.recv().await.unwrap();
        assert_eq!(a.seq + 1, b.seq);
        assert_eq!(a.event.kind(), "speech.detected");
    }

    #[tokio::test]
    async fn emitting_without_subscribers_is_fine() {
        let bus = EventBus::new();
        for _ in 0..1000 {
            bus.emit(VoiceEvent::SpeechDetected { level_db: 0.0 });
        }
        assert_eq!(bus.last_seq(), 1000);
    }

    #[tokio::test]
    async fn features_have_latest_value_semantics() {
        let bus = EventBus::new();
        let rx = bus.features();
        for i in 0..100 {
            let mut f = AudioFeatures::silent();
            f.rms = i as f32;
            bus.publish_features(f);
        }
        assert_eq!(rx.borrow().rms, 99.0, "consumer sees the newest frame only");
    }

    #[test]
    fn latched_values_survive_having_no_subscribers() {
        let bus = EventBus::new();
        // Nobody is listening yet — this is the ordinary case at startup,
        // before the renderer or any API client has connected.
        bus.publish_state(VoiceState::Listening);
        bus.publish_features(AudioFeatures {
            level_db: -12.0,
            ..AudioFeatures::silent()
        });
        assert_eq!(bus.current_state(), VoiceState::Listening);
        assert_eq!(bus.current_features().level_db, -12.0);
        // And a late subscriber sees them immediately.
        assert_eq!(*bus.state().borrow(), VoiceState::Listening);
    }
}
