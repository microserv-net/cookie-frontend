//! The orchestrator.
//!
//! One task owns the state machine, the capture handle, the VAD and the
//! speaker. Everything slow — recognition, synthesis, disk, the backend —
//! happens in spawned tasks that report back over a bounded channel. Nothing
//! in here ever blocks, and nothing in here ever touches the renderer: the
//! only thing the orb sees is the event bus.
//!
//! ```text
//!  mic ─▶ CaptureHandle ─▶ analyser ─▶ VAD ─┐
//!                             │             │ utterance audio
//!                    AudioFeatures          ▼
//!                        (watch)        recogniser task
//!                             │             │ transcript
//!   commands ────────────────▶│◀────────────┘
//!   (API, backend, --test)    │
//!                         state machine
//!                             │ speak
//!                             ▼
//!                        speech task ─▶ speaker + AudioFeatures
//! ```
//!
//! ## Why one loop
//!
//! The alternative — a task per subsystem sharing a `Mutex<AppState>` — is
//! how audio applications end up with priority inversion and mysterious
//! stalls. Here the only mutable state lives in the loop, so a state
//! transition is an ordinary function call and there is nothing to contend
//! over. The one lock in the file guards the speaker, is held for
//! microseconds, and is never held across an `.await`.

mod speech;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

use crate::audio::{
    AudioAnalyzer, AudioBuffer, AudioInput, AudioOutput, CaptureHandle, EnergyVad, FeatureSource,
    MockInput, MockOutput, PlaybackHandle, Vad, VadEvent,
};
use crate::config::Config;
use crate::diagnostics::LiveStatus;
use crate::error::{Error, Result};
use crate::events::{Command, EventBus, SpeakRequest, VoiceEvent};
use crate::intent::{Intent, IntentEngine};
use crate::paths::Paths;
use crate::retention::RetentionManager;
use crate::state::{StateMachine, StateReason, Trigger};
use crate::stt::{SharedRecognizer, TranscribeOptions, Transcript};
use crate::tasks::TaskRegistry;
use crate::tts::SharedSynthesizer;
use crate::util::text::SentenceChunker;
use crate::VoiceState;

/// Audio devices the engine should use.
///
/// Separating this from [`Config`] is what makes the engine testable: the
/// test-suite passes mock devices and never opens a real one.
pub struct Devices {
    pub input: Box<dyn AudioInput>,
    pub output: Box<dyn AudioOutput>,
    /// False when we fell back to mocks because the hardware was unavailable.
    pub hardware: bool,
}

impl std::fmt::Debug for Devices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Devices")
            .field("input", &self.input.info().name)
            .field("output", &self.output.info().name)
            .field("hardware", &self.hardware)
            .finish()
    }
}

impl Devices {
    /// In-process devices. Used by tests, by `--no-audio`, and as the
    /// fallback when a machine has no working sound card.
    pub fn mock(sample_rate: u32) -> Self {
        Self {
            input: Box::new(MockInput::silence(sample_rate)),
            output: Box::new(MockOutput::new(sample_rate)),
            hardware: false,
        }
    }

    /// Real devices when the `audio-io` feature is on, mocks otherwise.
    ///
    /// A missing microphone is *not* a fatal error: the orb, the API and
    /// speech output all still work, and the user is told what happened.
    pub fn from_config(config: &Config) -> Self {
        #[cfg(feature = "audio-io")]
        {
            use crate::audio::{CpalInput, CpalOutput};
            let input = CpalInput::new(config.audio.input_device.clone(), config.audio.input_gain);
            let output =
                CpalOutput::new(config.audio.output_device.clone(), config.audio.output_gain);
            Self {
                input: Box::new(input),
                output: Box::new(output),
                hardware: true,
            }
        }
        #[cfg(not(feature = "audio-io"))]
        {
            Self::mock(config.audio.sample_rate)
        }
    }
}

/// Everything a client needs to know about how the engine was wired up.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineInfo {
    pub stt_provider: String,
    pub tts_provider: String,
    pub stt: crate::stt::SttCapabilities,
    pub tts: crate::tts::TtsCapabilities,
    pub input_device: String,
    pub output_device: String,
    pub hardware_audio: bool,
    pub backend_url: Option<String>,
}

/// Handle to a running engine.
///
/// Cheap to clone; every clone talks to the same loop.
#[derive(Clone)]
pub struct Engine {
    bus: Arc<EventBus>,
    commands: mpsc::Sender<Command>,
    config: Arc<Config>,
    info: Arc<EngineInfo>,
    retention: Arc<RetentionManager>,
    tasks: Arc<TaskRegistry>,
    /// Rolling health picture, updated as providers report in. Diagnostics
    /// reads it so "are you alright?" answers from what actually happened
    /// rather than from a fresh round of probing.
    status: Arc<Mutex<LiveStatus>>,
    paths: Arc<Paths>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("state", &self.bus.current_state())
            .field("stt", &self.info.stt_provider)
            .field("tts", &self.info.tts_provider)
            .finish()
    }
}

impl Engine {
    /// Boot the engine. Returns once the loop is running; models are prepared
    /// in the background so the orb appears immediately.
    pub async fn start(
        config: Arc<Config>,
        paths: Arc<Paths>,
        devices: Devices,
        retention: Arc<RetentionManager>,
    ) -> Result<Self> {
        let bus = Arc::new(EventBus::new());
        let paths_for_engine = paths.clone();
        let recognizer = crate::stt::build(&config.stt)?;
        let (synthesizer, tts_error) = crate::tts::build_with_fallback(&config.tts);

        let info = Arc::new(EngineInfo {
            stt_provider: recognizer.name(),
            tts_provider: synthesizer.name(),
            stt: recognizer.capabilities(),
            tts: synthesizer.capabilities(),
            input_device: devices.input.info().name.clone(),
            output_device: devices.output.info().name.clone(),
            hardware_audio: devices.hardware,
            backend_url: config.backend.enabled.then(|| config.backend.chat_url()),
        });

        let tasks = Arc::new(TaskRegistry::new());
        let status = Arc::new(Mutex::new(LiveStatus {
            stt_provider: Some(info.stt_provider.clone()),
            tts_provider: Some(info.tts_provider.clone()),
            input_device: Some(info.input_device.clone()),
            output_device: Some(info.output_device.clone()),
            hardware_audio: info.hardware_audio,
            ..Default::default()
        }));

        let (tx, rx) = mpsc::channel(64);
        let worker = Worker::new(
            config.clone(),
            paths,
            bus.clone(),
            devices,
            recognizer.clone(),
            synthesizer.clone(),
            retention.clone(),
            info.clone(),
            rx,
            tx.clone(),
            tasks.clone(),
            status.clone(),
        )?;
        tokio::spawn(worker.run());

        // Model preparation is deliberately off the boot path.
        {
            let bus = bus.clone();
            let status = status.clone();
            tokio::spawn(async move {
                if let Some(e) = tts_error {
                    bus.emit(VoiceEvent::ProviderStatus {
                        kind: "tts".into(),
                        provider: synthesizer.name(),
                        status: "degraded".into(),
                        detail: Some(e.to_string()),
                    });
                }
                for (kind, result, name) in [
                    ("stt", recognizer.prepare().await, recognizer.name()),
                    ("tts", synthesizer.prepare().await, synthesizer.name()),
                ] {
                    {
                        let mut guard = status.lock().expect("status");
                        let ready = result.is_ok();
                        if kind == "stt" {
                            guard.stt_ready = Some(ready);
                        } else {
                            guard.tts_ready = Some(ready);
                        }
                    }
                    match result {
                        Ok(()) => bus.emit(VoiceEvent::ProviderStatus {
                            kind: kind.into(),
                            provider: name,
                            status: "ready".into(),
                            detail: None,
                        }),
                        Err(e) => bus.emit(VoiceEvent::ProviderStatus {
                            kind: kind.into(),
                            provider: name,
                            status: "unavailable".into(),
                            detail: Some(e.to_string()),
                        }),
                    };
                }
            });
        }

        Ok(Self {
            bus,
            commands: tx,
            config,
            info,
            retention,
            tasks,
            status,
            paths: paths_for_engine,
        })
    }

    /// Backend work currently in flight.
    pub fn tasks(&self) -> &Arc<TaskRegistry> {
        &self.tasks
    }

    /// Snapshot of what diagnostics knows about this session.
    pub fn status(&self) -> LiveStatus {
        self.status.lock().expect("status").clone()
    }

    /// Told by the renderer whether it got a graphics surface, so
    /// "are you alright?" can mention a missing orb.
    pub fn set_renderer_ok(&self, ok: bool) {
        self.status.lock().expect("status").renderer_ok = Some(ok);
    }

    /// Run the capability checks. Shared by the CLI, the API and the spoken
    /// intent so all three can never disagree.
    pub async fn diagnose(&self, probe_devices: bool) -> crate::diagnostics::DiagnosticReport {
        let mut status = self.status();
        status.active_tasks = self.tasks.active().len();
        crate::diagnostics::run(&self.config, &self.paths, &status, probe_devices).await
    }

    /// The event bus. The renderer and the API both subscribe here.
    pub fn bus(&self) -> &Arc<EventBus> {
        &self.bus
    }

    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    pub fn info(&self) -> &Arc<EngineInfo> {
        &self.info
    }

    pub fn retention(&self) -> &Arc<RetentionManager> {
        &self.retention
    }

    pub fn state(&self) -> VoiceState {
        self.bus.current_state()
    }

    /// Queue a command. Applies backpressure rather than growing memory.
    pub async fn send(&self, command: Command) -> Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| Error::Other("the voice engine has stopped".into()))
    }

    /// Non-blocking variant for callers that must not await (audio callbacks
    /// never call this; the renderer does).
    pub fn try_send(&self, command: Command) -> Result<()> {
        self.commands
            .try_send(command)
            .map_err(|e| Error::Other(format!("could not queue command: {e}")))
    }

    /// Ask the loop to stop and wait briefly for it to do so.
    pub async fn shutdown(&self) {
        let _ = self.commands.send(Command::Shutdown).await;
        let deadline = Instant::now() + Duration::from_millis(500);
        while !self.commands.is_closed() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// Messages the loop receives from its own spawned tasks.
#[derive(Debug)]
pub(crate) enum Internal {
    Transcript {
        utterance_id: String,
        interim: bool,
        result: Box<Result<Transcript>>,
        offset_ms: u64,
    },
    SpeechFinished {
        utterance_id: String,
        duration_ms: u64,
        reason: &'static str,
    },
    Sweep(crate::retention::SweepReport),
}

struct SpeakingJob {
    utterance_id: String,
    cancel: Arc<AtomicBool>,
    chunker: Option<SentenceChunker>,
    started: Instant,
    echo_text: bool,
    persist: bool,
    voice: crate::config::VoiceSpec,
    task: Option<JoinHandle<()>>,
}

struct Worker {
    config: Arc<Config>,
    paths: Arc<Paths>,
    bus: Arc<EventBus>,
    machine: StateMachine,
    capture: Option<CaptureHandle>,
    input: Box<dyn AudioInput>,
    playback: Arc<Mutex<PlaybackHandle>>,
    analyzer: AudioAnalyzer,
    vad: EnergyVad,
    recognizer: SharedRecognizer,
    synthesizer: SharedSynthesizer,
    retention: Arc<RetentionManager>,
    info: Arc<EngineInfo>,
    commands: mpsc::Receiver<Command>,
    command_tx: mpsc::Sender<Command>,
    internal_tx: mpsc::Sender<Internal>,
    internal_rx: mpsc::Receiver<Internal>,
    /// Audio of the utterance currently being spoken *to* us.
    utterance: Vec<f32>,
    utterance_id: String,
    last_partial: Instant,
    listening: bool,
    continuous: bool,
    speaking: Option<SpeakingJob>,
    frame: Vec<f32>,
    scratch: Vec<f32>,
    backend: Option<Arc<crate::backend::BackendClient>>,
    /// One recognition at a time.
    ///
    /// Whisper on this machine is a whole CPU for a second or two. Letting a
    /// second run start before the first finishes does not make anything
    /// faster — it makes both slower, starves the audio callback into
    /// underruns, and pushes each run past its own timeout.
    stt_permit: Arc<Semaphore>,
    tasks: Arc<TaskRegistry>,
    status: Arc<Mutex<LiveStatus>>,
    intents: IntentEngine,
    last_sweep: Instant,
    shutdown: bool,
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: Arc<Config>,
        paths: Arc<Paths>,
        bus: Arc<EventBus>,
        devices: Devices,
        recognizer: SharedRecognizer,
        synthesizer: SharedSynthesizer,
        retention: Arc<RetentionManager>,
        info: Arc<EngineInfo>,
        commands: mpsc::Receiver<Command>,
        command_tx: mpsc::Sender<Command>,
        tasks: Arc<TaskRegistry>,
        status: Arc<Mutex<LiveStatus>>,
    ) -> Result<Self> {
        let rate = config.audio.sample_rate;
        let playback = devices.output.open()?;
        let (internal_tx, internal_rx) = mpsc::channel(32);
        let backend = crate::backend::BackendClient::from_config(&config.backend).map(Arc::new);
        Ok(Self {
            machine: StateMachine::new(),
            capture: None,
            input: devices.input,
            playback: Arc::new(Mutex::new(playback)),
            analyzer: AudioAnalyzer::new(rate, FeatureSource::Input),
            vad: EnergyVad::new(config.vad.clone(), rate, config.audio.frame_ms),
            recognizer,
            synthesizer,
            retention,
            info,
            commands,
            command_tx,
            internal_tx,
            internal_rx,
            utterance: Vec::new(),
            utterance_id: String::new(),
            last_partial: Instant::now(),
            listening: false,
            continuous: config.audio.listen_on_start,
            speaking: None,
            frame: Vec::with_capacity(config.audio.frame_samples()),
            scratch: Vec::with_capacity(config.audio.frame_samples() * 4),
            backend,
            stt_permit: Arc::new(Semaphore::new(1)),
            tasks,
            status,
            intents: IntentEngine::default(),
            last_sweep: Instant::now(),
            config,
            paths,
            bus,
            shutdown: false,
        })
    }

    async fn run(mut self) {
        self.bus.publish_state(VoiceState::Idle);
        self.bus.emit(VoiceEvent::Ready {
            version: crate::VERSION.to_string(),
            stt_provider: self.info.stt_provider.clone(),
            tts_provider: self.info.tts_provider.clone(),
            audio_available: self.info.hardware_audio,
        });
        if self.continuous {
            self.start_listening("startup".into(), true);
        }

        let frame_ms = self.config.audio.frame_ms.max(1) as u64;
        let mut ticker = tokio::time::interval(Duration::from_millis(frame_ms));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        while !self.shutdown {
            tokio::select! {
                biased;
                Some(command) = self.commands.recv() => self.on_command(command).await,
                Some(message) = self.internal_rx.recv() => self.on_internal(message),
                _ = ticker.tick() => self.on_tick(),
            }
        }

        self.stop_capture("shutdown");
        if let Some(job) = self.speaking.take() {
            job.cancel.store(true, Ordering::Release);
        }
        self.commands.close();
        tracing::info!("voice engine stopped");
    }

    // ---------------------------------------------------------------- timing

    fn on_tick(&mut self) {
        self.pump_capture();
        if let Some(transition) = self.machine.tick() {
            self.bus.publish_state(transition.to);
            self.bus.emit(VoiceEvent::StateChanged {
                from: transition.from,
                to: transition.to,
                reason: transition.reason,
            });
        }
        // The mic analyser must keep decaying when nothing is captured, or the
        // orb would freeze mid-gesture whenever the device goes quiet.
        if self.capture.is_none() && self.speaking.is_none() {
            let features = self.analyzer.decay_only();
            self.bus.publish_features(features);
        }
        self.maybe_sweep();
    }

    fn maybe_sweep(&mut self) {
        let every = Duration::from_secs(self.config.retention.sweep_interval_minutes.max(1) * 60);
        if self.last_sweep.elapsed() < every {
            return;
        }
        self.last_sweep = Instant::now();
        let retention = self.retention.clone();
        let tx = self.internal_tx.clone();
        // Disk work never happens on the loop.
        tokio::task::spawn_blocking(move || match retention.sweep() {
            Ok(report) => {
                let _ = tx.blocking_send(Internal::Sweep(report));
            }
            Err(e) => tracing::warn!("retention sweep failed: {e}"),
        });
    }

    // --------------------------------------------------------------- capture

    fn start_listening(&mut self, source: String, continuous: bool) {
        self.continuous = continuous;
        if self.capture.is_none() {
            match self.input.start(self.config.audio.sample_rate) {
                Ok(handle) => {
                    tracing::info!(device = %handle.info().name, "microphone open");
                    self.capture = Some(handle);
                }
                Err(e) => {
                    self.report(&e, false);
                    self.bus.emit(VoiceEvent::ProviderStatus {
                        kind: "audio-in".into(),
                        provider: "capture".into(),
                        status: "unavailable".into(),
                        detail: Some(e.to_string()),
                    });
                    return;
                }
            }
        }
        self.listening = true;
        self.vad.reset();
        self.analyzer.reset();
        self.apply(Trigger::StartListening);
        self.bus
            .emit(VoiceEvent::ListeningStarted { source, continuous });
    }

    fn stop_capture(&mut self, reason: &str) {
        self.listening = false;
        self.continuous = false;
        if let Some(handle) = self.capture.take() {
            handle.stop();
        }
        self.utterance.clear();
        self.vad.reset();
        self.apply(Trigger::StopListening);
        self.bus.emit(VoiceEvent::ListeningStopped {
            reason: reason.to_string(),
        });
    }

    /// Drain the microphone, analyse it a frame at a time, and feed the VAD.
    fn pump_capture(&mut self) {
        let Some(capture) = self.capture.as_mut() else {
            return;
        };
        if let Some(error) = capture.take_error() {
            tracing::warn!("capture: {error}");
        }
        self.scratch.clear();
        capture.drain(&mut self.scratch);
        if self.scratch.is_empty() {
            return;
        }
        // While Cookie is talking the microphone is either ignored (ducking)
        // or used only to detect barge-in. Either way the samples are not
        // added to the utterance buffer: nobody wants the assistant
        // transcribing itself.
        let speaking = self.speaking.is_some();
        let ducked = speaking && self.config.audio.duck_input_while_speaking;

        let frame_len = self.config.audio.frame_samples();
        let mut taken = 0usize;
        while taken < self.scratch.len() {
            let end = (taken + frame_len - self.frame.len()).min(self.scratch.len());
            self.frame.extend_from_slice(&self.scratch[taken..end]);
            taken = end;
            if self.frame.len() < frame_len {
                break;
            }
            let frame = std::mem::take(&mut self.frame);
            self.on_frame(&frame, ducked, speaking);
            self.frame = frame;
            self.frame.clear();
        }
    }

    fn on_frame(&mut self, frame: &[f32], ducked: bool, speaking: bool) {
        let features = self.analyzer.process(frame);
        if !speaking {
            // During speech the output analyser owns the feature channel, so
            // the orb reacts to the voice rather than to room noise.
            self.bus.publish_features(features.clone());
        }
        if ducked {
            return;
        }
        let decision = self.vad.push(frame, &features);
        if self.vad.is_speaking() && !speaking {
            self.utterance.extend_from_slice(frame);
        }
        if let Some(event) = decision.event {
            match event {
                VadEvent::SpeechStart { level_db } => self.on_speech_start(level_db, speaking),
                VadEvent::SpeechEnd {
                    audio,
                    duration_ms,
                    truncated,
                } => {
                    let rate = self.config.audio.sample_rate;
                    self.on_speech_end(AudioBuffer::mono(audio, rate), duration_ms, truncated)
                }
                VadEvent::Discarded { duration_ms } => {
                    tracing::debug!("vad discarded a {duration_ms}ms utterance as too short");
                    self.utterance.clear();
                }
            }
        }
        self.maybe_partial();
    }

    fn on_speech_start(&mut self, level_db: f32, speaking: bool) {
        self.utterance.clear();
        self.utterance_id = uuid::Uuid::new_v4().to_string();
        self.last_partial = Instant::now();
        if speaking {
            // Barge-in: the human wins, always.
            self.interrupt(None, "barge-in");
            self.apply_with(Trigger::SpeechStarted, StateReason::BargeIn);
        } else {
            self.apply(Trigger::SpeechStarted);
        }
        self.bus.emit(VoiceEvent::SpeechDetected { level_db });
    }

    fn on_speech_end(&mut self, audio: AudioBuffer, duration_ms: u64, truncated: bool) {
        self.bus.emit(VoiceEvent::SpeechEnded { duration_ms });
        self.apply(Trigger::SpeechEnded);
        if truncated {
            tracing::debug!("utterance hit the maximum length and was cut");
        }
        self.utterance.clear();
        let id = std::mem::take(&mut self.utterance_id);
        let id = if id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            id
        };
        self.spawn_recognition(audio, id, false, duration_ms);
        if !self.continuous {
            self.listening = false;
        }
    }

    /// Ask for an interim transcript if enough new audio has arrived.
    fn maybe_partial(&mut self) {
        if !self.config.stt.partials || !self.vad.is_speaking() {
            return;
        }
        // A provider that has to re-run a local model to produce a partial is
        // not asked to. The interim text is a nicety; the final transcript is
        // the point, and racing them costs the thing that matters.
        if !self.recognizer.capabilities().cheap_partials {
            return;
        }
        // And even a cheap one waits its turn: if a recognition is already
        // running, this partial would only be queued behind it and arrive
        // after the final it was meant to precede.
        if self.stt_permit.available_permits() == 0 {
            return;
        }
        let interval = Duration::from_millis(self.config.stt.partial_interval_ms.max(150) as u64);
        if self.last_partial.elapsed() < interval || self.utterance.len() < 4_000 {
            return;
        }
        self.last_partial = Instant::now();
        let audio = AudioBuffer::mono(self.utterance.clone(), self.config.audio.sample_rate);
        let offset = audio.duration_ms();
        let id = self.utterance_id.clone();
        self.spawn_recognition(audio, id, true, offset);
    }

    fn spawn_recognition(
        &self,
        audio: AudioBuffer,
        utterance_id: String,
        interim: bool,
        offset_ms: u64,
    ) {
        let recognizer = self.recognizer.clone();
        let tx = self.internal_tx.clone();
        let permit = self.stt_permit.clone();
        let mut options = TranscribeOptions::from_config(&self.config.stt);
        if interim {
            options = options.interim();
        }
        tokio::spawn(async move {
            // Interim passes give up rather than queue; a final one waits,
            // because losing it would lose the utterance.
            let _guard = if interim {
                match permit.clone().try_acquire_owned() {
                    Ok(guard) => guard,
                    Err(_) => return,
                }
            } else {
                match permit.acquire_owned().await {
                    Ok(guard) => guard,
                    Err(_) => return,
                }
            };
            let result = recognizer.transcribe(audio, options).await;
            let _ = tx
                .send(Internal::Transcript {
                    utterance_id,
                    interim,
                    result: Box::new(result),
                    offset_ms,
                })
                .await;
        });
    }

    // -------------------------------------------------------------- commands

    async fn on_command(&mut self, command: Command) {
        match command {
            Command::Speak(request) => self.speak(*request, None).await,
            Command::SpeakStreamOpen(request) => {
                let chunker = SentenceChunker::new();
                self.speak(*request, Some(chunker)).await;
            }
            Command::SpeakStreamDelta { utterance_id, text } => {
                self.speak_delta(&utterance_id, &text).await
            }
            Command::SpeakStreamEnd { utterance_id } => self.speak_end(&utterance_id).await,
            Command::Interrupt {
                utterance_id,
                source,
            } => self.interrupt(utterance_id, &source),
            Command::StartListening { continuous, source } => {
                self.start_listening(source, continuous)
            }
            Command::StopListening { source } => {
                if self.capture.is_some() {
                    self.stop_capture(&source);
                }
            }
            Command::PushAudio { samples } => {
                // Audio injected over the API: transcribed directly, bypassing
                // the VAD, because the caller already decided where the
                // utterance starts and ends.
                let audio = AudioBuffer::mono(samples, self.config.audio.sample_rate);
                let ms = audio.duration_ms();
                self.apply(Trigger::SpeechEnded);
                self.spawn_recognition(audio, uuid::Uuid::new_v4().to_string(), false, ms);
            }
            Command::CancelTask { task_id, source } => self.cancel_task(task_id, &source),
            Command::RunDiagnostics { speak } => self.run_diagnostics(speak),
            Command::Shutdown => self.shutdown = true,
        }
    }

    fn on_internal(&mut self, message: Internal) {
        match message {
            Internal::Transcript {
                utterance_id,
                interim,
                result,
                offset_ms,
            } => self.on_transcript(utterance_id, interim, *result, offset_ms),
            Internal::SpeechFinished {
                utterance_id,
                duration_ms,
                reason,
            } => {
                let still_current = self
                    .speaking
                    .as_ref()
                    .is_some_and(|j| j.utterance_id == utterance_id);
                if still_current {
                    self.speaking = None;
                    self.apply(Trigger::PlaybackFinished);
                }
                self.bus.emit(VoiceEvent::SpeakFinished {
                    utterance_id,
                    duration_ms,
                    reason: reason.to_string(),
                });
            }
            Internal::Sweep(report) => {
                if report.deleted > 0 || report.failed > 0 {
                    self.bus.emit(VoiceEvent::RetentionSwept {
                        deleted: report.deleted,
                        freed_bytes: report.freed_bytes,
                        failed: report.failed,
                    });
                }
            }
        }
    }

    fn on_transcript(
        &mut self,
        utterance_id: String,
        interim: bool,
        result: Result<Transcript>,
        offset_ms: u64,
    ) {
        let transcript = match result {
            Ok(t) => t,
            Err(e) => {
                self.report(&e, false);
                if !interim {
                    self.apply(Trigger::RecognitionFinished);
                }
                return;
            }
        };
        if transcript.is_empty() {
            if !interim {
                self.apply(Trigger::RecognitionFinished);
            }
            return;
        }
        if interim {
            self.bus.emit(VoiceEvent::TranscriptPartial {
                utterance_id,
                text: transcript.text,
                confidence: transcript.confidence,
                stability: None,
                offset_ms,
            });
            return;
        }
        self.bus.emit(VoiceEvent::TranscriptFinal {
            utterance_id: utterance_id.clone(),
            text: transcript.text.clone(),
            confidence: transcript.confidence,
            language: transcript.language.clone(),
            duration_ms: transcript.audio_ms,
            segments: transcript.segments.clone(),
        });
        self.apply(Trigger::RecognitionFinished);
        self.route_transcript(utterance_id, transcript.text);
    }

    /// Decide what a finished transcript is *for*.
    ///
    /// Almost everything goes straight to the backend. The exceptions are the
    /// handful of things the interface owns — being told to be quiet, being
    /// asked whether it is alright, being told to cancel — and they are
    /// handled here so they still work when the backend is unreachable, which
    /// is exactly when they matter most. See [`crate::intent`].
    fn route_transcript(&mut self, utterance_id: String, text: String) {
        let backend_available = self.backend.is_some();
        let matched = self
            .intents
            .infer(&text, self.machine.current(), backend_available);

        if let Some(found) = matched {
            let forwarded = !found.intent.consumes_utterance() && backend_available;
            self.bus.emit(VoiceEvent::Intent {
                intent: found.intent.as_str().to_string(),
                confidence: found.score,
                forwarded,
            });
            tracing::info!(
                intent = found.intent.as_str(),
                score = found.score,
                "handled locally"
            );
            self.act_on_intent(found.intent);
            if !forwarded {
                return;
            }
        }
        self.forward_to_backend(utterance_id, text);
    }

    fn act_on_intent(&mut self, intent: Intent) {
        match intent {
            // Being told to stop talking stops the *talking*. Whatever the
            // backend is working on carries on untouched: the two are
            // different requests and conflating them would make Cookie
            // unusable, because you could never interrupt her mid-sentence
            // without losing the work you asked for.
            Intent::StopSpeaking => self.interrupt(None, "voice"),
            Intent::CancelTask => {
                self.interrupt(None, "voice");
                self.cancel_task(None, "voice");
            }
            Intent::Diagnostics => self.run_diagnostics(true),
            Intent::Sleep => {
                if self.capture.is_some() {
                    self.stop_capture("voice");
                }
            }
            Intent::Wake => self.start_listening("voice".into(), true),
            Intent::ShowActivity => {
                let summary = self.tasks.spoken_summary();
                self.say(summary);
            }
        }
    }

    /// Speak a sentence Cookie generated about herself.
    ///
    /// This is *not* the backend talking: it is the interface reporting on
    /// its own state, which is the only kind of speech this crate originates.
    fn say(&self, text: String) {
        let commands = self.command_tx.clone();
        let voice = self.config.tts.voice.clone();
        tokio::spawn(async move {
            let _ = commands
                .send(Command::Speak(Box::new(SpeakRequest {
                    utterance_id: format!("interface-{}", uuid::Uuid::new_v4()),
                    text,
                    voice,
                    interrupt: true,
                    echo_text: false,
                    persist: false,
                })))
                .await;
        });
    }

    fn run_diagnostics(&mut self, speak: bool) {
        let config = self.config.clone();
        let paths = self.paths.clone();
        let bus = self.bus.clone();
        let commands = self.command_tx.clone();
        let mut status = self.status.lock().expect("status").clone();
        status.active_tasks = self.tasks.active().len();
        let backend = self.backend.clone();
        let shared_status = self.status.clone();
        let voice = self.config.tts.voice.clone();

        tokio::spawn(async move {
            // Probe the backend as part of answering, rather than reporting a
            // stale verdict: "are you alright?" is usually asked *because*
            // something feels wrong.
            if let Some(backend) = backend {
                let reachable = backend.health().await;
                let mut guard = shared_status.lock().expect("status");
                guard.backend_ok = Some(reachable.is_ok());
                guard.backend_detail = reachable.err().map(|e| e.to_string());
                status.backend_ok = guard.backend_ok;
                status.backend_detail = guard.backend_detail.clone();
            }
            let report = crate::diagnostics::run(&config, &paths, &status, true).await;
            let summary = report.spoken_summary();
            bus.emit(VoiceEvent::Diagnostics {
                overall: format!("{:?}", report.overall).to_lowercase(),
                problems: report
                    .problems()
                    .iter()
                    .map(|c| c.capability.clone())
                    .collect(),
                summary: summary.clone(),
            });
            if speak {
                let _ = commands
                    .send(Command::Speak(Box::new(SpeakRequest {
                        utterance_id: format!("diagnostics-{}", uuid::Uuid::new_v4()),
                        text: summary,
                        voice,
                        interrupt: true,
                        echo_text: false,
                        persist: false,
                    })))
                    .await;
            }
        });
    }

    /// Abandon backend work. Only ever reached from an explicit request.
    fn cancel_task(&mut self, task_id: Option<String>, source: &str) {
        let targets = match &task_id {
            Some(id) => vec![id.clone()],
            None => self.tasks.active_ids(),
        };
        if targets.is_empty() {
            tracing::debug!("cancel requested but nothing is running");
            return;
        }
        for id in &targets {
            let task = self.tasks.upsert(
                id,
                None,
                crate::tasks::TaskState::Cancelled,
                None,
                Some(format!("cancelled by {source}")),
            );
            self.bus.emit(VoiceEvent::Task {
                id: task.id,
                state: task.state.as_str().to_string(),
                title: task.title,
                detail: task.detail,
                weight: format!("{:?}", task.weight).to_lowercase(),
            });
        }
        if let Some(backend) = self.backend.clone() {
            tokio::spawn(async move {
                if let Err(e) = backend.cancel(task_id.as_deref()).await {
                    tracing::warn!("backend cancellation failed: {e}");
                }
            });
        }
    }

    /// Hand a finished transcript to the Cookie backend, if one is configured.
    ///
    /// This is the only place the interface calls *out*. Everything it gets
    /// back arrives as ordinary commands, so a backend reply is
    /// indistinguishable from an API caller asking for speech.
    ///
    /// The turn carries a scheduling hint (see [`crate::tasks`]): a short
    /// sentence spoken while something heavy is running is marked
    /// `interactive` with `preempt`, which asks the backend to pause at its
    /// next checkpoint, answer, and resume — not to give up.
    fn forward_to_backend(&self, utterance_id: String, text: String) {
        let Some(backend) = self.backend.clone() else {
            return;
        };
        let commands = self.command_tx.clone();
        let bus = self.bus.clone();
        let tasks = self.tasks.clone();
        let (priority, preempt) = self.tasks.classify(&text);
        let active = self.tasks.active_ids();
        let options = crate::backend::TurnOptions { priority, preempt };
        tokio::spawn(async move {
            if let Err(e) = backend
                .turn(
                    &utterance_id,
                    &text,
                    options,
                    active,
                    commands,
                    tasks,
                    bus.clone(),
                )
                .await
            {
                tracing::warn!("backend turn failed: {e}");
                bus.emit(VoiceEvent::Error {
                    code: e.code().to_string(),
                    message: e.to_string(),
                    fatal: false,
                    hint: e.hint().map(str::to_owned),
                });
            }
        });
    }

    // ---------------------------------------------------------------- speech

    async fn speak(&mut self, request: SpeakRequest, chunker: Option<SentenceChunker>) {
        if request.interrupt {
            self.interrupt(None, "new-utterance");
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let job = SpeakingJob {
            utterance_id: request.utterance_id.clone(),
            cancel: cancel.clone(),
            chunker,
            started: Instant::now(),
            echo_text: request.echo_text,
            persist: request.persist && self.config.tts.persist_audio,
            voice: request.voice.clone(),
            task: None,
        };
        let streaming = job.chunker.is_some();
        self.bus.emit(VoiceEvent::SpeakStarted {
            utterance_id: request.utterance_id.clone(),
            characters: request.text.chars().count(),
            voice: request.voice.id.clone(),
            streaming,
        });
        self.apply(Trigger::SynthesisStarted);
        self.speaking = Some(job);

        if !request.text.trim().is_empty() {
            let text = request.text.clone();
            self.enqueue_speech(&text).await;
        } else if streaming {
            // An opened stream with no initial text just waits for deltas.
        } else {
            self.finish_speech("completed");
        }
    }

    async fn speak_delta(&mut self, utterance_id: &str, text: &str) {
        let chunks = match self.speaking.as_mut() {
            Some(job) if job.utterance_id == utterance_id => match job.chunker.as_mut() {
                Some(chunker) => chunker.push(text),
                None => {
                    tracing::warn!("delta for a non-streaming utterance was ignored");
                    return;
                }
            },
            _ => {
                tracing::debug!("delta for an unknown utterance {utterance_id} was ignored");
                return;
            }
        };
        for chunk in chunks {
            self.enqueue_speech(&chunk).await;
        }
    }

    async fn speak_end(&mut self, utterance_id: &str) {
        let tail = match self.speaking.as_mut() {
            Some(job) if job.utterance_id == utterance_id => {
                job.chunker.as_mut().and_then(|c| c.flush())
            }
            _ => return,
        };
        if let Some(text) = tail {
            self.enqueue_speech(&text).await;
        }
        self.finish_speech("completed");
    }

    /// Spawn (or extend) the speech task for the current utterance.
    ///
    /// Each chunk of text gets its own task, chained through the job so that
    /// chunks are spoken strictly in order. The task does the synthesis, the
    /// playback, the feature extraction and the optional disk write; the loop
    /// only ever hands it a string.
    async fn enqueue_speech(&mut self, text: &str) {
        let Some(job) = self.speaking.as_mut() else {
            return;
        };
        let previous = job.task.take();
        let task = speech::SpeechTask {
            utterance_id: job.utterance_id.clone(),
            text: text.to_string(),
            voice: job.voice.clone(),
            echo_text: job.echo_text,
            persist: job.persist,
            cancel: job.cancel.clone(),
            synthesizer: self.synthesizer.clone(),
            playback: self.playback.clone(),
            bus: self.bus.clone(),
            retention: self.retention.clone(),
            paths: self.paths.clone(),
            frame_ms: self.config.audio.frame_ms.max(5) as u64,
            internal: self.internal_tx.clone(),
            streaming: job.chunker.is_some(),
        };
        job.task = Some(tokio::spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            task.run().await;
        }));
    }

    fn finish_speech(&mut self, reason: &'static str) {
        let Some(job) = self.speaking.take() else {
            return;
        };
        let tx = self.internal_tx.clone();
        let utterance_id = job.utterance_id.clone();
        let duration_ms = job.started.elapsed().as_millis() as u64;
        let task = job.task;
        // Wait for the audio already queued to finish before announcing the
        // end; otherwise the orb would settle while Cookie is still talking.
        tokio::spawn(async move {
            if let Some(task) = task {
                let _ = task.await;
            }
            let _ = tx
                .send(Internal::SpeechFinished {
                    utterance_id,
                    duration_ms,
                    reason,
                })
                .await;
        });
    }

    fn interrupt(&mut self, utterance_id: Option<String>, source: &str) {
        let Some(job) = self.speaking.as_ref() else {
            return;
        };
        if let Some(wanted) = &utterance_id {
            if wanted != &job.utterance_id {
                return;
            }
        }
        job.cancel.store(true, Ordering::Release);
        if let Ok(mut playback) = self.playback.lock() {
            playback.clear();
        }
        let elapsed_ms = job.started.elapsed().as_millis() as u64;
        let id = job.utterance_id.clone();
        self.bus.emit(VoiceEvent::Interrupted {
            utterance_id: Some(id.clone()),
            source: source.to_string(),
            elapsed_ms,
        });
        self.speaking = None;
        self.apply_with(Trigger::Interrupt, StateReason::UserInterrupt);
        self.bus.emit(VoiceEvent::SpeakFinished {
            utterance_id: id,
            duration_ms: elapsed_ms,
            reason: "interrupted".into(),
        });
    }

    // ----------------------------------------------------------------- state

    fn apply(&mut self, trigger: Trigger) {
        if let Some(transition) = self.machine.apply(trigger) {
            self.bus.publish_state(transition.to);
            self.bus.emit(VoiceEvent::StateChanged {
                from: transition.from,
                to: transition.to,
                reason: transition.reason,
            });
        }
    }

    fn apply_with(&mut self, trigger: Trigger, reason: StateReason) {
        if let Some(mut transition) = self.machine.apply(trigger) {
            transition.reason = reason;
            self.bus.publish_state(transition.to);
            self.bus.emit(VoiceEvent::StateChanged {
                from: transition.from,
                to: transition.to,
                reason,
            });
        }
    }

    fn report(&mut self, error: &Error, fatal: bool) {
        tracing::warn!("{error}");
        self.bus.emit(VoiceEvent::Error {
            code: error.code().to_string(),
            message: error.to_string(),
            fatal,
            hint: error.hint().map(str::to_owned),
        });
        if fatal {
            self.apply(Trigger::Failed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SttProviderKind, TtsProviderKind};
    use crate::events::VoiceEvent as Ev;
    use tokio::time::timeout;

    async fn engine_with(config: Config) -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Arc::new(Paths::rooted(dir.path()));
        paths.ensure().unwrap();
        let retention = Arc::new(RetentionManager::open(&paths, config.retention.clone()).unwrap());
        let devices = Devices::mock(config.audio.sample_rate);
        let engine = Engine::start(Arc::new(config), paths, devices, retention)
            .await
            .unwrap();
        (engine, dir)
    }

    fn test_config() -> Config {
        let mut config = Config::default();
        config.stt.provider = SttProviderKind::Mock;
        config.tts.provider = TtsProviderKind::Mock;
        config.tts.persist_audio = false;
        config.audio.listen_on_start = false;
        config
    }

    async fn wait_for<F>(
        rx: &mut tokio::sync::broadcast::Receiver<crate::events::EventEnvelope>,
        mut pred: F,
    ) -> Option<Ev>
    where
        F: FnMut(&Ev) -> bool,
    {
        timeout(Duration::from_secs(5), async {
            loop {
                match rx.recv().await {
                    Ok(envelope) => {
                        if pred(&envelope.event) {
                            return Some(envelope.event);
                        }
                    }
                    Err(_) => return None,
                }
            }
        })
        .await
        .ok()
        .flatten()
    }

    #[tokio::test]
    async fn engine_announces_itself() {
        let (engine, _dir) = engine_with(test_config()).await;
        let mut rx = engine.bus().subscribe();
        // `ready` may have been emitted before we subscribed; state is a
        // latched watch, so check that instead.
        assert_eq!(engine.state(), VoiceState::Idle);
        engine
            .send(Command::Speak(Box::new(SpeakRequest {
                utterance_id: "u1".into(),
                text: "hello".into(),
                voice: Default::default(),
                interrupt: false,
                echo_text: false,
                persist: false,
            })))
            .await
            .unwrap();
        let started = wait_for(&mut rx, |e| matches!(e, Ev::SpeakStarted { .. })).await;
        assert!(started.is_some());
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn speaking_reaches_the_finished_event() {
        let (engine, _dir) = engine_with(test_config()).await;
        let mut rx = engine.bus().subscribe();
        engine
            .send(Command::Speak(Box::new(SpeakRequest {
                utterance_id: "u2".into(),
                text: "good evening".into(),
                voice: Default::default(),
                interrupt: false,
                echo_text: false,
                persist: false,
            })))
            .await
            .unwrap();
        let finished = wait_for(&mut rx, |e| matches!(e, Ev::SpeakFinished { .. })).await;
        assert!(matches!(finished, Some(Ev::SpeakFinished { .. })));
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn interrupt_stops_speech_and_reports_it() {
        let mut config = test_config();
        config
            .tts
            .options
            .insert("realtime".into(), toml::Value::Boolean(true));
        let (engine, _dir) = engine_with(config).await;
        let mut rx = engine.bus().subscribe();
        engine
            .send(Command::Speak(Box::new(SpeakRequest {
                utterance_id: "u3".into(),
                text: "a long sentence that keeps going for a while".into(),
                voice: Default::default(),
                interrupt: false,
                echo_text: false,
                persist: false,
            })))
            .await
            .unwrap();
        wait_for(&mut rx, |e| matches!(e, Ev::SpeakStarted { .. })).await;
        engine
            .send(Command::Interrupt {
                utterance_id: None,
                source: "test".into(),
            })
            .await
            .unwrap();
        let event = wait_for(&mut rx, |e| matches!(e, Ev::Interrupted { .. })).await;
        assert!(event.is_some());
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn pushed_audio_produces_a_final_transcript() {
        let (engine, _dir) = engine_with(test_config()).await;
        let mut rx = engine.bus().subscribe();
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.02).sin() * 0.4).collect();
        engine.send(Command::PushAudio { samples }).await.unwrap();
        let event = wait_for(&mut rx, |e| matches!(e, Ev::TranscriptFinal { .. })).await;
        match event {
            Some(Ev::TranscriptFinal { text, .. }) => assert!(!text.is_empty()),
            other => panic!("expected a final transcript, got {other:?}"),
        }
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn listening_commands_move_the_state_machine() {
        let (engine, _dir) = engine_with(test_config()).await;
        let mut rx = engine.bus().subscribe();
        engine
            .send(Command::StartListening {
                continuous: true,
                source: "test".into(),
            })
            .await
            .unwrap();
        assert!(
            wait_for(&mut rx, |e| matches!(e, Ev::ListeningStarted { .. }))
                .await
                .is_some()
        );
        engine
            .send(Command::StopListening {
                source: "test".into(),
            })
            .await
            .unwrap();
        assert!(
            wait_for(&mut rx, |e| matches!(e, Ev::ListeningStopped { .. }))
                .await
                .is_some()
        );
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn streamed_text_is_spoken_sentence_by_sentence() {
        let (engine, _dir) = engine_with(test_config()).await;
        let mut rx = engine.bus().subscribe();
        engine
            .send(Command::SpeakStreamOpen(Box::new(SpeakRequest {
                utterance_id: "s1".into(),
                text: String::new(),
                voice: Default::default(),
                interrupt: true,
                echo_text: true,
                persist: false,
            })))
            .await
            .unwrap();
        for delta in ["Good evening. ", "How can ", "I help? "] {
            engine
                .send(Command::SpeakStreamDelta {
                    utterance_id: "s1".into(),
                    text: delta.into(),
                })
                .await
                .unwrap();
        }
        engine
            .send(Command::SpeakStreamEnd {
                utterance_id: "s1".into(),
            })
            .await
            .unwrap();
        assert!(wait_for(&mut rx, |e| matches!(e, Ev::SpeakChunk { .. }))
            .await
            .is_some());
        assert!(wait_for(&mut rx, |e| matches!(e, Ev::SpeakFinished { .. }))
            .await
            .is_some());
        engine.shutdown().await;
    }
}
