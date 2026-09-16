//! End-to-end tests over the public API surface only.
//!
//! These use the mock providers and mock devices, so they run on a machine
//! with no microphone, no speakers, no GPU and no downloaded model — which is
//! the whole point: the pipeline is exercised on every `cargo test`, and only
//! the hardware itself is left to `--test` and `--doctor`.

use std::sync::Arc;
use std::time::Duration;

use cookie_interface::config::{Config, SttProviderKind, TtsProviderKind};
use cookie_interface::engine::{Devices, Engine};
use cookie_interface::events::{Command, SpeakRequest, VoiceEvent};
use cookie_interface::paths::Paths;
use cookie_interface::retention::{RetentionManager, RetentionPolicy};
use cookie_interface::state::VoiceState;

struct Harness {
    engine: Engine,
    _dir: tempfile::TempDir,
}

async fn harness(configure: impl FnOnce(&mut Config)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let paths = Arc::new(Paths::rooted(dir.path()));
    paths.ensure().unwrap();

    let mut config = Config::default();
    config.stt.provider = SttProviderKind::Mock;
    config.tts.provider = TtsProviderKind::Mock;
    config.audio.listen_on_start = false;
    configure(&mut config);

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

    Harness { engine, _dir: dir }
}

async fn wait_for<F>(
    rx: &mut tokio::sync::broadcast::Receiver<cookie_interface::events::EventEnvelope>,
    mut matches: F,
) -> Option<VoiceEvent>
where
    F: FnMut(&VoiceEvent) -> bool,
{
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let envelope = rx.recv().await.ok()?;
            if matches(&envelope.event) {
                return Some(envelope.event);
            }
        }
    })
    .await
    .ok()
    .flatten()
}

#[tokio::test]
async fn a_spoken_utterance_runs_the_whole_pipeline() {
    let h = harness(|c| c.tts.persist_audio = false).await;
    let mut events = h.engine.bus().subscribe();

    h.engine
        .send(Command::Speak(Box::new(SpeakRequest {
            utterance_id: "e2e-1".into(),
            text: "Good evening. The kettle has boiled.".into(),
            voice: Default::default(),
            interrupt: true,
            echo_text: false,
            persist: false,
        })))
        .await
        .unwrap();

    assert!(matches!(
        wait_for(&mut events, |e| matches!(
            e,
            VoiceEvent::SpeakStarted { .. }
        ))
        .await,
        Some(VoiceEvent::SpeakStarted { .. })
    ));
    // Chunks mean audio actually reached the speaker, not just that the job
    // was accepted.
    assert!(matches!(
        wait_for(&mut events, |e| matches!(e, VoiceEvent::SpeakChunk { .. })).await,
        Some(VoiceEvent::SpeakChunk { .. })
    ));
    match wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::SpeakFinished { .. })
    })
    .await
    {
        Some(VoiceEvent::SpeakFinished { reason, .. }) => assert_eq!(reason, "completed"),
        other => panic!("expected completion, got {other:?}"),
    }

    h.engine.shutdown().await;
}

#[tokio::test]
async fn pushed_audio_becomes_a_final_transcript() {
    let h = harness(|_| {}).await;
    let mut events = h.engine.bus().subscribe();

    let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.02).sin() * 0.4).collect();
    h.engine.send(Command::PushAudio { samples }).await.unwrap();

    match wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::TranscriptFinal { .. })
    })
    .await
    {
        Some(VoiceEvent::TranscriptFinal { text, .. }) => assert!(!text.trim().is_empty()),
        other => panic!("expected a transcript, got {other:?}"),
    }

    h.engine.shutdown().await;
}

#[tokio::test]
async fn interrupting_speech_does_not_cancel_work() {
    let h = harness(|c| {
        c.tts
            .options
            .insert("realtime".into(), toml::Value::Boolean(true));
    })
    .await;

    // A task the "backend" is working on.
    h.engine.tasks().upsert(
        "task-1",
        Some("fixing the failing tests".into()),
        cookie_interface::tasks::TaskState::Running,
        Some(cookie_interface::tasks::Weight::Heavy),
        None,
    );

    let mut events = h.engine.bus().subscribe();
    h.engine
        .send(Command::Speak(Box::new(SpeakRequest {
            utterance_id: "e2e-2".into(),
            text: "This is a long sentence that keeps going for quite a while.".into(),
            voice: Default::default(),
            interrupt: false,
            echo_text: false,
            persist: false,
        })))
        .await
        .unwrap();
    wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::SpeakStarted { .. })
    })
    .await;

    h.engine
        .send(Command::Interrupt {
            utterance_id: None,
            source: "test".into(),
        })
        .await
        .unwrap();

    assert!(matches!(
        wait_for(&mut events, |e| matches!(e, VoiceEvent::Interrupted { .. })).await,
        Some(VoiceEvent::Interrupted { .. })
    ));
    // The point of the test: the work survived being told to be quiet.
    assert_eq!(h.engine.tasks().active().len(), 1);
    assert!(h.engine.tasks().heavy_in_flight());

    h.engine.shutdown().await;
}

#[tokio::test]
async fn cancelling_work_is_a_separate_request() {
    let h = harness(|_| {}).await;
    h.engine.tasks().upsert(
        "task-2",
        Some("building the project".into()),
        cookie_interface::tasks::TaskState::Running,
        Some(cookie_interface::tasks::Weight::Heavy),
        None,
    );
    let mut events = h.engine.bus().subscribe();

    h.engine
        .send(Command::CancelTask {
            task_id: None,
            source: "test".into(),
        })
        .await
        .unwrap();

    assert!(matches!(
        wait_for(&mut events, |e| matches!(e, VoiceEvent::Task { .. })).await,
        Some(VoiceEvent::Task { .. })
    ));
    assert!(h.engine.tasks().active().is_empty());

    h.engine.shutdown().await;
}

#[tokio::test]
async fn diagnostics_answer_in_a_sentence_a_person_would_say() {
    let h = harness(|_| {}).await;
    let report = h.engine.diagnose(false).await;
    let spoken = report.spoken_summary();

    // No identifiers, no field names: this gets read aloud.
    for forbidden in ["audio.input", "stt", "tts", "_", "{"] {
        assert!(!spoken.contains(forbidden), "{spoken}");
    }
    assert!(spoken.ends_with('.') || spoken.ends_with('!'), "{spoken}");
    assert!(report.checks.len() >= 8);

    h.engine.shutdown().await;
}

#[tokio::test]
async fn generated_audio_obeys_the_retention_policy() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Arc::new(Paths::rooted(dir.path()));
    paths.ensure().unwrap();

    let mut config = Config::default();
    config.stt.provider = SttProviderKind::Mock;
    config.tts.provider = TtsProviderKind::Mock;
    config.tts.persist_audio = true;
    config.audio.listen_on_start = false;
    let config = Arc::new(config);

    let retention = Arc::new(RetentionManager::open(&paths, config.retention.clone()).unwrap());
    let engine = Engine::start(
        config.clone(),
        paths.clone(),
        Devices::mock(config.audio.sample_rate),
        retention.clone(),
    )
    .await
    .unwrap();

    let mut events = engine.bus().subscribe();
    engine
        .send(Command::Speak(Box::new(SpeakRequest {
            utterance_id: "kept".into(),
            text: "Remember this.".into(),
            voice: Default::default(),
            interrupt: true,
            echo_text: false,
            persist: true,
        })))
        .await
        .unwrap();
    wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::SpeakFinished { .. })
    })
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    let wavs = |dir: &std::path::Path| {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .count()
    };
    assert!(wavs(retention.directory()) > 0, "nothing was persisted");

    // Shortening the policy applies retroactively, not just to new audio.
    retention.set_policy(RetentionPolicy::Immediate);
    let report = tokio::task::spawn_blocking({
        let retention = retention.clone();
        move || retention.sweep()
    })
    .await
    .unwrap()
    .unwrap();

    assert!(report.deleted > 0, "{report:?}");
    assert_eq!(wavs(retention.directory()), 0);
    // The ledger itself is not audio and must survive the sweep, or the next
    // startup would treat everything as an orphan.
    assert!(paths.audio_ledger().exists());

    engine.shutdown().await;
}

#[tokio::test]
async fn listening_moves_through_the_state_machine() {
    let h = harness(|_| {}).await;
    assert_eq!(h.engine.state(), VoiceState::Idle);

    let mut events = h.engine.bus().subscribe();
    h.engine
        .send(Command::StartListening {
            continuous: true,
            source: "test".into(),
        })
        .await
        .unwrap();
    wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::ListeningStarted { .. })
    })
    .await;
    assert_eq!(h.engine.state(), VoiceState::Listening);

    h.engine
        .send(Command::StopListening {
            source: "test".into(),
        })
        .await
        .unwrap();
    wait_for(&mut events, |e| {
        matches!(e, VoiceEvent::ListeningStopped { .. })
    })
    .await;
    assert_eq!(h.engine.state(), VoiceState::Idle);

    h.engine.shutdown().await;
}
