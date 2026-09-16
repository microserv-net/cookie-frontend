//! `cookie-interface --test`.
//!
//! One short conversation that exercises every piece of real hardware and
//! every layer of the pipeline in the order they are actually used: speak,
//! listen, detect speech, transcribe, react, speak again. If this works, the
//! installation works; if it hangs at a particular step, that step tells you
//! exactly which part is wrong.
//!
//! It is intentionally the *only* place in this crate that decides what to
//! say based on what was heard. That is a six-line greeting, not an assistant
//! — the thinking belongs to the backend.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::timeout;

use crate::config::Config;
use crate::engine::{Devices, Engine};
use crate::error::{Error, Result};
use crate::events::{Command, SpeakRequest, VoiceEvent};
use crate::paths::Paths;
use crate::retention::RetentionManager;
use crate::util::text::extract_name;

/// How long to wait for the user to say something.
const LISTEN_TIMEOUT: Duration = Duration::from_secs(30);

/// How long to wait for the recogniser once you have stopped speaking.
///
/// Separate from the listening deadline, and much longer, because they are
/// different questions: the first is "did anybody say anything", the second
/// is "how long does this machine take". A local Whisper on a cold cache can
/// take most of a minute for its first utterance, and a check that gives up
/// at twenty seconds reports "nothing was transcribed" for a machine that was
/// about to transcribe it perfectly.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(150);
/// How long to wait for a piece of speech to finish playing.
const SPEAK_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of the check, so the caller can set an exit code.
#[derive(Debug, PartialEq)]
pub struct TestOutcome {
    pub heard: Option<String>,
    pub name: Option<String>,
    pub spoke: bool,
}

/// Run the end-to-end check against an already-running engine.
pub async fn run(engine: &Engine) -> Result<TestOutcome> {
    let mut events = engine.bus().subscribe();
    let voice = engine.config().tts.voice.clone();

    println!("  1/5  speaking a prompt");
    engine
        .send(Command::Speak(Box::new(SpeakRequest {
            utterance_id: "test-prompt".into(),
            text: "What's your name?".into(),
            voice: voice.clone(),
            interrupt: true,
            echo_text: false,
            persist: false,
        })))
        .await?;

    // Wait for playback to finish before opening the microphone; otherwise
    // the first thing we transcribe is ourselves.
    let mut spoke = false;
    let deadline = tokio::time::Instant::now() + SPEAK_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        match timeout(Duration::from_secs(2), events.recv()).await {
            Ok(Ok(envelope)) => match envelope.event {
                VoiceEvent::SpeakFinished { utterance_id, .. } if utterance_id == "test-prompt" => {
                    spoke = true;
                    break;
                }
                VoiceEvent::Error { message, .. } => {
                    println!("       ! {message}");
                }
                _ => {}
            },
            Ok(Err(_)) => break,
            Err(_) => continue,
        }
    }
    if !spoke {
        println!("       ! the prompt never finished playing");
    }

    println!("  2/5  listening — say your name");
    // `StartListening` also wakes her, so the check does not require saying
    // "Cookie" into your own diagnostic.
    engine
        .send(Command::StartListening {
            continuous: false,
            source: "test".into(),
        })
        .await?;

    let mut heard = None;
    let mut detected = false;
    let mut deadline = tokio::time::Instant::now() + LISTEN_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        match timeout(Duration::from_secs(1), events.recv()).await {
            Ok(Ok(envelope)) => match envelope.event {
                VoiceEvent::SpeechDetected { level_db } => {
                    if !detected {
                        detected = true;
                        println!("  3/5  speech detected at {level_db:.0} dB");
                    }
                }
                VoiceEvent::SpeechEnded { duration_ms } => {
                    println!("       {duration_ms} ms of speech; transcribing…");
                    // The clock now measures the recogniser, not you.
                    deadline = tokio::time::Instant::now() + TRANSCRIBE_TIMEOUT;
                }
                VoiceEvent::Ignored { text, .. } => {
                    // The wake word is bypassed for `--test`, so this should
                    // not happen — but if it does, silence would be the worst
                    // possible report.
                    println!("       heard but discarded: {text}");
                }
                VoiceEvent::TranscriptPartial { text, .. } => {
                    println!("       … {text}");
                }
                VoiceEvent::TranscriptFinal { text, .. } => {
                    println!("  4/5  heard: {text}");
                    heard = Some(text);
                    break;
                }
                VoiceEvent::Error { message, .. } => println!("       ! {message}"),
                _ => {}
            },
            Ok(Err(_)) => break,
            Err(_) => continue,
        }
    }

    let _ = engine
        .send(Command::StopListening {
            source: "test".into(),
        })
        .await;

    let Some(transcript) = heard.clone() else {
        return Err(Error::Stt(if detected {
            "your speech was heard but never transcribed. The recogniser may \
             still be loading — try again, and check `--doctor`."
                .into()
        } else {
            "nothing was heard at all. Check the microphone with `--doctor`, \
             and that the input level is not muted."
                .to_string()
        }));
    };

    // A stand-in provider produces a transcript that is not a transcription.
    // Reporting "heard: My name is Alex" in that state is how this check came
    // to pass on a machine where nothing was being recognised at all.
    if transcript.contains("stand-in recogniser") {
        return Err(Error::ModelUnavailable {
            name: "speech recognition".into(),
            reason: "no recogniser is installed, so nothing was transcribed. \
                     Run `cookie-interface --setup` to fetch one."
                .into(),
        });
    }

    let name = extract_name(&transcript);
    let reply = match &name {
        Some(name) => format!("Nice to meet you, {name}."),
        // Being wrong gracefully matters more here than being clever.
        None => "Nice to meet you.".to_string(),
    };

    println!("  5/5  replying: {reply}");
    engine
        .send(Command::Speak(Box::new(SpeakRequest {
            utterance_id: "test-reply".into(),
            text: reply,
            voice,
            interrupt: true,
            echo_text: false,
            persist: false,
        })))
        .await?;

    let mut replied = false;
    let deadline = tokio::time::Instant::now() + SPEAK_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        match timeout(Duration::from_secs(2), events.recv()).await {
            Ok(Ok(envelope)) => {
                if let VoiceEvent::SpeakFinished { utterance_id, .. } = envelope.event {
                    if utterance_id == "test-reply" {
                        replied = true;
                        break;
                    }
                }
            }
            Ok(Err(_)) => break,
            Err(_) => continue,
        }
    }

    Ok(TestOutcome {
        heard,
        name,
        spoke: spoke && replied,
    })
}

/// Boot an engine, run the check, and shut down again.
///
/// Used when `--test` is passed without a full session.
pub async fn run_standalone(config: Arc<Config>, paths: Arc<Paths>) -> Result<TestOutcome> {
    let retention = Arc::new(RetentionManager::open(&paths, config.retention.clone())?);
    let devices = Devices::from_config(&config);
    let engine = Engine::start(config, paths, devices, retention).await?;
    // Give the providers a moment to report in before we start talking.
    tokio::time::sleep(Duration::from_millis(250)).await;
    let outcome = run(&engine).await;
    engine.shutdown().await;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SttProviderKind, TtsProviderKind};

    async fn engine_with_script(script: &str) -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Arc::new(Paths::rooted(dir.path()));
        paths.ensure().unwrap();
        let mut config = Config::default();
        config.stt.provider = SttProviderKind::Mock;
        config.tts.provider = TtsProviderKind::Mock;
        config.tts.persist_audio = false;
        config.audio.listen_on_start = false;
        config.stt.options.insert(
            "script".into(),
            toml::Value::Array(vec![toml::Value::String(script.into())]),
        );
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
        (engine, dir)
    }

    #[tokio::test]
    async fn the_flow_fails_clearly_when_nothing_is_heard() {
        // Mock devices produce silence, so the VAD never fires and no
        // transcript arrives: exactly the "your microphone is dead" case.
        let (engine, _dir) = engine_with_script("My name is Alex").await;
        let result = tokio::time::timeout(Duration::from_secs(40), run(&engine)).await;
        match result {
            Ok(Err(e)) => assert_eq!(e.code(), "stt_error"),
            Ok(Ok(outcome)) => assert!(outcome.heard.is_some()),
            Err(_) => panic!("the test flow hung instead of timing out cleanly"),
        }
        engine.shutdown().await;
    }

    #[test]
    fn names_are_pulled_out_of_ordinary_answers() {
        assert_eq!(extract_name("My name is Alex").as_deref(), Some("Alex"));
        assert_eq!(extract_name("I'm Robin").as_deref(), Some("Robin"));
        assert_eq!(extract_name("Sam").as_deref(), Some("Sam"));
    }
}
