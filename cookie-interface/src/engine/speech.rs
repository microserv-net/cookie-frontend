//! The task that turns one piece of text into sound.
//!
//! It is deliberately the only place in the crate that touches the speaker
//! while Cookie is talking. Three things happen here in lockstep:
//!
//! 1. audio is pulled from the synthesiser chunk by chunk,
//! 2. it is queued for playback in small pieces so an interruption is felt
//!    within a frame rather than at the end of a sentence,
//! 3. the same audio is analysed and published to the feature channel *in
//!    real time*, which is what makes the orb move with the voice instead of
//!    ahead of it.
//!
//! Step 3 is also the pacing mechanism: walking the audio at wall-clock speed
//! naturally stops us from queueing more than a fraction of a second ahead of
//! the speaker, with no timers or watermarks to get wrong.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::audio::{AudioAnalyzer, AudioBuffer, FeatureSource, PlaybackHandle};
use crate::config::VoiceSpec;
use crate::events::{EventBus, VoiceEvent};
use crate::paths::Paths;
use crate::retention::RetentionManager;
use crate::tts::{SharedSynthesizer, SynthesisRequest};

use super::Internal;

/// Longest slice of audio handed to the speaker at once.
///
/// Short enough that `clear()` on barge-in cuts the voice off effectively
/// immediately, long enough that we are not waking up constantly.
const PIECE_MS: u64 = 120;

/// One unit of speech work.
pub(crate) struct SpeechTask {
    pub(crate) utterance_id: String,
    pub(crate) text: String,
    pub(crate) voice: VoiceSpec,
    pub(crate) echo_text: bool,
    pub(crate) persist: bool,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) synthesizer: SharedSynthesizer,
    pub(crate) playback: Arc<Mutex<PlaybackHandle>>,
    pub(crate) bus: Arc<EventBus>,
    pub(crate) retention: Arc<RetentionManager>,
    pub(crate) paths: Arc<Paths>,
    pub(crate) frame_ms: u64,
    pub(crate) internal: mpsc::Sender<Internal>,
    /// Part of a streaming utterance: the engine, not this task, decides when
    /// the utterance as a whole is over.
    pub(crate) streaming: bool,
}

impl std::fmt::Debug for SpeechTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechTask")
            .field("utterance_id", &self.utterance_id)
            .field("characters", &self.text.len())
            .finish()
    }
}

impl SpeechTask {
    /// Synthesise and play. Returns when the audio has been played, the task
    /// was cancelled, or synthesis failed.
    pub async fn run(self) {
        let started = std::time::Instant::now();
        let request = SynthesisRequest {
            utterance_id: self.utterance_id.clone(),
            text: self.text.clone(),
            voice: self.voice.clone(),
        };

        let mut stream = match self.synthesizer.synthesize(request).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("synthesis failed: {e}");
                self.bus.emit(VoiceEvent::Error {
                    code: e.code().to_string(),
                    message: e.to_string(),
                    fatal: false,
                    hint: e.hint().map(str::to_owned),
                });
                self.report_finished(started, "failed").await;
                return;
            }
        };

        let mut analyzer: Option<AudioAnalyzer> = None;
        let mut keep: Option<(Vec<f32>, u32)> = self.persist.then(|| (Vec::new(), 0));
        let mut index = 0u32;
        let mut spoken_ms = 0u64;

        while let Some(chunk) = stream.next_chunk().await {
            if self.cancelled() {
                break;
            }
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("synthesis stream: {e}");
                    self.bus.emit(VoiceEvent::Error {
                        code: e.code().to_string(),
                        message: e.to_string(),
                        fatal: false,
                        hint: None,
                    });
                    break;
                }
            };
            if chunk.is_empty() {
                continue;
            }
            if let Some((buffer, rate)) = keep.as_mut() {
                if *rate == 0 {
                    *rate = chunk.sample_rate;
                }
                if *rate == chunk.sample_rate {
                    buffer.extend_from_slice(&chunk.samples);
                } else {
                    buffer.extend_from_slice(&chunk.resampled(*rate).samples);
                }
            }
            let analyzer = analyzer.get_or_insert_with(|| {
                AudioAnalyzer::new(chunk.sample_rate, FeatureSource::Output)
            });

            spoken_ms += chunk.duration_ms();
            self.bus.emit(VoiceEvent::SpeakChunk {
                utterance_id: self.utterance_id.clone(),
                index,
                duration_ms: chunk.duration_ms(),
                text: self.echo_text.then(|| self.text.clone()),
            });
            index += 1;

            if !self.play_chunk(&chunk, analyzer).await {
                break;
            }
        }

        // Let the tail of the audio actually reach the speaker before we call
        // the utterance finished.
        self.drain().await;

        if let Some((samples, rate)) = keep {
            if !samples.is_empty() && !self.cancelled() {
                self.persist_audio(samples, rate).await;
            }
        }

        self.report_finished(
            started,
            if self.cancelled() {
                "interrupted"
            } else {
                "completed"
            },
        )
        .await;
        let _ = spoken_ms;
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    /// Queue one synthesised chunk in real-time-sized pieces.
    ///
    /// Returns `false` if the utterance was cancelled part-way through.
    async fn play_chunk(&self, chunk: &AudioBuffer, analyzer: &mut AudioAnalyzer) -> bool {
        let rate = chunk.sample_rate.max(1) as u64;
        let piece = ((rate * PIECE_MS) / 1000).max(1) as usize;
        let frame = ((rate * self.frame_ms) / 1000).max(1) as usize;

        for part in chunk.samples.chunks(piece) {
            if self.cancelled() {
                return false;
            }
            {
                // Short, `await`-free critical section: the guard never
                // crosses a suspension point.
                let mut playback = match self.playback.lock() {
                    Ok(p) => p,
                    Err(poisoned) => poisoned.into_inner(),
                };
                playback.write(part, chunk.sample_rate);
            }
            // Walk the piece at wall-clock speed, publishing what the user is
            // hearing right now.
            for window in part.chunks(frame) {
                if self.cancelled() {
                    return false;
                }
                let features = analyzer.process(window);
                self.bus.publish_features(features);
                tokio::time::sleep(Duration::from_millis(self.frame_ms)).await;
            }
        }
        true
    }

    /// Wait for queued audio to finish playing, with a ceiling so a stuck
    /// device cannot wedge the utterance forever.
    async fn drain(&self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if self.cancelled() || std::time::Instant::now() > deadline {
                return;
            }
            let queued = {
                let playback = match self.playback.lock() {
                    Ok(p) => p,
                    Err(poisoned) => poisoned.into_inner(),
                };
                playback.queued_ms()
            };
            if queued <= self.frame_ms {
                return;
            }
            tokio::time::sleep(Duration::from_millis(self.frame_ms)).await;
        }
    }

    /// Write the utterance to the managed audio directory under the retention
    /// ledger. All of it happens on a blocking thread.
    async fn persist_audio(&self, samples: Vec<f32>, rate: u32) {
        let retention = self.retention.clone();
        let id = self.utterance_id.clone();
        let duration_ms = (samples.len() as u64 * 1000) / rate.max(1) as u64;
        let paths = self.paths.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _ = paths; // the manager owns the directory; kept for clarity
            let pending = retention.begin(&id)?;
            match crate::audio::wav::write_mono_wav(&pending.path, &samples, rate) {
                Ok(bytes) => retention.commit(pending, bytes, duration_ms),
                Err(e) => {
                    retention.abort(pending);
                    Err(e)
                }
            }
        })
        .await;
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("could not store generated audio: {e}"),
            Err(e) => tracing::warn!("audio persistence task failed: {e}"),
        }
    }

    async fn report_finished(&self, started: std::time::Instant, reason: &'static str) {
        if self.streaming {
            // A streaming utterance ends when the caller closes it.
            return;
        }
        let _ = self
            .internal
            .send(Internal::SpeechFinished {
                utterance_id: self.utterance_id.clone(),
                duration_ms: started.elapsed().as_millis() as u64,
                reason,
            })
            .await;
    }
}
