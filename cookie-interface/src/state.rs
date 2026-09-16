//! The interface state machine.
//!
//! This is the single source of truth for "what is Cookie doing right now".
//! The renderer *observes* it and never writes to it; the engine drives it.
//! Keeping the rule that direction-of-flow is one-way is what stops the orb
//! and the audio pipeline from ever disagreeing.
//!
//! Invalid transitions are rejected rather than applied, because a half-valid
//! transition is exactly how you end up with a "speaking" animation playing
//! over silence.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceState {
    /// Nothing in flight. The orb breathes.
    Idle,
    /// Microphone open, VAD armed.
    Listening,
    /// Audio captured, recognition or synthesis warm-up in flight.
    Processing,
    /// Audio is being played out.
    Speaking,
    /// Speech was cut off. A short, visible beat before returning to Idle —
    /// it exists as a state so the interruption is *legible* rather than the
    /// orb just snapping silent.
    Interrupted,
    /// Something failed. Recoverable; the engine leaves this state on the next
    /// successful operation or after a timeout.
    Error,
}

impl VoiceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            VoiceState::Idle => "idle",
            VoiceState::Listening => "listening",
            VoiceState::Processing => "processing",
            VoiceState::Speaking => "speaking",
            VoiceState::Interrupted => "interrupted",
            VoiceState::Error => "error",
        }
    }

    /// States that resolve on their own after a short time.
    pub fn auto_exit_after(&self) -> Option<Duration> {
        match self {
            VoiceState::Interrupted => Some(Duration::from_millis(900)),
            VoiceState::Error => Some(Duration::from_secs(6)),
            _ => None,
        }
    }
}

impl std::fmt::Display for VoiceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a transition happened. Carried in the `state` event so a client can
/// distinguish "stopped listening because you went quiet" from "stopped
/// listening because someone called /v1/interrupt".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateReason {
    Startup,
    VoiceActivity,
    SilenceTimeout,
    ApiRequest,
    RecognitionComplete,
    SynthesisReady,
    PlaybackComplete,
    UserInterrupt,
    BargeIn,
    Failure,
    Timeout,
    Shutdown,
}

/// Events that drive the machine. Modelling the *trigger* rather than letting
/// callers set a state directly is what makes the legal-transition table
/// meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    Boot,
    StartListening,
    SpeechStarted,
    SpeechEnded,
    RecognitionFinished,
    SynthesisStarted,
    PlaybackStarted,
    PlaybackFinished,
    Interrupt,
    StopListening,
    Failed,
    Recovered,
    /// Timed states settling back to Idle.
    Settle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    pub from: VoiceState,
    pub to: VoiceState,
    pub reason: StateReason,
}

#[derive(Debug)]
pub struct StateMachine {
    current: VoiceState,
    since: Instant,
    /// Small ring of recent transitions, for `--doctor` and bug reports.
    history: Vec<(VoiceState, StateReason)>,
    history_limit: usize,
    transitions: u64,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            current: VoiceState::Idle,
            since: Instant::now(),
            history: Vec::new(),
            history_limit: 32,
            transitions: 0,
        }
    }

    pub fn current(&self) -> VoiceState {
        self.current
    }

    pub fn time_in_state(&self) -> Duration {
        self.since.elapsed()
    }

    pub fn transition_count(&self) -> u64 {
        self.transitions
    }

    pub fn history(&self) -> &[(VoiceState, StateReason)] {
        &self.history
    }

    /// Where a trigger would take us, or `None` if it is not legal here.
    ///
    /// The table is intentionally written out rather than derived: every entry
    /// is a product decision about what the orb should do.
    pub fn resolve(&self, trigger: Trigger) -> Option<(VoiceState, StateReason)> {
        use StateReason as R;
        use Trigger as T;
        use VoiceState as S;

        let next = match (self.current, trigger) {
            (S::Idle, T::Boot) => (S::Idle, R::Startup),

            // Listening can be entered from anywhere that isn't already it —
            // including mid-speech, which is barge-in.
            (S::Idle, T::StartListening) => (S::Listening, R::ApiRequest),
            (S::Error, T::StartListening) => (S::Listening, R::ApiRequest),
            (S::Interrupted, T::StartListening) => (S::Listening, R::ApiRequest),
            (S::Processing, T::StartListening) => (S::Listening, R::ApiRequest),
            (S::Speaking, T::StartListening) => (S::Listening, R::BargeIn),

            (S::Listening, T::SpeechStarted) => (S::Listening, R::VoiceActivity),
            (S::Idle, T::SpeechStarted) => (S::Listening, R::VoiceActivity),
            (S::Speaking, T::SpeechStarted) => (S::Listening, R::BargeIn),

            (S::Listening, T::SpeechEnded) => (S::Processing, R::SilenceTimeout),
            (S::Listening, T::StopListening) => (S::Idle, R::ApiRequest),

            (S::Processing, T::RecognitionFinished) => (S::Idle, R::RecognitionComplete),
            // A recogniser result that arrives while we are already speaking
            // must not drag us out of Speaking.
            (S::Speaking, T::RecognitionFinished) => return None,

            (S::Idle, T::SynthesisStarted) => (S::Processing, R::ApiRequest),
            (S::Listening, T::SynthesisStarted) => (S::Processing, R::ApiRequest),
            (S::Processing, T::SynthesisStarted) => (S::Processing, R::ApiRequest),
            (S::Error, T::SynthesisStarted) => (S::Processing, R::ApiRequest),
            (S::Interrupted, T::SynthesisStarted) => (S::Processing, R::ApiRequest),
            // Already speaking and more synthesis starts: that's a queued
            // chunk of the same reply, not a state change.
            (S::Speaking, T::SynthesisStarted) => return None,

            (S::Processing, T::PlaybackStarted) => (S::Speaking, R::SynthesisReady),
            (S::Idle, T::PlaybackStarted) => (S::Speaking, R::SynthesisReady),
            (S::Speaking, T::PlaybackStarted) => return None,

            (S::Speaking, T::PlaybackFinished) => (S::Idle, R::PlaybackComplete),
            (S::Processing, T::PlaybackFinished) => (S::Idle, R::PlaybackComplete),

            (S::Speaking, T::Interrupt) => (S::Interrupted, R::UserInterrupt),
            (S::Processing, T::Interrupt) => (S::Interrupted, R::UserInterrupt),
            (S::Listening, T::Interrupt) => (S::Idle, R::UserInterrupt),
            // Interrupting silence is a no-op, not an error.
            (S::Idle, T::Interrupt) => return None,
            (S::Interrupted, T::Interrupt) => return None,
            (S::Error, T::Interrupt) => (S::Idle, R::UserInterrupt),

            (S::Interrupted, T::Settle) => (S::Idle, R::Timeout),
            (S::Error, T::Settle) => (S::Idle, R::Timeout),

            (_, T::Failed) => (S::Error, R::Failure),
            (S::Error, T::Recovered) => (S::Idle, R::Timeout),

            _ => return None,
        };
        Some(next)
    }

    pub fn can(&self, trigger: Trigger) -> bool {
        self.resolve(trigger).is_some()
    }

    /// Apply a trigger. Returns the transition if one occurred; `None` means
    /// the trigger was ignored (either illegal or a no-op).
    pub fn apply(&mut self, trigger: Trigger) -> Option<Transition> {
        let (to, reason) = self.resolve(trigger)?;
        let from = self.current;
        if from == to && !matches!(trigger, Trigger::Boot) {
            // Same-state triggers (speech continuing while listening) are real
            // events but not transitions; don't spam the bus with them.
            return None;
        }
        self.current = to;
        self.since = Instant::now();
        self.transitions += 1;
        self.history.push((to, reason));
        if self.history.len() > self.history_limit {
            self.history.remove(0);
        }
        Some(Transition { from, to, reason })
    }

    /// Drive timed states (`Interrupted`, `Error`) back to `Idle`. Call from
    /// the engine tick.
    pub fn tick(&mut self) -> Option<Transition> {
        let timeout = self.current.auto_exit_after()?;
        if self.since.elapsed() >= timeout {
            self.apply(Trigger::Settle)
        } else {
            None
        }
    }

    /// Force a state. Only for shutdown paths and tests; bypasses the table on
    /// purpose and is deliberately awkward to reach for.
    pub fn force(&mut self, state: VoiceState, reason: StateReason) -> Transition {
        let from = self.current;
        self.current = state;
        self.since = Instant::now();
        self.transitions += 1;
        self.history.push((state, reason));
        Transition {
            from,
            to: state,
            reason,
        }
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(triggers: &[Trigger]) -> StateMachine {
        let mut m = StateMachine::new();
        for t in triggers {
            m.apply(*t);
        }
        m
    }

    #[test]
    fn happy_path_cycles_back_to_idle() {
        let m = run(&[
            Trigger::StartListening,
            Trigger::SpeechStarted,
            Trigger::SpeechEnded,
            Trigger::RecognitionFinished,
            Trigger::SynthesisStarted,
            Trigger::PlaybackStarted,
            Trigger::PlaybackFinished,
        ]);
        assert_eq!(m.current(), VoiceState::Idle);
    }

    #[test]
    fn each_stage_lands_where_expected() {
        let mut m = StateMachine::new();
        assert_eq!(m.current(), VoiceState::Idle);
        m.apply(Trigger::StartListening);
        assert_eq!(m.current(), VoiceState::Listening);
        m.apply(Trigger::SpeechEnded);
        assert_eq!(m.current(), VoiceState::Processing);
        m.apply(Trigger::SynthesisStarted);
        assert_eq!(m.current(), VoiceState::Processing);
        m.apply(Trigger::PlaybackStarted);
        assert_eq!(m.current(), VoiceState::Speaking);
    }

    #[test]
    fn barge_in_moves_speaking_to_listening() {
        let mut m = run(&[Trigger::SynthesisStarted, Trigger::PlaybackStarted]);
        assert_eq!(m.current(), VoiceState::Speaking);
        let t = m.apply(Trigger::SpeechStarted).expect("barge-in is legal");
        assert_eq!(t.to, VoiceState::Listening);
        assert_eq!(t.reason, StateReason::BargeIn);
    }

    #[test]
    fn interrupting_speech_passes_through_interrupted() {
        let mut m = run(&[Trigger::SynthesisStarted, Trigger::PlaybackStarted]);
        let t = m.apply(Trigger::Interrupt).unwrap();
        assert_eq!(t.to, VoiceState::Interrupted);
        assert_eq!(t.reason, StateReason::UserInterrupt);
    }

    #[test]
    fn interrupting_idle_is_a_no_op_not_an_error() {
        let mut m = StateMachine::new();
        assert!(m.apply(Trigger::Interrupt).is_none());
        assert_eq!(m.current(), VoiceState::Idle);
        assert_eq!(m.transition_count(), 0);
    }

    #[test]
    fn recognition_result_cannot_yank_us_out_of_speaking() {
        let mut m = run(&[Trigger::SynthesisStarted, Trigger::PlaybackStarted]);
        assert!(m.apply(Trigger::RecognitionFinished).is_none());
        assert_eq!(m.current(), VoiceState::Speaking);
    }

    #[test]
    fn failure_is_reachable_from_every_state() {
        for start in [
            VoiceState::Idle,
            VoiceState::Listening,
            VoiceState::Processing,
            VoiceState::Speaking,
            VoiceState::Interrupted,
        ] {
            let mut m = StateMachine::new();
            m.force(start, StateReason::Startup);
            let t = m.apply(Trigger::Failed).expect("failure must always apply");
            assert_eq!(t.to, VoiceState::Error);
        }
    }

    #[test]
    fn interrupted_settles_back_to_idle() {
        let mut m = run(&[
            Trigger::SynthesisStarted,
            Trigger::PlaybackStarted,
            Trigger::Interrupt,
        ]);
        assert_eq!(m.current(), VoiceState::Interrupted);
        assert!(m.apply(Trigger::Settle).is_some());
        assert_eq!(m.current(), VoiceState::Idle);
    }

    #[test]
    fn no_trigger_sequence_can_reach_an_undefined_state() {
        // Exhaustive breadth-first walk: every reachable state must be one of
        // the six, and no transition may report from == to.
        let triggers = [
            Trigger::Boot,
            Trigger::StartListening,
            Trigger::SpeechStarted,
            Trigger::SpeechEnded,
            Trigger::RecognitionFinished,
            Trigger::SynthesisStarted,
            Trigger::PlaybackStarted,
            Trigger::PlaybackFinished,
            Trigger::Interrupt,
            Trigger::StopListening,
            Trigger::Failed,
            Trigger::Recovered,
            Trigger::Settle,
        ];
        let mut seen = std::collections::HashSet::new();
        let mut queue = vec![VoiceState::Idle];
        while let Some(state) = queue.pop() {
            if !seen.insert(state) {
                continue;
            }
            for t in triggers {
                // `Boot` is the one deliberate self-transition: it exists so
                // startup can announce the initial state, and announcing
                // Idle *from* Idle is exactly what it is for.
                if matches!(t, Trigger::Boot) {
                    continue;
                }
                let mut m = StateMachine::new();
                m.force(state, StateReason::Startup);
                if let Some(tr) = m.apply(t) {
                    assert_ne!(
                        tr.from, tr.to,
                        "{t:?} from {state} produced a self-transition"
                    );
                    queue.push(tr.to);
                }
            }
        }
        assert_eq!(seen.len(), 6, "reached {seen:?}");
    }

    #[test]
    fn history_is_bounded() {
        let mut m = StateMachine::new();
        for _ in 0..200 {
            m.apply(Trigger::StartListening);
            m.apply(Trigger::StopListening);
        }
        assert!(m.history().len() <= 32);
    }
}
