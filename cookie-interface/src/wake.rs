//! Waiting to be called.
//!
//! The microphone is open all the time. That is not the same as Cookie
//! listening to you, and the difference is the whole point of this module:
//! until you say her name, everything heard is transcribed, checked for the
//! name, and thrown away. Nothing is emitted, nothing reaches the backend,
//! and the orb does not appear.
//!
//! Once called, she stays attentive for a while, so a conversation does not
//! require saying "Cookie" before every sentence. The attention lapses on its
//! own, which is the property that makes an always-open microphone tolerable:
//! forgetting to dismiss her is the normal case, so forgetting must be safe.
//!
//! Matching is fuzzy for the same reason the intent engine is: recognition
//! renders the name as "cookie", "cook he", "cooky" and "quickie" depending
//! on the room, and a wake word that only works when enunciated is a wake
//! word people stop using.

use std::time::{Duration, Instant};

use crate::config::WakeConfig;

/// What to do with something that was heard.
#[derive(Debug, Clone, PartialEq)]
pub enum Heard {
    /// Nobody called her. Discard it silently.
    Ignore,
    /// Called, with nothing else said — "Cookie?" on its own.
    Woken,
    /// Act on this text. Either she was already attentive, or the name and
    /// the request arrived together and this is what was left after it.
    Act(String),
}

/// Tracks whether Cookie is being spoken to.
#[derive(Debug)]
pub struct WakeGate {
    config: WakeConfig,
    /// When attention lapses. `None` means asleep.
    until: Option<Instant>,
}

impl WakeGate {
    pub fn new(config: WakeConfig) -> Self {
        Self {
            config,
            until: None,
        }
    }

    /// True while Cookie is listening *to you* rather than merely listening.
    pub fn is_awake(&self) -> bool {
        match self.until {
            Some(until) => Instant::now() < until,
            None => false,
        }
    }

    /// Call when attention should lapse on its own. Returns true if it just
    /// did, so the caller can announce it once rather than every tick.
    pub fn tick(&mut self) -> bool {
        if self.until.is_some() && !self.is_awake() {
            self.until = None;
            return true;
        }
        false
    }

    /// Wake without being called — used by the API, and by `--test`.
    pub fn wake(&mut self) {
        self.until = Some(Instant::now() + Duration::from_secs(self.config.attention_secs.max(1)));
    }

    pub fn sleep(&mut self) {
        self.until = None;
    }

    /// Decide what to do with a finished transcript.
    pub fn consider(&mut self, transcript: &str) -> Heard {
        if !self.config.enabled {
            return Heard::Act(transcript.to_string());
        }

        match strip_wake_word(transcript, &self.config.word) {
            Some(rest) => {
                self.wake();
                if rest.is_empty() {
                    Heard::Woken
                } else {
                    Heard::Act(rest)
                }
            }
            None if self.is_awake() => {
                // Already in a conversation: every reply extends it, so you
                // do not have to keep saying her name.
                self.wake();
                Heard::Act(transcript.to_string())
            }
            None => Heard::Ignore,
        }
    }
}

/// Find the wake word and return whatever was said after it.
///
/// Returns `None` when the name is not there. The search covers the first few
/// words only: "the cookie recipe is on the counter" is somebody talking
/// about biscuits, not to an assistant, and treating it as a summons is how
/// an always-on microphone becomes a nuisance.
pub fn strip_wake_word(transcript: &str, wake_word: &str) -> Option<String> {
    let wake_word = wake_word.trim().to_lowercase();
    if wake_word.is_empty() {
        return None;
    }
    let words: Vec<&str> = transcript.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    // "Hey Cookie", "okay Cookie", "so, Cookie —" all put the name a word or
    // two in.
    let horizon = words.len().min(3);
    for (index, word) in words.iter().take(horizon).enumerate() {
        if !sounds_like(word, &wake_word) {
            continue;
        }
        let rest = words[index + 1..].join(" ");
        return Some(tidy(&rest));
    }
    None
}

/// Whether a heard word is plausibly the wake word.
///
/// Punctuation is stripped first because recognition adds it freely, and the
/// comparison tolerates one edit, because "cooky", "cook he" and "cookey" are
/// all what a microphone in a kitchen makes of "Cookie".
fn sounds_like(heard: &str, wake_word: &str) -> bool {
    let heard: String = heard
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    if heard == wake_word {
        return true;
    }
    if heard.len() + 2 < wake_word.len() || heard.len() > wake_word.len() + 2 {
        return false;
    }
    // Two edits for a name of six letters or more, one for a short one. Any
    // more and ordinary words start waking her; any less and "cooky" — two
    // edits from "cookie", and what a kitchen microphone routinely produces —
    // is missed.
    let allowed = if wake_word.chars().count() >= 6 { 2 } else { 1 };
    edit_distance(&heard, wake_word) <= allowed
}

/// Levenshtein distance, bounded by the lengths involved (a wake word is one
/// short token, so the quadratic cost is a few dozen operations).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            current[j + 1] = (previous[j] + cost)
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// Clean up what is left after the name: leading commas, a stray "please".
fn tidy(rest: &str) -> String {
    rest.trim()
        .trim_start_matches([',', '.', '!', '?', '-', '—', ':'])
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> WakeGate {
        WakeGate::new(WakeConfig::default())
    }

    #[test]
    fn nothing_happens_until_she_is_called() {
        let mut gate = gate();
        assert_eq!(gate.consider("what time is it"), Heard::Ignore);
        assert_eq!(
            gate.consider("I was telling him about the meeting"),
            Heard::Ignore
        );
        assert!(!gate.is_awake());
    }

    #[test]
    fn the_name_wakes_her_and_the_rest_is_the_request() {
        let mut gate = gate();
        assert_eq!(
            gate.consider("Cookie, what time is it"),
            Heard::Act("what time is it".into())
        );
        assert!(gate.is_awake());
    }

    #[test]
    fn the_name_on_its_own_is_just_an_answer() {
        let mut gate = gate();
        assert_eq!(gate.consider("Cookie?"), Heard::Woken);
        assert!(gate.is_awake());
    }

    #[test]
    fn the_name_is_found_a_word_or_two_in() {
        for phrase in [
            "Hey Cookie, open the project",
            "okay Cookie open the project",
        ] {
            let mut gate = gate();
            assert_eq!(gate.consider(phrase), Heard::Act("open the project".into()));
        }
    }

    #[test]
    fn recognition_errors_on_the_name_still_wake_her() {
        // What a laptop microphone in a real room makes of "Cookie".
        for heard in ["cooky", "Cookie.", "COOKIE", "cookies"] {
            let mut gate = gate();
            assert!(
                matches!(
                    gate.consider(&format!("{heard} what time is it")),
                    Heard::Act(_)
                ),
                "{heard} should have woken her"
            );
        }
    }

    #[test]
    fn ordinary_words_do_not_wake_her() {
        // The tolerance has to stop somewhere short of "everything".
        for word in ["coffee", "look", "could", "book", "okay", "kitchen"] {
            let mut gate = gate();
            assert_eq!(
                gate.consider(&format!("{word} is what I meant")),
                Heard::Ignore,
                "{word} woke her"
            );
        }
    }

    #[test]
    fn talking_about_biscuits_is_not_a_summons() {
        let mut gate = gate();
        // The name late in a sentence is somebody talking *about* something.
        assert_eq!(
            gate.consider("I left the last cookie on the counter"),
            Heard::Ignore
        );
    }

    #[test]
    fn a_conversation_does_not_need_her_name_every_time() {
        let mut gate = gate();
        gate.consider("Cookie, open the project");
        assert_eq!(
            gate.consider("now run the tests"),
            Heard::Act("now run the tests".into())
        );
    }

    #[test]
    fn attention_lapses_on_its_own() {
        let mut gate = WakeGate::new(WakeConfig {
            attention_secs: 1,
            ..Default::default()
        });
        gate.wake();
        assert!(gate.is_awake());
        // Forgetting to dismiss her is the normal case, so it has to be safe.
        gate.until = Some(Instant::now() - Duration::from_millis(1));
        assert!(gate.tick(), "the lapse is announced once");
        assert!(!gate.tick(), "and not repeatedly");
        assert_eq!(gate.consider("what time is it"), Heard::Ignore);
    }

    #[test]
    fn disabling_the_gate_lets_everything_through() {
        let mut gate = WakeGate::new(WakeConfig {
            enabled: false,
            ..Default::default()
        });
        assert_eq!(
            gate.consider("what time is it"),
            Heard::Act("what time is it".into())
        );
    }

    #[test]
    fn a_different_name_can_be_configured() {
        let mut gate = WakeGate::new(WakeConfig {
            word: "jarvis".into(),
            ..Default::default()
        });
        assert_eq!(gate.consider("Cookie, hello"), Heard::Ignore);
        assert_eq!(gate.consider("Jarvis, hello"), Heard::Act("hello".into()));
    }
}
