//! Working out what a spoken line is *for*, without a language model.
//!
//! Almost everything you say to Cookie belongs to the backend: that is where
//! the intelligence lives, and this crate has no business guessing at it. But
//! a handful of requests are about the interface itself, and three of them
//! cannot wait for a round trip:
//!
//! * **"Cookie, are you alright?"** — a health question. If the backend is
//!   down, the backend obviously cannot answer it. This is precisely the
//!   moment the interface has to speak for itself.
//! * **"Stop."** — stop *talking*. Barge-in is a reflex; sending it over the
//!   network and waiting is the difference between an assistant that listens
//!   and one that talks over you.
//! * **"Cancel that."** — abandon the work in progress. Deliberately a
//!   different intent from the one above: interrupting speech must never
//!   cancel a task, and cancelling a task must be something you asked for.
//!
//! ## This is inference, not a command list
//!
//! There is no table of exact strings to memorise. Matching is fuzzy on
//! purpose — speech recognition mangles words, people phrase things
//! differently every time, and "are you alright", "you feeling okay?", "run
//! full diagnostics" and "is everything still working" are the same question.
//! Scoring combines phrase similarity (character bigram overlap, which
//! survives transcription errors) with keyword evidence, and anything that
//! does not clear a confident margin is handed to the backend untouched.
//!
//! ## The authority is the backend
//!
//! When a backend is connected the bar is high: the interface only claims an
//! utterance it is nearly certain about, because the backend can do better.
//! With no backend reachable the bar drops, since otherwise nothing would
//! answer at all. [`IntentEngine::infer`] takes that context as an argument
//! rather than deciding on its own.

use crate::state::VoiceState;

/// Something the interface can act on by itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// "Are you alright?", "run diagnostics", "is everything working?"
    Diagnostics,
    /// "Stop", "quiet", "enough" — stop speaking, nothing more.
    StopSpeaking,
    /// "Cancel that", "forget what you're doing", "abort".
    CancelTask,
    /// "Go to sleep", "stop listening", "that'll be all".
    Sleep,
    /// "Wake up", "are you there?", "Cookie?"
    Wake,
    /// "What are you working on?", "show me what you're doing."
    ShowActivity,
}

impl Intent {
    /// Stable identifier used in events and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Intent::Diagnostics => "diagnostics",
            Intent::StopSpeaking => "stop_speaking",
            Intent::CancelTask => "cancel_task",
            Intent::Sleep => "sleep",
            Intent::Wake => "wake",
            Intent::ShowActivity => "show_activity",
        }
    }

    /// Whether acting on this locally means the backend should *not* see the
    /// utterance. "Stop talking" is ours alone; a diagnostics question is
    /// also worth telling the backend about, since it may know more.
    pub fn consumes_utterance(self) -> bool {
        !matches!(self, Intent::Diagnostics)
    }
}

/// A match, with the confidence that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntentMatch {
    pub intent: Intent,
    /// Roughly 0..=1. Compare against the thresholds in [`IntentEngine`].
    pub score: f32,
}

/// One inferrable intent: a few ways of saying it, plus the words that carry
/// the meaning.
struct Pattern {
    intent: Intent,
    /// Example phrasings. These are *examples*, not a whitelist — similarity
    /// against them is one input to the score.
    phrases: &'static [&'static str],
    /// Words that individually suggest this intent.
    keywords: &'static [&'static str],
    /// Words that must not be present (they indicate a different intent).
    blockers: &'static [&'static str],
    /// Word pairs that together carry the meaning even though neither word
    /// does alone. "Stop" is ambiguous and "doing" is meaningless, but
    /// "stop … doing" is unmistakably a request to abandon the work.
    pairs: &'static [(&'static str, &'static str)],
    /// Utterances longer than this many words are almost certainly a real
    /// request for the backend rather than a control phrase.
    max_words: usize,
}

const PATTERNS: &[Pattern] = &[
    Pattern {
        intent: Intent::Diagnostics,
        phrases: &[
            "are you alright",
            "are you okay",
            "are you feeling okay",
            "is everything working",
            "is everything alright",
            "run diagnostics",
            "run a full diagnostic",
            "run full diagnostics",
            "system check",
            "self test",
            "check yourself",
            "check your systems",
            "status report",
            "how are you doing",
            "is anything broken",
            "what is broken",
            "can you hear me",
            "is your microphone working",
            "are all your systems working",
            "diagnostics",
        ],
        keywords: &[
            "diagnostic",
            "diagnostics",
            "alright",
            "okay",
            "ok",
            "broken",
            "working",
            "health",
            "status",
            "systems",
            "self-test",
        ],
        blockers: &["weather", "project", "file", "repository", "code"],
        pairs: &[
            ("everything", "working"),
            ("all", "working"),
            ("hear", "me"),
        ],
        max_words: 9,
    },
    Pattern {
        intent: Intent::StopSpeaking,
        phrases: &[
            "stop",
            "stop talking",
            "be quiet",
            "quiet",
            "hush",
            "shush",
            "enough",
            "that is enough",
            "okay stop",
            "alright stop",
            "shut up",
            "pause",
            "wait",
            "hold on",
        ],
        keywords: &["stop", "quiet", "hush", "shush", "enough", "pause", "wait"],
        // If they named the work, they mean cancel, not "stop talking".
        blockers: &[
            "cancel",
            "task",
            "job",
            "working",
            "doing",
            "abort",
            "forget",
            "listening",
        ],
        pairs: &[],
        max_words: 4,
    },
    Pattern {
        intent: Intent::CancelTask,
        phrases: &[
            "cancel that",
            "cancel it",
            "cancel the task",
            "cancel what you are doing",
            "stop what you are doing",
            "stop working on that",
            "abort",
            "abort that",
            "forget it",
            "forget about it",
            "never mind",
            "drop it",
            "give up on that",
        ],
        keywords: &["cancel", "abort", "forget", "nevermind", "drop", "scrap"],
        blockers: &["listening"],
        pairs: &[
            ("stop", "doing"),
            ("stop", "working"),
            ("stop", "task"),
            ("stop", "that"),
            ("give", "up"),
            ("never", "mind"),
        ],
        max_words: 8,
    },
    Pattern {
        intent: Intent::Sleep,
        phrases: &[
            "go to sleep",
            "stop listening",
            "go away",
            "that will be all",
            "thats all for now",
            "dismiss",
            "you can go",
            "leave me alone",
            "standby",
        ],
        keywords: &["sleep", "standby", "dismiss", "away"],
        blockers: &["wake", "up"],
        pairs: &[("stop", "listening"), ("thats", "all")],
        max_words: 7,
    },
    Pattern {
        intent: Intent::Wake,
        phrases: &[
            "wake up",
            "cookie",
            "are you there",
            "hey cookie",
            "cookie are you there",
            "start listening",
            "listen to me",
        ],
        keywords: &["wake", "cookie", "there", "listen"],
        blockers: &["sleep", "stop", "alright", "okay", "diagnostics"],
        pairs: &[("wake", "up"), ("start", "listening")],
        max_words: 5,
    },
    Pattern {
        intent: Intent::ShowActivity,
        phrases: &[
            "what are you doing",
            "what are you working on",
            "show me what you are working on",
            "show me what you are doing",
            "show your activity",
            "what is happening",
            "whats going on",
            "show me the log",
        ],
        keywords: &[
            "doing",
            "working",
            "activity",
            "progress",
            "log",
            "happening",
        ],
        blockers: &["stop", "cancel"],
        pairs: &[("what", "doing"), ("what", "working"), ("show", "doing")],
        max_words: 9,
    },
];

/// Fuzzy intent inference over a small, fixed set of interface commands.
#[derive(Debug, Clone)]
pub struct IntentEngine {
    /// Score needed when a backend is available to answer instead.
    pub backend_threshold: f32,
    /// Score needed when nothing else could possibly answer.
    pub standalone_threshold: f32,
}

impl Default for IntentEngine {
    fn default() -> Self {
        Self {
            // Tuned so that clear control phrases land and ordinary requests
            // ("open the project I was working on") never do.
            backend_threshold: 0.62,
            standalone_threshold: 0.48,
        }
    }
}

impl IntentEngine {
    /// Best guess at what the user meant, or `None` to let the backend decide.
    ///
    /// `state` disambiguates the genuinely ambiguous cases: a bare "stop"
    /// while Cookie is talking means *stop talking*, and the same word while
    /// she is working means cancel the work.
    pub fn infer(
        &self,
        transcript: &str,
        state: VoiceState,
        backend_available: bool,
    ) -> Option<IntentMatch> {
        let normalised = normalise(transcript);
        if normalised.is_empty() {
            return None;
        }
        let words: Vec<&str> = normalised.split(' ').filter(|w| !w.is_empty()).collect();

        let mut best: Option<IntentMatch> = None;
        for pattern in PATTERNS {
            if words.len() > pattern.max_words {
                continue;
            }
            if pattern
                .blockers
                .iter()
                .any(|b| words.iter().any(|w| w == b))
            {
                continue;
            }
            let similarity = pattern
                .phrases
                .iter()
                .map(|phrase| similarity(&normalised, phrase))
                .fold(0.0f32, f32::max);
            let keyword_hits = pattern
                .keywords
                .iter()
                .filter(|k| words.iter().any(|w| w == *k || fuzzy_word(w, k)))
                .count();
            let has_pair = pattern
                .pairs
                .iter()
                .any(|(a, b)| words.iter().any(|w| w == a) && words.iter().any(|w| w == b));
            if similarity < 0.3 && keyword_hits == 0 && !has_pair {
                continue;
            }
            // Similarity carries most of the weight; keywords rescue phrasings
            // nobody thought to list.
            let pair_hit = pattern
                .pairs
                .iter()
                .any(|(a, b)| words.iter().any(|w| w == a) && words.iter().any(|w| w == b));
            let keyword_score = (keyword_hits as f32 / 2.0).min(1.0);
            let mut score = similarity * 0.72 + keyword_score * 0.38;
            if pair_hit {
                score += 0.35;
            }
            // Short utterances are much more likely to be control phrases.
            if words.len() <= 3 {
                score += 0.06;
            }
            score = score.min(1.0);
            if best.map(|b| score > b.score).unwrap_or(true) {
                best = Some(IntentMatch {
                    intent: pattern.intent,
                    score,
                });
            }
        }

        let mut candidate = best?;
        let threshold = if backend_available {
            self.backend_threshold
        } else {
            self.standalone_threshold
        };
        if candidate.score < threshold {
            return None;
        }

        // Context resolves the one real ambiguity in the set.
        candidate.intent = match (candidate.intent, state) {
            (Intent::StopSpeaking, VoiceState::Processing) => Intent::CancelTask,
            (Intent::StopSpeaking, VoiceState::Idle) => Intent::StopSpeaking,
            (other, _) => other,
        };
        Some(candidate)
    }
}

/// Lowercase, strip punctuation, collapse whitespace, drop filler.
fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else if ch.is_whitespace() || ch == '-' {
            out.push(' ');
        } else if ch == '\'' {
            // "you're" -> "youre", which matches the stored phrasings.
        } else {
            out.push(' ');
        }
    }
    let filler = [
        "cookie", "please", "hey", "um", "uh", "er", "like", "just", "can", "you", "could",
        "would", "is", "it", "the", "a", "my", "your",
    ];
    let words: Vec<&str> = out
        .split_whitespace()
        .filter(|w| !filler.contains(w))
        .collect();
    if words.is_empty() {
        // Everything was filler — "Cookie?" is itself meaningful.
        return out.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    words.join(" ")
}

/// Sørensen–Dice coefficient over character bigrams.
///
/// Chosen because speech recognition errors are usually one or two characters
/// ("diagnostics" → "diagnostic", "alright" → "all right"), which bigram
/// overlap shrugs off and exact matching does not.
fn similarity(a: &str, b: &str) -> f32 {
    let b = normalise(b);
    if a == b {
        return 1.0;
    }
    let (x, y) = (bigrams(a), bigrams(&b));
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let mut shared = 0usize;
    let mut used = vec![false; y.len()];
    for bigram in &x {
        for (i, other) in y.iter().enumerate() {
            if !used[i] && bigram == other {
                used[i] = true;
                shared += 1;
                break;
            }
        }
    }
    (2.0 * shared as f32) / (x.len() + y.len()) as f32
}

fn bigrams(text: &str) -> Vec<[char; 2]> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    chars.windows(2).map(|w| [w[0], w[1]]).collect()
}

/// Cheap single-edit tolerance for individual words.
fn fuzzy_word(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.len() < 4 || b.len() < 4 {
        return false;
    }
    if a.len().abs_diff(b.len()) > 2 {
        return false;
    }
    // Prefix agreement catches the common tail errors: diagnostic(s),
    // work/working, cancel/cancelled.
    let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    shared >= a.len().min(b.len()).saturating_sub(2) && shared >= 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer(text: &str) -> Option<Intent> {
        IntentEngine::default()
            .infer(text, VoiceState::Idle, true)
            .map(|m| m.intent)
    }

    fn infer_in(text: &str, state: VoiceState) -> Option<Intent> {
        IntentEngine::default()
            .infer(text, state, true)
            .map(|m| m.intent)
    }

    #[test]
    fn health_questions_are_recognised_in_many_phrasings() {
        for phrase in [
            "Cookie, are you alright?",
            "are you okay?",
            "Cookie are you feeling okay",
            "run full diagnostics",
            "run a diagnostic",
            "is everything working",
            "can you hear me",
            "is anything broken",
            "system check please",
        ] {
            assert_eq!(infer(phrase), Some(Intent::Diagnostics), "{phrase}");
        }
    }

    #[test]
    fn transcription_errors_still_match() {
        // What Whisper actually does to these on a bad microphone.
        assert_eq!(infer("are you all right"), Some(Intent::Diagnostics));
        assert_eq!(infer("run full diagnostic"), Some(Intent::Diagnostics));
        assert_eq!(infer("is every thing working"), Some(Intent::Diagnostics));
    }

    #[test]
    fn stopping_speech_is_not_cancelling_work() {
        assert_eq!(infer("stop"), Some(Intent::StopSpeaking));
        assert_eq!(infer("be quiet"), Some(Intent::StopSpeaking));
        assert_eq!(infer("that's enough"), Some(Intent::StopSpeaking));
        assert_eq!(infer("cancel that"), Some(Intent::CancelTask));
        assert_eq!(infer("stop what you're doing"), Some(Intent::CancelTask));
        assert_eq!(infer("forget it"), Some(Intent::CancelTask));
    }

    #[test]
    fn a_bare_stop_means_cancel_only_while_working() {
        assert_eq!(
            infer_in("stop", VoiceState::Speaking),
            Some(Intent::StopSpeaking)
        );
        assert_eq!(
            infer_in("stop", VoiceState::Processing),
            Some(Intent::CancelTask)
        );
    }

    #[test]
    fn ordinary_requests_are_left_for_the_backend() {
        for phrase in [
            "open the project I was working on yesterday",
            "what's the weather like in Bengaluru",
            "create a GitHub repository for this",
            "find the file I saved last week",
            "fix the failing tests in the auth module",
            "research how this error is normally fixed",
        ] {
            assert_eq!(infer(phrase), None, "{phrase} was claimed locally");
        }
    }

    #[test]
    fn the_bar_is_lower_with_no_backend_to_defer_to() {
        let engine = IntentEngine::default();
        let borderline = "everything working";
        let with_backend = engine.infer(borderline, VoiceState::Idle, true);
        let alone = engine.infer(borderline, VoiceState::Idle, false);
        assert!(alone.is_some());
        assert!(with_backend.is_none() || with_backend.unwrap().score >= engine.backend_threshold);
    }

    #[test]
    fn activity_and_sleep_are_distinguished() {
        assert_eq!(infer("what are you working on"), Some(Intent::ShowActivity));
        assert_eq!(
            infer("show me what you're doing"),
            Some(Intent::ShowActivity)
        );
        assert_eq!(infer("go to sleep"), Some(Intent::Sleep));
        assert_eq!(infer("stop listening"), Some(Intent::Sleep));
    }

    #[test]
    fn diagnostics_still_reaches_the_backend_but_stop_does_not() {
        assert!(!Intent::Diagnostics.consumes_utterance());
        assert!(Intent::StopSpeaking.consumes_utterance());
        assert!(Intent::CancelTask.consumes_utterance());
    }

    #[test]
    fn empty_and_noise_transcripts_infer_nothing() {
        assert!(infer("").is_none());
        assert!(infer("   ...  ").is_none());
        assert!(infer("mmm hmm yeah so anyway").is_none());
    }
}
