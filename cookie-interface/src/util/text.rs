//! Text helpers used by the streaming speech path.
//!
//! When a caller streams text into `/v1/speak/stream` one token at a time we
//! cannot wait for the whole message before synthesising — that would defeat
//! the point of streaming — but we also cannot hand single tokens to a TTS
//! engine, because prosody needs at least a clause to work with. The chunker
//! below sits in the middle: it emits a chunk as soon as one is *speakable*.

/// Accumulates streamed text and yields speakable chunks.
#[derive(Debug, Default)]
pub struct SentenceChunker {
    buffer: String,
    /// Emit early once the buffer passes this length even without punctuation,
    /// so a caller that streams a long unpunctuated run still gets audio.
    soft_limit: usize,
    /// Never hold back more than this; protects against pathological input.
    hard_limit: usize,
    /// Don't emit a chunk shorter than this unless flushing, otherwise
    /// "Yes." becomes its own audio file with a full engine warm-up.
    min_chunk: usize,
}

impl SentenceChunker {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            soft_limit: 180,
            hard_limit: 600,
            min_chunk: 12,
        }
    }

    pub fn with_limits(min_chunk: usize, soft_limit: usize, hard_limit: usize) -> Self {
        Self {
            buffer: String::new(),
            soft_limit,
            hard_limit,
            min_chunk,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.trim().is_empty()
    }

    pub fn pending(&self) -> &str {
        &self.buffer
    }

    /// Feed a delta. Returns zero or more chunks that are ready to speak.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut out = Vec::new();
        while let Some(chunk) = self.take_ready() {
            out.push(chunk);
        }
        out
    }

    /// Emit whatever is left, regardless of punctuation. Call at end of stream.
    pub fn flush(&mut self) -> Option<String> {
        let rest = self.buffer.trim().to_string();
        self.buffer.clear();
        if rest.is_empty() {
            None
        } else {
            Some(rest)
        }
    }

    fn take_ready(&mut self) -> Option<String> {
        let len = self.buffer.chars().count();

        if let Some(idx) = self.sentence_boundary() {
            if idx >= self.min_chunk || len >= self.soft_limit {
                return Some(self.split_at(idx));
            }
        }

        if len >= self.soft_limit {
            if let Some(idx) = self.clause_boundary().or_else(|| self.word_boundary()) {
                return Some(self.split_at(idx));
            }
        }

        if len >= self.hard_limit {
            let idx = self.buffer.len();
            return Some(self.split_at(idx));
        }

        None
    }

    /// Byte index just past a sentence terminator, skipping decimals,
    /// abbreviations and ellipses that are not really sentence ends.
    fn sentence_boundary(&self) -> Option<usize> {
        let bytes: Vec<(usize, char)> = self.buffer.char_indices().collect();
        for (pos, (idx, ch)) in bytes.iter().enumerate() {
            if !matches!(ch, '.' | '!' | '?' | '…' | '。' | '！' | '？') {
                continue;
            }
            let next = bytes.get(pos + 1).map(|(_, c)| *c);
            let prev = if pos > 0 {
                Some(bytes[pos - 1].1)
            } else {
                None
            };

            // "3.14" — a dot between digits is not a sentence end.
            if *ch == '.'
                && prev.map(|c| c.is_ascii_digit()).unwrap_or(false)
                && next.map(|c| c.is_ascii_digit()).unwrap_or(false)
            {
                continue;
            }
            // "Dr." / "e.g." — a dot right after a single letter usually isn't.
            if *ch == '.' && self.looks_like_abbreviation(pos, &bytes) {
                continue;
            }
            // Consume trailing quotes/brackets and the run of terminators.
            let mut end = idx + ch.len_utf8();
            let mut k = pos + 1;
            while let Some((i, c)) = bytes.get(k) {
                if matches!(c, '.' | '!' | '?' | '"' | '\'' | '”' | '’' | ')' | ']') {
                    end = i + c.len_utf8();
                    k += 1;
                } else {
                    break;
                }
            }
            // Require whitespace (or end of buffer) after, so "cookie.interface"
            // is not split mid-token while text is still arriving.
            match bytes.get(k) {
                None => return Some(end),
                Some((_, c)) if c.is_whitespace() => return Some(end),
                _ => continue,
            }
        }
        None
    }

    fn looks_like_abbreviation(&self, pos: usize, bytes: &[(usize, char)]) -> bool {
        // Single capital letter before the dot ("J. Smith"), or a known short form.
        if pos == 0 {
            return false;
        }
        let prev = bytes[pos - 1].1;
        if !prev.is_alphabetic() {
            return false;
        }
        let before = if pos >= 2 {
            Some(bytes[pos - 2].1)
        } else {
            None
        };
        if before.map(|c| !c.is_alphabetic()).unwrap_or(true) {
            return true;
        }
        let start = bytes[..pos]
            .iter()
            .rposition(|(_, c)| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        let word: String = bytes[start..=pos].iter().map(|(_, c)| *c).collect();
        const ABBREVIATIONS: &[&str] = &[
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.",
            "i.e.", "approx.", "no.", "fig.",
        ];
        ABBREVIATIONS.contains(&word.to_ascii_lowercase().as_str())
    }

    fn clause_boundary(&self) -> Option<usize> {
        let mut best = None;
        for (idx, ch) in self.buffer.char_indices() {
            if matches!(ch, ',' | ';' | ':' | '—' | '–') {
                best = Some(idx + ch.len_utf8());
            }
        }
        best
    }

    fn word_boundary(&self) -> Option<usize> {
        self.buffer
            .char_indices()
            .rfind(|(_, c)| c.is_whitespace())
            .map(|(i, c)| i + c.len_utf8())
    }

    fn split_at(&mut self, idx: usize) -> String {
        let idx = idx.min(self.buffer.len());
        let rest = self.buffer.split_off(idx);
        let chunk = std::mem::replace(&mut self.buffer, rest);
        chunk.trim().to_string()
    }
}

/// Pull a plausible first name out of a spoken reply.
///
/// Used only by `--test` ("My name is Alex" -> "Alex"). This is deliberately
/// dumb string handling, *not* natural language understanding: anything
/// resembling comprehension belongs in the Cookie backend, not here.
pub fn extract_name(transcript: &str) -> Option<String> {
    const LEAD_INS: &[&str] = &[
        "my name is",
        "my name's",
        "i am called",
        "i'm called",
        "you can call me",
        "call me",
        "this is",
        "i am",
        "i'm",
        "it's",
        "its",
        "name is",
    ];

    let cleaned = transcript
        .trim()
        .trim_end_matches(['.', '!', '?', ','])
        .to_string();
    let lower = cleaned.to_lowercase();

    let tail = LEAD_INS
        .iter()
        .filter_map(|lead| lower.find(lead).map(|at| at + lead.len()))
        .min()
        .map(|at| cleaned[at..].trim())
        .unwrap_or(cleaned.as_str());

    let candidate = tail
        .split_whitespace()
        .find(|w| w.chars().any(char::is_alphabetic))?;

    let name: String = candidate
        .chars()
        .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
        .collect();
    if name.is_empty() {
        return None;
    }

    let mut chars = name.chars();
    let first = chars.next()?.to_uppercase().collect::<String>();
    Some(format!("{first}{}", chars.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_emits_on_sentence_end() {
        let mut c = SentenceChunker::new();
        assert!(c.push("Good evening. ").len() == 1);
        assert!(c.push("How can ").is_empty());
        let out = c.push("I help you today? ");
        assert_eq!(out, vec!["How can I help you today?"]);
    }

    #[test]
    fn chunker_does_not_split_decimals_or_abbreviations() {
        let mut c = SentenceChunker::new();
        assert!(c.push("The value is 3.14159 approximately ").is_empty());
        let mut c2 = SentenceChunker::new();
        assert!(c2.push("Speaking with Dr. Chandra now ").is_empty());
    }

    #[test]
    fn chunker_emits_at_soft_limit_without_punctuation() {
        let mut c = SentenceChunker::with_limits(4, 40, 100);
        let out = c.push("this is a long run of words with no punctuation at all in it ");
        assert!(!out.is_empty(), "expected a soft-limit emission");
    }

    #[test]
    fn chunker_flushes_remainder() {
        let mut c = SentenceChunker::new();
        c.push("no terminator here");
        assert_eq!(c.flush().as_deref(), Some("no terminator here"));
        assert_eq!(c.flush(), None);
    }

    #[test]
    fn chunker_is_lossless() {
        let mut c = SentenceChunker::with_limits(2, 30, 60);
        let source = "Hello there. This is a test of the streaming chunker, which should not \
                      drop or duplicate any words! Right? Right.";
        let mut got = Vec::new();
        for ch in source.chars() {
            got.extend(c.push(&ch.to_string()));
        }
        got.extend(c.flush());
        let joined: String = got.join(" ");
        let a: Vec<&str> = source.split_whitespace().collect();
        let b: Vec<&str> = joined.split_whitespace().collect();
        assert_eq!(a, b);
    }

    #[test]
    fn name_extraction() {
        assert_eq!(extract_name("My name is Alex.").as_deref(), Some("Alex"));
        assert_eq!(extract_name("I'm Priya").as_deref(), Some("Priya"));
        assert_eq!(extract_name("Sam").as_deref(), Some("Sam"));
        assert_eq!(extract_name("you can call me Jo!").as_deref(), Some("Jo"));
        assert_eq!(extract_name("   ").as_deref(), None);
    }
}
