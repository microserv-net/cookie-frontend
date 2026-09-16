//! Provider-independent description of *how Cookie should sound*.
//!
//! The important property: a `VoiceSpec` is a request, not a command. Each
//! synthesiser reports which fields it honours (`SynthesisCapabilities`), and
//! the API never advertises a knob the active provider cannot actually turn —
//! a pitch slider that silently does nothing is worse than no slider.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Gender {
    Female,
    Male,
    Neutral,
    Unspecified,
}

/// The default target voice, per the brief: a warm, natural, sophisticated
/// British female voice — conversational rather than announcer-like.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceSpec {
    /// Provider-specific voice identifier. `"auto"` asks the provider to pick
    /// the closest match to `language` + `gender` + `style` on this machine,
    /// which is what keeps one config file working across all three platforms.
    pub id: String,
    /// BCP-47. Drives voice selection and, where supported, the accent.
    pub language: String,
    pub gender: Gender,
    /// Free-form style/emotion tag ("warm", "calm", "bright"). Passed through
    /// only to providers that advertise style support.
    pub style: Option<String>,
    /// Speaking rate multiplier, 1.0 = the provider's natural pace.
    pub rate: f32,
    /// Pitch offset in semitones, 0.0 = natural.
    pub pitch: f32,
    /// Linear gain applied to synthesised audio, 1.0 = unchanged.
    pub volume: f32,
    /// Provider-specific extras (speaker id, seed, emotion vector, ...).
    /// Anything a provider does not recognise is rejected loudly rather than
    /// silently dropped.
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for VoiceSpec {
    fn default() -> Self {
        Self {
            id: "auto".into(),
            language: "en-GB".into(),
            gender: Gender::Female,
            style: Some("warm".into()),
            rate: 1.0,
            pitch: 0.0,
            volume: 1.0,
            extra: BTreeMap::new(),
        }
    }
}

impl VoiceSpec {
    pub fn is_auto(&self) -> bool {
        self.id.eq_ignore_ascii_case("auto") || self.id.is_empty()
    }

    /// Apply a partial override from an API request. Only fields the caller
    /// actually sent are changed.
    pub fn overlaid(&self, patch: &VoicePatch) -> Self {
        let mut out = self.clone();
        if let Some(v) = &patch.voice {
            out.id = v.clone();
        }
        if let Some(v) = &patch.language {
            out.language = v.clone();
        }
        if let Some(v) = patch.gender {
            out.gender = v;
        }
        if let Some(v) = &patch.style {
            out.style = Some(v.clone());
        }
        if let Some(v) = patch.rate {
            out.rate = v;
        }
        if let Some(v) = patch.pitch {
            out.pitch = v;
        }
        if let Some(v) = patch.volume {
            out.volume = v;
        }
        for (k, v) in &patch.extra {
            out.extra.insert(k.clone(), v.clone());
        }
        out
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.25..=4.0).contains(&self.rate) {
            return Err(format!("rate {} outside 0.25..=4.0", self.rate));
        }
        if !(-12.0..=12.0).contains(&self.pitch) {
            return Err(format!("pitch {} outside -12..=12 semitones", self.pitch));
        }
        if !(0.0..=4.0).contains(&self.volume) {
            return Err(format!("volume {} outside 0.0..=4.0", self.volume));
        }
        if self.language.len() > 32 {
            return Err("language tag is implausibly long".into());
        }
        if self.id.len() > 128 {
            return Err("voice id is implausibly long".into());
        }
        Ok(())
    }

    /// Two-letter primary language subtag, lowercased.
    pub fn primary_language(&self) -> String {
        self.language
            .split(['-', '_'])
            .next()
            .unwrap_or("en")
            .to_lowercase()
    }

    /// Region subtag if present, uppercased ("GB").
    pub fn region(&self) -> Option<String> {
        self.language
            .split(['-', '_'])
            .nth(1)
            .map(|s| s.to_uppercase())
    }
}

/// A sparse patch coming from `POST /v1/speak`. Absent fields mean "leave the
/// configured value alone", which is different from "reset to default".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VoicePatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gender: Option<Gender>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f32>,
    /// Provider-specific extras. Validated against the active provider's
    /// declared capabilities before use.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn default_voice_is_british_female() {
        let v = VoiceSpec::default();
        assert_eq!(v.language, "en-GB");
        assert_eq!(v.gender, Gender::Female);
        assert_eq!(v.primary_language(), "en");
        assert_eq!(v.region().as_deref(), Some("GB"));
        assert!(v.is_auto());
        v.validate().unwrap();
    }

    #[test]
    fn patch_only_touches_supplied_fields() {
        let base = VoiceSpec::default();
        let patch = VoicePatch {
            rate: Some(1.1),
            ..Default::default()
        };
        let out = base.overlaid(&patch);
        assert_eq!(out.rate, 1.1);
        assert_eq!(out.language, base.language);
        assert_eq!(out.style, base.style);
    }

    #[test]
    fn validation_catches_out_of_range() {
        let mut v = VoiceSpec::default();
        v.rate = 10.0;
        assert!(v.validate().is_err());
        v.rate = 1.0;
        v.pitch = -40.0;
        assert!(v.validate().is_err());
    }

    #[test]
    fn patch_deserialises_from_sparse_json() {
        let p: VoicePatch = serde_json::from_str(r#"{"voice":"serena","rate":0.95}"#).unwrap();
        assert_eq!(p.voice.as_deref(), Some("serena"));
        assert_eq!(p.rate, Some(0.95));
        assert!(p.pitch.is_none());
    }
}
