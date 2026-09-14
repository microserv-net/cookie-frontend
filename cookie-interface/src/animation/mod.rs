//! The animation system.
//!
//! The orb is not four canned loops bound to four states. It is a parameter
//! block — thirty-odd floats describing shape, motion, colour and turbulence —
//! that a *behaviour* writes into every frame, and a director that decides
//! which behaviour is running, when to swap it for another, and how to
//! crossfade between them.
//!
//! ```text
//!   VoiceState ──▶ director ──▶ behaviour ──▶ OrbParams ──▶ WGSL uniforms
//!   AudioFeatures ─────┴───────────┘
//! ```
//!
//! Why this shape:
//!
//! * **Variety.** Each state owns several behaviours with different weights.
//!   Idle alone has six; you will not see the same one twice in a row.
//! * **Reproducibility.** Selection runs off a seeded [`Rng`], so a session
//!   started with `--seed 42` animates identically every time. Bugs in a
//!   visual system are otherwise almost impossible to report.
//! * **Extensibility.** Adding a behaviour means writing one struct and
//!   pushing it into the registry. Nothing else changes.
//! * **Continuity.** Behaviours never snap. Swaps crossfade in parameter
//!   space, so the entity appears to *change its mind* rather than cut.

pub mod behaviors;

use crate::audio::AudioFeatures;
use crate::config::{AnimationConfig, ThemeConfig};
use crate::util::{approach, lerp, Rng};
use crate::VoiceState;

pub use behaviors::registry;

/// Everything the shader needs for one frame.
///
/// Values are normalised: `1.0` is "the natural amount", not a pixel count.
/// Keeping the units abstract is what lets one behaviour be reused at any
/// window size and lets the renderer change without touching behaviours.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbParams {
    /// Seconds since the orb came alive. Shader time base.
    pub time: f32,
    /// Base radius, `1.0` = the design size.
    pub radius: f32,
    /// Low-frequency shape deformation.
    pub wobble: f32,
    /// High-frequency internal noise.
    pub turbulence: f32,
    /// Rotational flow of the interior.
    pub swirl: f32,
    /// Speed the internal noise field evolves at.
    pub flow_speed: f32,
    /// Spatial frequency of the noise field.
    pub noise_scale: f32,
    /// Weight of the finer FBM octaves.
    pub detail: f32,
    /// Outer bloom strength.
    pub glow: f32,
    /// Overall emission.
    pub brightness: f32,
    /// Opacity of the whole entity.
    pub alpha: f32,
    /// Base hue in degrees.
    pub hue: f32,
    /// How far the hue travels across the body.
    pub hue_spread: f32,
    pub saturation: f32,
    /// Thickness of the bright shell relative to the core.
    pub shell: f32,
    /// Density of the dark core.
    pub core: f32,
    /// Radial distortion of the silhouette.
    pub distortion: f32,
    /// Off-centre drift, in radii.
    pub offset: [f32; 2],
    /// Rotation of the whole field, radians.
    pub spin: f32,
    /// Smoothed audio energy, `0..=1`.
    pub energy: f32,
    /// Band energies, `0..=1`.
    pub bands: [f32; 3],
    /// Transient response, decays fast.
    pub onset: f32,
    /// Per-behaviour random constant, so two runs of the same behaviour
    /// differ in their details.
    pub seed: f32,
}

impl Default for OrbParams {
    fn default() -> Self {
        Self {
            time: 0.0,
            radius: 1.0,
            wobble: 0.12,
            turbulence: 0.35,
            swirl: 0.25,
            flow_speed: 0.25,
            noise_scale: 2.4,
            detail: 0.55,
            glow: 0.9,
            brightness: 1.0,
            alpha: 1.0,
            hue: 276.0,
            hue_spread: 26.0,
            saturation: 0.72,
            shell: 0.55,
            core: 0.45,
            distortion: 0.2,
            offset: [0.0, 0.0],
            spin: 0.0,
            energy: 0.0,
            bands: [0.0; 3],
            onset: 0.0,
            seed: 0.0,
        }
    }
}

impl OrbParams {
    /// Parameters that match the configured theme.
    pub fn from_theme(theme: &ThemeConfig) -> Self {
        Self {
            hue: theme.hue,
            hue_spread: (theme.hue_secondary - theme.hue).abs().clamp(0.0, 180.0),
            saturation: theme.saturation,
            brightness: theme.intensity,
            glow: theme.glow,
            alpha: 1.0,
            ..Default::default()
        }
    }

    /// Linear blend, used for crossfades between behaviours.
    ///
    /// Hue interpolates the short way round the colour wheel; blending 350°
    /// and 10° through 180° would send the orb green mid-transition.
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut hue_delta = other.hue - self.hue;
        if hue_delta > 180.0 {
            hue_delta -= 360.0;
        } else if hue_delta < -180.0 {
            hue_delta += 360.0;
        }
        Self {
            time: other.time,
            radius: lerp(self.radius, other.radius, t),
            wobble: lerp(self.wobble, other.wobble, t),
            turbulence: lerp(self.turbulence, other.turbulence, t),
            swirl: lerp(self.swirl, other.swirl, t),
            flow_speed: lerp(self.flow_speed, other.flow_speed, t),
            noise_scale: lerp(self.noise_scale, other.noise_scale, t),
            detail: lerp(self.detail, other.detail, t),
            glow: lerp(self.glow, other.glow, t),
            brightness: lerp(self.brightness, other.brightness, t),
            alpha: lerp(self.alpha, other.alpha, t),
            hue: (self.hue + hue_delta * t).rem_euclid(360.0),
            hue_spread: lerp(self.hue_spread, other.hue_spread, t),
            saturation: lerp(self.saturation, other.saturation, t),
            shell: lerp(self.shell, other.shell, t),
            core: lerp(self.core, other.core, t),
            distortion: lerp(self.distortion, other.distortion, t),
            offset: [
                lerp(self.offset[0], other.offset[0], t),
                lerp(self.offset[1], other.offset[1], t),
            ],
            spin: lerp(self.spin, other.spin, t),
            energy: lerp(self.energy, other.energy, t),
            bands: [
                lerp(self.bands[0], other.bands[0], t),
                lerp(self.bands[1], other.bands[1], t),
                lerp(self.bands[2], other.bands[2], t),
            ],
            onset: lerp(self.onset, other.onset, t),
            seed: if t < 0.5 { self.seed } else { other.seed },
        }
    }

    /// Clamp everything into ranges the shader can cope with. Called once per
    /// frame so a misbehaving behaviour cannot produce a black screen or a
    /// NaN.
    pub fn sanitise(&mut self) {
        fn fix(v: &mut f32, min: f32, max: f32, fallback: f32) {
            if !v.is_finite() {
                *v = fallback;
            }
            *v = v.clamp(min, max);
        }
        fix(&mut self.radius, 0.2, 2.0, 1.0);
        fix(&mut self.wobble, 0.0, 1.5, 0.1);
        fix(&mut self.turbulence, 0.0, 2.5, 0.35);
        fix(&mut self.swirl, -4.0, 4.0, 0.25);
        fix(&mut self.flow_speed, 0.0, 4.0, 0.25);
        fix(&mut self.noise_scale, 0.5, 12.0, 2.4);
        fix(&mut self.detail, 0.0, 1.5, 0.55);
        fix(&mut self.glow, 0.0, 3.0, 0.9);
        fix(&mut self.brightness, 0.05, 3.0, 1.0);
        fix(&mut self.alpha, 0.0, 1.0, 1.0);
        fix(&mut self.hue_spread, 0.0, 180.0, 26.0);
        fix(&mut self.saturation, 0.0, 1.0, 0.72);
        fix(&mut self.shell, 0.05, 1.5, 0.55);
        fix(&mut self.core, 0.0, 1.5, 0.45);
        fix(&mut self.distortion, 0.0, 1.5, 0.2);
        fix(&mut self.offset[0], -0.6, 0.6, 0.0);
        fix(&mut self.offset[1], -0.6, 0.6, 0.0);
        fix(&mut self.energy, 0.0, 1.0, 0.0);
        for b in &mut self.bands {
            fix(b, 0.0, 1.0, 0.0);
        }
        fix(&mut self.onset, 0.0, 1.0, 0.0);
        if !self.hue.is_finite() {
            self.hue = 276.0;
        }
        self.hue = self.hue.rem_euclid(360.0);
        if !self.spin.is_finite() {
            self.spin = 0.0;
        }
        if !self.time.is_finite() {
            self.time = 0.0;
        }
    }
}

/// What a behaviour is told each frame.
#[derive(Debug, Clone)]
pub struct BehaviorContext {
    /// Seconds since this behaviour started.
    pub elapsed: f32,
    /// Seconds since the last frame.
    pub dt: f32,
    /// Absolute animation time.
    pub time: f32,
    /// Voice state the behaviour was selected for.
    pub state: VoiceState,
    /// Smoothed audio features. Already normalised into `0..=1`.
    pub audio: AudioLevels,
    /// How strongly the user wants motion, from configuration.
    pub intensity: f32,
    /// Per-instance random constant in `0..=1`.
    pub seed: f32,
}

/// Audio, reduced to the handful of numbers an animation actually wants.
///
/// Raw [`AudioFeatures`] are noisy and unbounded; these are smoothed with
/// asymmetric attack/release so the orb responds instantly to a consonant and
/// relaxes gracefully afterwards.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AudioLevels {
    pub energy: f32,
    pub low: f32,
    pub mid: f32,
    pub high: f32,
    pub onset: f32,
    /// Slow envelope: roughly "is there sound at all right now".
    pub presence: f32,
    /// `true` while the signal looks like voiced speech.
    pub voiced: bool,
}

impl AudioLevels {
    fn absorb(&mut self, features: &AudioFeatures, dt: f32) {
        // Map dB onto 0..1 with -60 dB as silence; linear amplitude looks
        // wrong on a display because hearing is logarithmic.
        let level = ((features.level_db + 60.0) / 60.0).clamp(0.0, 1.0);
        let attack = 0.03;
        let release = 0.25;
        let half_life = |target: f32, current: f32| {
            if target > current {
                attack
            } else {
                release
            }
        };
        self.energy = approach(self.energy, level, half_life(level, self.energy), dt);
        let bands = [features.low, features.mid, features.high];
        let normalise = |v: f32| (v * 6.0).clamp(0.0, 1.0);
        self.low = approach(
            self.low,
            normalise(bands[0]),
            half_life(normalise(bands[0]), self.low),
            dt,
        );
        self.mid = approach(
            self.mid,
            normalise(bands[1]),
            half_life(normalise(bands[1]), self.mid),
            dt,
        );
        self.high = approach(
            self.high,
            normalise(bands[2]),
            half_life(normalise(bands[2]), self.high),
            dt,
        );
        // Onsets are events: jump on arrival, decay quickly.
        self.onset = (self.onset - dt * 4.0)
            .max(0.0)
            .max(features.onset.clamp(0.0, 1.0));
        self.presence = approach(self.presence, features.envelope.clamp(0.0, 1.0), 0.4, dt);
        self.voiced = features.voiced > 0.5;
    }
}

/// One visual behaviour.
///
/// Implementations are pure: they own their own phase state and write into
/// `params`, but never read global state or allocate per frame.
pub trait Behavior: Send {
    /// Stable identifier, surfaced in logs and in `--doctor`.
    fn name(&self) -> &'static str;

    /// States this behaviour is eligible for.
    fn states(&self) -> &'static [VoiceState];

    /// Relative selection weight within a state.
    fn weight(&self) -> f32 {
        1.0
    }

    /// How long it may run before the director considers a change, in
    /// seconds. `None` means "until the state changes" (used by transient
    /// states like Interrupted).
    fn duration(&self) -> Option<(f32, f32)> {
        Some((6.0, 22.0))
    }

    /// Called once when the behaviour is selected.
    fn start(&mut self, _ctx: &BehaviorContext) {}

    /// Called every frame. `params` starts from the theme baseline.
    fn update(&mut self, ctx: &BehaviorContext, params: &mut OrbParams);
}

/// Runs the show.
pub struct AnimationDirector {
    theme: ThemeConfig,
    config: AnimationConfig,
    rng: Rng,
    seed: u64,
    pool: Vec<Box<dyn Behavior>>,
    current: usize,
    previous: Option<usize>,
    blend: f32,
    blend_rate: f32,
    previous_params: OrbParams,
    elapsed: f32,
    hold_until: f32,
    time: f32,
    state: VoiceState,
    audio: AudioLevels,
    ctx_seed: f32,
}

impl std::fmt::Debug for AnimationDirector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationDirector")
            .field("seed", &self.seed)
            .field("state", &self.state)
            .field("behavior", &self.pool[self.current].name())
            .finish()
    }
}

impl AnimationDirector {
    /// Build with the standard behaviour registry.
    pub fn new(theme: ThemeConfig, config: AnimationConfig) -> Self {
        Self::with_behaviors(theme, config, registry())
    }

    /// Build with a custom behaviour set. This is the extension point.
    pub fn with_behaviors(
        theme: ThemeConfig,
        config: AnimationConfig,
        pool: Vec<Box<dyn Behavior>>,
    ) -> Self {
        assert!(!pool.is_empty(), "the behaviour pool must not be empty");
        let seed = config.seed.unwrap_or_else(|| {
            let mut entropy = Rng::from_entropy();
            entropy.next_u64()
        });
        tracing::info!(seed, "animation seed");
        let mut me = Self {
            rng: Rng::new(seed),
            seed,
            pool,
            current: 0,
            previous: None,
            blend: 1.0,
            blend_rate: 2.5,
            previous_params: OrbParams::from_theme(&theme),
            elapsed: 0.0,
            hold_until: 0.0,
            time: 0.0,
            state: VoiceState::Idle,
            audio: AudioLevels::default(),
            ctx_seed: 0.0,
            theme,
            config,
        };
        me.select(VoiceState::Idle, true);
        me
    }

    /// The seed this session is running with. Printed at startup so a visual
    /// bug can be reproduced exactly.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Name of the behaviour currently on screen.
    pub fn current_behavior(&self) -> &'static str {
        self.pool[self.current].name()
    }

    /// Advance and produce this frame's parameters.
    pub fn update(&mut self, dt: f32, state: VoiceState, features: &AudioFeatures) -> OrbParams {
        let dt = dt.clamp(0.0, 0.1);
        self.time += dt;
        self.elapsed += dt;
        self.audio.absorb(features, dt);

        if state != self.state {
            self.state = state;
            self.select(state, false);
        } else if self.elapsed >= self.hold_until && self.blend >= 1.0 {
            // Same state, but this behaviour has had its turn.
            self.select(state, false);
        }

        let ctx = BehaviorContext {
            elapsed: self.elapsed,
            dt,
            time: self.time,
            state: self.state,
            audio: self.audio,
            intensity: if self.config.reduce_motion {
                0.35
            } else {
                self.config.reactivity
            },
            seed: self.ctx_seed,
        };

        let mut params = OrbParams::from_theme(&self.theme);
        params.time = self.time;
        params.seed = self.ctx_seed;
        params.energy = self.audio.energy;
        params.bands = [self.audio.low, self.audio.mid, self.audio.high];
        params.onset = self.audio.onset;
        self.pool[self.current].update(&ctx, &mut params);
        params.sanitise();

        if self.blend < 1.0 {
            self.blend = (self.blend + dt * self.blend_rate).min(1.0);
            let smooth = crate::util::smoothstep(0.0, 1.0, self.blend);
            let mut blended = self.previous_params.blend(&params, smooth);
            blended.sanitise();
            self.previous_params = blended.clone();
            if self.blend >= 1.0 {
                self.previous = None;
            }
            return blended;
        }

        self.previous_params = params.clone();
        params
    }

    /// Choose a behaviour for `state`, avoiding an immediate repeat.
    fn select(&mut self, state: VoiceState, initial: bool) {
        let candidates: Vec<usize> = self
            .pool
            .iter()
            .enumerate()
            .filter(|(i, b)| {
                b.states().contains(&state)
                    && (initial || *i != self.current || self.only_one(state))
            })
            .map(|(i, _)| i)
            .collect();
        let candidates = if candidates.is_empty() {
            // Never leave the orb without a behaviour; Idle always has one.
            (0..self.pool.len())
                .filter(|i| self.pool[*i].states().contains(&VoiceState::Idle))
                .collect()
        } else {
            candidates
        };
        if candidates.is_empty() {
            return;
        }
        // `variation` biases between "always the highest-weighted behaviour"
        // and "uniformly random". 0.85 gives noticeable variety without
        // making the character feel unstable.
        let variation = self.config.variation.clamp(0.0, 1.0);
        let weights: Vec<f32> = candidates
            .iter()
            .map(|i| {
                let w = self.pool[*i].weight().max(0.01);
                lerp(w, 1.0, variation)
            })
            .collect();
        let pick = candidates[self.rng.weighted(&weights)];

        if !initial {
            self.previous = Some(self.current);
            self.blend = 0.0;
            // Alarming states swap fast; calm ones ease across.
            self.blend_rate = match state {
                VoiceState::Interrupted | VoiceState::Error => 6.0,
                VoiceState::Speaking | VoiceState::Listening => 3.5,
                _ => 1.8,
            };
        }
        self.current = pick;
        self.elapsed = 0.0;
        self.ctx_seed = self.rng.f32();
        let (min, max) = self.pool[pick].duration().unwrap_or((f32::MAX, f32::MAX));
        let (min, max) = if min == f32::MAX {
            (f32::MAX, f32::MAX)
        } else {
            (
                min.max(self.config.min_behavior_seconds),
                max.min(self.config.max_behavior_seconds).max(min),
            )
        };
        self.hold_until = if min == f32::MAX {
            f32::MAX
        } else {
            self.rng.range(min, max)
        };
        let ctx = BehaviorContext {
            elapsed: 0.0,
            dt: 0.0,
            time: self.time,
            state,
            audio: self.audio,
            intensity: self.config.reactivity,
            seed: self.ctx_seed,
        };
        self.pool[pick].start(&ctx);
        tracing::debug!(
            behavior = self.pool[pick].name(),
            ?state,
            "animation behaviour selected"
        );
    }

    fn only_one(&self, state: VoiceState) -> bool {
        self.pool
            .iter()
            .filter(|b| b.states().contains(&state))
            .count()
            <= 1
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn director(seed: Option<u64>) -> AnimationDirector {
        let mut config = AnimationConfig::default();
        config.seed = seed;
        AnimationDirector::new(ThemeConfig::default(), config)
    }

    fn silence() -> AudioFeatures {
        AudioFeatures::silent()
    }

    #[test]
    fn same_seed_produces_the_same_sequence() {
        let mut a = director(Some(7));
        let mut b = director(Some(7));
        let mut names_a = Vec::new();
        let mut names_b = Vec::new();
        for i in 0..900 {
            let state = if i % 300 < 150 {
                VoiceState::Idle
            } else {
                VoiceState::Listening
            };
            a.update(1.0 / 30.0, state, &silence());
            b.update(1.0 / 30.0, state, &silence());
            names_a.push(a.current_behavior());
            names_b.push(b.current_behavior());
        }
        assert_eq!(names_a, names_b);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = director(Some(1));
        let mut b = director(Some(2));
        let mut same = true;
        for _ in 0..2_000 {
            a.update(1.0 / 60.0, VoiceState::Idle, &silence());
            b.update(1.0 / 60.0, VoiceState::Idle, &silence());
            if a.current_behavior() != b.current_behavior() {
                same = false;
                break;
            }
        }
        assert!(!same, "two seeds produced identical behaviour sequences");
    }

    #[test]
    fn idle_uses_several_behaviours_over_time() {
        let mut d = director(Some(11));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..20_000 {
            d.update(1.0 / 60.0, VoiceState::Idle, &silence());
            seen.insert(d.current_behavior());
        }
        assert!(seen.len() >= 3, "only saw {seen:?}");
    }

    #[test]
    fn every_state_has_a_behaviour() {
        for state in [
            VoiceState::Idle,
            VoiceState::Listening,
            VoiceState::Processing,
            VoiceState::Speaking,
            VoiceState::Interrupted,
            VoiceState::Error,
        ] {
            let mut d = director(Some(3));
            d.update(0.016, state, &silence());
            assert!(
                d.pool[d.current].states().contains(&state),
                "no behaviour for {state:?}"
            );
        }
    }

    #[test]
    fn parameters_stay_in_range_under_hostile_audio() {
        let mut d = director(Some(5));
        let mut features = AudioFeatures::silent();
        features.level_db = f32::NAN;
        features.low = 1e9;
        features.onset = 42.0;
        for _ in 0..200 {
            let p = d.update(0.016, VoiceState::Speaking, &features);
            assert!(p.radius.is_finite() && p.radius <= 2.0);
            assert!((0.0..=1.0).contains(&p.alpha));
            assert!((0.0..=360.0).contains(&p.hue));
        }
    }

    #[test]
    fn state_changes_crossfade_rather_than_snap() {
        let mut d = director(Some(9));
        for _ in 0..60 {
            d.update(0.016, VoiceState::Idle, &silence());
        }
        let before = d.update(0.016, VoiceState::Idle, &silence());
        let after = d.update(0.016, VoiceState::Speaking, &silence());
        // One frame after a state change the visible parameters must still be
        // close to the old ones.
        assert!((before.radius - after.radius).abs() < 0.25);
    }

    #[test]
    fn blend_interpolates_hue_the_short_way() {
        let a = OrbParams {
            hue: 350.0,
            ..Default::default()
        };
        let b = OrbParams {
            hue: 10.0,
            ..Default::default()
        };
        let mid = a.blend(&b, 0.5);
        assert!(mid.hue > 355.0 || mid.hue < 5.0, "{}", mid.hue);
    }

    #[test]
    fn reduce_motion_lowers_intensity() {
        let mut config = AnimationConfig::default();
        config.seed = Some(4);
        config.reduce_motion = true;
        let mut d = AnimationDirector::new(ThemeConfig::default(), config);
        let mut loud = AudioFeatures::silent();
        loud.level_db = -6.0;
        loud.low = 0.8;
        let mut max_radius: f32 = 0.0;
        for _ in 0..300 {
            max_radius = max_radius.max(d.update(0.016, VoiceState::Speaking, &loud).radius);
        }
        assert!(max_radius < 1.35, "{max_radius}");
    }
}
