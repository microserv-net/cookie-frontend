//! The behaviour library.
//!
//! Every behaviour here is a small, self-contained idea about how a living
//! thing might move. The director picks between them; none of them knows the
//! others exist.
//!
//! Adding one is deliberately trivial: implement [`Behavior`], push it into
//! [`registry`], done. Nothing else in the crate needs to change, which is
//! the whole reason the animation layer is separate from the renderer.

use std::f32::consts::{PI, TAU};

use crate::util::{approach, lerp};
use crate::VoiceState;

use super::{Behavior, BehaviorContext, OrbParams};

/// Every behaviour the application ships with.
///
/// Weights are relative *within* a state. Idle deliberately favours the quiet
/// ones: the orb spends most of its life here and should not perform.
pub fn registry() -> Vec<Box<dyn Behavior>> {
    vec![
        // Idle — six ways of doing almost nothing.
        Box::new(Breathing::default()),
        Box::new(Drift::default()),
        Box::new(InternalWeather::default()),
        Box::new(SlowMorph::default()),
        Box::new(DormantFlicker::default()),
        Box::new(OccasionalPulse::default()),
        // Listening — the microphone drives the surface.
        Box::new(Ripples::default()),
        Box::new(Gather::default()),
        Box::new(EdgeResponse::default()),
        Box::new(DirectionalFlow::default()),
        // Processing — energy turned inward.
        Box::new(Vortex::default()),
        Box::new(Winding::default()),
        Box::new(Orbits::default()),
        // Speaking — driven by the voice itself.
        Box::new(VoiceBloom::default()),
        Box::new(Filaments::default()),
        Box::new(Resonance::default()),
        // Transient states.
        Box::new(Recoil::default()),
        Box::new(Fault::default()),
    ]
}

const IDLE: &[VoiceState] = &[VoiceState::Idle];
const LISTENING: &[VoiceState] = &[VoiceState::Listening];
const PROCESSING: &[VoiceState] = &[VoiceState::Processing];
const SPEAKING: &[VoiceState] = &[VoiceState::Speaking];
const INTERRUPTED: &[VoiceState] = &[VoiceState::Interrupted];
const ERROR: &[VoiceState] = &[VoiceState::Error];

/// Smooth pseudo-noise from a couple of sines. Cheap, and good enough for
/// parameter-space motion (the interesting noise lives in the shader).
fn wave(t: f32, seed: f32, rate: f32) -> f32 {
    let a = (t * rate + seed * TAU).sin();
    let b = (t * rate * 0.37 + seed * 11.0).sin();
    (a * 0.65 + b * 0.35).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Idle
// ---------------------------------------------------------------------------

/// Slow, even breathing. The baseline against which everything else reads as
/// activity.
#[derive(Debug, Default)]
pub struct Breathing {
    phase: f32,
}

impl Behavior for Breathing {
    fn name(&self) -> &'static str {
        "idle.breathing"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn weight(&self) -> f32 {
        1.6
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        // ~5.5 s per breath: slower than a resting human, which reads as calm
        // rather than asleep.
        self.phase += ctx.dt * TAU / lerp(5.5, 7.5, ctx.seed);
        let breath = self.phase.sin();
        p.radius = 1.0 + breath * 0.045 * ctx.intensity;
        p.brightness *= 1.0 + breath * 0.08;
        p.glow *= 1.0 + breath * 0.12;
        p.turbulence = 0.28;
        p.flow_speed = 0.14;
        p.swirl = 0.12 + breath * 0.03;
        p.shell = 0.58 + breath * 0.03;
    }
}

/// Asymmetric drift: the body wanders slightly off centre, like something
/// suspended in liquid.
#[derive(Debug, Default)]
pub struct Drift {
    t: f32,
}

impl Behavior for Drift {
    fn name(&self) -> &'static str {
        "idle.drift"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn weight(&self) -> f32 {
        1.2
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let x = wave(self.t, ctx.seed, 0.21);
        let y = wave(self.t, ctx.seed + 0.5, 0.17);
        p.offset = [x * 0.07 * ctx.intensity, y * 0.05 * ctx.intensity];
        p.radius = 1.0 + wave(self.t, ctx.seed + 0.2, 0.13) * 0.03;
        p.wobble = 0.14 + x.abs() * 0.05;
        p.flow_speed = 0.18;
        p.swirl = 0.16 * x.signum();
        p.spin += ctx.dt * 0.05;
    }
}

/// Weather on the inside: the surface stays put while the interior churns.
#[derive(Debug, Default)]
pub struct InternalWeather {
    t: f32,
}

impl Behavior for InternalWeather {
    fn name(&self) -> &'static str {
        "idle.weather"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let churn = wave(self.t, ctx.seed, 0.33);
        p.turbulence = 0.45 + churn * 0.18 * ctx.intensity;
        p.detail = 0.75;
        p.noise_scale = 3.1 + churn * 0.5;
        p.flow_speed = 0.32;
        p.swirl = 0.35 * churn;
        p.core = 0.52;
        p.radius = 1.0 + churn * 0.015;
    }
}

/// The silhouette itself slowly changes shape.
#[derive(Debug, Default)]
pub struct SlowMorph {
    t: f32,
}

impl Behavior for SlowMorph {
    fn name(&self) -> &'static str {
        "idle.morph"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let a = wave(self.t, ctx.seed, 0.11);
        let b = wave(self.t, ctx.seed + 0.3, 0.07);
        p.wobble = 0.2 + a * 0.12 * ctx.intensity;
        p.distortion = 0.28 + b * 0.14;
        p.noise_scale = 2.0 + b * 0.6;
        p.flow_speed = 0.12;
        p.hue += a * 4.0;
        p.hue_spread *= 1.0 + b * 0.25;
    }
}

/// Dormant, with the occasional flicker — the visual equivalent of a
/// standby light. Suppressed when the user asked for reduced motion.
#[derive(Debug, Default)]
pub struct DormantFlicker {
    t: f32,
    next: f32,
    flash: f32,
}

impl Behavior for DormantFlicker {
    fn name(&self) -> &'static str {
        "idle.dormant"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn weight(&self) -> f32 {
        0.7
    }
    fn start(&mut self, ctx: &BehaviorContext) {
        self.t = 0.0;
        self.flash = 0.0;
        self.next = 1.5 + ctx.seed * 4.0;
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        if self.t >= self.next {
            self.next = self.t + 2.0 + wave(self.t, ctx.seed, 1.3).abs() * 5.0;
            self.flash = 1.0;
        }
        self.flash = (self.flash - ctx.dt * 2.5).max(0.0);
        let f = self.flash * self.flash * ctx.intensity;
        p.brightness *= 0.72 + f * 0.6;
        p.glow *= 0.7 + f * 0.9;
        p.alpha = 0.88 + f * 0.12;
        p.turbulence = 0.2;
        p.flow_speed = 0.08;
        p.radius = 0.94 + f * 0.04;
        p.core = 0.6;
    }
}

/// Mostly still, with a single slow pulse that travels outward.
#[derive(Debug, Default)]
pub struct OccasionalPulse {
    t: f32,
    since: f32,
    period: f32,
}

impl Behavior for OccasionalPulse {
    fn name(&self) -> &'static str {
        "idle.pulse"
    }
    fn states(&self) -> &'static [VoiceState] {
        IDLE
    }
    fn weight(&self) -> f32 {
        0.9
    }
    fn start(&mut self, ctx: &BehaviorContext) {
        self.t = 0.0;
        self.since = 0.0;
        self.period = lerp(4.0, 9.0, ctx.seed);
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        self.since += ctx.dt;
        if self.since > self.period {
            self.since = 0.0;
        }
        // A half-sine that runs for the first second of each period.
        let pulse = if self.since < 1.0 {
            (self.since * PI).sin()
        } else {
            0.0
        };
        p.radius = 1.0 + pulse * 0.06 * ctx.intensity;
        p.shell = 0.5 + pulse * 0.22;
        p.glow *= 1.0 + pulse * 0.5;
        p.turbulence = 0.26 + pulse * 0.2;
        p.flow_speed = 0.16 + pulse * 0.3;
    }
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------

/// Concentric ripples driven by the microphone: the orb visibly *hears*.
#[derive(Debug, Default)]
pub struct Ripples {
    t: f32,
    ring: f32,
}

impl Behavior for Ripples {
    fn name(&self) -> &'static str {
        "listening.ripples"
    }
    fn states(&self) -> &'static [VoiceState] {
        LISTENING
    }
    fn weight(&self) -> f32 {
        1.4
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((5.0, 14.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let a = ctx.audio;
        self.ring = approach(self.ring, a.energy, 0.06, ctx.dt);
        p.radius = 1.02 + self.ring * 0.1 * ctx.intensity + a.onset * 0.04;
        p.shell = 0.42 + self.ring * 0.35;
        p.wobble = 0.1 + a.high * 0.25;
        p.turbulence = 0.3 + a.mid * 0.5;
        p.flow_speed = 0.3 + a.energy * 0.8;
        p.swirl = 0.2 + a.low * 0.4;
        p.glow *= 1.0 + self.ring * 0.5;
        p.hue += a.high * 8.0;
    }
}

/// The body gathers itself inward and brightens: attention, not excitement.
#[derive(Debug, Default)]
pub struct Gather {
    tension: f32,
}

impl Behavior for Gather {
    fn name(&self) -> &'static str {
        "listening.gather"
    }
    fn states(&self) -> &'static [VoiceState] {
        LISTENING
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        let a = ctx.audio;
        let target = 0.35 + a.energy * 0.65;
        self.tension = approach(self.tension, target, 0.12, ctx.dt);
        p.radius = 0.94 + a.energy * 0.08 * ctx.intensity;
        p.core = 0.3 + self.tension * 0.35;
        p.shell = 0.7 - self.tension * 0.2;
        p.brightness *= 1.05 + self.tension * 0.25;
        p.turbulence = 0.22 + a.mid * 0.3;
        p.noise_scale = 3.4 + self.tension;
        p.flow_speed = 0.25 + a.energy * 0.5;
        p.saturation = (p.saturation + 0.08).min(1.0);
    }
}

/// Only the rim responds: a quiet, precise way to show input level.
#[derive(Debug, Default)]
pub struct EdgeResponse {
    edge: f32,
}

impl Behavior for EdgeResponse {
    fn name(&self) -> &'static str {
        "listening.edge"
    }
    fn states(&self) -> &'static [VoiceState] {
        LISTENING
    }
    fn weight(&self) -> f32 {
        0.9
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        let a = ctx.audio;
        self.edge = approach(self.edge, a.energy.max(a.onset), 0.05, ctx.dt);
        p.radius = 1.0 + self.edge * 0.03;
        p.shell = 0.3 + self.edge * 0.5;
        p.distortion = 0.14 + self.edge * 0.35 * ctx.intensity;
        p.wobble = 0.08 + self.edge * 0.3;
        p.turbulence = 0.24;
        p.glow *= 1.0 + self.edge * 0.8;
        p.core = 0.5;
        p.flow_speed = 0.22;
    }
}

/// Internal flow leans in one direction, as if turned toward the speaker.
#[derive(Debug, Default)]
pub struct DirectionalFlow {
    t: f32,
    lean: [f32; 2],
}

impl Behavior for DirectionalFlow {
    fn name(&self) -> &'static str {
        "listening.lean"
    }
    fn states(&self) -> &'static [VoiceState] {
        LISTENING
    }
    fn weight(&self) -> f32 {
        0.8
    }
    fn start(&mut self, ctx: &BehaviorContext) {
        let angle = ctx.seed * TAU;
        self.lean = [angle.cos(), angle.sin()];
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let a = ctx.audio;
        let amount = (0.25 + a.energy * 0.75) * ctx.intensity;
        p.offset = [self.lean[0] * 0.06 * amount, self.lean[1] * 0.06 * amount];
        p.swirl = 0.5 * self.lean[0] + a.low * 0.6;
        p.flow_speed = 0.35 + a.energy * 0.7;
        p.turbulence = 0.3 + a.mid * 0.35;
        p.wobble = 0.12 + a.high * 0.2;
        p.radius = 1.0 + a.energy * 0.05;
    }
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

/// A vortex forms inside. Reads unmistakably as "working".
#[derive(Debug, Default)]
pub struct Vortex {
    spin: f32,
}

impl Behavior for Vortex {
    fn name(&self) -> &'static str {
        "processing.vortex"
    }
    fn states(&self) -> &'static [VoiceState] {
        PROCESSING
    }
    fn weight(&self) -> f32 {
        1.3
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((3.0, 10.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        // Spin ramps up rather than starting at speed: acceleration is what
        // makes it look like effort.
        self.spin = approach(self.spin, 1.0, 0.5, ctx.dt);
        p.spin += ctx.dt * 1.6 * self.spin;
        p.swirl = 1.4 * self.spin * ctx.intensity;
        p.turbulence = 0.5 + 0.4 * self.spin;
        p.noise_scale = 3.6;
        p.detail = 0.85;
        p.flow_speed = 0.9 + self.spin * 0.8;
        p.core = 0.65;
        p.radius = 0.98;
        p.brightness *= 1.0 + self.spin * 0.15;
    }
}

/// Everything winds tighter and tighter, then holds.
#[derive(Debug, Default)]
pub struct Winding {
    t: f32,
}

impl Behavior for Winding {
    fn name(&self) -> &'static str {
        "processing.winding"
    }
    fn states(&self) -> &'static [VoiceState] {
        PROCESSING
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((3.0, 9.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let wind = (self.t * 0.8).tanh();
        p.radius = 1.0 - wind * 0.06;
        p.noise_scale = 2.4 + wind * 4.0;
        p.detail = 0.6 + wind * 0.5;
        p.turbulence = 0.4 + wind * 0.6 * ctx.intensity;
        p.swirl = 0.4 + wind * 1.0;
        p.flow_speed = 0.6 + wind * 1.2;
        p.core = 0.4 + wind * 0.35;
        p.hue -= wind * 6.0;
    }
}

/// Bright structures orbit inside the body.
#[derive(Debug, Default)]
pub struct Orbits {
    t: f32,
}

impl Behavior for Orbits {
    fn name(&self) -> &'static str {
        "processing.orbits"
    }
    fn states(&self) -> &'static [VoiceState] {
        PROCESSING
    }
    fn weight(&self) -> f32 {
        0.9
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((3.0, 10.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let a = (self.t * 1.1 + ctx.seed * TAU).sin();
        p.spin += ctx.dt * (0.9 + ctx.seed * 0.6);
        p.offset = [a * 0.03, (self.t * 0.9).cos() * 0.03];
        p.swirl = 0.8 + a * 0.5;
        p.turbulence = 0.55;
        p.detail = 0.9;
        p.noise_scale = 4.2 + a * 0.8;
        p.flow_speed = 1.0;
        p.shell = 0.45 + a.abs() * 0.2;
        p.glow *= 1.15;
    }
}

// ---------------------------------------------------------------------------
// Speaking
// ---------------------------------------------------------------------------

/// The default speaking behaviour: the body blooms with the voice.
///
/// Deliberately *not* "scale by volume". Low frequencies push the radius,
/// mids drive interior turbulence, highs sharpen the rim, and onsets fire a
/// short bright flash — so consonants, vowels and pauses all look different.
#[derive(Debug, Default)]
pub struct VoiceBloom {
    bloom: f32,
    flash: f32,
}

impl Behavior for VoiceBloom {
    fn name(&self) -> &'static str {
        "speaking.bloom"
    }
    fn states(&self) -> &'static [VoiceState] {
        SPEAKING
    }
    fn weight(&self) -> f32 {
        1.5
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((4.0, 12.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        let a = ctx.audio;
        self.bloom = approach(self.bloom, a.low * 0.7 + a.energy * 0.3, 0.05, ctx.dt);
        self.flash = (self.flash - ctx.dt * 3.0).max(a.onset);
        let k = ctx.intensity;
        p.radius = 1.0 + self.bloom * 0.14 * k;
        p.wobble = 0.12 + a.high * 0.4 * k;
        p.turbulence = 0.35 + a.mid * 0.9 * k;
        p.detail = 0.6 + a.high * 0.6;
        p.noise_scale = 2.6 + a.mid * 2.0;
        p.flow_speed = 0.4 + a.energy * 1.6;
        p.swirl = 0.25 + a.low * 0.8;
        p.shell = 0.45 + self.bloom * 0.3;
        p.core = 0.45 - self.bloom * 0.15;
        p.brightness *= 1.0 + self.bloom * 0.35 + self.flash * 0.3;
        p.glow *= 1.0 + a.energy * 0.7 + self.flash * 0.6;
        p.hue += a.high * 10.0 - a.low * 6.0;
        p.hue_spread *= 1.0 + a.energy * 0.5;
        // Pauses are visible: without sound the body eases back but the
        // interior keeps moving, so it reads as "still talking, mid-thought".
        if a.energy < 0.05 {
            p.radius = lerp(p.radius, 0.99, 0.5);
            p.flow_speed = lerp(p.flow_speed, 0.5, 0.5);
        }
    }
}

/// Filaments reach outward on transients: a more theatrical speaking mode.
#[derive(Debug, Default)]
pub struct Filaments {
    reach: f32,
}

impl Behavior for Filaments {
    fn name(&self) -> &'static str {
        "speaking.filaments"
    }
    fn states(&self) -> &'static [VoiceState] {
        SPEAKING
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((4.0, 12.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        let a = ctx.audio;
        self.reach = approach(self.reach, a.high * 0.6 + a.onset, 0.04, ctx.dt);
        p.distortion = 0.25 + self.reach * 0.7 * ctx.intensity;
        p.wobble = 0.18 + self.reach * 0.5;
        p.radius = 1.0 + a.energy * 0.07;
        p.turbulence = 0.45 + a.mid * 0.7;
        p.detail = 0.9;
        p.noise_scale = 3.8 + a.high * 2.5;
        p.flow_speed = 0.6 + a.energy * 1.4;
        p.shell = 0.35 + a.energy * 0.3;
        p.glow *= 1.0 + self.reach * 0.9;
    }
}

/// Standing waves: quieter, more musical, suits slow speech.
#[derive(Debug, Default)]
pub struct Resonance {
    phase: f32,
}

impl Behavior for Resonance {
    fn name(&self) -> &'static str {
        "speaking.resonance"
    }
    fn states(&self) -> &'static [VoiceState] {
        SPEAKING
    }
    fn weight(&self) -> f32 {
        0.8
    }
    fn duration(&self) -> Option<(f32, f32)> {
        Some((4.0, 12.0))
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        let a = ctx.audio;
        self.phase += ctx.dt * (2.0 + a.energy * 6.0);
        let ring = self.phase.sin() * a.energy;
        p.radius = 1.0 + ring * 0.06 * ctx.intensity;
        p.shell = 0.5 + ring * 0.25;
        p.turbulence = 0.3 + a.mid * 0.5;
        p.swirl = 0.3 + ring * 0.5;
        p.flow_speed = 0.45 + a.energy * 1.0;
        p.noise_scale = 2.2 + a.low * 1.5;
        p.brightness *= 1.0 + a.energy * 0.3;
        p.glow *= 1.0 + a.low * 0.6;
    }
}

// ---------------------------------------------------------------------------
// Interrupted / Error
// ---------------------------------------------------------------------------

/// A sharp contraction, then a quick settle. Being cut off should *look*
/// like being cut off.
#[derive(Debug, Default)]
pub struct Recoil {
    t: f32,
}

impl Behavior for Recoil {
    fn name(&self) -> &'static str {
        "interrupted.recoil"
    }
    fn states(&self) -> &'static [VoiceState] {
        INTERRUPTED
    }
    fn duration(&self) -> Option<(f32, f32)> {
        None
    }
    fn start(&mut self, _ctx: &BehaviorContext) {
        self.t = 0.0;
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let snap = (-self.t * 4.0).exp();
        p.radius = 1.0 - snap * 0.16 * ctx.intensity;
        p.core = 0.35 + snap * 0.4;
        p.shell = 0.75 - snap * 0.3;
        p.turbulence = 0.6 * snap + 0.25;
        p.flow_speed = 1.4 * snap + 0.2;
        p.brightness *= 1.0 - snap * 0.25;
        p.saturation *= 1.0 - snap * 0.25;
        p.spin -= ctx.dt * snap * 2.0;
    }
}

/// Something is wrong: the hue shifts off the house violet toward red, the
/// body loses coherence, and a slow warning pulse runs underneath. No text
/// needed to know the assistant is unhappy.
#[derive(Debug, Default)]
pub struct Fault {
    t: f32,
}

impl Behavior for Fault {
    fn name(&self) -> &'static str {
        "error.fault"
    }
    fn states(&self) -> &'static [VoiceState] {
        ERROR
    }
    fn duration(&self) -> Option<(f32, f32)> {
        None
    }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        let pulse = (self.t * 2.2).sin() * 0.5 + 0.5;
        // Toward red, the short way round from violet.
        p.hue = lerp(p.hue, 352.0, 0.75);
        p.hue_spread = 12.0;
        p.saturation = (p.saturation * 0.9).min(0.8);
        p.brightness *= 0.75 + pulse * 0.25;
        p.radius = 0.96 + pulse * 0.02;
        p.wobble = 0.3;
        p.distortion = 0.45;
        p.turbulence = 0.7;
        p.detail = 0.4;
        p.flow_speed = 0.5;
        p.shell = 0.35;
        p.core = 0.6;
        p.alpha = 0.9;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::AudioLevels;

    fn ctx(state: VoiceState, energy: f32) -> BehaviorContext {
        BehaviorContext {
            elapsed: 0.5,
            dt: 1.0 / 60.0,
            time: 0.5,
            state,
            audio: AudioLevels {
                energy,
                low: energy * 0.8,
                mid: energy * 0.6,
                high: energy * 0.4,
                onset: 0.0,
                presence: energy,
                voiced: energy > 0.1,
            },
            intensity: 1.0,
            seed: 0.37,
        }
    }

    #[test]
    fn registry_covers_every_state() {
        let pool = registry();
        for state in [
            VoiceState::Idle,
            VoiceState::Listening,
            VoiceState::Processing,
            VoiceState::Speaking,
            VoiceState::Interrupted,
            VoiceState::Error,
        ] {
            assert!(
                pool.iter().any(|b| b.states().contains(&state)),
                "no behaviour for {state:?}"
            );
        }
    }

    #[test]
    fn behaviour_names_are_unique() {
        let pool = registry();
        let mut names: Vec<_> = pool.iter().map(|b| b.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn idle_and_speaking_each_have_several_variants() {
        let pool = registry();
        let idle = pool
            .iter()
            .filter(|b| b.states().contains(&VoiceState::Idle))
            .count();
        let speaking = pool
            .iter()
            .filter(|b| b.states().contains(&VoiceState::Speaking))
            .count();
        assert!(idle >= 4, "{idle}");
        assert!(speaking >= 3, "{speaking}");
    }

    #[test]
    fn every_behaviour_produces_finite_parameters() {
        for mut behavior in registry() {
            let state = behavior.states()[0];
            let context = ctx(state, 0.6);
            behavior.start(&context);
            let mut params = OrbParams::default();
            for _ in 0..120 {
                behavior.update(&context, &mut params);
                assert!(params.radius.is_finite(), "{}", behavior.name());
                assert!(params.turbulence.is_finite(), "{}", behavior.name());
                assert!(params.hue.is_finite(), "{}", behavior.name());
            }
        }
    }

    #[test]
    fn speaking_behaviours_actually_react_to_audio() {
        for mut behavior in registry()
            .into_iter()
            .filter(|b| b.states().contains(&VoiceState::Speaking))
        {
            let quiet = ctx(VoiceState::Speaking, 0.0);
            let loud = ctx(VoiceState::Speaking, 0.9);
            let mut a = OrbParams::default();
            let mut b = OrbParams::default();
            for _ in 0..60 {
                behavior.update(&quiet, &mut a);
            }
            for _ in 0..60 {
                behavior.update(&loud, &mut b);
            }
            assert!(
                a != b,
                "{} ignored the audio signal entirely",
                behavior.name()
            );
        }
    }

    #[test]
    fn error_behaviour_leaves_the_house_hue() {
        let mut fault = Fault::default();
        let mut params = OrbParams::default();
        fault.update(&ctx(VoiceState::Error, 0.0), &mut params);
        assert!(params.hue > 330.0 || params.hue < 20.0, "{}", params.hue);
    }

    #[test]
    fn transient_behaviours_run_until_the_state_changes() {
        assert!(Recoil::default().duration().is_none());
        assert!(Fault::default().duration().is_none());
        assert!(Breathing::default().duration().is_some());
    }
}
