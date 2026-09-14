# The orb

## Not an animation

There is no video, no sprite sheet, no imported asset. Every frame is generated
from about thirty floats, which is what lets the entity react to a voice rather
than replay a loop.

```
VoiceState ─┐
            ├─▶ AnimationDirector ─▶ behaviour ─▶ OrbParams ─▶ WGSL uniforms
AudioFeatures ┘        (seeded scheduler, crossfades)
```

## Behaviours

Eighteen of them, several per state, each a small self-contained idea about how
a living thing might move. Idle has six — breathing, drift, internal weather,
slow morph, dormant flicker, occasional pulse — and favours the quiet ones,
because the orb spends most of its life there and should not perform.

The director picks by weight, holds a behaviour for a randomised interval, then
crossfades to another **in parameter space**, so the entity appears to change
its mind rather than cut. Alarming states swap fast; calm ones ease across.

Adding one is a struct and a line in `registry()`:

```rust
pub struct Shiver { t: f32 }

impl Behavior for Shiver {
    fn name(&self) -> &'static str { "idle.shiver" }
    fn states(&self) -> &'static [VoiceState] { &[VoiceState::Idle] }
    fn weight(&self) -> f32 { 0.5 }
    fn update(&mut self, ctx: &BehaviorContext, p: &mut OrbParams) {
        self.t += ctx.dt;
        p.wobble = 0.15 + (self.t * 9.0).sin().abs() * 0.05 * ctx.intensity;
        p.flow_speed = 0.3;
    }
}
```

Nothing else changes. Parameters are normalised — `1.0` is "the natural
amount", not a pixel count — so a behaviour works at any window size, and
`sanitise()` clamps everything each frame so a misbehaving one cannot produce a
black screen or a NaN.

## Speaking is not volume

Scaling the whole orb by amplitude is what every audio visualiser does and it
reads as a meter, not a voice. Instead the bands are mapped to different
things:

| Signal | Effect |
|---|---|
| low frequencies | body radius, swirl |
| mid | interior turbulence, noise scale |
| high | rim sharpness, silhouette detail, hue drift |
| onsets | a short bright flash |
| envelope | overall bloom |
| silence *during* speech | body eases back, interior keeps moving |

So consonants, vowels, loud speech, quiet speech and mid-sentence pauses all
look different. The features come from the actual synthesised audio, walked at
wall-clock speed as it plays, so the orb moves *with* the voice rather than
ahead of it.

## Reproducibility

Every session logs its seed; `--seed 42` replays the exact visual sequence.
Visual bugs are otherwise almost impossible to report, and "it did something
odd about a minute in" is not a bug report.

```bash
cookie-interface --seed 42
```

## The shader

`shaders/orb.wgsl`. One full-screen triangle; the body is a sphere raymarched
in screen space with a domain-warped FBM field standing in for plasma, plus a
shell mask, a rim light and an outer bloom. Eighteen march steps, four octaves
— enough to look alive, cheap enough for integrated graphics at 60 fps while a
model loads on the CPU.

Output is premultiplied alpha, which is what a transparent window needs to
composite correctly. Where a compositor refuses alpha, the renderer says so and
draws opaque instead of producing a black box.

## Theme

```toml
[ui.theme]
hue = 28.0            # brown. 276 is the violet of the original reference.
hue_secondary = 42.0  # amber core
saturation = 0.78
intensity = 1.0
glow = 1.0

[ui.animation]
variation = 0.85      # 0 = calmest behaviour always, 1 = uniformly random
reactivity = 1.0      # scales every audio-driven displacement
reduce_motion = false # slower, smaller, no flicker
```

`reduce_motion` is honoured by every behaviour through `ctx.intensity`, not
bolted on afterwards.
