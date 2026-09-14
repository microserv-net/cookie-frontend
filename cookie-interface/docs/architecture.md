# Architecture

## The shape of it

```
   microphone ─▶ CpalInput ─▶ lock-free ring ─▶ CaptureHandle
                                                     │ f32 mono @16k
                                                     ▼
                                    AudioAnalyzer ─▶ EnergyVad
                                          │               │ utterance
                              AudioFeatures (watch)       ▼
                                          │       SpeechRecognizer
                                          │               │ transcript
                                          │               ▼
                                          │         IntentEngine ──▶ diagnostics
                                          │               │            / interrupt
                                          │               │            / cancel
                                          │               ▼
   your app ──HTTP──▶ api ──commands──▶ engine loop ──▶ BackendClient
                          ◀──events───     │  ▲              │ replies
                                           ▼  │              │
                                   SpeechSynthesizer ◀───────┘
                                           │ chunks
                                           ▼
                                   PlaybackHandle ─▶ ring ─▶ speaker
                                           │
                              AudioFeatures (watch)
                                           ▼
                            AnimationDirector ─▶ OrbParams ─▶ WGSL
```

Every arrow is a channel, not a lock.

## One loop, not shared state

The alternative — a task per subsystem sharing a `Mutex<AppState>` — is how
audio applications end up with priority inversion and stalls nobody can
reproduce. Here one task owns the state machine, the capture handle, the VAD
and the speaker. A state transition is an ordinary function call; there is
nothing to contend over.

Everything slow happens in spawned tasks that report back over a bounded
channel: recognition, synthesis, disk, the backend, retention sweeps
(`spawn_blocking`). Nothing in the loop blocks, and nothing in the loop touches
the renderer.

There is exactly one lock in the engine. It guards the speaker, is held for
microseconds, and never crosses an `await`.

## Three transports on the bus

Different data wants different delivery:

- **`broadcast`** for discrete events. Bounded; a slow subscriber lags and is
  told so rather than growing memory.
- **`watch`** for audio features. Only the latest frame matters — a renderer
  that missed three frames wants the current one, not a backlog.
- **`watch`** for voice state. Latched, so a client that connects late still
  learns the state immediately.

## Real-time discipline

The audio callbacks allocate nothing, lock nothing, and touch neither the
filesystem nor the network. They move samples through a lock-free SPSC ring —
the crate's only `unsafe`, about forty lines, with the soundness argument
written out above it. Downmixing, resampling and analysis all happen on the
async side of that ring.

Device formats (i16, u16, i32, f64…) are converted exactly once, at the device
boundary. Above it everything is f32 mono at one rate, which removes a whole
class of "why is it chipmunk speed" bugs.

## Speech pacing

The speech task walks synthesised audio at wall-clock speed, publishing
features as the user hears them. That is both the pacing mechanism and the
alignment mechanism: because we only fetch the next chunk after finishing the
current one in real time, the speaker queue never runs more than ~120 ms ahead,
so barge-in is felt within a frame and the orb moves *with* the voice rather
than ahead of it. No watermarks to get wrong.

## Providers

`SpeechRecognizer` and `SpeechSynthesizer` are object-safe traits returning
boxed futures. Four implementations each; swapping one is a config change.

Whisper-family models are chunk models, not streaming models. Rather than make
every provider fake a streaming API, the trait exposes one honest operation —
"turn this audio into text" — and the engine produces partials by
re-transcribing the utterance-so-far at an interval, which is how streaming
Whisper UIs actually work. A provider that does better advertises
`native_partials`.

## State

An explicit state machine with a hand-written transition table. Every entry is
a product decision: barge-in moves Speaking → Listening, recognition finishing
cannot yank you out of Speaking, interrupting while idle is a no-op. A test
walks every reachable state with every trigger and asserts no sequence produces
a self-transition or an undefined state.

## The orb

Parameters, not animations. Eighteen behaviours write ~30 floats; a seeded
weighted scheduler picks between them and crossfades in parameter space. The
shader raymarches a domain-warped FBM field — 18 steps, 4 octaves — which runs
at 60 fps on integrated graphics.

Losing the GPU is a degradation, never a failure: speech, hearing and the API
carry on, and diagnostics reports that Cookie has no face.

## Where the boundary is

This crate does hearing, speaking, visual presence, state and the protocol.

It does not do thinking. The one exception is
[`intent`](../src/intent.rs), which recognises three requests the interface
owns — "are you alright", "stop talking", "cancel that" — because they must
work when the backend is unreachable, which is exactly when they are asked.
Matching is fuzzy and the confidence bar rises when a backend is available,
because the backend can do better.
