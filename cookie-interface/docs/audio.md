# Audio

## The path a sound takes

```
microphone
   │  device format: i16 / u16 / i32 / f32 / f64, 1-8 channels, 44.1k or 48k
   ▼
cpal callback  ── real-time context: no allocation, no locks, no I/O
   │  converted to f32 once, here, at the boundary
   ▼
lock-free SPSC ring        ← the only `unsafe` in the crate
   │
   ▼
CaptureHandle.drain()  ── downmix to mono, resample to 16 kHz
   │
   ├─▶ AudioAnalyzer ─▶ AudioFeatures (watch) ─▶ the orb
   │
   ▼
EnergyVad ─▶ utterance ─▶ SpeechRecognizer
```

and back out:

```
SpeechSynthesizer ─▶ chunks ─▶ speech task
                                   │  walked at wall-clock speed
                                   ├─▶ AudioFeatures (watch) ─▶ the orb
                                   ▼
                         PlaybackHandle.write() ── resample per chunk
                                   │
                                   ▼
                            ring ─▶ cpal callback ─▶ speaker
```

## Why there is `unsafe`

The audio callback is a real-time context. It runs on a thread the operating
system will not wait for, and if it allocates, takes a lock, touches the
filesystem or blocks on a channel, you hear it — a click, a dropout, or in the
bad case a stall that outlives the callback.

An `Arc<Mutex<Vec<f32>>>` is not safe there, because the other side might be
holding the lock while it does something slow. A `tokio::mpsc` is not safe
there either, because sending can allocate.

So samples cross the boundary through a lock-free single-producer,
single-consumer ring: two atomics, a preallocated buffer, and about forty
lines. The soundness argument is written out above the code, the invariants
are asserted in tests, and it is the only `unsafe` block in the project.
Everything above the ring is ordinary async Rust.

Overflow drops the oldest samples and increments a counter rather than
blocking the producer. If the consumer is that far behind, the old audio is
already useless, and a real-time thread must never wait.

## One format, once

Devices offer whatever they offer. The conversion to f32 mono happens exactly
once, at the device boundary, and everything above it is f32 mono at
`audio.sample_rate` (16 kHz by default, which is what Whisper-family models
want).

This removes a whole class of bugs. If a rate or a channel count leaks upward,
you get chipmunk speech or half-speed speech, and the symptom appears a long
way from the cause.

## Resampling

Cubic Hermite, with carried phase and a three-sample history so chunk
boundaries do not click. Two forms: a one-shot `resample()` and a stateful
`Resampler` for streams.

The stateful one had a bug worth remembering. Phase was carried with the
history offset added twice, so three input samples were skipped per chunk — at
48k→16k that is one output sample per chunk, 0.3% of the audio, silently. It
was found by a test asserting 48000 samples in produced 16000 out, and it got
15954. Exact-count assertions on resamplers earn their keep.

## Features

A 512-point Hann-windowed FFT every frame, producing:

| | |
|---|---|
| `rms`, `peak`, `level_db` | loudness, the last one in dB for display |
| `envelope` | fast attack, slow release |
| `low` / `mid` / `high` | band energy, split at 60 / 300 / 2000 Hz |
| `centroid` | spectral brightness |
| `flux`, `onset` | transients — consonants, plosives |
| `zcr`, `voiced` | voicing heuristic |
| `noise_floor_db` | adaptive, so a noisy room self-calibrates |

Cheap enough to run every 20 ms on one core, detailed enough for the orb to
tell a consonant from a vowel.

`decay_only()` exists so that when capture stops, the features keep decaying
instead of freezing — otherwise the orb stops mid-gesture, which looks like a
crash.

## Voice activity detection

Energy against an adaptive noise floor, smoothed into an evidence signal
(65% loudness, 35% voicing), with four phases: Silence, Onset, Speech,
Hangover.

Three settings matter:

- `speech_ms` (120) — evidence must persist this long before declaring speech.
  Stops a door slam from opening an utterance.
- `silence_ms` (700) — how long a pause may be before the utterance is
  considered finished. Too short and you cut people off mid-thought; too long
  and the assistant feels slow.
- `preroll_ms` (300) — audio kept from *before* the onset, so the first
  syllable is not lost. Without it, "Cookie" is transcribed as "ookie".

`min_utterance_ms` is judged on **voiced** audio, not on buffer length. The
buffer also holds the pre-roll and the hangover tail, so measuring it would
let a 200 ms cough report 850 ms and sail past a 600 ms threshold — which is
exactly what it was doing before a test caught it.

## Ducking and barge-in

While Cookie speaks, capture is suppressed (`duck_input_while_speaking`) so
she does not transcribe herself through the speakers. With a headset you can
turn it off and get true barge-in: speech detected while speaking interrupts
her immediately, and the human always wins.

The interruption is felt within a frame because the speech task never queues
more than about 120 ms of audio ahead — see the pacing note in
[architecture.md](architecture.md).

## Testing without hardware

`MockInput` replays silence, fixed samples, or scripted utterances;
`MockOutput` records what it was asked to play. Both can run in real time or as
fast as possible. They are not scaffolding bolted on for tests — they are what
`--no-ui` and a missing microphone fall back to, and what makes `cargo test`
run on a machine with no sound card.
