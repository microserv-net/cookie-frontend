# Development

```bash
cargo test --no-default-features --features audio-io,http-providers   # 248 tests
cargo clippy --all-targets -- -D warnings
cargo fmt
cargo run -- --doctor
```

Skipping `ui` locally is worth it: wgpu and winit dominate build time, and
almost nothing you will change depends on them.

## The rule about tests

`cargo test` must never require a microphone, a speaker, a GPU, a network or a
downloaded model. If a test starts needing one, something has leaked out of
the provider abstraction and that is the bug — not the test.

Hardware is exercised by `--test` (a full spoken round trip) and `--doctor`
(every capability, non-destructively).

## Where things are

```
src/
  audio/       capture, playback, ring buffer, resampling, features, VAD
  stt/ tts/    provider traits and four implementations each
  engine/      the orchestrator loop and the speech task
  api/         HTTP and WebSocket
  animation/   behaviours and the director
  renderer/    winit + wgpu; shaders/orb.wgsl
  intent.rs    local intent inference
  diagnostics.rs  the checks behind --doctor and "are you alright?"
  tasks.rs     backend task registry and scheduling hints
  backend.rs   the outbound client
  retention/   the audio ledger
```

## Adding things

**A model provider.** Implement `SpeechRecognizer` or `SpeechSynthesizer`, add
a variant to the provider enum in `config`, and wire it in `build()`. Report
capabilities honestly — a provider that ignores `pitch` says `pitch: false`,
and `/v1/state` publishes that to clients so they never send a parameter into
the void.

**An animation behaviour.** One struct implementing `Behavior`, one line in
`registry()`. See [animation.md](animation.md). Nothing else changes.

**An API field.** Additive only. Within `/v1`, fields may be added and new
event types may appear; nothing is removed or repurposed. Request bodies use
`deny_unknown_fields` so a typo tells you; clients must ignore unknown
response fields.

**A local intent.** Think hard first. The default answer is that it belongs to
the backend, and `intent.rs` explains the three-part test something has to
pass to live here instead.

## Conventions worth knowing

- No blocking calls on async runtime threads. Disk and CPU-bound work goes
  through `spawn_blocking`; the retention sweep is the canonical example.
- No lock held across an `await`. There is one lock in the engine and it
  guards the speaker for microseconds.
- Bounded channels everywhere, so an overloaded producer applies backpressure
  instead of growing memory.
- `unsafe` is confined to `audio/ring.rs`. Adding a second block needs a
  reason as good as that one's.
- Errors carry a `code()` for machines and a `hint()` for people. If a failure
  has an obvious next step, say it.

## Debugging

```bash
RUST_LOG=cookie_interface=debug cargo run
cargo run -- --seed 42            # reproduce a visual bug exactly
cargo run -- --paths              # where everything lives
curl -sN localhost:8787/v1/events # watch the pipeline live
```

Every session logs its animation seed at startup. "It did something odd about
a minute in" is not a bug report; `--seed 42` is.

## CI

Four jobs: fmt and clippy; the test suite on Linux, macOS and Windows; a
`cargo check` of the renderer on all three; and rustdoc with `-D warnings`.

The renderer is only compiled, never run — there is no display on a runner,
and a job that pretended otherwise would be worse than no job at all.
