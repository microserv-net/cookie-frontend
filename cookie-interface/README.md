# cookie-interface

The front-facing voice interface for the Cookie assistant: a microphone, a
voice, and a small brown orb that lives beside your mouse pointer.

It contains **no language model, no agent, no memory and no reasoning.** It
gives Cookie ears, a voice and a face. The mind is a separate program — see
[docs/backend.md](docs/backend.md) — and this works perfectly well with no
backend at all, driven entirely over its local HTTP API.

```
        you
         │  speech
         ▼
 ┌───────────────────────────────────────────┐
 │  microphone → VAD → recognition           │
 │        ↕                                  │
 │  local HTTP API  ←→  your application     │
 │        ↕              or the Cookie       │
 │  synthesis → speaker      backend         │
 │        ↕                                  │
 │  audio features → orb (wgpu / WGSL)       │
 └───────────────────────────────────────────┘
```

## Running it

```bash
cargo run --release                      # the orb, the API, the microphone
cargo run --release -- --doctor          # is everything working?
cargo run --release -- --test            # ask my name, listen, greet me
cargo run --release -- --port 8787       # somewhere else
cargo run --release -- --no-ui           # headless
```

Nothing is downloaded during `cargo build`. The binary works immediately using
your operating system's voice — but it cannot transcribe until you fetch a
recogniser:

```bash
cargo run --release -- --setup           # ~700 MB, once
```

That fetches a prebuilt sherpa-onnx, `whisper-large-v3-turbo` for hearing, and
Kokoro for the voice, then writes a configuration that uses them. The default
voice is `bf_emma`: British, female, unhurried. See
[docs/models.md](docs/models.md).

### Linux needs ALSA headers

```bash
sudo apt install libasound2-dev        # Debian/Ubuntu
sudo dnf install alsa-lib-devel        # Fedora
```

macOS and Windows need nothing extra.

## The orb

By default Cookie is a 160×160 transparent, click-through, always-on-top window
pinned to your mouse pointer with no interpolation — where your eyes already
are, rather than in a window somewhere else.

**She is not always there.** The orb is absent while idle and appears when
there is a reason: you called her, she is working, she is speaking, she is
watching the screen, or something is wrong. It fades out a beat later. All of
that is configurable under `[ui.visibility]`.

The orb is not an animation. It is a raymarched plasma field driven by ~30
parameters that behaviours write every frame — eighteen of them, several per
state — chosen by a seeded scheduler and crossfaded, so it does not repeat.
While Cookie speaks it reacts to the actual synthesised audio: low frequencies
push the body, mids drive interior turbulence, highs sharpen the rim, onsets
flash. Quiet speech, loud speech, consonants and pauses all look different.

`--seed 42` makes the whole visual sequence reproducible.

## Asking her things

Three requests are handled by the interface itself, because they cannot wait
for a round trip and must work when the backend is down:

- **"Cookie, are you alright?"** — runs full diagnostics and answers in plain
  English. So does "run diagnostics", "is everything working", "can you hear
  me", "is anything broken". It is fuzzy matching, not a command list.
- **"Stop."** — stops *talking*. It never cancels work.
- **"Cancel that."** — abandons the work. A different thing, deliberately.

Everything else goes to the backend untouched.

## The API

Streaming runs both directions. Full reference in [docs/api.md](docs/api.md).

```bash
curl -s localhost:8787/v1/health
curl -s localhost:8787/v1/diagnostics | jq .summary

curl -s localhost:8787/v1/speak -H 'content-type: application/json' \
     -d '{"text":"Good evening."}'

# Pipe tokens in; she starts speaking at the first sentence boundary.
printf '{"text":"Good evening. "}\n{"text":"The kettle has boiled."}\n{"end":true}\n' \
  | curl -s -X POST localhost:8787/v1/speak/stream --data-binary @-

# Hear what was said to her.
curl -sN localhost:8787/v1/transcripts
```

## Replacing the models

Recognition and synthesis are traits. Four providers ship: `mock` (no model),
`system` (the OS voice), `http` (any OpenAI-compatible endpoint, Kokoro,
Qwen3-TTS) and `sidecar` (a local process speaking four lines of JSON —
whisper.cpp, sherpa-onnx, anything Python). See
[docs/models.md](docs/models.md).

```toml
[stt]
provider = "http"
model    = "whisper-large-v3-turbo"
endpoint = "http://127.0.0.1:8080/v1/audio/transcriptions"
```

## Where things live

`--paths` prints them. Configuration, cache, data, generated audio and logs are
separated, and every path comes from the `directories` crate rather than being
hard-coded.

Generated speech is governed by a retention policy that is actually enforced:
an append-only ledger, a sweep at every startup, orphan adoption, and recovery
from a session that crashed mid-write. Shortening the policy applies
retroactively. Nothing outside the managed audio directory is ever deleted.

## Development

```bash
cargo test --no-default-features --features audio-io,http-providers   # 240 tests, no hardware needed
cargo clippy --all-targets
cargo fmt
```

`cargo test` never requires a microphone, a speaker, a GPU or a downloaded
model. Hardware is exercised by `--test` and `--doctor`. More in
[docs/development.md](docs/development.md); the audio path is explained in
[docs/audio.md](docs/audio.md).

## Licence

MIT OR Apache-2.0.
