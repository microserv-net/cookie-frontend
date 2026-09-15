# Models

## What `--setup` installs

```bash
cookie-interface --setup      # about 700 MB, once
```

| | | |
|---|---|---|
| runtime | sherpa-onnx, prebuilt for your platform | 20–45 MB |
| hearing | `whisper-large-v3-turbo` | 564 MB |
| voice | Kokoro, British female (`bf_emma`) | 103 MB |

Downloads are checksummed where upstream publishes a digest, written to a
`.part` file and renamed only when the checksum matches, so an interrupted
setup resumes rather than leaving a half-model that looks complete. Archives
are deleted after extraction; re-fetching is one command and half a gigabyte
of cache nobody knows about is not.

Everything is then located by *searching* the extracted tree rather than by
hard-coded filenames, because upstream renames files between releases and a
setup that breaks on a rename is a setup that breaks. The paths are written
into your config, so you can see exactly what is being used and change it.

`--setup --config-only` prepares directories and configuration without
downloading anything, for a machine that points at a model server instead.

### The voice

Kokoro identifies speakers by index. 7 is `bf_emma` — the warm, unhurried
British voice — and it is the default. `tts.voice.id` accepts the index or
the name:

```toml
[tts.voice]
id = "bf_isabella"   # or "8", or "bf_emma", or "7"
```

`rate` works; `pitch` does not, because Kokoro has no pitch control and a knob
that does nothing is worse than no knob. `GET /v1/state` says so.

### Why a subprocess

The models run in sherpa-onnx's own command-line tools rather than through its
C API. Linking would mean this crate could not build without the shared
libraries present and `cargo test` would need a 600 MB download; a process per
utterance costs tens of milliseconds against a model that takes hundreds. That
is a bad trade only when transcribing continuously, which a voice assistant
does not — it transcribes one utterance, after you stop speaking.

## Why there is no model in the binary

The best speech runtimes are C++ (whisper.cpp, sherpa-onnx) or Python
(Qwen3-TTS, Kokoro). Linking either would mean `cargo build` needs CMake, a C++
toolchain or a Python environment — on three operating systems, for a project
whose point is being easy to run. And downloading gigabytes during a build
makes builds fail on aeroplanes and behind proxies.

So models live outside the process and speak one of three protocols.

## `system` — no setup at all

The default. macOS `say` (Serena), Windows SAPI (Hazel), Linux `espeak-ng`
(`en-gb+f3`). Quality is below a modern model, but macOS and Windows both ship
convincing British female voices, and it works on a machine you just sat down
at.

## `http` — a model server

```toml
[stt]
provider = "http"
model    = "whisper-large-v3-turbo"
endpoint = "http://127.0.0.1:8080/v1/audio/transcriptions"

[tts]
provider = "http"
dialect  = "openai-compatible"    # or "qwen3"
model    = "kokoro"
endpoint = "http://127.0.0.1:8081/v1/audio/speech"
```

Works with whisper.cpp's `server`, faster-whisper-server, Speaches, vLLM,
Kokoro-FastAPI, openedai-speech, LocalAI, or a hosted API. Point it at
`127.0.0.1` and no audio leaves the machine.

Two TTS dialects are supported: `openai-compatible`
(`{model, input, voice, response_format, speed}`) and `qwen3` (DashScope's
`{model, input:{text, voice}, parameters:{…}}`). When the server returns raw
PCM the audio is forwarded chunk by chunk as it arrives.

`https://` needs `--features tls` (it costs a C toolchain, which is why it is
opt-in).

## `sidecar` — a local process

```toml
[stt]
provider = "sidecar"
sidecar_command = ["python3", "/opt/cookie/whisper_sidecar.py"]
```

One JSON object per line, both directions. Four message types. Here is a
complete conforming sidecar:

```python
import sys, json, base64, numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", compute_type="int8")
print(json.dumps({"type": "result", "ready": True}), flush=True)

for line in sys.stdin:
    req = json.loads(line)
    if req.get("op") == "hello":
        print(json.dumps({"id": req["id"], "type": "result",
                          "model": "large-v3-turbo"}), flush=True)
        continue
    audio = np.frombuffer(base64.b64decode(req["audio"]), dtype=np.float32)
    segments, info = model.transcribe(audio, language=req.get("language"))
    text = " ".join(s.text for s in segments).strip()
    print(json.dumps({"id": req["id"], "type": "result", "text": text,
                      "language": info.language,
                      "confidence": info.language_probability}), flush=True)
```

A TTS sidecar answers `{"op":"synthesize"}` with one or more
`{"type":"chunk","sample_rate":24000,"encoding":"f32le","audio":"<base64>"}`
lines followed by `{"type":"end"}`. Streaming chunks start the voice — and the
orb — hundreds of milliseconds earlier.

`{"type":"error","message":"…"}` ends any exchange. Anything on stderr is
forwarded to the log, which is how model-loading progress and Python
tracebacks reach the user. A sidecar that dies is re-spawned automatically;
the cost is one failed request, not the session.

## `mock` — no model

Deterministic. The recogniser replays a script; the synthesiser produces
speech-*shaped* audio — a pitched source with two formants, an envelope driven
by the real text's syllables, pauses at punctuation. Not words, but enough to
exercise the orb, the ducking, the barge-in and the whole streaming path with
nothing downloaded. This is what `cargo test` uses.

## Downloading

```toml
[stt.options]
model_url    = "https://example.invalid/whisper-large-v3-turbo.onnx"
model_sha256 = "9f2c…"
```

`cookie-interface --setup` fetches it to a `.part` file and renames it only
after the checksum matches, so an interrupted setup never leaves a half-model
that looks complete. A file whose checksum stops matching is re-fetched.
