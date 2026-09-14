# Configuration

`cookie-interface --config` prints the file and its path; `--paths` prints
every directory. Edit the file, or use `--set`:

```bash
cookie-interface --set tts.voice.rate=0.95 --set ui.theme.hue=28
```

Unknown keys are rejected rather than ignored, so a typo tells you.

## Where it lives

| | Linux | macOS | Windows |
|---|---|---|---|
| config | `~/.config/cookie-interface/` | `~/Library/Application Support/…` | `%APPDATA%\…\config` |
| data | `~/.local/share/cookie-interface/` | `~/Library/Application Support/…` | `%APPDATA%\…\data` |
| cache | `~/.cache/cookie-interface/` | `~/Library/Caches/…` | `%LOCALAPPDATA%\…\cache` |

Set `COOKIE_INTERFACE_HOME` to put everything under one directory (useful for
testing multiple instances).

## `[api]`

```toml
bind = "127.0.0.1"      # loopback. Changing this also needs allow_remote.
port = 8787
allow_remote = false    # exposing a live microphone must be deliberate
auth_token = ""         # required for any non-loopback bind
max_body_bytes = 1048576
max_subscribers = 32
```

## `[backend]`

See [backend.md](backend.md). Off by default — an interface that silently
phoned home would be a surprising default for a microphone application.

## `[audio]` and `[vad]`

```toml
[audio]
sample_rate = 16000        # what Whisper-family models want
frame_ms = 20
duck_input_while_speaking = true   # turn off with a headset for true barge-in
listen_on_start = true             # off for push-to-talk via POST /v1/listen

[vad]
threshold_db = 9.0     # above the adaptive noise floor
speech_ms = 120        # before declaring speech started
silence_ms = 700       # before declaring it finished
preroll_ms = 300       # kept so the first syllable survives
min_utterance_ms = 250 # judged on voiced audio, not buffer length
```

## `[stt]` and `[tts]`

See [models.md](models.md). The voice:

```toml
[tts.voice]
id       = "auto"     # let the provider pick for this platform
language = "en-GB"
gender   = "female"
style    = "warm"     # only sent to providers that support style
rate     = 1.0
pitch    = 0.0        # semitones
```

Check `/v1/state` for what your provider actually honours.

## `[retention]`

```toml
policy = "24h"          # immediate | 1h | 24h | 7d | 30d | forever
sweep_interval_minutes = 30
max_files = 500
max_bytes = 536870912
```

Enforced at every startup before normal operation, including after a crash.
Shortening applies retroactively.

## `[ui]`

```toml
presentation = "cursor-companion"   # window | floating-orb | docked
width = 160
height = 160
transparent = true
click_through = true
always_on_top = true
cursor_offset = [34.0, 26.0]
cursor_follow_lag = 0.0    # 0 = pinned to the pointer, no interpolation
target_fps = 60
unfocused_fps = 30

[ui.visibility]
when_idle = false          # the setting that decides presence vs clutter
when_listening = true
when_working = true
when_speaking = true
when_error = true
when_observing = true
linger_seconds = 1.2
fade_seconds = 0.35

[ui.theme]
hue = 28.0                 # brown, as in cookie. 276 is the violet reference.
hue_secondary = 42.0
saturation = 0.78

[ui.animation]
seed = 0                   # omit for random; it is logged either way
variation = 0.85           # 0 = calmest behaviour always, 1 = full variety
reactivity = 1.0
reduce_motion = false
```
