# Troubleshooting

Start here:

```bash
cookie-interface --doctor
```

Or just ask her: **"Cookie, are you alright?"** — same checks, spoken answer.

## She can't hear me

`--doctor` will say which. In order of likelihood: no input device (plug one
in); the wrong one is selected (`--set audio.input_device="Yeti"`, substring
match); the recogniser is the mock (`--set stt.provider=http`); or the VAD
never fires, which means `vad.threshold_db` is too high for a quiet
microphone — try 6.

Watch it live: `curl -sN localhost:8787/v1/events?types=speech.detected`.

## She hears herself

`audio.duck_input_while_speaking = true` (the default) suppresses capture while
speaking. If you have a headset and want true barge-in, turn it off.

## No orb

Not fatal — everything else still works, and `--doctor` will say "my face"
failed. Usually a compositor that refuses transparent always-on-top surfaces
(the renderer falls back to opaque), or no usable GPU. `--no-ui` makes it
explicit.

On Wayland, cursor-following depends on the compositor allowing a client to
position its own window; some do not. `presentation = "floating-orb"` is the
fallback.

## The orb won't leave / won't appear

`[ui.visibility]`. By default she is absent while idle and appears when there
is a reason. `when_idle = true` pins her on screen permanently.

## Port already in use

`--port 8788`. `--doctor` distinguishes "in use by me" from "in use by
something else".

## The backend is unreachable

The interface still starts, still listens, still speaks. `--doctor` reports it,
and so does she. Check `backend.base_url` — it needs a scheme
(`http://192.168.1.42:8080/api`), and `https://` needs `--features tls`.

## "Stop" isn't stopping the task

Correct. "Stop" stops the talking. To abandon work say "cancel that", or
`POST /v1/cancel`. They are separate on purpose — otherwise you could never
interrupt a sentence without losing the job.

## Audio files are piling up

`GET /v1/retention` shows the policy and count. The sweep runs at startup and
every 30 minutes. `--clear-audio` empties it now; `--set
retention.policy=immediate` stops keeping any.

## A visual bug I can't reproduce

Every session logs its animation seed. Re-run with `--seed <n>` and the whole
visual sequence repeats exactly.

## Build fails on Linux

`alsa-sys` needs ALSA headers: `sudo apt install libasound2-dev`. To build
without any audio hardware support: `--no-default-features --features
http-providers`.
