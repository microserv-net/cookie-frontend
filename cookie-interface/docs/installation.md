# Installation

## What you need

Rust 1.80 or newer. That is genuinely the whole list on macOS and Windows.

On Linux, cpal needs ALSA headers:

```bash
sudo apt install libasound2-dev     # Debian, Ubuntu
sudo dnf install alsa-lib-devel     # Fedora
sudo pacman -S alsa-lib             # Arch
```

Nothing is downloaded during the build. No model, no CMake, no Python. The
binary you get works immediately using your operating system's own voice.

## Build and run

```bash
git clone <your fork>
cd cookie-interface
cargo build --release
./target/release/cookie-interface --doctor    # check before you commit to it
./target/release/cookie-interface             # the orb, the API, the microphone
```

`--doctor` tells you, in plain English, which capabilities work on this
machine and what to do about the ones that do not.

## Feature flags

The default build is `ui,audio-io,http-providers`.

| Flag | What it adds | Cost |
|---|---|---|
| `ui` | The orb: wgpu, winit, the shader | slower build; needs a GPU at runtime, degrades gracefully without one |
| `audio-io` | Real microphone and speaker via cpal | ALSA headers on Linux |
| `http-providers` | HTTP model providers and the backend client | reqwest, plain HTTP only |
| `tls` | `https://` endpoints | a C toolchain, for aws-lc |

Useful combinations:

```bash
# Headless: API, speech and hearing, no window. Good for a server or a Pi.
cargo build --release --no-default-features --features audio-io,http-providers

# CI: everything testable, no GPU, no sound card, no network.
cargo test --no-default-features --features audio-io,http-providers

# Reaching a hosted model over TLS.
cargo build --release --features tls
```

`tls` is opt-in because rustls pulls aws-lc-rs, which needs a C compiler — a
surprising thing to discover halfway through a build when all you wanted was
localhost.

## First run

```bash
cookie-interface --setup
```

Creates the directories, writes a default configuration, and fetches any model
you have declared a `model_url` for (verifying the checksum, resuming rather
than restarting). It is idempotent; run it as often as you like.

```bash
cookie-interface --paths      # where everything lives on this platform
cookie-interface --config     # the current configuration and its path
```

## Then

```bash
cookie-interface --test
```

Speaks a prompt, listens, transcribes, and greets you by name. It exercises the
microphone, the VAD, recognition, synthesis, playback and the state machine in
the order they are actually used — so if it stops at a particular step, that
step is the thing that is wrong.

## Running more than one

```bash
COOKIE_INTERFACE_HOME=/tmp/cookie-b cookie-interface --port 8788
```

The environment variable relocates config, data and cache together, so two
instances share nothing.

## As a service

The interface wants a user session (audio devices, a display for the orb), so
prefer a user service over a system one.

```ini
# ~/.config/systemd/user/cookie-interface.service
[Unit]
Description=Cookie voice interface
After=graphical-session.target

[Service]
ExecStart=%h/.cargo/bin/cookie-interface
Restart=on-failure
Environment=RUST_LOG=cookie_interface=info

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now cookie-interface
```

On macOS use a `launchd` `LaunchAgent` (not a `LaunchDaemon`, which has no
audio session). For headless use, add `--no-ui`.
