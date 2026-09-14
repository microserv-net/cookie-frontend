# HTTP API

Everything lives under `/v1` and binds to `127.0.0.1:8787` by default.

## Compatibility

Within `v1`: fields may be **added** to any request or response, and new event
types may appear at any time. Nothing is removed or repurposed. **Clients must
ignore unknown fields and unknown event types.**

Reserved for later: a `visual` object on `/v1/speak`, `ui.*` event types, and a
`capabilities` block on `/v1/state` describing what the running interface can
render. None of it exists yet; the extension points are named so that clients
written today keep working.

## Security

Loopback by default. Binding elsewhere requires **both** `api.allow_remote =
true` and `api.auth_token` — two locks on the same door, because the thing
behind it is a live microphone. When a token is set, every endpoint except
`/v1/health` requires `Authorization: Bearer <token>`. Bodies are capped at
`api.max_body_bytes` (1 MiB).

## Status

### `GET /v1/health`
```json
{"status":"ok","version":"0.1.0","api_version":"v1","uptime_seconds":412}
```

### `GET /v1/state`
```json
{
  "state": "idle",
  "since_seq": 219,
  "providers": {
    "stt": {"name":"http:whisper-large-v3-turbo",
            "capabilities":{"timestamps":true,"confidence":false,
                            "language_detection":true,"prompt":true,
                            "native_partials":false,"sample_rate":16000}},
    "tts": {"name":"system:say",
            "capabilities":{"streaming":false,"rate":true,"pitch":false,
                            "style":false,"voice_listing":true,
                            "sample_rate":22050}}
  },
  "audio": {"input_device":"MacBook Pro Microphone","hardware":true,"level_db":-58.2},
  "backend": "http://192.168.1.42:8080/api/v1/chat",
  "voice": {"id":"auto","language":"en-GB","gender":"female","rate":1.0}
}
```

`capabilities` is not aspirational. A provider that ignores `pitch` reports
`pitch: false`, so you never send a parameter into the void.

### `GET /v1/diagnostics?probe_devices=true&speak=false`

The same checks as `--doctor` and as "Cookie, are you alright?".

```json
{
  "overall": "degraded",
  "summary": "Mostly. My hearing is running on a fallback. I'm using a simulated microphone, so I can't actually hear the room.",
  "checks": [
    {"name":"audio.input","capability":"my hearing","health":"degraded",
     "detail":"…","hint":"…","elapsed_ms":3}
  ],
  "elapsed_ms": 64
}
```

`health` is `ok`, `degraded`, `failed`, `disabled` or `unknown`. `unknown`
never sets `overall` — it means a check could not be settled without being
intrusive, not that something is wrong.

## Speaking

### `POST /v1/speak`
```bash
curl -s localhost:8787/v1/speak -H 'content-type: application/json' -d '{
  "text": "The build finished. Two tests are still failing.",
  "voice": {"rate": 0.95},
  "interrupt": true,
  "persist": true
}'
```
```json
{"utterance_id":"9f2a…","accepted":true,"events":"/v1/events"}
```

Unknown fields are **rejected**, so a typo is not silently ignored. Follow
progress on `/v1/events` filtered by `utterance_id`.

### `POST /v1/speak/stream`

Newline-delimited JSON in, newline-delimited JSON out. The request body is
consumed as it arrives — pipe model tokens straight in and Cookie speaks the
first sentence while the rest is still being generated.

```
{"text":"Good evening. ","voice":{"rate":0.95}}
{"text":"The kettle has boiled."}
{"end":true}
```
```
{"type":"started","utterance_id":"1c0…"}
{"type":"accepted","characters":14}
{"type":"end","utterance_id":"1c0…"}
```

Closing the connection without `{"end":true}` still speaks the tail; a dropped
client does not leave Cookie holding half a sentence.

### `POST /v1/interrupt`
```bash
curl -s -X POST localhost:8787/v1/interrupt -d '{}' -H 'content-type: application/json'
```
Stops speech. **Never touches backend work.**

## Listening

- `POST /v1/listen` — `{"continuous": true}`
- `POST /v1/listen/stop`
- `POST /v1/audio?sample_rate=16000` — raw 16-bit little-endian mono PCM for
  callers with their own microphone. Bypasses the VAD: you decide where the
  utterance begins and ends.

## Events out

### `GET /v1/events?types=transcript.final,state`

Server-sent events. `Last-Event-ID` resumes; events already delivered are
skipped. This is a live stream, not a log — a long absence loses events.

```
id: 412
event: transcript.final
data: {"seq":412,"ts_ms":1739,"type":"transcript.final","utterance_id":"…",
       "text":"open the project I was working on","confidence":0.94,
       "language":"en","duration_ms":1820,"segments":[]}
```

Event types: `ready`, `state`, `listening.started`, `speech.detected`,
`speech.ended`, `transcript.partial`, `transcript.final`, `listening.stopped`,
`speak.started`, `speak.chunk`, `speak.finished`, `interrupted`,
`provider.status`, `task`, `intent`, `diagnostics`, `retention.swept`, `error`.

### `GET /v1/transcripts`

The same stream narrowed to speech — the common case for a backend.

### `GET /v1/stream` (WebSocket)

Both directions on one connection. Events arrive as text frames; commands go
the other way:

```json
{"type":"speak","text":"Good evening."}
{"type":"speak.open","utterance_id":"u1"}
{"type":"speak.delta","utterance_id":"u1","text":"one sentence at a time. "}
{"type":"speak.end","utterance_id":"u1"}
{"type":"interrupt"}
{"type":"cancel","task_id":"task-18"}
{"type":"listen","continuous":true}
{"type":"diagnostics","speak":true}
{"type":"audio","sample_rate":16000,"pcm":"<base64 s16le>"}
```

Binary frames are raw 16-bit PCM at the pipeline rate. Unknown types are
ignored, not fatal — that is the surface future visual commands will extend.

## Work

### `GET /v1/tasks`
```json
{"active":[{"id":"task-18","title":"fixing the failing tests","state":"running",
            "weight":"heavy","priority":"normal","detail":"running the test suite",
            "age_seconds":41}],
 "summary":"I'm fixing the failing tests. Right now: running the test suite",
 "heavy_in_flight":true}
```

### `POST /v1/cancel`
```json
{"task_id": "task-18"}   →   {"cancelled":["task-18"]}
```
Omit `task_id` to cancel everything. Separate from `/v1/interrupt` on purpose.

## Retention

- `GET /v1/retention` — current policy, the options, the directory, the count.
- `POST /v1/retention` — `{"policy":"7d","sweep":true}`

Changing the policy applies **retroactively**: shortening it deletes old audio
rather than only affecting future recordings. `"persisted": false` in the
response means the running policy changed but the config file did not — use
`--set retention.policy=7d` for that.

## Errors

Every failure has the same shape:

```json
{"error":"text must not be empty","code":"bad_request",
 "hint":"send `Authorization: Bearer <api.auth_token>`"}
```

`400` bad request, `401` unauthorized, `503` model unavailable, `500`
otherwise.

## A minimal Rust client

```rust
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    client.post("http://127.0.0.1:8787/v1/speak")
        .json(&serde_json::json!({"text": "Good evening."}))
        .send().await?;

    let mut events = client
        .get("http://127.0.0.1:8787/v1/transcripts")
        .send().await?
        .bytes_stream();

    while let Some(chunk) = events.next().await {
        for line in String::from_utf8_lossy(&chunk?).lines() {
            if let Some(json) = line.strip_prefix("data: ") {
                let event: serde_json::Value = serde_json::from_str(json)?;
                if event["type"] == "transcript.final" {
                    println!("heard: {}", event["text"]);
                }
            }
        }
    }
    Ok(())
}
```
