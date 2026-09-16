# The backend protocol

This is the contract between `cookie-interface` (the face) and the Cookie
backend (the mind). It is deliberately small: one POST that streams
newline-delimited JSON back, plus a cancel endpoint. You can implement a
conforming backend in about forty lines of any language, and the interface
works with no backend at all.

The interface dials **out**. The backend never needs to know where the
interface is, so this works unchanged behind NAT, over Tailscale, or on
localhost.

```toml
[backend]
enabled  = true
base_url = "http://192.168.1.42:8080/api"
chat_path   = "/v1/chat"
health_path = "/v1/health"
cancel_path = "/v1/cancel"
auth_token_env = "COOKIE_BACKEND_TOKEN"
```

## A turn

```http
POST /api/v1/chat
Content-Type: application/json
Accept: application/x-ndjson, text/event-stream, application/json
Authorization: Bearer <token>          # when auth_token_env is set

{
  "protocol": "cookie-interface/1",
  "session_id": "6b1c…",
  "utterance_id": "9f2a…",
  "text": "fix the failing tests in the auth module",
  "final": true,
  "interface": { "speech": true, "listening": true, "visual": "orb",
                 "version": "0.1.0" },
  "scheduling": { "priority": "interactive", "preempt": true },
  "active_tasks": ["task-17"]
}
```

`interface` describes what the front end can do. It will gain keys as Cookie
gains visual capabilities; ignore the ones you do not recognise.

## The reply

Three shapes are accepted. Newline-delimited JSON is the one to implement.

```
Content-Type: application/x-ndjson

{"type":"task","id":"task-18","state":"running","title":"fixing the failing tests","weight":"heavy"}
{"type":"delta","text":"Give me a moment. "}
{"type":"task","id":"task-18","state":"running","detail":"running the test suite"}
{"type":"delta","text":"Three tests were failing because the token refresh "}
{"type":"delta","text":"expired early. I've fixed it and they pass now."}
{"type":"task","id":"task-18","state":"completed"}
{"type":"end"}
```

Each `delta` is appended to a buffer and spoken as soon as a sentence boundary
is reached, so Cookie starts talking before you have finished thinking. Server-
sent events (`data: {…}`) work identically, and a plain
`{"reply":"…"}` object is spoken in one go.

### Message types

| `type` | Effect |
|---|---|
| `delta` / `text` / `token` | Append `text` to the spoken reply. |
| `task` / `task.update` / `progress` | Record progress. Fields: `id`, `state`, `title`, `detail`, `weight`. |
| `interrupt` | Stop speaking now. |
| `listen` | Open the microphone. `continuous` optional. |
| `error` | Abort the turn; `message` is logged and surfaced. |
| `end` / `done` | Finished. |

Task `state` is one of `queued`, `running`, `suspended`, `completed`,
`failed`, `cancelled`. `weight` is `light`, `normal` or `heavy`.

**Anything else is ignored rather than treated as an error.** `ui` is reserved:
it is where future visual instructions will arrive. A backend written today
keeps working when that lands.

## Scheduling: one heavy task at a time

The backend cannot tell from the text whether you are adding to the job it is
already doing or interrupting with something small. The interface can — it
knows a human is waiting and how long the sentence was — so every turn carries
a hint:

| Situation | `priority` | `preempt` |
|---|---|---|
| nothing running | `normal` | `false` |
| heavy task running, ≤12 word utterance | `interactive` | `true` |
| heavy task running, longer utterance | `normal` | `false` |

`preempt: true` does **not** mean abandon the task. It means: suspend at your
next natural boundary, answer this, then carry on. The boundaries are the
points where you are already between things —

* a model call has just returned,
* you are about to swap models,
* a tool is running and you are waiting on it,
* you are between steps of a plan.

Report the suspension as `{"type":"task","id":…,"state":"suspended"}` and the
resumption as `state":"running"`. The interface shows a suspended task as
paused rather than stuck, and stops asking for preemption while it is
suspended — a suspended heavy task is not occupying the machine.

A backend that ignores the hint is still correct. It is just less pleasant on
hardware that can hold one large model in memory at a time.

## Cancelling

```http
POST /api/v1/cancel
{"protocol":"cookie-interface/1","session_id":"6b1c…","task_id":"task-18"}
```

`task_id: null` cancels everything in flight.

**Cancelling is not interrupting.** "Stop" means stop *talking* and never
reaches this endpoint. Only an explicit cancellation — "cancel that", "forget
it", `POST /v1/cancel` — abandons work. Conflating the two would make Cookie
unusable: you could never interrupt a sentence without losing the job you
asked for.

## Health

```http
GET /api/v1/health   →   200, any JSON body
```

Probed by `--doctor` and whenever somebody asks Cookie whether she is alright.
