//! A complete client, in about a hundred lines.
//!
//! Speaks a line, streams a second one in token by token, listens for what
//! you say back, and asks Cookie whether she is alright. This is the whole
//! integration surface — everything the Cookie backend needs to do, minus the
//! thinking.
//!
//! ```bash
//! cookie-interface &                       # in one terminal
//! cargo run --example client               # in another
//! ```

use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{json, Value};

const BASE: &str = "http://127.0.0.1:8787/v1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // --- is anyone home? --------------------------------------------------
    let health: Value = client
        .get(format!("{BASE}/health"))
        .send()
        .await?
        .json()
        .await?;
    println!("cookie-interface {} is up", health["version"]);

    // --- what does this installation actually support? --------------------
    let state: Value = client
        .get(format!("{BASE}/state"))
        .send()
        .await?
        .json()
        .await?;
    let tts = &state["providers"]["tts"];
    println!(
        "voice: {} (streaming: {}, rate: {}, pitch: {})",
        tts["name"],
        tts["capabilities"]["streaming"],
        tts["capabilities"]["rate"],
        tts["capabilities"]["pitch"]
    );
    // Never send a parameter the provider ignores; ask first.
    let honours_rate = tts["capabilities"]["rate"].as_bool().unwrap_or(false);

    // --- say something ----------------------------------------------------
    let mut body = json!({"text": "Good evening. Everything is ready."});
    if honours_rate {
        body["voice"] = json!({"rate": 0.95});
    }
    let accepted: Value = client
        .post(format!("{BASE}/speak"))
        .json(&body)
        .send()
        .await?
        .json()
        .await?;
    println!("speaking as {}", accepted["utterance_id"]);

    // --- stream a reply in, the way a language model would ----------------
    //
    // Each line is spoken as soon as it completes a sentence, so the voice
    // starts before the last token exists.
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(8);
    tokio::spawn(async move {
        for token in [
            "I've had a look at the build. ",
            "Three tests were failing because the token refresh ",
            "expired a second early. ",
            "I've fixed it and they pass now.",
        ] {
            let _ = tx.send(format!("{}\n", json!({"text": token}))).await;
            tokio::time::sleep(Duration::from_millis(120)).await;
        }
        let _ = tx.send(format!("{}\n", json!({"end": true}))).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<_, std::io::Error>);
    let mut progress = client
        .post(format!("{BASE}/speak/stream"))
        .header("content-type", "application/x-ndjson")
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await?
        .bytes_stream();

    while let Some(chunk) = progress.next().await {
        for line in String::from_utf8_lossy(&chunk?)
            .lines()
            .filter(|l| !l.is_empty())
        {
            println!("  ← {line}");
        }
    }

    // --- ask how she is ---------------------------------------------------
    let report: Value = client
        .get(format!("{BASE}/diagnostics"))
        .send()
        .await?
        .json()
        .await?;
    println!("diagnostics: {}", report["summary"].as_str().unwrap_or(""));

    // --- listen ------------------------------------------------------------
    client
        .post(format!("{BASE}/listen"))
        .json(&json!({"continuous": true}))
        .send()
        .await?;
    println!("listening — say something (ctrl-c to stop)");

    let mut events = client
        .get(format!("{BASE}/transcripts"))
        .send()
        .await?
        .bytes_stream();

    while let Some(chunk) = events.next().await {
        for line in String::from_utf8_lossy(&chunk?).lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            let event: Value = serde_json::from_str(payload)?;
            match event["type"].as_str() {
                Some("transcript.partial") => println!("  … {}", event["text"]),
                Some("transcript.final") => {
                    println!("  → {}", event["text"]);
                    // This is where a real backend would think about it, and
                    // send the answer back through /v1/speak/stream.
                }
                // Unknown types are ignored, not fatal: that is what keeps
                // this client working as the protocol grows.
                _ => {}
            }
        }
    }

    Ok(())
}
