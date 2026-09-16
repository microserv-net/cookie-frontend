// Apple's on-device speech recogniser, wrapped in the JSON-lines protocol the
// rest of Cookie already speaks.
//
// Why this exists: Whisper large-v3-turbo through onnxruntime costs eleven
// seconds for a two-second sentence on an M-series laptop, because Whisper
// pads every utterance to thirty seconds and the quantised graph cannot use
// the Neural Engine. Apple's recogniser is already on the machine, already
// on-device, already tuned for exactly this hardware, and returns partial
// results while you are still talking. On macOS it is simply the better tool.
//
// It is Swift rather than Rust because `SFSpeechRecognizer` is an Objective-C
// API whose Rust bindings would mean unsafe blocks and a large dependency;
// this crate is `#![forbid(unsafe_code)]` and the boundary is a pipe.
//
// Protocol, one JSON object per line in each direction:
//
//   → {"id":"1","op":"hello"}
//   ← {"id":"1","type":"result","model":"apple-on-device","on_device":true}
//
//   → {"id":"2","op":"transcribe","sample_rate":16000,"audio":"<base64 f32le>"}
//   ← {"id":"2","type":"result","text":"open the project"}
//
// Anything it cannot do is reported as {"type":"error"} with a sentence, so
// the Rust side can fall back to Whisper rather than going silent.

import AVFoundation
import Foundation
import Speech

/// One line of JSON on stdout, flushed immediately — the reader is blocking
/// on it and buffering would look like a hang.
func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let line = String(data: data, encoding: .utf8) else { return }
    print(line)
    fflush(stdout)
}

func fail(_ id: String, _ message: String) {
    emit(["id": id, "type": "error", "message": message])
}

/// Ask once, at startup. The prompt is the user's first sight of this, so it
/// happens before anything is waiting on a result.
func authorise() -> Bool {
    if SFSpeechRecognizer.authorizationStatus() == .authorized { return true }
    let gate = DispatchSemaphore(value: 0)
    var granted = false
    SFSpeechRecognizer.requestAuthorization { status in
        granted = status == .authorized
        gate.signal()
    }
    // Generous: this is a dialogue a person has to read and click.
    _ = gate.wait(timeout: .now() + 120)
    return granted
}

final class Transcriber {
    private let recogniser: SFSpeechRecognizer

    init?(locale: String) {
        guard let recogniser = SFSpeechRecognizer(locale: Locale(identifier: locale)) else {
            return nil
        }
        self.recogniser = recogniser
    }

    var isAvailable: Bool { recogniser.isAvailable }

    /// True when the model runs on this machine rather than Apple's servers.
    ///
    /// Reported honestly rather than assumed: a locale without a downloaded
    /// model would otherwise send the user's voice to a server, which is not
    /// a thing to do by accident.
    var supportsOnDevice: Bool { recogniser.supportsOnDeviceRecognition }

    /// Transcribe one utterance of 32-bit float samples.
    func transcribe(samples: [Float], sampleRate: Double, id: String) {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false
        request.requiresOnDeviceRecognition = recogniser.supportsOnDeviceRecognition
        if #available(macOS 13.0, *) {
            request.addsPunctuation = true
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            fail(id, "could not build an audio buffer")
            return
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channel = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { source in
                channel.update(from: source.baseAddress!, count: samples.count)
            }
        }

        let gate = DispatchSemaphore(value: 0)
        var answered = false
        let task = recogniser.recognitionTask(with: request) { result, error in
            if let result, result.isFinal {
                answered = true
                emit([
                    "id": id,
                    "type": "result",
                    "text": result.bestTranscription.formattedString,
                ])
                gate.signal()
                return
            }
            if let error {
                // "No speech detected" is an ordinary outcome for a room
                // noise that cleared the threshold, not a failure.
                let text = (error as NSError).code == 1110 ? "" : nil
                if let text {
                    answered = true
                    emit(["id": id, "type": "result", "text": text])
                } else {
                    fail(id, error.localizedDescription)
                    answered = true
                }
                gate.signal()
            }
        }

        request.append(buffer)
        request.endAudio()

        // The recogniser is fast, but a hung task must not hold the pipe.
        if gate.wait(timeout: .now() + 20) == .timedOut, !answered {
            task.cancel()
            fail(id, "the recogniser did not answer in time")
        }
    }
}

// --- main -------------------------------------------------------------------

let locale = ProcessInfo.processInfo.environment["COOKIE_SPEECH_LOCALE"] ?? "en-GB"

guard authorise() else {
    emit([
        "type": "error",
        "message": "speech recognition was not permitted. Allow it in System "
            + "Settings → Privacy & Security → Speech Recognition.",
    ])
    exit(1)
}

guard let transcriber = Transcriber(locale: locale) else {
    emit(["type": "error", "message": "no recogniser for \(locale)"])
    exit(1)
}

guard transcriber.isAvailable else {
    emit(["type": "error", "message": "the recogniser for \(locale) is not available yet"])
    exit(1)
}

while let line = readLine(strippingNewline: true) {
    guard !line.isEmpty,
          let data = line.data(using: .utf8),
          let request = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { continue }

    let id = (request["id"] as? String) ?? ""
    switch request["op"] as? String ?? "" {
    case "hello":
        emit([
            "id": id,
            "type": "result",
            "model": "apple-on-device",
            "locale": locale,
            "on_device": transcriber.supportsOnDevice,
        ])
    case "transcribe":
        guard let encoded = request["audio"] as? String,
              let audio = Data(base64Encoded: encoded)
        else {
            fail(id, "no audio in that request")
            continue
        }
        let sampleRate = (request["sample_rate"] as? Double) ?? 16000
        let samples = audio.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        transcriber.transcribe(samples: samples, sampleRate: sampleRate, id: id)
    default:
        fail(id, "unknown operation")
    }
}
