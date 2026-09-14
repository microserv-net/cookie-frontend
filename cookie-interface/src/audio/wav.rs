//! WAV reading and writing, plus raw-PCM helpers.
//!
//! Only WAV and raw PCM are handled on purpose. Every provider we speak to can
//! emit one of them, and pulling in an MP3/Opus decoder would add a
//! non-trivial dependency (and licensing surface) to a component whose job is
//! to be small. When a provider returns something else we say so plainly
//! instead of playing noise — see `decode_audio`.

use std::io::Cursor;
use std::path::Path;

use super::AudioBuffer;
use crate::error::{Error, Result};

/// Write mono f32 as 16-bit PCM WAV. 16-bit keeps generated-speech files at a
/// sensible size on disk; the pipeline itself never round-trips through this.
pub fn write_mono_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<u64> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| wav_error(path, "create", e))?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(v)
            .map_err(|e| wav_error(path, "write", e))?;
    }
    writer
        .finalize()
        .map_err(|e| wav_error(path, "finalize", e))?;
    let size = std::fs::metadata(path)
        .map_err(|e| Error::io(path, e))?
        .len();
    Ok(size)
}

/// Encode mono f32 as a 16-bit PCM WAV **in memory**.
///
/// Used by the HTTP recognisers, which must hand a complete file to a
/// multipart request; nothing touches the disk on the way.
pub fn encode_mono_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| Error::Other(format!("wav encode: {e}")))?;
        for s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(v)
                .map_err(|e| Error::Other(format!("wav encode: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| Error::Other(format!("wav encode: {e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Read a WAV file into mono f32 at its native rate.
pub fn read_wav(path: &Path) -> Result<AudioBuffer> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    decode_wav(&bytes)
}

/// Decode WAV bytes (any bit depth hound supports) into mono f32.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer> {
    let reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| Error::UnsupportedAudioFormat(format!("not a readable WAV: {e}")))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1);
    let max = match spec.bits_per_sample {
        8 => i8::MAX as f32,
        16 => i16::MAX as f32,
        24 => 8_388_607.0,
        32 => i32::MAX as f32,
        other => {
            return Err(Error::UnsupportedAudioFormat(format!(
                "{other}-bit WAV is not supported"
            )))
        }
    };

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::UnsupportedAudioFormat(format!("bad float WAV data: {e}")))?,
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| v as f32 / max))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::UnsupportedAudioFormat(format!("bad int WAV data: {e}")))?,
    };

    let buffer = AudioBuffer {
        samples,
        sample_rate: spec.sample_rate,
        channels,
    };
    Ok(buffer.to_mono())
}

/// Decode signed 16-bit little-endian PCM (what most TTS streaming endpoints
/// emit when asked for `pcm`).
pub fn decode_pcm_s16le(bytes: &[u8], sample_rate: u32, channels: u16) -> AudioBuffer {
    let samples: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / i16::MAX as f32)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: channels.max(1),
    }
    .to_mono()
}

/// Best-effort decode of whatever a provider returned.
///
/// `content_type` is the HTTP header when we have one. Anything we cannot
/// decode produces an error naming the format, because "the assistant made a
/// horrible buzzing noise" is a much worse failure mode than an error message.
pub fn decode_audio(
    bytes: &[u8],
    content_type: Option<&str>,
    assumed_rate: u32,
) -> Result<AudioBuffer> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return decode_wav(bytes);
    }
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    if ct.contains("wav") || ct.contains("wave") {
        return decode_wav(bytes);
    }
    if ct.contains("pcm") || ct.contains("l16") || ct.is_empty() {
        if bytes.len() % 2 != 0 {
            return Err(Error::UnsupportedAudioFormat(
                "odd-length buffer is not 16-bit PCM".into(),
            ));
        }
        return Ok(decode_pcm_s16le(bytes, assumed_rate, 1));
    }
    let named = if ct.contains("mpeg") || ct.contains("mp3") {
        "MP3"
    } else if ct.contains("opus") || ct.contains("ogg") {
        "Opus/Ogg"
    } else if ct.contains("flac") {
        "FLAC"
    } else if ct.contains("aac") || ct.contains("m4a") {
        "AAC"
    } else {
        return Err(Error::UnsupportedAudioFormat(format!(
            "unrecognised audio content-type {ct:?}"
        )));
    };
    Err(Error::UnsupportedAudioFormat(format!(
        "{named} is not decoded by cookie-interface; ask the provider for wav or pcm \
         (set response_format/format accordingly in tts.options)"
    )))
}

fn wav_error(path: &Path, op: &str, e: hound::Error) -> Error {
    match e {
        hound::Error::IoError(io) => Error::io(path, io),
        other => Error::Other(format!("WAV {op} failed for {}: {other}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / 20.0).sin() * 0.5).collect()
    }

    #[test]
    fn wav_roundtrip_preserves_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        let samples = tone(4000);
        let size = write_mono_wav(&path, &samples, 16_000).unwrap();
        assert!(size > 8000, "suspiciously small file: {size}");

        let back = read_wav(&path).unwrap();
        assert_eq!(back.sample_rate, 16_000);
        assert_eq!(back.channels, 1);
        assert_eq!(back.samples.len(), samples.len());
        for (a, b) in samples.iter().zip(back.samples.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_detects_wav_by_magic_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.wav");
        write_mono_wav(&path, &tone(800), 22_050).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let buf = decode_audio(&bytes, Some("application/octet-stream"), 16_000).unwrap();
        assert_eq!(buf.sample_rate, 22_050);
    }

    #[test]
    fn raw_pcm_decodes_at_the_assumed_rate() {
        let pcm: Vec<u8> = (0..1000i16).flat_map(|v| (v * 30).to_le_bytes()).collect();
        let buf = decode_audio(&pcm, Some("audio/pcm"), 24_000).unwrap();
        assert_eq!(buf.sample_rate, 24_000);
        assert_eq!(buf.samples.len(), 1000);
    }

    #[test]
    fn undecodable_formats_fail_loudly_and_usefully() {
        let err = decode_audio(&[0xff, 0xfb, 0x00, 0x00], Some("audio/mpeg"), 16_000).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("MP3"), "{msg}");
        assert!(msg.contains("wav"), "error should say what to do: {msg}");
    }

    #[test]
    fn stereo_wav_is_downmixed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..100 {
            w.write_sample(16384i16).unwrap();
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();

        let buf = read_wav(&path).unwrap();
        assert_eq!(buf.channels, 1);
        assert_eq!(buf.samples.len(), 100);
        assert!((buf.samples[0] - 0.25).abs() < 0.01, "{}", buf.samples[0]);
    }
}
