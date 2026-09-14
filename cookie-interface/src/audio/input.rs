//! Microphone capture.
//!
//! The cpal `Stream` is not `Send` on every platform, so it is built on, and
//! owned by, a dedicated thread that does nothing but hold it alive. Everything
//! else talks to that thread through the lock-free ring and a stop flag. This
//! is also why `CaptureHandle` does the channel downmix and resampling: doing
//! it in the callback would mean allocating on the real-time thread.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::resample::{interleaved_to_mono, Resampler};
use super::ring::{ring, RingConsumer};
use super::AudioDeviceInfo;
use crate::error::{Error, Result};

/// Replaceable capture device.
pub trait AudioInput: Send + Sync {
    fn info(&self) -> AudioDeviceInfo;
    /// Open the device and start filling a ring. Returns a handle that pops
    /// mono audio at `target_rate`.
    fn start(&self, target_rate: u32) -> Result<CaptureHandle>;
}

/// Live capture session. Dropping it stops the device.
pub struct CaptureHandle {
    consumer: RingConsumer,
    info: AudioDeviceInfo,
    target_rate: u32,
    resampler: Resampler,
    raw: Vec<f32>,
    mono: Vec<f32>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    error: Arc<Mutex<Option<String>>>,
    overflows: Arc<AtomicUsize>,
}

impl CaptureHandle {
    pub fn info(&self) -> &AudioDeviceInfo {
        &self.info
    }

    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Samples the device had to discard because this side fell behind.
    /// A non-zero value here is the signature of a blocked capture task.
    pub fn dropped_samples(&self) -> usize {
        self.consumer.dropped() + self.overflows.load(Ordering::Relaxed)
    }

    /// Take any error the audio thread reported (device unplugged, etc).
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    /// Move everything currently buffered into `out` as mono at `target_rate`.
    /// Appends; the caller owns the buffer so nothing is allocated per call.
    pub fn drain(&mut self, out: &mut Vec<f32>) -> usize {
        let available = self.consumer.len();
        if available == 0 {
            return 0;
        }
        if self.raw.len() < available {
            self.raw.resize(available, 0.0);
        }
        let n = self.consumer.pop_slice(&mut self.raw[..available]);
        if n == 0 {
            return 0;
        }
        self.mono.clear();
        interleaved_to_mono(&self.raw[..n], self.info.channels, &mut self.mono);
        let before = out.len();
        let mono = std::mem::take(&mut self.mono);
        self.resampler.process_into(&mono, out);
        self.mono = mono;
        self.mono.clear();
        out.len() - before
    }

    /// Discard buffered audio (used when we stop caring about what was said).
    pub fn clear(&mut self) {
        self.consumer.clear();
        self.resampler.reset();
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------

/// Signal a `MockInput` produces. Lets the whole pipeline — VAD, recognition,
/// state machine, orb — be exercised with no hardware at all, which is what
/// keeps `cargo test` free of a microphone requirement.
#[derive(Debug, Clone)]
pub enum MockSignal {
    Silence,
    /// Repeating scripted samples (e.g. a recorded phrase).
    Samples(Arc<Vec<f32>>),
    /// Speech-shaped tone bursts: `speech_ms` of voiced signal followed by
    /// `gap_ms` of silence, forever.
    Utterances {
        speech_ms: u32,
        gap_ms: u32,
        amplitude: f32,
    },
}

pub struct MockInput {
    signal: MockSignal,
    sample_rate: u32,
    /// When false, audio is produced as fast as it is consumed, so tests run
    /// at full speed instead of in real time.
    realtime: bool,
    name: String,
}

impl MockInput {
    pub fn new(signal: MockSignal, sample_rate: u32) -> Self {
        Self {
            signal,
            sample_rate,
            realtime: true,
            name: "mock-input".into(),
        }
    }

    pub fn silence(sample_rate: u32) -> Self {
        Self::new(MockSignal::Silence, sample_rate)
    }

    pub fn samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(MockSignal::Samples(Arc::new(samples)), sample_rate)
    }

    pub fn speaking(sample_rate: u32) -> Self {
        Self::new(
            MockSignal::Utterances {
                speech_ms: 1200,
                gap_ms: 1500,
                amplitude: 0.4,
            },
            sample_rate,
        )
    }

    pub fn with_realtime(mut self, realtime: bool) -> Self {
        self.realtime = realtime;
        self
    }
}

impl AudioInput for MockInput {
    fn info(&self) -> AudioDeviceInfo {
        AudioDeviceInfo::mock(&self.name, self.sample_rate)
    }

    fn start(&self, target_rate: u32) -> Result<CaptureHandle> {
        let (mut producer, consumer) = ring(self.sample_rate as usize * 2);
        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let overflows = Arc::new(AtomicUsize::new(0));

        let signal = self.signal.clone();
        let rate = self.sample_rate;
        let realtime = self.realtime;
        let stop_thread = Arc::clone(&stop);
        let frame = (rate / 50).max(1) as usize; // 20 ms

        let thread = std::thread::Builder::new()
            .name("cookie-mock-input".into())
            .spawn(move || {
                let mut cursor = 0usize;
                let mut buf = vec![0.0f32; frame];
                while !stop_thread.load(Ordering::Acquire) {
                    fill_mock(&signal, rate, cursor, &mut buf);
                    cursor = cursor.wrapping_add(frame);
                    producer.push_slice(&buf);
                    if realtime {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    } else if producer.free() < frame * 2 {
                        // Non-realtime mode still must not spin the CPU when
                        // the consumer is slow.
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            })
            .map_err(|e| Error::AudioDevice(format!("could not spawn mock input thread: {e}")))?;

        Ok(CaptureHandle {
            consumer,
            info: self.info(),
            target_rate,
            resampler: Resampler::new(self.sample_rate, target_rate),
            raw: vec![0.0; 4096],
            mono: Vec::with_capacity(4096),
            stop,
            thread: Some(thread),
            error,
            overflows,
        })
    }
}

fn fill_mock(signal: &MockSignal, rate: u32, cursor: usize, out: &mut [f32]) {
    match signal {
        MockSignal::Silence => out.iter_mut().for_each(|s| *s = 0.0),
        MockSignal::Samples(data) => {
            if data.is_empty() {
                out.iter_mut().for_each(|s| *s = 0.0);
                return;
            }
            for (i, s) in out.iter_mut().enumerate() {
                *s = data[(cursor + i) % data.len()];
            }
        }
        MockSignal::Utterances {
            speech_ms,
            gap_ms,
            amplitude,
        } => {
            let period = ((*speech_ms + *gap_ms) as usize * rate as usize) / 1000;
            let speech = (*speech_ms as usize * rate as usize) / 1000;
            for (i, s) in out.iter_mut().enumerate() {
                let phase = (cursor + i) % period.max(1);
                *s = if phase < speech {
                    let t = (cursor + i) as f32 / rate as f32;
                    // Envelope the bursts so they look like speech to the VAD
                    // rather than like a square-edged beep.
                    let env = ((phase as f32 / speech as f32) * std::f32::consts::PI).sin();
                    amplitude
                        * env
                        * (0.6 * (2.0 * std::f32::consts::PI * 140.0 * t).sin()
                            + 0.3 * (2.0 * std::f32::consts::PI * 430.0 * t).sin()
                            + 0.1 * (2.0 * std::f32::consts::PI * 950.0 * t).sin())
                } else {
                    0.0
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cpal
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-io")]
mod cpal_impl {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{Sample, SampleFormat};

    /// Real microphone capture through cpal (WASAPI / CoreAudio / ALSA /
    /// PipeWire — whichever the platform provides; we never name one).
    pub struct CpalInput {
        /// Substring matched against device names. `None` = system default.
        pub device_name: Option<String>,
        pub gain: f32,
    }

    impl CpalInput {
        pub fn new(device_name: Option<String>, gain: f32) -> Self {
            Self { device_name, gain }
        }

        fn pick_device() -> Result<(cpal::Device, cpal::SupportedStreamConfig, String)> {
            Self::pick(None)
        }

        fn pick(
            wanted: Option<&str>,
        ) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String)> {
            let host = cpal::default_host();
            let device = match wanted {
                Some(want) => host
                    .input_devices()
                    .map_err(|e| Error::AudioDevice(e.to_string()))?
                    .find(|d| {
                        cpal_device_name(d)
                            .to_lowercase()
                            .contains(&want.to_lowercase())
                    })
                    .ok_or_else(|| {
                        Error::AudioDevice(format!("no input device matching {want:?}"))
                    })?,
                None => host
                    .default_input_device()
                    .ok_or_else(|| Error::AudioDevice("no default input device".into()))?,
            };
            let config = device
                .default_input_config()
                .map_err(|e| Error::AudioDevice(format!("no usable input config: {e}")))?;
            let name = cpal_device_name(&device);
            Ok((device, config, name))
        }

        /// Device names for `--doctor`.
        pub fn list() -> Result<Vec<String>> {
            let host = cpal::default_host();
            Ok(host
                .input_devices()
                .map_err(|e| Error::AudioDevice(e.to_string()))?
                .map(|d| cpal_device_name(&d))
                .collect())
        }
    }

    pub(crate) fn cpal_device_name(d: &cpal::Device) -> String {
        d.description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "unknown".into())
    }

    impl AudioInput for CpalInput {
        fn info(&self) -> AudioDeviceInfo {
            match Self::pick(self.device_name.as_deref()) {
                Ok((_, config, name)) => AudioDeviceInfo {
                    name,
                    is_default: self.device_name.is_none(),
                    sample_rate: config.sample_rate(),
                    channels: config.channels(),
                    backend: "cpal".into(),
                },
                Err(_) => AudioDeviceInfo {
                    name: "unavailable".into(),
                    is_default: true,
                    sample_rate: 0,
                    channels: 0,
                    backend: "cpal".into(),
                },
            }
        }

        fn start(&self, target_rate: u32) -> Result<CaptureHandle> {
            let (_, probe, name) = Self::pick(self.device_name.as_deref())?;
            let device_rate = probe.sample_rate();
            let channels = probe.channels();
            let info = AudioDeviceInfo {
                name: name.clone(),
                is_default: self.device_name.is_none(),
                sample_rate: device_rate,
                channels,
                backend: "cpal".into(),
            };

            // Two seconds of slack: long enough that a GC-like stall in the
            // async runtime does not lose audio, short enough that we never
            // transcribe something the user said a minute ago.
            let (producer, consumer) = ring(device_rate as usize * channels as usize * 2);
            let stop = Arc::new(AtomicBool::new(false));
            let error = Arc::new(Mutex::new(None));
            let overflows = Arc::new(AtomicUsize::new(0));

            let wanted = self.device_name.clone();
            let gain = self.gain;
            let stop_thread = Arc::clone(&stop);
            let error_thread = Arc::clone(&error);
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

            let thread = std::thread::Builder::new()
                .name("cookie-audio-in".into())
                .spawn(move || {
                    let built = (|| -> Result<cpal::Stream> {
                        let (device, config, _) = CpalInput::pick(wanted.as_deref())?;
                        build_input_stream(&device, &config, producer, gain, error_thread.clone())
                    })();

                    let stream = match built {
                        Ok(s) => {
                            let _ = ready_tx.send(Ok(()));
                            s
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    if let Err(e) = stream.play() {
                        if let Ok(mut slot) = error_thread.lock() {
                            *slot = Some(format!("could not start input stream: {e}"));
                        }
                        return;
                    }
                    // Own the stream until asked to stop. Parking rather than
                    // busy-waiting keeps this thread off the CPU entirely.
                    while !stop_thread.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    drop(stream);
                })
                .map_err(|e| Error::AudioDevice(format!("could not spawn audio thread: {e}")))?;

            match ready_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    stop.store(true, Ordering::Release);
                    let _ = thread.join();
                    return Err(e);
                }
                Err(_) => {
                    stop.store(true, Ordering::Release);
                    return Err(Error::AudioDevice(
                        "timed out opening the input device".into(),
                    ));
                }
            }

            let _ = CpalInput::pick_device();
            Ok(CaptureHandle {
                consumer,
                info,
                target_rate,
                resampler: Resampler::new(device_rate, target_rate),
                raw: vec![0.0; 8192],
                mono: Vec::with_capacity(8192),
                stop,
                thread: Some(thread),
                error,
                overflows,
            })
        }
    }

    fn build_input_stream(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        mut producer: super::super::ring::RingProducer,
        gain: f32,
        error_slot: Arc<Mutex<Option<String>>>,
    ) -> Result<cpal::Stream> {
        let stream_config: cpal::StreamConfig = cpal::StreamConfig {
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };
        let err_fn = move |e: cpal::Error| {
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
        };

        // Scratch buffer for formats that need conversion. Allocated once,
        // here, outside the callback; the callback only ever writes into it.
        let mut scratch: Vec<f32> = Vec::with_capacity(16_384);

        macro_rules! build {
            ($t:ty) => {{
                device
                    .build_input_stream(
                        stream_config,
                        move |data: &[$t], _: &cpal::InputCallbackInfo| {
                            // REAL-TIME SECTION: no allocation, no locks, no I/O.
                            if scratch.capacity() < data.len() {
                                // Only reachable if the driver suddenly hands
                                // us a far larger block; better one allocation
                                // than dropped audio.
                                scratch.reserve(data.len() - scratch.capacity());
                            }
                            scratch.clear();
                            for s in data {
                                scratch.push(((*s).to_sample::<f32>() * gain).clamp(-1.0, 1.0));
                            }
                            producer.push_slice(&scratch);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| Error::AudioDevice(format!("could not build input stream: {e}")))
            }};
        }

        match config.sample_format() {
            SampleFormat::F32 => build!(f32),
            SampleFormat::I16 => build!(i16),
            SampleFormat::U16 => build!(u16),
            SampleFormat::I32 => build!(i32),
            SampleFormat::I8 => build!(i8),
            SampleFormat::U8 => build!(u8),
            other => Err(Error::UnsupportedAudioFormat(format!("{other:?}"))),
        }
    }
}

#[cfg(feature = "audio-io")]
pub(crate) use cpal_impl::cpal_device_name;
#[cfg(feature = "audio-io")]
pub use cpal_impl::CpalInput;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_input_delivers_audio_at_the_target_rate() {
        let input = MockInput::speaking(48_000).with_realtime(false);
        let mut handle = input.start(16_000).unwrap();
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while out.len() < 16_000 && std::time::Instant::now() < deadline {
            handle.drain(&mut out);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(out.len() >= 16_000, "only got {} samples", out.len());
        assert!(
            out.iter().any(|s| s.abs() > 0.05),
            "mock produced no signal"
        );
        assert_eq!(handle.target_rate(), 16_000);
    }

    #[test]
    fn mock_silence_stays_silent() {
        let input = MockInput::silence(16_000).with_realtime(false);
        let mut handle = input.start(16_000).unwrap();
        let mut out = Vec::new();
        std::thread::sleep(std::time::Duration::from_millis(50));
        handle.drain(&mut out);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn dropping_the_handle_stops_the_device() {
        let input = MockInput::silence(16_000).with_realtime(false);
        let handle = input.start(16_000).unwrap();
        drop(handle); // must not hang: the thread is joined in Drop
    }

    #[test]
    fn scripted_samples_are_replayed() {
        let script: Vec<f32> = (0..1600).map(|i| (i % 100) as f32 / 100.0).collect();
        let input = MockInput::samples(script, 16_000).with_realtime(false);
        let mut handle = input.start(16_000).unwrap();
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while out.len() < 1600 && std::time::Instant::now() < deadline {
            handle.drain(&mut out);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(out.len() >= 1600);
        assert!(out.iter().any(|s| *s > 0.9));
    }
}
