//! Speaker playback.
//!
//! Same shape as capture, mirrored: the async side writes mono audio at
//! whatever rate the synthesiser produced, `PlaybackHandle` resamples and
//! interleaves it, and the real-time callback does nothing but copy out of the
//! ring (filling with silence on underrun rather than glitching or blocking).
//!
//! `clear()` exists for interruption: when speech is cut off, the audio already
//! queued in the ring must be dropped immediately, otherwise Cookie keeps
//! talking for a second after being told to stop, which feels broken.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::resample::{mono_to_interleaved, Resampler};
use super::ring::{ring, RingProducer};
use super::AudioDeviceInfo;
use crate::error::{Error, Result};

/// Replaceable playback device.
pub trait AudioOutput: Send + Sync {
    fn info(&self) -> AudioDeviceInfo;
    fn open(&self) -> Result<PlaybackHandle>;
}

/// Shared counters the callback updates and the writer reads.
#[derive(Debug, Default)]
pub struct PlaybackStats {
    /// Frames handed to the device.
    pub frames_played: AtomicU64,
    /// Times the callback found the ring empty.
    pub underruns: AtomicUsize,
    /// Set by the writer, read by the callback: play silence but keep running.
    pub muted: AtomicBool,
}

/// Live playback session. Dropping it stops the device.
pub struct PlaybackHandle {
    producer: RingProducer,
    info: AudioDeviceInfo,
    resampler: Resampler,
    source_rate: u32,
    scratch_mono: Vec<f32>,
    scratch_inter: Vec<f32>,
    stats: Arc<PlaybackStats>,
    stop: Arc<AtomicBool>,
    /// Set by the callback when it should drop everything (interruption).
    flush_flag: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    error: Arc<Mutex<Option<String>>>,
    gain: f32,
}

impl PlaybackHandle {
    pub fn info(&self) -> &AudioDeviceInfo {
        &self.info
    }

    pub fn device_rate(&self) -> u32 {
        self.info.sample_rate
    }

    /// Queue mono audio recorded at `sample_rate`. Returns the number of
    /// device frames queued. Never blocks: if the ring is full the *oldest*
    /// queued audio is dropped, which for speech means we favour staying in
    /// sync over playing every sample.
    pub fn write(&mut self, samples: &[f32], sample_rate: u32) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if sample_rate != self.source_rate {
            self.resampler = Resampler::new(sample_rate, self.info.sample_rate);
            self.source_rate = sample_rate;
        }
        self.scratch_mono.clear();
        self.resampler.process_into(samples, &mut self.scratch_mono);
        if self.gain != 1.0 {
            for s in &mut self.scratch_mono {
                *s = (*s * self.gain).clamp(-1.0, 1.0);
            }
        }
        self.scratch_inter.clear();
        mono_to_interleaved(
            &self.scratch_mono,
            self.info.channels,
            &mut self.scratch_inter,
        );
        self.producer.push_slice(&self.scratch_inter);
        self.scratch_mono.len()
    }

    /// Milliseconds of audio still waiting to be played.
    pub fn queued_ms(&self) -> u64 {
        let frames = self.producer.len() / self.info.channels.max(1) as usize;
        (frames as u64 * 1000) / self.info.sample_rate.max(1) as u64
    }

    pub fn is_draining(&self) -> bool {
        !self.producer.is_empty()
    }

    /// Drop all queued audio immediately (interrupt / barge-in).
    pub fn clear(&mut self) {
        self.flush_flag.store(true, Ordering::Release);
        self.resampler.reset();
    }

    pub fn set_muted(&self, muted: bool) {
        self.stats.muted.store(muted, Ordering::Relaxed);
    }

    pub fn underruns(&self) -> usize {
        self.stats.underruns.load(Ordering::Relaxed)
    }

    pub fn frames_played(&self) -> u64 {
        self.stats.frames_played.load(Ordering::Relaxed)
    }

    pub fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut e| e.take())
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

impl Drop for PlaybackHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------

/// Playback sink that records instead of making noise.
///
/// `realtime` consumes at wall-clock speed (so `--test` and the engine behave
/// like the real thing), while non-realtime drains as fast as possible so the
/// test-suite is not waiting on audio durations.
pub struct MockOutput {
    pub sample_rate: u32,
    pub channels: u16,
    pub realtime: bool,
    recorded: Arc<Mutex<Vec<f32>>>,
}

impl MockOutput {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 1,
            realtime: false,
            recorded: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_realtime(mut self, realtime: bool) -> Self {
        self.realtime = realtime;
        self
    }

    /// Everything that has been "played", in device order.
    pub fn recorded(&self) -> Arc<Mutex<Vec<f32>>> {
        Arc::clone(&self.recorded)
    }
}

impl AudioOutput for MockOutput {
    fn info(&self) -> AudioDeviceInfo {
        AudioDeviceInfo {
            name: "mock-output".into(),
            is_default: true,
            sample_rate: self.sample_rate,
            channels: self.channels,
            backend: "mock".into(),
        }
    }

    fn open(&self) -> Result<PlaybackHandle> {
        let (producer, mut consumer) = ring(self.sample_rate as usize * 4);
        let stop = Arc::new(AtomicBool::new(false));
        let flush_flag = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PlaybackStats::default());
        let error = Arc::new(Mutex::new(None));

        let recorded = Arc::clone(&self.recorded);
        let stop_thread = Arc::clone(&stop);
        let flush_thread = Arc::clone(&flush_flag);
        let stats_thread = Arc::clone(&stats);
        let realtime = self.realtime;
        let rate = self.sample_rate;
        let chunk = (rate / 50).max(1) as usize;

        let thread = std::thread::Builder::new()
            .name("cookie-mock-output".into())
            .spawn(move || {
                let mut buf = vec![0.0f32; chunk];
                while !stop_thread.load(Ordering::Acquire) {
                    if flush_thread.swap(false, Ordering::AcqRel) {
                        consumer.clear();
                    }
                    let n = consumer.pop_slice(&mut buf);
                    if n > 0 {
                        if let Ok(mut rec) = recorded.lock() {
                            rec.extend_from_slice(&buf[..n]);
                        }
                        stats_thread
                            .frames_played
                            .fetch_add(n as u64, Ordering::Relaxed);
                    } else {
                        stats_thread.underruns.fetch_add(1, Ordering::Relaxed);
                    }
                    if realtime {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            })
            .map_err(|e| Error::AudioDevice(format!("could not spawn mock output thread: {e}")))?;

        Ok(PlaybackHandle {
            producer,
            info: self.info(),
            resampler: Resampler::new(self.sample_rate, self.sample_rate),
            source_rate: self.sample_rate,
            scratch_mono: Vec::with_capacity(4096),
            scratch_inter: Vec::with_capacity(4096),
            stats,
            stop,
            flush_flag,
            thread: Some(thread),
            error,
            gain: 1.0,
        })
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

    pub struct CpalOutput {
        pub device_name: Option<String>,
        pub gain: f32,
    }

    impl CpalOutput {
        pub fn new(device_name: Option<String>, gain: f32) -> Self {
            Self { device_name, gain }
        }

        fn pick(
            wanted: Option<&str>,
        ) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String)> {
            let host = cpal::default_host();
            let device = match wanted {
                Some(want) => host
                    .output_devices()
                    .map_err(|e| Error::AudioDevice(e.to_string()))?
                    .find(|d| {
                        super::super::input::cpal_device_name(d)
                            .to_lowercase()
                            .contains(&want.to_lowercase())
                    })
                    .ok_or_else(|| {
                        Error::AudioDevice(format!("no output device matching {want:?}"))
                    })?,
                None => host
                    .default_output_device()
                    .ok_or_else(|| Error::AudioDevice("no default output device".into()))?,
            };
            let config = device
                .default_output_config()
                .map_err(|e| Error::AudioDevice(format!("no usable output config: {e}")))?;
            let name = super::super::input::cpal_device_name(&device);
            Ok((device, config, name))
        }

        /// The device that will actually be used when nothing is configured.
        ///
        /// Not the first one enumerated: on macOS that is whatever sorts
        /// first, which is how `--doctor` came to report a virtual loopback
        /// device nobody had selected while playback was correctly using the
        /// built-in speakers. Diagnostics must report what will happen, not
        /// what is merely present.
        pub fn default_name() -> Option<String> {
            cpal::default_host()
                .default_output_device()
                .map(|d| super::super::input::cpal_device_name(&d))
        }

        /// Device names for `--doctor`.
        pub fn list() -> Result<Vec<String>> {
            let host = cpal::default_host();
            Ok(host
                .output_devices()
                .map_err(|e| Error::AudioDevice(e.to_string()))?
                .map(|d| super::super::input::cpal_device_name(&d))
                .collect())
        }
    }

    impl AudioOutput for CpalOutput {
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

        fn open(&self) -> Result<PlaybackHandle> {
            let (_, probe, name) = Self::pick(self.device_name.as_deref())?;
            let device_rate = probe.sample_rate();
            let channels = probe.channels();
            let info = AudioDeviceInfo {
                name,
                is_default: self.device_name.is_none(),
                sample_rate: device_rate,
                channels,
                backend: "cpal".into(),
            };

            let (producer, consumer) = ring(device_rate as usize * channels as usize * 2);
            let stop = Arc::new(AtomicBool::new(false));
            let flush_flag = Arc::new(AtomicBool::new(false));
            let stats = Arc::new(PlaybackStats::default());
            let error = Arc::new(Mutex::new(None));

            let wanted = self.device_name.clone();
            let stop_thread = Arc::clone(&stop);
            let flush_thread = Arc::clone(&flush_flag);
            let stats_thread = Arc::clone(&stats);
            let error_thread = Arc::clone(&error);
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

            let thread = std::thread::Builder::new()
                .name("cookie-audio-out".into())
                .spawn(move || {
                    let built = (|| -> Result<cpal::Stream> {
                        let (device, config, _) = CpalOutput::pick(wanted.as_deref())?;
                        build_output_stream(
                            &device,
                            &config,
                            consumer,
                            flush_thread,
                            stats_thread,
                            error_thread.clone(),
                        )
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
                            *slot = Some(format!("could not start output stream: {e}"));
                        }
                        return;
                    }
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
                        "timed out opening the output device".into(),
                    ));
                }
            }

            Ok(PlaybackHandle {
                producer,
                info,
                resampler: Resampler::new(device_rate, device_rate),
                source_rate: device_rate,
                scratch_mono: Vec::with_capacity(8192),
                scratch_inter: Vec::with_capacity(8192),
                stats,
                stop,
                flush_flag,
                thread: Some(thread),
                error,
                gain: self.gain,
            })
        }
    }

    fn build_output_stream(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        mut consumer: super::super::ring::RingConsumer,
        flush_flag: Arc<AtomicBool>,
        stats: Arc<PlaybackStats>,
        error_slot: Arc<Mutex<Option<String>>>,
    ) -> Result<cpal::Stream> {
        let stream_config = cpal::StreamConfig {
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };
        let err_fn = move |e: cpal::Error| {
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
        };
        let mut scratch: Vec<f32> = vec![0.0; 16_384];

        macro_rules! build {
            ($t:ty) => {{
                device
                    .build_output_stream(
                        stream_config,
                        move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                            // REAL-TIME SECTION: no allocation, no locks, no I/O.
                            if flush_flag.swap(false, Ordering::AcqRel) {
                                consumer.clear();
                            }
                            let want = data.len().min(scratch.len());
                            let got = if stats.muted.load(Ordering::Relaxed) {
                                0
                            } else {
                                consumer.pop_slice(&mut scratch[..want])
                            };
                            if got < data.len() {
                                stats.underruns.fetch_add(1, Ordering::Relaxed);
                            }
                            for (i, out) in data.iter_mut().enumerate() {
                                let v = if i < got { scratch[i] } else { 0.0 };
                                *out = <$t>::from_sample(v);
                            }
                            stats.frames_played.fetch_add(got as u64, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| Error::AudioDevice(format!("could not build output stream: {e}")))
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
pub use cpal_impl::CpalOutput;

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for(recorded: &Arc<Mutex<Vec<f32>>>, at_least: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let n = recorded.lock().unwrap().len();
            if n >= at_least || std::time::Instant::now() > deadline {
                return n;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    #[test]
    fn mock_output_records_what_it_plays() {
        let out = MockOutput::new(16_000);
        let recorded = out.recorded();
        let mut handle = out.open().unwrap();
        let tone: Vec<f32> = (0..1600).map(|i| (i as f32 / 100.0).sin()).collect();
        handle.write(&tone, 16_000);
        assert!(wait_for(&recorded, 1600) >= 1600);
    }

    #[test]
    fn write_resamples_to_the_device_rate() {
        let out = MockOutput::new(48_000);
        let recorded = out.recorded();
        let mut handle = out.open().unwrap();
        // One second at 24 kHz must become ~one second at 48 kHz.
        handle.write(&vec![0.25; 24_000], 24_000);
        // `wait_for` returns as soon as the threshold is met, so it has to be
        // close to the expected count — otherwise this measures how fast the
        // callback thread happened to run rather than the resampler.
        let n = wait_for(&recorded, 47_900);
        assert!((n as i64 - 48_000).abs() < 500, "got {n} samples");
    }

    #[test]
    fn clear_drops_queued_audio() {
        let out = MockOutput::new(16_000).with_realtime(true);
        let recorded = out.recorded();
        let mut handle = out.open().unwrap();
        handle.write(&vec![0.5; 16_000 * 3], 16_000);
        handle.clear();
        std::thread::sleep(std::time::Duration::from_millis(120));
        let played = recorded.lock().unwrap().len();
        assert!(
            played < 16_000,
            "interrupt should have dropped most of 3s, played {played}"
        );
    }

    #[test]
    fn queued_ms_reports_backlog() {
        let out = MockOutput::new(16_000).with_realtime(true);
        let mut handle = out.open().unwrap();
        handle.write(&vec![0.1; 16_000], 16_000);
        assert!(handle.queued_ms() > 100, "got {}", handle.queued_ms());
    }

    #[test]
    fn dropping_the_handle_stops_the_device() {
        let out = MockOutput::new(16_000);
        let handle = out.open().unwrap();
        drop(handle);
    }
}
