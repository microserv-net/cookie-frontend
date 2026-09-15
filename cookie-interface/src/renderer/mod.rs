//! The orb on screen.
//!
//! ## Presentation
//!
//! The default is [`PresentationMode::CursorCompanion`]: a small, borderless,
//! transparent, click-through, always-on-top window that rides beside the
//! mouse pointer. That choice is deliberate and worth defending, because the
//! obvious alternative — a normal window somewhere on the desktop — is worse
//! in every way that matters. An assistant in a window is somewhere *else*;
//! you have to go and look at it. An assistant beside the pointer is already
//! where your eyes are, because your eyes follow your hand.
//!
//! Following is **not smoothed by default**. A trailing, easing orb looks
//! like laggy software rather than a companion, so the window is moved to the
//! pointer every frame with no interpolation (`ui.cursor_follow_lag = 0.0`
//! turns smoothing off; raise it if you prefer the floatier feel).
//!
//! ## Presence, not permanence
//!
//! The orb is absent while Cookie is idle. It appears when there is a reason
//! — you called her, she is working, she is speaking, something went wrong,
//! she is observing the screen — and fades out a beat after the reason ends.
//! See [`VisibilityConfig`]. An always-on glow next to the cursor would stop
//! being a presence within a day and start being clutter.
//!
//! ## Never in the way
//!
//! Hit-testing is off (`click_through`), so the orb cannot swallow a click,
//! and the window never takes focus. On platforms where a transparent or
//! always-on-top surface is refused, the renderer degrades to a normal window
//! and says so through diagnostics rather than failing to start.
//!
//! ## Thread boundary
//!
//! The renderer owns its window and its GPU resources and nothing else. It
//! reads state and audio features from watch channels and sends commands the
//! same way any other client would. It cannot block the audio pipeline, and
//! the audio pipeline cannot block it: a frame is never waiting on a model.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

use crate::animation::AnimationDirector;
use crate::audio::AudioFeatures;
use crate::config::{Config, PresentationMode};
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::events::Command;
use crate::VoiceState;

mod gpu;

pub use gpu::GpuState;

/// Run the window and render loop on the calling thread.
///
/// Must be the main thread on macOS and Windows; winit enforces it. The async
/// runtime keeps running in the background throughout.
pub fn run(engine: Engine, config: Arc<Config>) -> Result<()> {
    let event_loop = EventLoop::new()
        .map_err(|e| Error::Renderer(format!("could not create an event loop: {e}")))?;
    // `Poll` rather than `Wait`: the orb animates continuously, and the frame
    // pacer below decides how often. `Wait` would tie the animation to input.
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = OrbApp::new(engine, config);
    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Renderer(format!("the event loop stopped: {e}")))
}

/// Why the orb is currently on screen, or not.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Presence {
    /// Target opacity, 0 or 1.
    target: f32,
    /// Current opacity, eased toward the target.
    current: f32,
    /// When the last reason to be visible went away.
    since_reason: Option<Instant>,
}

impl Presence {
    fn new(visible_at_start: bool) -> Self {
        Self {
            target: if visible_at_start { 1.0 } else { 0.0 },
            current: if visible_at_start { 1.0 } else { 0.0 },
            since_reason: None,
        }
    }

    /// Decide whether the orb should be on screen this frame.
    fn update(&mut self, wanted: bool, dt: f32, config: &crate::config::VisibilityConfig) -> bool {
        let now = Instant::now();
        if wanted {
            self.since_reason = None;
            self.target = 1.0;
        } else {
            let since = *self.since_reason.get_or_insert(now);
            // Linger briefly so a short exchange does not flicker the orb in
            // and out twice in a second.
            if since.elapsed().as_secs_f32() >= config.linger_seconds {
                self.target = 0.0;
            }
        }
        let fade = config.fade_seconds.max(0.01);
        let step = dt / fade;
        if self.current < self.target {
            self.current = (self.current + step).min(self.target);
        } else if self.current > self.target {
            self.current = (self.current - step).max(self.target);
        }
        self.current > 0.001
    }
}

struct OrbApp {
    engine: Engine,
    config: Arc<Config>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    director: AnimationDirector,
    state_rx: tokio::sync::watch::Receiver<VoiceState>,
    features_rx: tokio::sync::watch::Receiver<AudioFeatures>,
    last_frame: Instant,
    last_present: Instant,
    presence: Presence,
    /// Last known pointer position, kept only as a fallback for platforms
    /// where the system query fails (some Wayland compositors).
    cursor: Option<LogicalPosition<f64>>,
    smoothed_cursor: Option<LogicalPosition<f64>>,
    scale_factor: f64,
    focused: bool,
    /// Set once the GPU is up, so diagnostics can report the truth about
    /// whether Cookie actually has a face on this machine.
    reported: bool,
    degraded_reason: Option<String>,
}

impl OrbApp {
    fn new(engine: Engine, config: Arc<Config>) -> Self {
        let director = AnimationDirector::new(config.ui.theme.clone(), config.ui.animation.clone());
        tracing::info!(
            seed = director.seed(),
            mode = ?config.ui.presentation,
            "orb starting"
        );
        Self {
            state_rx: engine.bus().state(),
            features_rx: engine.bus().features(),
            presence: Presence::new(config.ui.visibility.when_idle),
            director,
            engine,
            config,
            window: None,
            gpu: None,
            last_frame: Instant::now(),
            last_present: Instant::now(),
            cursor: None,
            smoothed_cursor: None,
            scale_factor: 1.0,
            focused: false,
            reported: false,
            degraded_reason: None,
        }
    }

    fn window_attributes(&self) -> WindowAttributes {
        let ui = &self.config.ui;
        let companion = matches!(
            ui.presentation,
            PresentationMode::CursorCompanion
                | PresentationMode::FloatingOrb
                | PresentationMode::Docked
        );
        let mut attributes = Window::default_attributes()
            .with_title("Cookie")
            .with_inner_size(LogicalSize::new(ui.width as f64, ui.height as f64))
            .with_decorations(!companion)
            .with_transparent(ui.transparent)
            .with_resizable(!companion);
        if ui.always_on_top && companion {
            attributes = attributes.with_window_level(WindowLevel::AlwaysOnTop);
        }
        if matches!(ui.presentation, PresentationMode::CursorCompanion) {
            // Never steal focus: you are typing somewhere else, and the orb
            // appearing must not interrupt that.
            attributes = attributes.with_active(false);
        }
        attributes
    }

    /// Where the pointer is, in logical screen coordinates.
    ///
    /// Queried from the windowing system rather than accumulated from motion
    /// events. winit reports motion relative to a window, so accumulating
    /// deltas drifted, needed an arbitrary origin, and stopped entirely while
    /// the pointer was over another application — which is where it spends
    /// almost all of its time. Asking the system costs a microsecond and is
    /// always right.
    fn pointer(&self) -> Option<LogicalPosition<f64>> {
        use mouse_position::mouse_position::Mouse;
        let Mouse::Position { x, y } = Mouse::get_mouse_position() else {
            return None;
        };
        // macOS reports the pointer in points, which are already logical —
        // dividing by the scale factor there put the orb at half the
        // coordinates, which is why it floated up and to the left of the
        // pointer on a Retina display and drifted further the further right
        // you went. Windows and X11 report device pixels.
        if cfg!(target_os = "macos") {
            Some(LogicalPosition::new(x as f64, y as f64))
        } else {
            Some(
                winit::dpi::PhysicalPosition::new(x as f64, y as f64).to_logical(self.scale_factor),
            )
        }
    }

    /// Where the orb should sit this frame, in logical screen coordinates.
    fn desired_position(&mut self, dt: f32) -> Option<LogicalPosition<f64>> {
        if !matches!(
            self.config.ui.presentation,
            PresentationMode::CursorCompanion
        ) {
            return None;
        }
        let cursor = self.pointer().or(self.cursor)?;
        // Centre the orb on the offset point. A window is positioned by its
        // top-left corner, so without this it sits up and to the left of
        // where it was asked to go — which is what made it look like it was
        // floating away from the pointer.
        let half_width = self.config.ui.width as f64 / 2.0;
        let half_height = self.config.ui.height as f64 / 2.0;
        let lag = self.config.ui.cursor_follow_lag;
        let target = LogicalPosition::new(
            cursor.x + self.config.ui.cursor_offset[0] as f64 - half_width,
            cursor.y + self.config.ui.cursor_offset[1] as f64 - half_height,
        );
        if lag <= 0.0 {
            self.smoothed_cursor = Some(target);
            return Some(target);
        }
        // Exponential approach with a half-life, so the feel is independent
        // of the frame rate.
        let previous = self.smoothed_cursor.unwrap_or(target);
        let t = 1.0 - (-dt / lag).exp();
        let smoothed = LogicalPosition::new(
            previous.x + (target.x - previous.x) * t as f64,
            previous.y + (target.y - previous.y) * t as f64,
        );
        self.smoothed_cursor = Some(smoothed);
        Some(smoothed)
    }

    /// Whether anything currently justifies being on screen.
    fn wants_presence(&self) -> bool {
        let visibility = &self.config.ui.visibility;
        let state = *self.state_rx.borrow();
        if visibility.wants(state) {
            return true;
        }
        // Backend work counts even when the voice pipeline is idle: that is
        // the "Cookie is thinking about something you asked for" case.
        visibility.when_working && !self.engine.tasks().active().is_empty()
    }

    fn frame_interval(&self) -> Duration {
        let ui = &self.config.ui;
        let fps = if self.focused || self.presence.target > 0.0 {
            ui.target_fps
        } else {
            ui.unfocused_fps.min(ui.target_fps)
        };
        Duration::from_secs_f32(1.0 / fps.max(1) as f32)
    }

    fn redraw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let visible = self
            .presence
            .update(self.wants_presence(), dt, &self.config.ui.visibility);

        let Some(window) = self.window.clone() else {
            return;
        };

        // Hiding rather than drawing nothing: a transparent always-on-top
        // window still costs a compositor pass on every platform.
        if !visible {
            window.set_visible(false);
            return;
        }
        window.set_visible(true);

        if let Some(position) = self.desired_position(dt) {
            window.set_outer_position(position);
        }

        let state = *self.state_rx.borrow();
        let features = self.features_rx.borrow().clone();
        let mut params = self.director.update(dt, state, &features);
        // The fade is applied on top of whatever the behaviour asked for, so
        // appearing and disappearing never fight the animation.
        params.alpha *= self.presence.current;

        if let Some(gpu) = self.gpu.as_mut() {
            match gpu.render(&params, self.config.ui.theme.background_alpha) {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!("frame dropped: {e}");
                }
            }
        }
    }
}

impl ApplicationHandler for OrbApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop.create_window(self.window_attributes()) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("could not create the orb window: {e}");
                self.degraded_reason = Some(e.to_string());
                self.engine.set_renderer_ok(false);
                return;
            }
        };
        if self.config.ui.click_through {
            // Not fatal if refused: some compositors simply do not offer it.
            if let Err(e) = window.set_cursor_hittest(false) {
                tracing::info!("this platform kept the orb clickable: {e}");
            }
        }
        self.scale_factor = window.scale_factor();
        window.set_visible(self.presence.current > 0.0);

        match pollster::block_on(GpuState::new(window.clone(), &self.config)) {
            Ok(gpu) => {
                tracing::info!(backend = %gpu.backend(), "renderer ready");
                self.engine.set_renderer_ok(true);
                self.gpu = Some(gpu);
            }
            Err(e) => {
                // No GPU is a degradation, not a failure: speech, hearing and
                // the API all still work, and diagnostics will say the orb is
                // missing if anybody asks.
                tracing::warn!("no graphics surface, running without the orb: {e}");
                self.degraded_reason = Some(e.to_string());
                self.engine.set_renderer_ok(false);
            }
        }
        self.reported = true;
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                let engine = self.engine.clone();
                tokio::spawn(async move { engine.shutdown().await });
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(focused) => self.focused = focused,
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Only fires when the pointer is over our own window, which
                // is rare for a click-through orb — but it costs nothing and
                // keeps the fallback fresh.
                let scale = self.scale_factor;
                self.cursor = Some(LogicalPosition::new(position.x / scale, position.y / scale));
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Only reachable when click-through is off. Clicking the orb
                // is the mouse equivalent of saying her name.
                let _ = self.engine.try_send(Command::StartListening {
                    continuous: true,
                    source: "orb".into(),
                });
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.state() == VoiceState::Error && self.window.is_none() {
            event_loop.exit();
            return;
        }
        // Frame pacing lives here rather than in a sleep inside `redraw`, so
        // input is still processed promptly between frames.
        if self.last_present.elapsed() >= self.frame_interval() {
            self.last_present = Instant::now();
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        } else if self.presence.current > 0.0 {
            // Following must not wait for the next frame: the orb moving at
            // the animation rate reads as lag, while moving the window is
            // nearly free.
            let dt = self.last_frame.elapsed().as_secs_f32();
            if let (Some(window), Some(position)) = (self.window.clone(), self.desired_position(dt))
            {
                window.set_outer_position(position);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VisibilityConfig;

    #[test]
    fn presence_appears_immediately_and_leaves_after_lingering() {
        let config = VisibilityConfig {
            linger_seconds: 0.0,
            fade_seconds: 0.1,
            ..Default::default()
        };
        let mut presence = Presence::new(false);
        // A reason appears: fade in over roughly `fade_seconds`.
        for _ in 0..12 {
            presence.update(true, 0.016, &config);
        }
        assert!(presence.current > 0.9);
        // Reason gone: fade out.
        for _ in 0..12 {
            presence.update(false, 0.016, &config);
        }
        assert!(presence.current < 0.1);
    }

    #[test]
    fn a_short_gap_does_not_flicker_the_orb() {
        let config = VisibilityConfig {
            linger_seconds: 1.0,
            fade_seconds: 0.1,
            ..Default::default()
        };
        let mut presence = Presence::new(true);
        // Half a second without a reason is inside the linger window.
        for _ in 0..30 {
            assert!(presence.update(false, 0.016, &config));
        }
        assert_eq!(presence.target, 1.0);
    }

    #[test]
    fn idle_is_invisible_by_default_but_working_is_not() {
        let config = VisibilityConfig::default();
        assert!(!config.wants(VoiceState::Idle));
        assert!(config.wants(VoiceState::Listening));
        assert!(config.wants(VoiceState::Processing));
        assert!(config.wants(VoiceState::Speaking));
        assert!(config.wants(VoiceState::Error));
    }
}
