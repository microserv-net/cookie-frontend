//! GPU plumbing for the orb.
//!
//! Small on purpose: one full-screen triangle, one uniform buffer, one
//! fragment shader. There is no mesh, no texture, no render graph and no
//! asset pipeline, because the entity is generated rather than played back.
//!
//! Two decisions are worth explaining:
//!
//! * **Alpha compositing is requested, not assumed.** A transparent window
//!   needs a surface alpha mode the compositor understands; where none is
//!   offered we fall back to opaque and tell the user, rather than drawing an
//!   orb in a black box and calling it done.
//! * **Nothing here is fatal.** Every failure path returns an error that the
//!   caller downgrades to "running without a face". Losing the visuals must
//!   never cost you the voice.

use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::animation::OrbParams;
use crate::config::Config;
use crate::error::{Error, Result};

/// The uniform block the shader reads. Layout must match `shaders/orb.wgsl`.
///
/// Packed into `vec4`s because WGSL uniform buffers align scalars to 16 bytes
/// anyway; grouping them keeps the block at 112 bytes instead of 448.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OrbUniform {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    e: [f32; 4],
    f: [f32; 4],
    g: [f32; 4],
}

impl OrbUniform {
    fn from_params(p: &OrbParams, aspect: f32, background_alpha: f32) -> Self {
        Self {
            a: [p.time, p.radius, p.wobble, p.turbulence],
            b: [p.swirl, p.flow_speed, p.noise_scale, p.detail],
            c: [p.glow, p.brightness, p.alpha, p.hue],
            d: [p.hue_spread, p.saturation, p.shell, p.core],
            e: [p.distortion, p.offset[0], p.offset[1], p.spin],
            f: [p.energy, p.bands[0], p.bands[1], p.bands[2]],
            g: [p.onset, p.seed, aspect, background_alpha],
        }
    }
}

/// Everything the renderer needs to draw a frame.
pub struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    backend: String,
    /// False when the compositor refused a transparent surface.
    transparent: bool,
}

impl std::fmt::Debug for GpuState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuState")
            .field("backend", &self.backend)
            .field("size", &(self.config.width, self.config.height))
            .field("transparent", &self.transparent)
            .finish()
    }
}

impl GpuState {
    /// Bring up an adapter, a device and a swapchain for `window`.
    pub async fn new(window: Arc<Window>, app_config: &Config) -> Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        // No backend is named: wgpu picks Vulkan, Metal, DX12 or GL as the
        // machine allows, which is the whole point of using it.
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance_descriptor);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| Error::Renderer(format!("no drawing surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // Low power on purpose: this is a small always-available orb,
                // not a game. On laptops it keeps us off the discrete GPU.
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Renderer(format!("no suitable graphics adapter: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cookie-orb"),
                // Defaults only: requesting features would exclude exactly the
                // integrated GPUs this is meant to run on.
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Renderer(format!("could not open the graphics device: {e}")))?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| Error::Renderer("this surface is not usable for drawing".into()))?;

        // Ask for a compositing mode that respects alpha; fall back quietly.
        let capabilities = surface.get_capabilities(&adapter);
        let wanted_alpha = if app_config.ui.transparent {
            [
                wgpu::CompositeAlphaMode::PreMultiplied,
                wgpu::CompositeAlphaMode::PostMultiplied,
                wgpu::CompositeAlphaMode::Inherit,
            ]
            .into_iter()
            .find(|mode| capabilities.alpha_modes.contains(mode))
        } else {
            None
        };
        let transparent = wanted_alpha.is_some();
        if let Some(mode) = wanted_alpha {
            config.alpha_mode = mode;
        } else if app_config.ui.transparent {
            tracing::info!(
                "this compositor will not blend the orb window; drawing it opaque instead"
            );
        }
        config.present_mode = if app_config.ui.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("orb"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/orb.wgsl").into()),
        });

        let uniform = OrbUniform::from_params(&OrbParams::default(), 1.0, 0.0);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("orb-uniforms"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("orb-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orb-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("orb-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("orb-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    // The shader outputs premultiplied alpha, which is what a
                    // transparent window needs to composite correctly.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            backend: format!("{:?}", adapter.get_info().backend),
            transparent,
        })
    }

    /// Graphics backend actually in use, for logs and diagnostics.
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Whether the window is really being blended with the desktop.
    pub fn is_transparent(&self) -> bool {
        self.transparent
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Draw one frame.
    pub fn render(&mut self, params: &OrbParams, background_alpha: f32) -> Result<()> {
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let uniform = OrbUniform::from_params(params, aspect, background_alpha);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        use wgpu::CurrentSurfaceTexture as Current;
        let frame = match self.surface.get_current_texture() {
            Current::Success(frame) => frame,
            // Suboptimal still draws; it just means the swapchain would
            // prefer to be reconfigured, which the next resize will do.
            Current::Suboptimal(frame) => frame,
            // A lost or outdated swapchain is routine (the window moved to
            // another monitor, the compositor restarted). Reconfigure and let
            // the next frame handle it.
            Current::Lost | Current::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            // Occluded means nothing is visible; skipping the frame is both
            // correct and the polite thing to do with somebody's battery.
            Current::Timeout | Current::Occluded => return Ok(()),
            Current::Validation => {
                return Err(Error::Renderer(
                    "the graphics driver rejected the surface".into(),
                ))
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orb-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("orb-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Clearing to fully transparent is what makes the
                        // window disappear around the orb.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        // In wgpu 30 presentation belongs to the queue, not the texture.
        self.queue.present(frame);
        Ok(())
    }
}
