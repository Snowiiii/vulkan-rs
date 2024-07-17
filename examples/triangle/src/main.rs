use std::{ffi::CStr, fs};

use ash::{
    ext::debug_utils,
    khr::swapchain,
    util::read_spv,
    vk::{self, CommandBufferResetFlags, Extent2D, SwapchainKHR, API_VERSION_1_0},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle},
    window::Window,
};

mod debug;

#[allow(dead_code)]
struct Triangle {
    // Our vulkan entry aka all our important Function pointers
    entry: ash::Entry,
    // Ash load all function pointers for our instance
    instance: ash::Instance,

    surface_loader: ash::khr::surface::Instance,
    surface_format: vk::SurfaceFormatKHR,
    surface: vk::SurfaceKHR,
    surface_resolution: vk::Extent2D,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    frame_buffers: Vec<vk::Framebuffer>,
    swapchain: vk::SwapchainKHR,

    fence: vk::Fence,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    pdevice: vk::PhysicalDevice,
    device: ash::Device,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    render_pass: vk::RenderPass,

    validation: bool,
    debug_utils: Option<debug_utils::Instance>,
    debug_utils_device: Option<debug_utils::Device>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Triangle {
    pub fn new(window: &winit::window::Window) -> Self {
        let display_handle = window.display_handle().unwrap();
        let window_handle = window.window_handle().unwrap();
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan") };
        let validation = cfg!(debug_assertions);
        let instance = Self::create_instance(&entry, display_handle, validation);

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .unwrap()
        };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let (pdevice, queue_family_index, present_family_index) =
            Self::create_physical_device(&instance, &surface_loader, surface);
        let surface_format =
            unsafe { surface_loader.get_physical_device_surface_formats(pdevice, surface) }
                .unwrap()[0];
        let device = Self::create_device(&instance, pdevice, queue_family_index);

        let (debug_utils, debug_utils_device, debug_messenger) = if validation {
            let (utils, device_utils, messenger) =
                debug::setup_debug_messenger(&entry, &instance, &device);
            (Some(utils), Some(device_utils), Some(messenger))
        } else {
            (None, None, None)
        };

        let render_pass = Self::create_render_pass(&device, surface_format);

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let window_size = window.inner_size();
        let (swapchain, surface_resolution) = Self::create_swapchain(
            &swapchain_loader,
            surface_format,
            &surface_loader,
            pdevice,
            true,
            surface,
            (window_size.width, window_size.height),
            None,
        );

        let (swapchain_image_views, swapchain_images) =
            Self::create_image_views(&device, &swapchain_loader, swapchain, surface_format);

        let frame_buffers = Self::create_frame_buffers(
            &swapchain_image_views,
            render_pass,
            &device,
            surface_resolution,
        );

        let pipeline_layout = Self::create_pipeline_layout(&device);

        let pipeline =
            Self::create_pipeline(&device, &render_pass, surface_resolution, pipeline_layout);

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        let command_pool = Self::create_command_pool(&device, queue_family_index);
        let command_buffer = Self::create_command_buffer(&device, &command_pool);

        let fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        };

        Self {
            entry,
            instance,
            surface_loader,
            surface_format,
            surface,
            surface_resolution,
            swapchain_loader,
            swapchain_images,
            swapchain_image_views,
            frame_buffers,
            swapchain,
            fence,
            pipeline_layout,
            pipeline,
            graphics_queue,
            present_queue,
            pdevice,
            device,
            command_pool,
            command_buffer,
            render_pass,
            validation,
            debug_utils,
            debug_utils_device,
            debug_messenger,
        }
    }

    pub unsafe fn draw(&self) {
        let (image_index, _sub_optimal) = unsafe {
            self.swapchain_loader
                .acquire_next_image(self.swapchain, u64::MAX, vk::Semaphore::null(), self.fence)
                .unwrap()
        };
        self.device
            .reset_command_buffer(self.command_buffer, CommandBufferResetFlags::empty())
            .unwrap();

        self.device
            .begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(self.surface_resolution);
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.2, 0.9, 1.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];

        let info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.frame_buffers[image_index as usize])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device
            .cmd_begin_render_pass(self.command_buffer, &info, vk::SubpassContents::INLINE);

        self.device.cmd_bind_pipeline(
            self.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        let viewport = vk::Viewport::default()
            .width(self.surface_resolution.width as f32)
            .height(self.surface_resolution.height as f32)
            .max_depth(1.0);

        self.device
            .cmd_set_scissor(self.command_buffer, 0, &[render_area]);
        self.device
            .cmd_set_viewport(self.command_buffer, 0, &[viewport]);

        // DRAWING
        self.device.cmd_draw(self.command_buffer, 3, 1, 0, 0);
        // END
        self.device.cmd_end_render_pass(self.command_buffer);

        self.device.end_command_buffer(self.command_buffer).unwrap();

        self.device
            .wait_for_fences(&[self.fence], true, u64::MAX)
            .unwrap();
        self.device.reset_fences(&[self.fence]).unwrap();

        // Present
        let binding = [self.command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&binding);
        self.device
            .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
            .unwrap();

        self.device.device_wait_idle().unwrap();

        let swapchains = &[self.swapchain];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.swapchain_loader
            .queue_present(self.present_queue, &present_info)
            .unwrap();
    }

    pub fn create_instance(
        entry: &ash::Entry,
        display_handle: DisplayHandle,
        validation: bool,
    ) -> ash::Instance {
        // Creates an Vulkan version which is used from our crate version
        let app_version = vk::make_api_version(
            0,
            env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        );

        let app_info = vk::ApplicationInfo::default()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"Triangle\n") })
            .application_version(app_version)
            .api_version(API_VERSION_1_0);

        let mut extension_names =
            ash_window::enumerate_required_extensions(display_handle.as_raw())
                .expect("Unsupported Surface Extension")
                .to_vec();

        if validation {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        let layer_names_ptrs = if validation {
            debug::check_validation_layer_support(entry);
            (debug::get_layer_names_and_pointers(),)
        } else {
            ((Vec::new(), Vec::new()),)
        };

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names_ptrs.0 .1)
            .flags(create_flags);

        unsafe {
            entry
                .create_instance(&instance_info, None)
                .expect("Failed to create Vulkan instance")
        }
    }

    ///
    /// Returns
    /// 1. Vulkan Physical Device
    /// 2. Graphics queue index
    /// 3. Present queue index
    pub fn create_physical_device(
        instance: &ash::Instance,
        surface_loader: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, u32, u32) {
        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };
        for pdevice in pdevices {
            let properties =
                unsafe { instance.get_physical_device_queue_family_properties(pdevice) };

            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

            let mut present = None;
            for (index, _properties) in properties.iter().enumerate() {
                if unsafe {
                    surface_loader.get_physical_device_surface_support(
                        pdevice,
                        index as u32,
                        surface,
                    )
                }
                .unwrap()
                {
                    present = Some(index as u32);
                    break;
                }
            }

            if let (Some(graphics), Some(present)) = (graphics, present) {
                return (pdevice, graphics, present);
            }
        }
        panic!("Missing required queue families")
    }

    fn create_device(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> ash::Device {
        let device_extension_names_raw = [swapchain::NAME.as_ptr()];

        let priorities = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw);

        unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .expect("Failed to create Vulkan device")
        }
    }

    fn create_frame_buffers(
        swapchain_image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        device: &ash::Device,
        surface_resolution: Extent2D,
    ) -> Vec<vk::Framebuffer> {
        swapchain_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&framebuffer_attachments)
                    .width(surface_resolution.width)
                    .height(surface_resolution.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<vk::Framebuffer>>()
    }

    fn create_render_pass(
        device: &ash::Device,
        surface_format: vk::SurfaceFormatKHR,
    ) -> vk::RenderPass {
        let renderpass_attachments = [vk::AttachmentDescription {
            format: surface_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        }];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        unsafe { device.create_render_pass(&create_info, None) }.unwrap()
    }

    fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        unsafe { device.create_command_pool(&create_info, None) }.unwrap()
    }

    fn create_command_buffer(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0]
    }

    pub fn create_pipeline_layout(device: &ash::Device) -> vk::PipelineLayout {
        let create_info = vk::PipelineLayoutCreateInfo::default();

        unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap()
    }
    fn create_pipeline(
        device: &ash::Device,
        render_pass: &vk::RenderPass,
        surface_resolution: Extent2D,
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let vertex_file = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/shader.vert.spv");
        let fragment_file = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/shader.frag.spv");
        let vertex_code =
            read_spv(&mut fs::File::open(vertex_file).expect("Failed to open Vertex File"))
                .unwrap();
        let vertex_module_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);
        let fragment_code =
            read_spv(&mut fs::File::open(fragment_file).expect("Failed to open Fragment File"))
                .unwrap();
        let fragment_module_info = vk::ShaderModuleCreateInfo::default().code(&fragment_code);

        let vertex_module =
            unsafe { device.create_shader_module(&vertex_module_info, None) }.unwrap();
        let fragment_module =
            unsafe { device.create_shader_module(&fragment_module_info, None) }.unwrap();

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let shader_stage_create_info = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                module: fragment_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default();

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: surface_resolution.width as f32,
            height: surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [surface_resolution.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .scissors(&scissors)
            .viewports(&viewports);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE, // TODO
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo::default();
        // .depth_test_enable(true)
        // .depth_write_enable(true)
        // .depth_compare_op(vk::CompareOp::LESS)
        // .max_depth_bounds(1.0);
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]; // TODO
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_info)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass);

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_info],
                None,
            )
        }
        .expect("Unable to create graphics pipeline");

        unsafe { device.destroy_shader_module(vertex_module, None) };
        unsafe { device.destroy_shader_module(fragment_module, None) };

        graphics_pipelines[0]
    }

    fn create_image_views(
        device: &ash::Device,
        swapchain_loader: &ash::khr::swapchain::Device,
        swapchain: SwapchainKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) -> (Vec<vk::ImageView>, Vec<vk::Image>) {
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();
        (
            images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    unsafe { device.create_image_view(&create_view_info, None) }.unwrap()
                })
                .collect::<Vec<vk::ImageView>>(),
            images,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_swapchain(
        swapchain_loader: &ash::khr::swapchain::Device,
        surface_format: vk::SurfaceFormatKHR,
        surface_loader: &ash::khr::surface::Instance,
        pdevice: vk::PhysicalDevice,
        vsync: bool,
        surface: vk::SurfaceKHR,
        size: (u32, u32),
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> (vk::SwapchainKHR, Extent2D) {
        let surface_capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(pdevice, surface) }
                .unwrap();
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }

        let width = size.0;
        let height = size.1;

        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(pdevice, surface) }
                .unwrap();

        let wanted_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::MAILBOX
        };
        let mut present_mode = vk::PresentModeKHR::FIFO; // Vsync, Always supported
        if present_modes.contains(&wanted_mode) {
            present_mode = wanted_mode;
        } else {
            println!("Swapchain: wanted mode is not supported, Using FIFO");
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain.unwrap_or_default());

        (
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }.unwrap(),
            surface_resolution,
        )
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_render_pass(self.render_pass, None);
            self.frame_buffers
                .drain(..)
                .for_each(|f| self.device.destroy_framebuffer(f, None));

            self.swapchain_image_views
                .drain(..)
                .for_each(|v| self.device.destroy_image_view(v, None));

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_fence(self.fence, None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            if let Some(debug_messenger) = self.debug_messenger {
                if let Some(debug_utils) = &self.debug_utils {
                    debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
                } else {
                    unreachable!()
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    let window_attributes = Window::default_attributes().with_title("Triangle");
    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut app = Triangle::new(&window);

    event_loop.run_app(&mut app).unwrap();
}

impl ApplicationHandler for Triangle {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if event == WindowEvent::RedrawRequested {
            unsafe { self.draw() };
        } else if event == WindowEvent::CloseRequested {
            event_loop.exit()
        }
    }
}

// 760+ Lines (without shaders) for a Triangle, Vulkan is great :D
