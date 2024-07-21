use std::{
    ffi::CStr,
    fs,
    mem::{align_of, offset_of},
};

use allocator::MemoryAllocator;
use ash::{
    ext::debug_utils,
    khr::swapchain,
    util::read_spv,
    vk::{
        self, CommandBufferResetFlags, Extent2D, Queue, SwapchainKHR, API_VERSION_1_0,
        API_VERSION_1_2, API_VERSION_1_3,
    },
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle},
    window::Window,
};
const DEFAULT_FENCE_TIMEOUT: u64 = 100000000000;

mod allocator;
mod debug;

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub fn input_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // offset_of macro got stabilized in rust 1.77
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Self, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Self, tex_coord) as u32),
        ]
    }
}

pub fn begin_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }[0];

    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Failed to begin recording Command Buffer at beginning!");
    }

    command_buffer
}

pub fn end_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
    }

    let binding = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&binding);

    unsafe {
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();

        device
            .queue_submit(submit_queue, &[submit_info], fence)
            .expect("Failed to Queue Submit!");
        device
            .wait_for_fences(&[fence], true, DEFAULT_FENCE_TIMEOUT)
            .expect("Failed to wait for Fence");

        device.destroy_fence(fence, None);
        device.free_command_buffers(command_pool, &[command_buffer]);
    }
}

#[allow(dead_code)]
struct TextureRendering {
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

    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    pdevice: vk::PhysicalDevice,
    device: ash::Device,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,

    render_pass: vk::RenderPass,

    validation: bool,
    debug_utils: Option<debug_utils::Instance>,
    debug_utils_device: Option<debug_utils::Device>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    memory_allocator: MemoryAllocator,

    vertex_buffer: VulkanBuffer,
    index_buffer: VulkanBuffer,
    index_count: u32,

    image: VulkanImage,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
}

impl VulkanBuffer {
    pub fn new(
        device: &ash::Device,
        memory_allocator: &MemoryAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.unwrap();

        let buffer_memory = memory_allocator.allocate_buffer(&device, buffer, flags);

        Self {
            buffer,
            buffer_memory,
        }
    }

    pub fn new_init<T: Copy>(
        device: &ash::Device,
        memory_allocator: &MemoryAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        data: &[T],
        flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer = Self::new(device, memory_allocator, size, usage, flags);
        let pointer = unsafe {
            device
                .map_memory(buffer.buffer_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap()
        };

        let mut align = unsafe { ash::util::Align::new(pointer, align_of::<T>() as _, size) };
        align.copy_from_slice(data);

        unsafe { device.unmap_memory(buffer.buffer_memory) };
        buffer
    }

    fn cpu_to_gpu<T: Copy>(
        device: &ash::Device,
        buffer_size: vk::DeviceSize,
        data: &[T],
        memory_allocator: &MemoryAllocator,
        usage: vk::BufferUsageFlags,
        command_pool: &vk::CommandPool,
        graphics_queue: Queue,
    ) -> Self {
        // buffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let final_buffer = unsafe { device.create_buffer(&buffer_info, None) }.unwrap();

        let final_buffer_memory = memory_allocator.allocate_buffer(
            &device,
            final_buffer,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // staging
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&buffer_info, None) }.unwrap();

        let staging_buffer_memory = memory_allocator.allocate_buffer(
            &device,
            staging_buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let pointer = unsafe {
            device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap()
        };

        let mut align =
            unsafe { ash::util::Align::new(pointer, align_of::<T>() as _, buffer_size) };
        align.copy_from_slice(data);

        let command_buffer = begin_single_time_command(&device, *command_pool);

        unsafe {
            let buffer_info = vk::BufferCopy::default().size(buffer_size);

            device.cmd_copy_buffer(command_buffer, staging_buffer, final_buffer, &[buffer_info]);
        }
        unsafe { device.unmap_memory(staging_buffer_memory) };

        end_single_time_command(&device, *command_pool, graphics_queue, command_buffer);
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }
        Self {
            buffer: final_buffer,
            buffer_memory: final_buffer_memory,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.buffer_memory, None); // Destroy buffer first
        }
    }
}

pub struct VulkanImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub memory: vk::DeviceMemory,
}

impl VulkanImage {
    pub fn from_image(
        device: &ash::Device,
        image: image::DynamicImage,
        command_pool: vk::CommandPool,
        memory_allocator: &MemoryAllocator,
        graphics_queue: Queue,
    ) -> Self {
        let image_size = Extent2D {
            width: image.width(),
            height: image.height(),
        };
        let image_data = match &image {
            image::DynamicImage::ImageLuma8(_) | image::DynamicImage::ImageRgb8(_) => {
                image.into_rgba8().into_raw()
            }
            image::DynamicImage::ImageLumaA8(_) | image::DynamicImage::ImageRgba8(_) => {
                image.into_bytes()
            }
            _ => image.into_rgb8().into_raw(),
        };
        let image_data_size = (image_size.width * image_size.height * 4) as vk::DeviceSize;

        let mut staging_buffer = VulkanBuffer::new_init(
            device,
            &memory_allocator,
            image_data_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &image_data,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mip_level = 1;

        let format = vk::Format::R8G8B8A8_UNORM;

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(image_size.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { device.create_image(&create_info, None) }.unwrap();

        let memory =
            memory_allocator.allocate_image(&device, image, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        let command_buffer = begin_single_time_command(device, command_pool);

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: mip_level,
                layer_count: 1,
                ..Default::default()
            });

        let binding = [image_barrier];
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(&binding)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::BufferImageCopy2::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource)
            .image_offset(vk::Offset3D::default())
            .image_extent(image_size.into());

        let binding = [region];
        let copy_image_info = vk::CopyBufferToImageInfo2::default()
            .src_buffer(staging_buffer.buffer)
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&binding);

        unsafe { device.cmd_copy_buffer_to_image2(command_buffer, &copy_image_info) };

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: mip_level,
                layer_count: 1,
                ..Default::default()
            });

        let binding = [image_barrier];
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(&binding)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        end_single_time_command(device, command_pool, graphics_queue, command_buffer);
        staging_buffer.destroy(&device);

        let image_view_info = vk::ImageViewCreateInfo::default()
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )
            .image(image)
            .format(format)
            .view_type(vk::ImageViewType::TYPE_2D);

        let image_view = unsafe { device.create_image_view(&image_view_info, None) }.unwrap();

        let sampler_info = vk::SamplerCreateInfo::default();
        let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

        Self {
            image,
            image_view,
            sampler,
            memory,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
            device.destroy_sampler(self.sampler, None);
            device.free_memory(self.memory, None);
        }
    }
}

impl TextureRendering {
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

        let descriptor_pool = Self::create_descriptor_pool(&device, swapchain_images.len() as u32);
        let descriptor_layout = Self::create_descriptor_layout(&device);

        let pipeline_layout = Self::create_pipeline_layout(&device, &descriptor_layout);

        let pipeline =
            Self::create_pipeline(&device, &render_pass, surface_resolution, pipeline_layout);

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        let command_pool = Self::create_command_pool(&device, queue_family_index);
        let command_buffer = Self::create_command_buffer(&device, &command_pool);

        let memory_allocator = MemoryAllocator::new(unsafe {
            instance.get_physical_device_memory_properties(pdevice)
        });

        // QUAD
        let vertices = [
            Vertex {
                position: [-1.0, -1.0],
                tex_coord: [0.0, 0.0],
            },
            Vertex {
                position: [-1.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0],
                tex_coord: [1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
                tex_coord: [1.0, 0.0],
            },
        ];

        let indices = [0u32, 1, 2, 2, 3, 0];

        let buffer_size = std::mem::size_of_val(&vertices) as vk::DeviceSize;
        let vertex_buffer = VulkanBuffer::cpu_to_gpu(
            &device,
            buffer_size,
            &vertices,
            &memory_allocator,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &command_pool,
            graphics_queue,
        );
        let buffer_size = std::mem::size_of_val(&indices) as vk::DeviceSize;
        let index_buffer = VulkanBuffer::cpu_to_gpu(
            &device,
            buffer_size,
            &vertices,
            &memory_allocator,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &command_pool,
            graphics_queue,
        );

        let layouts = (0..swapchain_images.len())
            .map(|_| descriptor_layout)
            .collect::<Vec<_>>();
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info) }.unwrap();
        let image_file = concat!(env!("CARGO_MANIFEST_DIR"), "/texture.png");

        let image = VulkanImage::from_image(
            &device,
            image::open(image_file).expect("Failed to parse Image"),
            command_pool,
            &memory_allocator,
            graphics_queue,
        );

        for descriptor_set in &descriptor_sets {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image.image_view)
                .sampler(image.sampler);

            let desc_sets = [vk::WriteDescriptorSet {
                dst_set: *descriptor_set,
                dst_binding: 0, // From DescriptorSetLayoutBinding
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &image_info,
                ..Default::default()
            }];

            unsafe {
                device.update_descriptor_sets(&desc_sets, &[]);
            }
        }

        let (image_available_semaphore, render_finished_semaphore, in_flight_fence) =
            Self::create_sync_objects(&device);

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
            image,
            frame_buffers,
            swapchain,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
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
            descriptor_pool,
            descriptor_layout,
            memory_allocator,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            descriptor_sets,
        }
    }

    pub unsafe fn draw(&self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.in_flight_fence]).unwrap();
        }

        let (image_index, _sub_optimal) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
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
        self.device.cmd_bind_descriptor_sets(
            self.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &self.descriptor_sets[image_index as usize..=image_index as usize],
            &[],
        );
        self.device.cmd_bind_vertex_buffers2(
            self.command_buffer,
            0,
            &[self.vertex_buffer.buffer],
            &[0],
            None,
            None,
        );
        self.device.cmd_bind_index_buffer(
            self.command_buffer,
            self.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.device
            .cmd_draw_indexed(self.command_buffer, self.index_count, 1, 0, 0, 0);
        // END
        self.device.cmd_end_render_pass(self.command_buffer);

        self.device.end_command_buffer(self.command_buffer).unwrap();

        // Present

        let wait_semaphores = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.image_available_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        let command_buffers =
            vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer);
        let signal_semaphores = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);

        let signal_infos = [signal_semaphores];
        let command_infos = [command_buffers];
        let wait_infos = [wait_semaphores];
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&command_infos)
            .signal_semaphore_infos(&signal_infos);

        self.device
            .queue_submit2(self.graphics_queue, &[submit_info], self.in_flight_fence)
            .unwrap();

        let swapchains = &[self.swapchain];
        let image_indices = &[image_index];
        let binding = [self.render_finished_semaphore];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&binding)
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
            .api_version(API_VERSION_1_3);

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

    fn create_sync_objects(device: &ash::Device) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
        let create_info = vk::SemaphoreCreateInfo::default();

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        (
            unsafe { device.create_semaphore(&create_info, None) }.unwrap(),
            unsafe { device.create_semaphore(&create_info, None) }.unwrap(),
            unsafe { device.create_fence(&fence_info, None) }.unwrap(),
        )
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

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut features_1_3);

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

    pub fn create_descriptor_pool(device: &ash::Device, count: u32) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: count,
        }];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(count);

        unsafe { device.create_descriptor_pool(&create_info, None) }.unwrap()
    }

    pub fn create_descriptor_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let desc_layout_bindings = [
            // Fragment
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_layout_bindings);
        unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap()
    }

    pub fn create_pipeline_layout(
        device: &ash::Device,
        descriptor_layout: &vk::DescriptorSetLayout,
    ) -> vk::PipelineLayout {
        let binding = [*descriptor_layout];
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&binding);

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

        let binding = Vertex::input_descriptions();
        let binding2 = [Vertex::binding_description()];
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&binding)
            .vertex_binding_descriptions(&binding2);

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
            .logic_op(vk::LogicOp::CLEAR)
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

impl Drop for TextureRendering {
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

            self.vertex_buffer.destroy(&self.device);
            self.index_buffer.destroy(&self.device);
            self.image.destroy(&self.device);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);

            self.device.destroy_fence(self.in_flight_fence, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);

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

    let window_attributes = Window::default_attributes().with_title("Texture Rendering");
    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut app = TextureRendering::new(&window);

    event_loop.run_app(&mut app).unwrap();
}

impl ApplicationHandler for TextureRendering {
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
