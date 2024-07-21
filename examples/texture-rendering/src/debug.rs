// Mostly From https://github.com/ventengine/Vent-Engine/blob/master/crates/vent-rendering/src/debug.rs

use ash::ext::debug_utils;
use ash::{vk, Entry, Instance};
use std::borrow::Cow;
use std::os::raw::c_void;
use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
};

const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
    let callback_data = *p_callback_data;

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };
    // Use log crate
    match flag {
        Flag::VERBOSE => println!("{:?} - {:?}", typ, message),
        Flag::INFO => println!("{:?} - {:?}", typ, message),
        Flag::WARNING => println!("{:?} - {:?}", typ, message),
        _ => println!("{:?} - {:?}", typ, message),
    }
    vk::FALSE
}

/// Get the pointers to the validation layers names.
/// Also return the corresponding `CString` to avoid dangling pointers.
pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const c_char>) {
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).unwrap())
        .collect::<Vec<_>>();
    let layer_names_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    (layer_names, layer_names_ptrs)
}

/// Check if the required validation set in `REQUIRED_LAYERS`
/// are supported by the Vulkan instance.
///
/// # Panics
///
/// Panic if at least one on the layer is not supported.
pub fn check_validation_layer_support(entry: &Entry) {
    for required in REQUIRED_LAYERS.iter() {
        let properties = unsafe { entry.enumerate_instance_layer_properties().unwrap() }; // TODO: Cache
        let layers = properties.iter().find(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            let name = name.to_str().expect("Failed to get layer name pointer");
            required == &name
        });

        if layers.is_none() {
            panic!("Layer not supported: {}", required);
        }
    }
}

// We are targetting Vulkan 1.0, So we can't use that
// pub fn get_validation_features() -> vk::ValidationFeaturesEXT<'static> {
//     return vk::ValidationFeaturesEXT::default()
//         .enabled_validation_features(&[
//             vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
//             vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
//         ])
//         .disabled_validation_features(&[]); // We need to give it an empty Array, If not we get an validation error
// }

/// Setup the debug message if validation layers are enabled.
#[must_use]
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
    device: &ash::Device,
) -> (
    debug_utils::Instance,
    debug_utils::Device,
    vk::DebugUtilsMessengerEXT,
) {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_utils_device = debug_utils::Device::new(instance, device);
    let debug_utils_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .unwrap()
    };

    (debug_utils, debug_utils_device, debug_utils_messenger)
}
