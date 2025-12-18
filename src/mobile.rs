use serde::de::DeserializeOwned;
use tauri::{
    plugin::{PluginApi, PluginHandle},
    AppHandle, Runtime,
};

use crate::models::*;

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_stt);

/// Initializes the Kotlin or Swift plugin classes
pub fn init<R: Runtime, C: DeserializeOwned>(
    _app: &AppHandle<R>,
    api: PluginApi<R, C>,
) -> crate::Result<Stt<R>> {
    #[cfg(target_os = "android")]
    let handle = api.register_android_plugin("io.affex.stt", "SttPlugin")?;
    #[cfg(target_os = "ios")]
    let handle = api.register_ios_plugin(init_plugin_stt)?;
    Ok(Stt(handle))
}

/// Access to the STT APIs.
pub struct Stt<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> Stt<R> {
    /// Start listening for speech
    pub fn start_listening(&self, config: ListenConfig) -> crate::Result<()> {
        self.0
            .run_mobile_plugin("startListening", config)
            .map_err(Into::into)
    }

    /// Stop listening for speech
    pub fn stop_listening(&self) -> crate::Result<()> {
        self.0
            .run_mobile_plugin::<()>("stopListening", ())
            .map_err(Into::into)
    }

    /// Check if STT is available
    pub fn is_available(&self) -> crate::Result<AvailabilityResponse> {
        self.0
            .run_mobile_plugin("isAvailable", ())
            .map_err(Into::into)
    }

    /// Get supported languages
    pub fn get_supported_languages(&self) -> crate::Result<SupportedLanguagesResponse> {
        self.0
            .run_mobile_plugin("getSupportedLanguages", ())
            .map_err(Into::into)
    }

    /// Check permission status
    pub fn check_permission(&self) -> crate::Result<PermissionResponse> {
        self.0
            .run_mobile_plugin("checkPermission", ())
            .map_err(Into::into)
    }

    /// Request permissions
    pub fn request_permission(&self) -> crate::Result<PermissionResponse> {
        self.0
            .run_mobile_plugin("requestPermission", ())
            .map_err(Into::into)
    }
}
