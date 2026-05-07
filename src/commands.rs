use tauri::{command, AppHandle, Runtime};

use crate::models::*;
use crate::Result;
use crate::SttExt;

/// Start listening for speech
#[command]
pub(crate) async fn start_listening<R: Runtime>(
    app: AppHandle<R>,
    config: Option<ListenConfig>,
) -> Result<()> {
    app.stt().start_listening(config.unwrap_or_default())
}

/// Stop listening for speech
#[command]
pub(crate) async fn stop_listening<R: Runtime>(app: AppHandle<R>) -> Result<()> {
    app.stt().stop_listening()
}

/// Check if STT is available on this device
#[command]
pub(crate) async fn is_available<R: Runtime>(app: AppHandle<R>) -> Result<AvailabilityResponse> {
    app.stt().is_available()
}

/// Get list of supported languages
#[command]
pub(crate) async fn get_supported_languages<R: Runtime>(
    app: AppHandle<R>,
) -> Result<SupportedLanguagesResponse> {
    app.stt().get_supported_languages()
}

/// Check permission status
#[command]
pub(crate) async fn check_permission<R: Runtime>(app: AppHandle<R>) -> Result<PermissionResponse> {
    app.stt().check_permission()
}

/// Request permissions
#[command]
pub(crate) async fn request_permission<R: Runtime>(
    app: AppHandle<R>,
) -> Result<PermissionResponse> {
    app.stt().request_permission()
}

/// Register a listener for plugin events (desktop only)
/// On mobile, this is handled by the Plugin base class
#[cfg(desktop)]
#[command]
pub(crate) async fn register_listener() -> Result<()> {
    // The mobile plugin handles listeners internally. This command exists
    // to satisfy the front-end call from `addPluginListener` on desktop.
    Ok(())
}

/// Remove a previously registered plugin listener (desktop only)
/// On mobile, this is handled by the Plugin base class
#[cfg(desktop)]
#[command]
pub(crate) async fn remove_listener() -> Result<()> {
    // No-op: mobile plugin manages its own listeners.
    Ok(())
}

/// List every Whisper model the plugin knows how to install, plus the
/// active selection and the total disk usage of installed models.
///
/// `include_advanced` (default `false`) hides large/oversized models
/// from the default catalogue. The Voice settings page passes `true`
/// behind a "Show advanced models" toggle.
#[command]
pub(crate) async fn list_models<R: Runtime>(
    app: AppHandle<R>,
    include_advanced: Option<bool>,
) -> Result<WhisperModelsResponse> {
    app.stt().list_models(include_advanced.unwrap_or(false))
}

/// Download a Whisper model into the app data directory. Streams
/// progress events on `stt://download-progress`. The first model
/// installed becomes active automatically.
#[command]
pub(crate) async fn install_model<R: Runtime>(app: AppHandle<R>, id: String) -> Result<()> {
    app.stt().install_model(id)
}

/// Delete a previously downloaded Whisper model. Clears the active
/// selection if the removed model was the active one.
#[command]
pub(crate) async fn remove_model<R: Runtime>(app: AppHandle<R>, id: String) -> Result<()> {
    app.stt().remove_model(id)
}

/// Set which installed model `start_listening` should load. Returns
/// `ModelNotInstalled` when the requested model is not on disk.
#[command]
pub(crate) async fn set_active_model<R: Runtime>(app: AppHandle<R>, id: String) -> Result<()> {
    app.stt().set_active_model(id)
}
