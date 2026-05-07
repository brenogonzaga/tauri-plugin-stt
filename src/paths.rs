// Path management utilities based on tauri-plugin-sql patterns
// See: https://github.com/tauri-apps/plugins-workspace/tree/v2/plugins/sql

use std::fs::create_dir_all;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, Runtime};

use crate::error::Error;

/// Default subdirectory for Whisper GGML models within app_data_dir
const MODELS_SUBDIR: &str = "whisper-models";

/// Gets the models directory for Whisper speech recognition models.
///
/// Uses `app_data_dir()` as base directory — these are large files
/// (75 MB – 3 GB) that should persist across app updates.
///
/// # Example
/// ```rust,ignore
/// let models_dir = get_models_dir(&app)?;
/// let model_path = models_dir.join("ggml-base.bin");
/// ```
#[allow(dead_code)]
pub fn get_models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, Error> {
    let base_path = app
        .path()
        .app_data_dir()
        .map_err(|e| Error::ConfigError(format!("Could not determine app data directory: {e}")))?;

    let full_path = base_path.join(MODELS_SUBDIR);

    create_dir_all(&full_path).map_err(|e| {
        Error::ConfigError(format!(
            "Could not create models directory {}: {}",
            full_path.display(),
            e
        ))
    })?;

    Ok(full_path)
}

/// Gets a specific model's path.
///
/// # Arguments
/// * `app` - The Tauri app handle
/// * `model_name` - Name of the model file (e.g., "ggml-base.bin")
#[allow(dead_code)]
pub fn get_model_path<R: Runtime>(app: &AppHandle<R>, model_name: &str) -> Result<PathBuf, Error> {
    validate_path(model_name)?;
    let models_dir = get_models_dir(app)?;
    Ok(models_dir.join(model_name))
}

/// Checks if a model file exists in the models directory.
#[allow(dead_code)]
pub fn model_exists<R: Runtime>(app: &AppHandle<R>, model_name: &str) -> Result<bool, Error> {
    let model_path = get_model_path(app, model_name)?;
    // Whisper models are single `.bin` files, not directories.
    Ok(model_path.exists() && model_path.is_file())
}

/// Lists available model files in the models directory. Returns the
/// raw filenames (`ggml-base.bin`, etc.); callers can match them
/// against the catalogue surfaced by `list_models`.
#[allow(dead_code)]
pub fn list_available_models<R: Runtime>(app: &AppHandle<R>) -> Result<Vec<String>, Error> {
    let models_dir = get_models_dir(app)?;
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                // Whisper models are single `.bin` files.
                if metadata.is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".bin") {
                            models.push(name.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(models)
}

/// Validates that a path doesn't contain path traversal attacks.
#[allow(dead_code)]
pub fn validate_path(path: &str) -> Result<(), Error> {
    let path_buf = PathBuf::from(path);

    for component in path_buf.components() {
        if let std::path::Component::ParentDir = component {
            return Err(Error::ConfigError(
                "Path traversal not allowed (contains '..')".to_string(),
            ));
        }
    }

    Ok(())
}
