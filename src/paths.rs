// Path management utilities based on tauri-plugin-sql patterns
// See: https://github.com/tauri-apps/plugins-workspace/tree/v2/plugins/sql

use std::fs::create_dir_all;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, Runtime};

use crate::error::Error;

/// Default subdirectory for Vosk models within app_data_dir
const MODELS_SUBDIR: &str = "vosk-models";

/// Gets the models directory for Vosk speech recognition models.
///
/// Uses `app_data_dir()` as base directory - these are large files that should persist.
///
/// # Example
/// ```rust,ignore
/// let models_dir = get_models_dir(&app)?;
/// let model_path = models_dir.join("vosk-model-en-us-0.22");
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

/// Gets a specific model's directory.
///
/// # Arguments
/// * `app` - The Tauri app handle
/// * `model_name` - Name of the model (e.g., "vosk-model-en-us-0.22")
#[allow(dead_code)]
pub fn get_model_path<R: Runtime>(app: &AppHandle<R>, model_name: &str) -> Result<PathBuf, Error> {
    validate_path(model_name)?;
    let models_dir = get_models_dir(app)?;
    Ok(models_dir.join(model_name))
}

/// Checks if a model exists in the models directory.
#[allow(dead_code)]
pub fn model_exists<R: Runtime>(app: &AppHandle<R>, model_name: &str) -> Result<bool, Error> {
    let model_path = get_model_path(app, model_name)?;
    Ok(model_path.exists() && model_path.is_dir())
}

/// Lists available models in the models directory.
#[allow(dead_code)]
pub fn list_available_models<R: Runtime>(app: &AppHandle<R>) -> Result<Vec<String>, Error> {
    let models_dir = get_models_dir(app)?;
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        models.push(name.to_string());
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
