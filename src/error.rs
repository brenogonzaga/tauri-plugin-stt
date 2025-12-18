use serde::{ser::Serializer, Serialize};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(mobile)]
    #[error(transparent)]
    PluginInvoke(#[from] tauri::plugin::mobile::PluginInvokeError),

    #[error("STT not available: {0}")]
    NotAvailable(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Recognition failed: {0}")]
    RecognitionFailed(String),

    #[error("Recording error: {0}")]
    Recording(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
