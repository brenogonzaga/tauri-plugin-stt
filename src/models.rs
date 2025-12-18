use serde::{Deserialize, Serialize};

/// Language code for speech recognition (e.g., "en-US", "pt-BR", "ja-JP")
pub type LanguageCode = String;

/// Configuration for starting speech recognition
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListenConfig {
    /// Language code for recognition (e.g., "en-US", "pt-BR")
    /// If not specified, uses device default language
    #[serde(default)]
    pub language: Option<LanguageCode>,

    /// Whether to return interim (partial) results
    #[serde(default, rename = "interimResults")]
    pub interim_results: bool,

    /// Whether to continue listening after getting a result
    /// If false, stops after first final result
    #[serde(default)]
    pub continuous: bool,

    /// Maximum duration to listen in milliseconds (0 = no limit)
    #[serde(default, rename = "maxDuration")]
    pub max_duration: u32,

    /// Maximum number of alternative transcriptions
    #[serde(default, rename = "maxAlternatives")]
    pub max_alternatives: Option<u32>,

    /// Use on-device recognition only (iOS 13+, no network required)
    /// When true, recognition works offline but may be less accurate
    #[serde(default, rename = "onDevice")]
    pub on_device: bool,
}

/// Recognition state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecognitionState {
    /// Not currently listening
    Idle,
    /// Actively listening for speech
    Listening,
    /// Processing audio (may briefly occur between utterances)
    Processing,
}

impl Default for RecognitionState {
    fn default() -> Self {
        Self::Idle
    }
}

/// A speech recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionResult {
    /// The recognized text
    pub transcript: String,

    /// Whether this is a final result (vs interim/partial)
    pub is_final: bool,

    /// Confidence score (0.0 to 1.0), if available
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// Current status of speech recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionStatus {
    /// Current state
    pub state: RecognitionState,

    /// Whether STT is available on this device
    pub is_available: bool,

    /// Current language being used
    #[serde(default)]
    pub language: Option<LanguageCode>,
}

/// Supported language information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupportedLanguage {
    /// Language code (e.g., "en-US")
    pub code: LanguageCode,

    /// Human-readable name (e.g., "English (United States)")
    pub name: String,

    /// Whether the model for this language is installed locally (desktop only)
    #[serde(default)]
    pub installed: Option<bool>,
}

/// Permission status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionStatus {
    /// Permission has been granted
    Granted,
    /// Permission has been denied
    Denied,
    /// Permission hasn't been requested yet
    Unknown,
}

/// Response for permission check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionResponse {
    /// Microphone permission status
    pub microphone: PermissionStatus,

    /// Speech recognition permission status (iOS/macOS specific)
    pub speech_recognition: PermissionStatus,
}

/// Response for availability check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailabilityResponse {
    /// Whether STT is available
    pub available: bool,

    /// Reason if not available
    #[serde(default)]
    pub reason: Option<String>,
}

/// Response for supported languages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupportedLanguagesResponse {
    /// List of supported languages
    pub languages: Vec<SupportedLanguage>,
}

/// Unified error codes for cross-platform consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SttErrorCode {
    /// No error
    None,
    /// Speech recognition service not available
    NotAvailable,
    /// Microphone permission denied
    PermissionDenied,
    /// Speech recognition permission denied (iOS)
    SpeechPermissionDenied,
    /// Network error (server-based recognition)
    NetworkError,
    /// Audio recording error
    AudioError,
    /// Recognition timed out (maxDuration reached)
    Timeout,
    /// No speech detected
    NoSpeech,
    /// Language not supported
    LanguageNotSupported,
    /// Recognition was cancelled
    Cancelled,
    /// Already listening
    AlreadyListening,
    /// Not currently listening
    NotListening,
    /// Service busy
    Busy,
    /// Unknown error
    Unknown,
}

impl Default for SttErrorCode {
    fn default() -> Self {
        Self::None
    }
}

impl SttErrorCode {
    /// Get a human-readable description of the error
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No error",
            Self::NotAvailable => "Speech recognition is not available on this device",
            Self::PermissionDenied => "Microphone permission was denied",
            Self::SpeechPermissionDenied => "Speech recognition permission was denied",
            Self::NetworkError => "Network error during recognition",
            Self::AudioError => "Error accessing audio input",
            Self::Timeout => "Recognition timed out",
            Self::NoSpeech => "No speech was detected",
            Self::LanguageNotSupported => "The requested language is not supported",
            Self::Cancelled => "Recognition was cancelled",
            Self::AlreadyListening => "Already listening for speech",
            Self::NotListening => "Not currently listening",
            Self::Busy => "Speech recognition service is busy",
            Self::Unknown => "An unknown error occurred",
        }
    }

    /// Get the numeric code for this error
    pub fn code(&self) -> i32 {
        match self {
            Self::None => 0,
            Self::NotAvailable => -1,
            Self::PermissionDenied => -2,
            Self::SpeechPermissionDenied => -3,
            Self::NetworkError => -4,
            Self::AudioError => -5,
            Self::Timeout => -6,
            Self::NoSpeech => -7,
            Self::LanguageNotSupported => -8,
            Self::Cancelled => -9,
            Self::AlreadyListening => -10,
            Self::NotListening => -11,
            Self::Busy => -12,
            Self::Unknown => -99,
        }
    }
}

/// Structured error event for frontend consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttError {
    /// Error code for programmatic handling
    pub code: SttErrorCode,
    /// Human-readable error message
    pub message: String,
    /// Platform-specific error details (optional)
    #[serde(default)]
    pub details: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listen_config_defaults() {
        let config: ListenConfig = serde_json::from_str("{}").unwrap();
        assert!(config.language.is_none());
        assert!(!config.interim_results);
        assert!(!config.continuous);
        assert_eq!(config.max_duration, 0);
    }

    #[test]
    fn test_listen_config_full() {
        let json = r#"{
            "language": "pt-BR",
            "interimResults": true,
            "continuous": true,
            "maxDuration": 30
        }"#;
        let config: ListenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.language, Some("pt-BR".to_string()));
        assert!(config.interim_results);
        assert!(config.continuous);
        assert_eq!(config.max_duration, 30);
    }

    #[test]
    fn test_recognition_state_serialization() {
        assert_eq!(
            serde_json::to_string(&RecognitionState::Idle).unwrap(),
            "\"idle\""
        );
        assert_eq!(
            serde_json::to_string(&RecognitionState::Listening).unwrap(),
            "\"listening\""
        );
        assert_eq!(
            serde_json::to_string(&RecognitionState::Processing).unwrap(),
            "\"processing\""
        );
    }

    #[test]
    fn test_recognition_result() {
        let result = RecognitionResult {
            transcript: "Hello world".to_string(),
            is_final: true,
            confidence: Some(0.95),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"transcript\":\"Hello world\""));
        assert!(json.contains("\"isFinal\":true"));
        assert!(json.contains("\"confidence\":0.95"));
    }

    #[test]
    fn test_permission_status_serialization() {
        assert_eq!(
            serde_json::to_string(&PermissionStatus::Granted).unwrap(),
            "\"granted\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionStatus::Denied).unwrap(),
            "\"denied\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionStatus::Unknown).unwrap(),
            "\"unknown\""
        );
    }
}
