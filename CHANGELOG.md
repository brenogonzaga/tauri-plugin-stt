# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12

### Added

- Initial release of Speech-to-Text plugin for Tauri 2.x
- Cross-platform support (Windows, macOS, Linux, iOS, Android)
- Speech recognition APIs
  - `startListening()` - Begin speech recognition
  - `stopListening()` - End recognition session
  - `isAvailable()` - Check STT availability
  - `getSupportedLanguages()` - List available languages
  - `checkPermission()` - Check microphone/speech permissions
  - `requestPermission()` - Request required permissions
- Real-time transcription with interim results
- Continuous listening mode
- Multi-language support (8+ languages)
- Partial result updates
- TypeScript API with full type definitions
- Comprehensive documentation and examples

### Platform Implementation

#### Desktop (Windows, macOS, Linux)

- **Vosk Integration**: Offline speech recognition using Vosk 0.3
- **Audio Capture**: cpal 0.15 for cross-platform microphone access
- **Real-time Processing**: 16kHz sample rate with continuous audio streaming
- **Model-based Recognition**: Supports multiple languages via Vosk models
- **Requirements**:
  - Vosk model must be downloaded and placed in app data directory
  - Recommended: `vosk-model-small-en-us-0.15` (40 MB)
  - See README for detailed setup instructions
- **Features**:
  - Partial and final results
  - Alternative transcriptions support
  - Model caching for performance
  - Graceful error handling with clear setup instructions

#### iOS

- SFSpeechRecognizer framework
- On-device and server-based recognition
- Real-time partial results
- Native permission handling

#### Android

- SpeechRecognizer API
- Supports multiple recognition services
- Continuous recognition mode
- Runtime permission handling

### Supported Languages

- English (US, UK)
- Portuguese (Brazil)
- Spanish (Spain)
- French (France)
- German (Germany)
- Japanese (Japan)
- Chinese (Simplified)
- More languages available based on platform

### Configuration Options

- `language`: Target recognition language (BCP-47 format)
- `continuous`: Enable continuous listening mode
- `interimResults`: Receive partial transcription updates
- `maxAlternatives`: Number of alternative transcriptions

### Requirements

- Tauri: 2.9+
- Rust: 1.77+
- Android SDK: 24+ (Android 7.0+)
- iOS: 14.0+

### Known Limitations

- Desktop STT requires additional platform-specific implementation
- Some languages may not be available on all platforms
- Network connection may be required for cloud-based recognition
