# Tauri Plugin STT (Speech-to-Text)

Cross-platform speech recognition plugin for Tauri 2.x applications. Provides real-time speech-to-text functionality for desktop (Windows, macOS, Linux) and mobile (iOS, Android).

## Features

- 🎤 **Real-time Speech Recognition** - Convert speech to text with low latency
- 📱 **Cross-platform Support** - iOS, Android, macOS, Windows, Linux
- 🌐 **Multi-language Support** - 9 languages with automatic model download
- 📝 **Interim Results** - Get partial transcriptions while speaking
- 🔄 **Continuous Mode** - Auto-restart recognition after each utterance
- 🔐 **Permission Handling** - Request and check microphone/speech permissions
- 📥 **Auto Model Download** - Vosk models are downloaded automatically on first use

## Platform Support

| Platform | Status  | API Used                              | Model Download |
| -------- | ------- | ------------------------------------- | -------------- |
| iOS      | ✅ Full | SFSpeechRecognizer (Speech framework) | Not required   |
| Android  | ✅ Full | SpeechRecognizer API                  | Not required   |
| macOS    | ✅ Full | Vosk (offline speech recognition)     | Automatic      |
| Windows  | ✅ Full | Vosk (offline speech recognition)     | Automatic      |
| Linux    | ✅ Full | Vosk (offline speech recognition)     | Automatic      |

## Supported Languages (Desktop)

| Language   | Code  | Model Size |
| ---------- | ----- | ---------- |
| English    | en-US | 40 MB      |
| Portuguese | pt-BR | 31 MB      |
| Spanish    | es-ES | 39 MB      |
| French     | fr-FR | 41 MB      |
| German     | de-DE | 45 MB      |
| Russian    | ru-RU | 45 MB      |
| Chinese    | zh-CN | 43 MB      |
| Japanese   | ja-JP | 48 MB      |
| Italian    | it-IT | 39 MB      |

Models are downloaded automatically from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) when you first use a language.

## Installation

### Rust

Add the plugin to your `Cargo.toml`:

```toml
[dependencies]
tauri-plugin-stt = "0.1"
```

### TypeScript

Install the JavaScript guest bindings:

```bash
npm install tauri-plugin-stt-api
# or
yarn add tauri-plugin-stt-api
# or
pnpm add tauri-plugin-stt-api
```

## Setup

### Register Plugin

In your Tauri app setup:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_stt::init())
        .run(tauri::generate_context!())
        .expect("error while running application");
}
```

### Permissions

Add permissions to your `capabilities/default.json`:

```json
{
  "permissions": ["stt:default"]
}
```

For granular permissions, you can specify individual commands:

```json
{
  "permissions": [
    "stt:allow-is-available",
    "stt:allow-get-supported-languages",
    "stt:allow-check-permission",
    "stt:allow-request-permission",
    "stt:allow-start-listening",
    "stt:allow-stop-listening",
    "stt:allow-register-listener",
    "stt:allow-remove-listener"
  ]
}
```

### iOS Configuration (Required)

For iOS apps, you **must** create an `Info.plist` file in your `src-tauri` directory with permission descriptions:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSMicrophoneUsageDescription</key>
    <string>This app needs access to the microphone for speech recognition.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>This app needs access to speech recognition to convert your voice to text.</string>
</dict>
</plist>
```

Then reference it in your `tauri.conf.json`:

```json
{
  "bundle": {
    "iOS": {
      "infoPlist": "Info.plist"
    }
  }
}
```

**Note:** Without these permission descriptions, the app will crash when requesting permissions on iOS.

### Android Configuration

Android permissions are automatically included from the plugin's `AndroidManifest.xml`. No additional configuration needed.

### Vosk Library (Desktop Only)

The Vosk runtime library must be installed on your system:

#### macOS

```bash
# Download and install libvosk
curl -LO https://github.com/alphacep/vosk-api/releases/download/v0.3.42/vosk-osx-0.3.42.zip
unzip vosk-osx-0.3.42.zip
sudo cp vosk-osx-0.3.42/libvosk.dylib /usr/local/lib/
```

#### Linux

```bash
wget https://github.com/alphacep/vosk-api/releases/download/v0.3.42/vosk-linux-x86_64-0.3.42.zip
unzip vosk-linux-x86_64-0.3.42.zip
sudo cp vosk-linux-x86_64-0.3.42/libvosk.so /usr/local/lib/
sudo ldconfig
```

#### Windows

Download from [GitHub Releases](https://github.com/alphacep/vosk-api/releases) and add to PATH.

## Usage

### TypeScript API

```typescript
import {
  isAvailable,
  getSupportedLanguages,
  startListening,
  stopListening,
  onResult,
  onStateChange,
  onError,
} from "tauri-plugin-stt-api";

// Check if STT is available
const result = await isAvailable();

// Get supported languages (with installed status)
const languages = await getSupportedLanguages();

// Listen for results
const resultListener = await onResult((result) => {
  console.log("Recognized:", result.transcript, result.isFinal);
});

// Listen for download progress (when model is being downloaded)
import { listen } from "@tauri-apps/api/event";
const downloadListener = await listen<{
  status: string;
  model: string;
  progress: number;
}>("stt://download-progress", (event) => {
  console.log(`${event.payload.status}: ${event.payload.progress}%`);
});

// Start listening
await startListening({
  language: "en-US",
  interimResults: true,
  continuous: true,
  // maxDuration and onDevice are supported by the guest SDK
});

// Stop listening
await stopListening();
```

### Configuration Options

```typescript
interface ListenConfig {
  language?: string; // Language code (e.g., "en-US", "pt-BR")
  interimResults?: boolean; // Return partial results while speaking
  continuous?: boolean; // Continue listening after utterance ends
  maxDuration?: number; // Max listening duration in milliseconds (0 = unlimited)
  onDevice?: boolean; // Prefer on-device recognition (iOS)
}
```

### Event Listeners

```typescript
// Listen for results
const unlistenResult = await onResult((result) => {
  console.log(result.transcript, result.isFinal);
});

// Listen for state changes
const unlistenState = await onStateChange((event) => {
  console.log("State:", event.state); // "idle" | "listening" | "processing"
});

// Listen for errors
const unlistenError = await onError((error) => {
  console.error(`[${error.code}] ${error.message}`);
});

// Clean up listeners
unlistenResult();
unlistenState();
unlistenError();
```

## Events

| Event                     | Payload                                | Description                              |
| ------------------------- | -------------------------------------- | ---------------------------------------- |
| `stt://result`            | `{ transcript, isFinal, confidence? }` | Recognition result                       |
| `stt://state-change`      | `{ state }`                            | State change (idle/listening/processing) |
| `stt://error`             | `{ code, message, details? }`          | Error event                              |
| `stt://download-progress` | `{ status, model, progress }`          | Model download progress                  |

## API Reference

### `startListening(config?: ListenConfig): Promise<void>`

Start speech recognition.

**Config Options:**

- `language`: Language code (e.g., "en-US", "pt-BR")
- `interimResults`: Return partial results (default: `false`)
- `continuous`: Continue listening after utterance ends (default: `false`)
- `maxDuration`: Max listening duration in ms (0 = unlimited)
- `onDevice`: Use on-device recognition (iOS only, default: `false`)

### `stopListening(): Promise<void>`

Stop current speech recognition session.

### `isAvailable(): Promise<AvailabilityResponse>`

Check if STT is available on the device.

**Returns:**

- `available`: Whether STT is available
- `reason`: Optional reason if unavailable

### `getSupportedLanguages(): Promise<SupportedLanguagesResponse>`

Get list of supported languages.

**Returns:** Array of languages with:

- `code`: Language code (e.g., "en-US")
- `name`: Display name
- `installed`: Whether model is installed (desktop only)

### `checkPermission(): Promise<PermissionResponse>`

Check current permission status.

**Returns:**

- `microphone`: "granted" | "denied" | "unknown"
- `speechRecognition`: "granted" | "denied" | "unknown"

### `requestPermission(): Promise<PermissionResponse>`

Request microphone and speech recognition permissions.

**Returns:** Same as `checkPermission()`

### `onResult(handler: (result: RecognitionResult) => void): Promise<UnlistenFn>`

Listen for recognition results.

**Result:**

- `transcript`: Recognized text
- `isFinal`: Whether this is a final result
- `confidence`: Confidence score (0.0-1.0, if available)

### `onStateChange(handler: (event: StateChangeEvent) => void): Promise<UnlistenFn>`

Listen for state changes.

**States:** `"idle"`, `"listening"`, `"processing"`

### `onError(handler: (error: SttError) => void): Promise<UnlistenFn>`

Listen for errors.

**Error Codes:**

- `NOT_AVAILABLE`: STT not available on device
- `PERMISSION_DENIED`: Microphone permission denied
- `SPEECH_PERMISSION_DENIED`: Speech recognition permission denied
- `NETWORK_ERROR`: Network error (server-based recognition)
- `AUDIO_ERROR`: Audio capture error
- `TIMEOUT`: Recognition timeout
- `NO_SPEECH`: No speech detected
- `LANGUAGE_NOT_SUPPORTED`: Requested language not supported
- `CANCELLED`: Recognition cancelled by user
- `ALREADY_LISTENING`: Already in listening state
- `NOT_LISTENING`: Not currently listening
- `BUSY`: Recognizer busy
- `UNKNOWN`: Unknown error

## Building

### Without STT (Default)

```bash
npm run dev
```

### With STT

```bash
npm run dev -- --features stt
# or
npm run dev:stt
```

## Troubleshooting

### Desktop: "library 'vosk' not found"

**Solution:** Install the Vosk library as described in the Vosk Library section.

```bash
# macOS
ls /usr/local/lib/libvosk.dylib  # Should exist

# Linux
ldconfig -p | grep vosk  # Should show libvosk.so

# Windows
where vosk.dll  # Should be in PATH
```

### Desktop: "Model not found" or automatic download fails

**Problem:** Vosk models are downloaded automatically on first use for each language.

**Solution:**

1. Ensure internet connectivity
2. Check app data directory: `~/.local/share/tauri-plugin-stt/models/` (Linux/macOS) or `%APPDATA%/tauri-plugin-stt/models/` (Windows)
3. Manual download: Download from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract to models directory
4. Model naming: Ensure folder name matches expected pattern (e.g., `vosk-model-small-en-us-0.15`)

### Mobile: "Speech recognition not available"

**iOS Solution:**

1. Ensure iOS 10+ (speech recognition requires iOS 10+)
2. Check Settings → Privacy → Speech Recognition → Enable for your app
3. For on-device recognition, iOS 13+ is required

**Android Solution:**

1. Install Google app (provides speech recognition service)
2. Check Settings → Apps → Default apps → Digital assistant app
3. Ensure internet connectivity for server-based recognition

### Permission denied errors

**Solution:** Call `requestPermission()` before `startListening()`

```typescript
const perm = await requestPermission();
if (perm.microphone !== "granted") {
  console.error("Microphone permission required");
  return;
}
await startListening();
```

### No audio input detected

**Checklist:**

- ✅ Microphone is working in other apps
- ✅ Correct microphone selected in system settings
- ✅ Microphone not muted (hardware or software)
- ✅ App has microphone permission
- ✅ No other app is using the microphone exclusively

### Interim results not showing

**Note:** Interim results availability varies by platform:

- **iOS/Android**: Full support
- **Desktop (Vosk)**: Partial support (depends on model)

```typescript
await startListening({
  interimResults: true, // Enable interim results
  continuous: true, // Keep listening
});
```

### Recognition accuracy is low

**Tips:**

- Use correct language code for your accent (e.g., "en-GB" vs "en-US")
- Speak clearly and avoid background noise
- On iOS, download enhanced voices in Settings → Accessibility → Spoken Content
- Desktop: Use larger Vosk models for better accuracy (at cost of size)

### "ALREADY_LISTENING" error

**Solution:** Stop current session before starting a new one:

```typescript
try {
  await stopListening();
} catch (e) {
  // Ignore if not listening
}
await startListening();
```

### Download progress events not firing

**Note:** Download progress events are only for desktop (Vosk models). Mobile uses native speech recognition without downloads.

```typescript
import { listen } from "@tauri-apps/api/event";

const unlisten = await listen("stt://download-progress", (event) => {
  console.log(`${event.payload.status}: ${event.payload.progress}%`);
});
```

## Examples

See the [examples/stt-example](./examples/stt-example) directory for a complete working demo with React + Material UI, featuring:

- Real-time transcription with interim results
- Language selection
- Permission handling
- Error handling with visual feedback
- Download progress monitoring
- Results history

## Platform-Specific Notes

### iOS

- Requires iOS 10+ for basic speech recognition
- iOS 13+ required for on-device recognition (`onDevice: true`)
- Must add `NSSpeechRecognitionUsageDescription` to Info.plist
- Must add `NSMicrophoneUsageDescription` to Info.plist

### Android

- Requires Android API 23+ (Android 6.0+)
- Google app must be installed for speech recognition
- Internet required for server-based recognition
- Must request `RECORD_AUDIO` permission in AndroidManifest.xml

### Desktop (Windows, macOS, Linux)

- Requires Vosk library installation (see Vosk Library section)
- Models downloaded automatically (40-50 MB per language)
- Fully offline after model download
- Models stored in app data directory

## License

MIT
