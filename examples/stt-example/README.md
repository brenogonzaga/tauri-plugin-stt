# STT Plugin Example

Complete demonstration of the `tauri-plugin-stt` functionality using React + TypeScript + Material UI.

## Features Demonstrated

- ✅ Real-time speech recognition with live transcription
- ✅ Interim (partial) results while speaking
- ✅ Continuous listening mode
- ✅ Language selection from supported languages
- ✅ Permission checking and requesting
- ✅ Download progress monitoring (desktop Vosk models)
- ✅ State change tracking (idle/listening/processing)
- ✅ Error handling with detailed error codes
- ✅ Results history with timestamps and confidence scores
- ✅ Responsive design (mobile-friendly)

## Running the Example

### Desktop

```bash
npm install
npm run tauri dev -- --features stt
```

**Note:** The `--features stt` flag is required to enable STT support. The plugin is optional by default.

### Mobile

```bash
npm install
npm run tauri android dev
# or
npm run tauri ios dev
```

## Project Structure

```
stt-example/
├── src/
│   ├── App.tsx          # Main demo component
│   └── main.tsx         # React entry point
├── src-tauri/
│   ├── src/
│   │   └── main.rs      # Tauri setup with STT plugin
│   ├── Cargo.toml       # Rust dependencies
│   └── capabilities/
│       └── default.json # Permissions configuration
└── package.json         # NPM dependencies
```

## Code Highlights

### Permission Flow

The example demonstrates proper permission handling:

```typescript
// Check permission status
const perm = await checkPermission();

// Request if not granted
if (perm.microphone !== "granted") {
  await requestPermission();
}

// Then start listening
await startListening();
```

### Event Listeners

Three types of listeners are set up:

```typescript
// Results
onResult(result => {
  if (result.isFinal) {
    setTranscript(prev => prev + " " + result.transcript);
  } else {
    setPartialTranscript(result.transcript);
  }
});

// State changes
onStateChange(event => {
  setIsListening(event.state === "listening");
});

// Errors
onError(error => {
  console.error(`[${error.code}] ${error.message}`);
});
```

### Download Progress (Desktop)

Monitors Vosk model downloads:

```typescript
listen("stt://download-progress", event => {
  setDownloadProgress({
    status: event.payload.status,
    model: event.payload.model,
    progress: event.payload.progress,
  });
});
```

## Technologies Used

- **Tauri 2.x** - Desktop/Mobile application framework
- **React 18** - UI library
- **TypeScript** - Type safety
- **Material UI 6** - Component library
- **Vite** - Build tool
- **Vosk** (Desktop) - Offline speech recognition

## Platform-Specific Features

### Desktop (Vosk)

- Automatic model download
- Fully offline after setup
- Download progress monitoring
- Multiple language models

### iOS

- Native SFSpeechRecognizer
- On-device and server-based recognition
- Requires iOS 10+
- Better with iOS 13+ for on-device

### Android

- Native SpeechRecognizer API
- Requires Google app
- Server-based recognition
- Works on Android 6.0+

## Learn More

- [tauri-plugin-stt Documentation](../../README.md)
- [Tauri Documentation](https://tauri.app/)
- [Material UI Documentation](https://mui.com/)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/)
- [Tauri Extension](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode)
- [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## Troubleshooting

### "Plugin not found" error

**Solution:** Build with STT feature enabled:

```bash
npm run tauri dev -- --features stt
```

### No voices showing on desktop

**Solution:**

1. Ensure internet connectivity
2. Models will download automatically on first use
3. Check app data directory for models

### Permission denied on mobile

**Solution:**

1. iOS: Add required keys to Info.plist
2. Android: Add RECORD_AUDIO permission to AndroidManifest.xml
3. Request permission before starting: `await requestPermission()`
