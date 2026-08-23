# Tauri Plugin STT (Speech-to-Text)

Cross-platform speech recognition for Tauri 2.x. Desktop targets use [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [`whisper-rs`](https://crates.io/crates/whisper-rs); mobile delegates to the native OS engine (`SFSpeechRecognizer` on iOS, `SpeechRecognizer` on Android).

## Highlights

- **One model, 99 languages** — Whisper is multilingual; a single GGML model file handles English, Portuguese, Mandarin, and more
- **No separate runtime to install** — `whisper-rs` builds whisper.cpp statically; there is no `.so`/`.dylib` to ship
- **Explicit model lifecycle** — the host app controls when a model is downloaded; `start_listening` returns `ModelNotInstalled` instead of pulling hundreds of MB silently
- **Hardware acceleration** — opt-in `metal` / `cuda` / `vulkan` features map to the matching whisper.cpp backend

## Platform Matrix

| Platform | Engine                                     | Model |
| -------- | ------------------------------------------ | ----- |
| iOS      | `SFSpeechRecognizer` (Speech.framework)    | OS    |
| Android  | `SpeechRecognizer`                         | OS    |
| macOS    | whisper.cpp via `whisper-rs` (Metal opt.)  | GGML  |
| Windows  | whisper.cpp via `whisper-rs` (CUDA opt.)   | GGML  |
| Linux    | whisper.cpp via `whisper-rs` (Vulkan opt.) | GGML  |

## Installation

### Rust

```toml
[dependencies]
tauri-plugin-stt = { version = "0.2", features = ["metal"] }  # macOS
# "cuda" for NVIDIA GPU, "vulkan" for cross-vendor GPU, omit for CPU
```

### TypeScript

```bash
npm install tauri-plugin-stt-api
```

Register the plugin:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_stt::init())
        .run(tauri::generate_context!())
        .unwrap();
}
```

### Permissions

```json
{ "permissions": ["stt:default"] }
```

## Model Catalogue

| id         | Size   | Notes         |
| ---------- | ------ | ------------- |
| `tiny`     | 75 MB  | fastest       |
| `base`     | 142 MB | balanced ⭐   |
| `small`    | 466 MB | accurate      |
| `medium`   | 1.5 GB | very accurate |
| `large-v3-turbo` | 1.6 GB | fast & accurate (advanced) |
| `large-v3` | 3.0 GB | most accurate |

Files are fetched from HuggingFace (`ggerganov/whisper.cpp`) and stored under `<app_data_dir>/whisper-models/`. The active model is persisted to `whisper-models/active.txt`.

## Commands

- `list_models()` → `{ models, active, total_disk_bytes }`
- `install_model(id)` — downloads and emits `stt://download-progress` events
- `remove_model(id)` — deletes file; clears active marker if needed
- `set_active_model(id)` — sets which installed model `start_listening` loads
- `unload_model()` — drops the loaded Whisper context from memory; fails while listening or transcribing
- `start_listening({ language?, max_duration? })` — begins a push-to-talk session
- `stop_listening()` — runs Whisper over captured audio and emits a final result
- `is_available()` — `true` only when a model is installed and ready
- `get_supported_languages()` — curated list of UI-facing locales
- `check_permission()` / `request_permission()` — microphone permission helpers

## Events

- `stt://download-progress` — `{ status, modelId, model, progress, downloaded?, total? }`
- `stt://result` — `{ transcript, isFinal, confidence }`
- `stt://error` / `plugin:stt:error` — `{ code, message, details? }` (codes follow the `SttErrorCode` union, e.g. `NO_SPEECH`, `AUDIO_ERROR`)
- `plugin:stt:stateChange` — `{ state, isAvailable, language }` (`idle` is emitted only after transcription finishes)

## Behaviour Notes

- Whisper is **not** a streaming recogniser. The plugin buffers audio during recording and runs a single inference pass on `stop_listening`. The UX is push-to-talk, not live transcription.
- Audio is captured at the device default rate, downmixed to mono, then decimated to 16 kHz with nearest-neighbour. Whisper is robust enough that a higher-quality resampler makes no measurable difference.
- Inference uses `min(available_parallelism(), 4)` threads — beyond that whisper.cpp shows diminishing returns, and we want headroom for the UI.

## Mobile

The mobile bridges expose the same JS API surface, but `list_models` returns an empty list and `install_model` / `remove_model` / `set_active_model` / `unload_model` are no-ops: the OS engine has no downloadable model concept. Use `is_available` to gate UI — on iOS/Android it reflects actual recogniser availability.

## Used By

See [USED_BY.md](./USED_BY.md) for projects using this plugin in production.

## License

MIT
