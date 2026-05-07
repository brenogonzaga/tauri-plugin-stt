# Tauri Plugin STT (Speech-to-Text)

Cross-platform speech recognition plugin for Tauri 2.x. Desktop targets
(Windows, macOS, Linux) use [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
via [`whisper-rs`](https://crates.io/crates/whisper-rs); mobile targets
delegate to the native OS engines (`SFSpeechRecognizer` on iOS,
`SpeechRecognizer` on Android).

## Highlights

- **One model, 99 languages** — Whisper is multilingual; users download a single
  GGML model and it works for English, Portuguese, Mandarin, …
- **No native runtime to ship** — `whisper-rs` builds whisper.cpp statically;
  there is no `libvosk.so` / `.dylib` to install separately.
- **Explicit model lifecycle** — the host app controls when (and whether) a
  model is downloaded. `start_listening` returns `ModelNotInstalled` instead of
  silently pulling hundreds of MB.
- **Hardware acceleration** — opt-in `metal` / `cuda` / `vulkan` features map
  straight to the matching whisper.cpp backend.

## Platform Matrix

| Platform | Engine                                    | Model |
| -------- | ----------------------------------------- | ----- |
| iOS      | `SFSpeechRecognizer` (Speech.framework)   | OS    |
| Android  | `SpeechRecognizer`                        | OS    |
| macOS    | whisper.cpp via `whisper-rs` (Metal opt.) | GGML  |
| Windows  | whisper.cpp via `whisper-rs` (CUDA opt.)  | GGML  |
| Linux    | whisper.cpp via `whisper-rs` (Vulkan opt.) | GGML  |

## Installation

```toml
[dependencies]
tauri-plugin-stt = { version = "0.2", features = ["metal"] } # macOS
# or "cuda" / "vulkan" — omit for plain CPU inference
```

Register the plugin and the four model-management commands:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_stt::init())
        .run(tauri::generate_context!())
        .unwrap();
}
```

Capability:

```json
{ "permissions": ["stt:default"] }
```

## Model Catalogue

| id         | display      | size   | tier           |
| ---------- | ------------ | ------ | -------------- |
| `tiny`     | Tiny         | 75 MB  | fastest        |
| `base`     | Base         | 142 MB | balanced ⭐    |
| `small`    | Small        | 466 MB | accurate       |
| `medium`   | Medium       | 1.5 GB | very accurate  |
| `large-v3` | Large v3     | 3.0 GB | most accurate  |

Files are fetched from
`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-<id>.bin`
and stored under `<app_data_dir>/whisper-models/`. The active selection
is persisted to `whisper-models/active.txt`.

## Commands

- `list_models()` → `{ models, active, total_disk_bytes }`
- `install_model(id)` — downloads the model, emits `stt://download-progress`
- `remove_model(id)` — deletes the file and clears the active marker if needed
- `set_active_model(id)` — picks which installed model `start_listening` loads
- `start_listening({ language?, max_duration? })` — push-to-talk session
- `stop_listening()` — runs Whisper over the captured audio and emits one final result
- `is_available()` — reports `available: true` only when a model is installed
- `get_supported_languages()` — curated list of UI-facing locales
- `check_permission()` / `request_permission()` — microphone permission helpers

## Events

- `stt://download-progress` — `{ status, modelId, model, progress, downloaded?, total? }`
- `stt://result` — `{ transcript, isFinal, confidence }`
- `stt://error` — `{ code, message }`
- `plugin:stt:result` — same payload as `stt://result` (legacy listener channel)
- `plugin:stt:stateChange` — `{ state, isAvailable, language }`

## Behaviour Notes

- Whisper is **not** a streaming recogniser. The plugin buffers audio while
  recording and runs a single pass on `stop_listening`. UX is push-to-talk.
- Audio is captured at the device default rate, downmixed to mono, then
  decimated to 16 kHz with nearest-neighbour. Whisper is robust enough that a
  high-quality resampler buys nothing measurable.
- Inference uses `min(available_parallelism(), 4)` threads — beyond that
  whisper.cpp shows diminishing returns and we want headroom for the UI.

## Mobile

The mobile bridges expose the same JS API surface but `list_models` returns an
empty list and `install_model` / `remove_model` / `set_active_model` are
no-ops: the OS engine has no downloadable model concept. Use `is_available`
to gate UI; on iOS / Android it reflects actual recognizer availability.

## License

MIT.
