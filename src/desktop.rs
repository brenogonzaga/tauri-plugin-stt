//! Desktop implementation backed by `whisper-rs` (whisper.cpp bindings).
//!
//! Why Whisper:
//!   * **One model covers 99 languages** — no per-language download.
//!   * `whisper.cpp` is statically linked through `whisper-rs`'s build
//!     script, so we don't ship any external runtime library.
//!
//! Trade-off: Whisper is *not* a streaming recognizer. Audio is buffered
//! while the user is speaking; on `stop_listening` we run the full
//! pipeline once and emit a single final result. The resulting UX is
//! push-to-talk (record → release → transcript in ≈100–500 ms for the
//! `tiny`/`base` models).
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Emitter, Manager, Runtime};
use whisper_rs::{
    convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

use crate::models::*;

/// Whisper consumes mono audio at exactly 16 kHz.
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Filename of the small marker that records which installed model is
/// currently selected as the active recognizer.
const ACTIVE_MARKER: &str = "active.txt";

/// Static catalogue of every Whisper.cpp GGML model we know how to
/// install. Kept ordered from smallest to largest so the UI renders in
/// a sensible default order without an extra sort.
struct ModelSpec {
    id: &'static str,
    display_name: &'static str,
    file_name: &'static str,
    url: &'static str,
    size_mb: u32,
    required_memory_mb: u32,
    tier: &'static str,
    recommended: bool,
    language: Option<&'static str>,
    advanced: bool,
}

/// Headroom multiplier on top of each model's published VRAM/RAM
/// requirement. Whisper.cpp's working set isn't the only thing on the
/// machine — the OS, the browser tab playing the lesson audio, and the
/// app itself all need room to breathe. 1.3× is the conservative
/// minimum we ask the user to have free; running a model right at its
/// floor reliably swaps on macOS and OOM-kills on low-end Linux.
const MEMORY_HEADROOM_NUMERATOR: u32 = 13;
const MEMORY_HEADROOM_DENOMINATOR: u32 = 10;

/// `true` when `host_mb` covers `required_mb` *with* the standard 30 %
/// headroom we promise the user. Saturating math so a stratospheric
/// `required_mb` can't wrap around.
fn host_fits(host_mb: u32, required_mb: u32) -> bool {
    let needed =
        required_mb.saturating_mul(MEMORY_HEADROOM_NUMERATOR) / MEMORY_HEADROOM_DENOMINATOR;
    host_mb >= needed
}

const CATALOGUE: &[ModelSpec] = &[
    // 39 M params · ~1 GB VRAM · ~10× realtime
    ModelSpec {
        id: "tiny",
        display_name: "Tiny",
        file_name: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        size_mb: 75,
        required_memory_mb: 1024,
        tier: "fastest",
        recommended: false,
        language: None,
        advanced: false,
    },
    ModelSpec {
        id: "tiny.en",
        display_name: "Tiny (English)",
        file_name: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        size_mb: 75,
        required_memory_mb: 1024,
        tier: "fastest",
        recommended: false,
        language: Some("en"),
        advanced: false,
    },
    // 74 M params · ~1 GB VRAM · ~7× realtime
    ModelSpec {
        id: "base",
        display_name: "Base",
        file_name: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        size_mb: 142,
        required_memory_mb: 1024,
        tier: "balanced",
        recommended: true,
        language: None,
        advanced: false,
    },
    ModelSpec {
        id: "base.en",
        display_name: "Base (English)",
        file_name: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        size_mb: 142,
        required_memory_mb: 1024,
        tier: "balanced",
        recommended: false,
        language: Some("en"),
        advanced: false,
    },
    // 244 M params · ~2 GB VRAM · ~4× realtime
    ModelSpec {
        id: "small",
        display_name: "Small",
        file_name: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        size_mb: 466,
        required_memory_mb: 2048,
        tier: "accurate",
        recommended: false,
        language: None,
        advanced: false,
    },
    ModelSpec {
        id: "small.en",
        display_name: "Small (English)",
        file_name: "ggml-small.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        size_mb: 466,
        required_memory_mb: 2048,
        tier: "accurate",
        recommended: false,
        language: Some("en"),
        advanced: false,
    },
    // 769 M params · ~5 GB VRAM · ~2× realtime
    ModelSpec {
        id: "medium",
        display_name: "Medium",
        file_name: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        size_mb: 1500,
        required_memory_mb: 5120,
        tier: "very accurate",
        recommended: false,
        language: None,
        advanced: false,
    },
    ModelSpec {
        id: "medium.en",
        display_name: "Medium (English)",
        file_name: "ggml-medium.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        size_mb: 1500,
        required_memory_mb: 5120,
        tier: "very accurate",
        recommended: false,
        language: Some("en"),
        advanced: false,
    },
    // 809 M params · ~6 GB VRAM · ~8× realtime — multilingual only.
    // Faster than `medium` and almost as accurate, but the 6 GB working
    // set rules out most laptops; advanced-only by default.
    ModelSpec {
        id: "large-v3-turbo",
        display_name: "Turbo",
        file_name: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        size_mb: 1620,
        required_memory_mb: 6144,
        tier: "fast & accurate",
        recommended: false,
        language: None,
        advanced: true,
    },
    // 1550 M params · ~10 GB VRAM · 1× realtime — multilingual only.
    ModelSpec {
        id: "large-v3",
        display_name: "Large v3",
        file_name: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        size_mb: 3000,
        required_memory_mb: 10240,
        tier: "most accurate",
        recommended: false,
        language: None,
        advanced: true,
    },
];

fn spec_by_id(id: &str) -> Option<&'static ModelSpec> {
    CATALOGUE.iter().find(|m| m.id == id)
}

/// Total physical RAM in megabytes, as reported by `sysinfo`. We use
/// this as a proxy for "what's the largest Whisper model that will
/// actually load on this machine" — both for the CPU path (working
/// set lives in RAM) and the GPU path (most of our users have unified
/// memory or VRAM ≤ system RAM).
fn system_memory_mb() -> u32 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    // sysinfo reports total memory in *kibibytes* historically; in 0.32
    // it returns bytes. Divide by 1024 * 1024 to land on MB regardless.
    let bytes = sys.total_memory();
    let mb = bytes / (1024 * 1024);
    u32::try_from(mb).unwrap_or(u32::MAX)
}

/// Mutable state shared across commands. Wrapped in `Arc<Mutex<…>>`
/// because the cpal audio thread also pushes samples into the buffer
/// while the user is speaking.
struct SttState {
    /// Loaded Whisper context. `None` until the first `start_listening`
    /// loads the active model. Reused for every subsequent session.
    context: Option<Arc<WhisperContext>>,
    /// Id of the model `context` was loaded from. Used to detect a
    /// model change between sessions and reload accordingly.
    loaded_model: Option<String>,
    /// Audio captured by the cpal stream during the current session.
    /// Already mono, 16-bit PCM, 16 kHz — ready to convert to f32.
    buffer: Arc<Mutex<Vec<i16>>>,
    /// `true` while recording. The cpal callback short-circuits when
    /// this flips to `false`, so we never drop the stream — we just
    /// stop appending samples.
    listening: Arc<AtomicBool>,
    /// Live cpal stream. Kept alive for the plugin's lifetime so the
    /// first `start_listening` can be reused (cpal stream creation is
    /// expensive and triggers the macOS microphone permission prompt).
    stream_alive: bool,
    /// Optional max-duration (ms) after which we auto-stop. `None`
    /// means "until the caller invokes `stop_listening`".
    max_duration_ms: Option<u64>,
    /// Wall-clock start of the current session — used for diagnostics.
    started_at: Option<Instant>,
    /// Locale tag (e.g. `pt-BR`) requested for the current session.
    /// Whisper expects ISO-639-1 codes, so we strip the region.
    language: Option<String>,
}

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
) -> crate::Result<Stt<R>> {
    let state = Arc::new(Mutex::new(SttState {
        context: None,
        loaded_model: None,
        buffer: Arc::new(Mutex::new(Vec::with_capacity(
            TARGET_SAMPLE_RATE as usize * 30,
        ))),
        listening: Arc::new(AtomicBool::new(false)),
        stream_alive: false,
        max_duration_ms: None,
        started_at: None,
        language: None,
    }));

    Ok(Stt {
        app: app.clone(),
        state,
    })
}

pub struct Stt<R: Runtime> {
    app: AppHandle<R>,
    state: Arc<Mutex<SttState>>,
}

impl<R: Runtime> Stt<R> {
    fn models_dir(&self) -> PathBuf {
        self.app
            .path()
            .app_data_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("whisper-models")
    }

    fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.models_dir().join(spec.file_name)
    }

    fn active_marker_path(&self) -> PathBuf {
        self.models_dir().join(ACTIVE_MARKER)
    }

    /// Returns the persisted active model id, falling back to the
    /// recommended model if it is installed, then to whatever happens
    /// to be on disk. `None` means "nothing usable installed".
    fn resolve_active_id(&self) -> Option<String> {
        if let Ok(raw) = fs::read_to_string(self.active_marker_path()) {
            let id = raw.trim();
            if let Some(spec) = spec_by_id(id) {
                if self.model_path(spec).exists() {
                    return Some(spec.id.to_string());
                }
            }
        }
        // Marker missing or stale — pick the first installed model in
        // catalogue order so the recommended one wins ties.
        for spec in CATALOGUE {
            if self.model_path(spec).exists() {
                return Some(spec.id.to_string());
            }
        }
        None
    }

    fn write_active_marker(&self, id: &str) -> crate::Result<()> {
        fs::create_dir_all(self.models_dir())?;
        fs::write(self.active_marker_path(), id)?;
        Ok(())
    }

    /// Builds the `WhisperModelInfo` list reflecting on-disk state.
    ///
    /// `include_advanced` controls visibility of two classes of model:
    ///   * **Advanced tier** — the `large` family. Multi-GB download
    ///     and multi-GB working set; massive overkill for spaced-rep
    ///     speaking practice. Hidden by default; only the Voice
    ///     settings page should opt in.
    ///   * **Doesn't fit in RAM** — anything whose `required_memory_mb`
    ///     exceeds the host's reported total RAM. Same rationale: by
    ///     default we don't tease the user with a model they can't run.
    ///
    /// Already-installed models are *always* listed so a power user
    /// who flipped the toggle, downloaded a model, then flipped it
    /// back can still see and remove what they have.
    pub fn list_models(&self, include_advanced: bool) -> crate::Result<WhisperModelsResponse> {
        let active = self.resolve_active_id();
        let host_mb = system_memory_mb();
        let mut total: u64 = 0;
        let models = CATALOGUE
            .iter()
            .filter_map(|spec| {
                let path = self.model_path(spec);
                let installed = path.exists();
                if installed {
                    if let Ok(meta) = fs::metadata(&path) {
                        total = total.saturating_add(meta.len());
                    }
                }
                let fits = spec.required_memory_mb <= host_mb;
                let visible = installed || include_advanced || (!spec.advanced && fits);
                if !visible {
                    return None;
                }
                Some(WhisperModelInfo {
                    id: spec.id.to_string(),
                    display_name: spec.display_name.to_string(),
                    size_mb: spec.size_mb,
                    required_memory_mb: spec.required_memory_mb,
                    installed,
                    active: Some(spec.id.to_string()) == active,
                    recommended: spec.recommended,
                    tier: spec.tier.to_string(),
                    language: spec.language.map(str::to_owned),
                    fits_in_memory: fits,
                    advanced: spec.advanced,
                })
            })
            .collect();
        Ok(WhisperModelsResponse {
            models,
            active,
            total_disk_bytes: total,
            system_memory_mb: host_mb,
        })
    }

    /// Downloads `id` into the app data directory if it is not already
    /// present. If no model is currently active, the freshly installed
    /// one becomes active automatically. Emits `stt://download-progress`
    /// events at ~4 Hz so the UI can render a progress bar.
    ///
    /// Refuses with `InsufficientMemory` when the host doesn't have
    /// enough physical RAM to load the model — better to fail fast
    /// here than to hand the user a 1.5 GB file they can never run.
    /// Already-installed models are exempt so we don't strand a model
    /// that fit on a previous machine.
    pub fn install_model(&self, id: String) -> crate::Result<()> {
        let spec = spec_by_id(&id).ok_or_else(|| crate::Error::UnknownModel(id.clone()))?;
        let dest = self.model_path(spec);
        if dest.exists() {
            // Already installed — make it the active model if nothing
            // else is, then return so the UI gets a fast path.
            if self.resolve_active_id().is_none() {
                self.write_active_marker(spec.id)?;
            }
            return Ok(());
        }
        let host_mb = system_memory_mb();
        if host_mb > 0 && !host_fits(host_mb, spec.required_memory_mb) {
            return Err(crate::Error::InsufficientMemory(format!(
                "{} needs ~{} MB (with 30% headroom) but this device reports {} MB total",
                spec.display_name, spec.required_memory_mb, host_mb
            )));
        }
        fs::create_dir_all(self.models_dir())
            .map_err(|e| crate::Error::Recording(format!("create models dir: {e}")))?;

        let _ = self.app.emit(
            "stt://download-progress",
            serde_json::json!({
                "status": "downloading",
                "modelId": spec.id,
                "model": spec.file_name,
                "progress": 0
            }),
        );

        let url = spec.url.to_string();
        let model_id = spec.id.to_string();
        let model_file = spec.file_name.to_string();
        let app_handle = self.app.clone();
        let dest_clone = dest.clone();

        let join = thread::spawn(move || -> Result<(), String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60 * 60))
                .build()
                .map_err(|e| format!("http client: {e}"))?;
            let mut response = client
                .get(&url)
                .send()
                .map_err(|e| format!("get {url}: {e}"))?
                .error_for_status()
                .map_err(|e| format!("http status: {e}"))?;

            let total = response.content_length();
            let tmp = dest_clone.with_extension("part");
            let mut file =
                fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
            let mut downloaded: u64 = 0;
            let mut last_emit = Instant::now();
            let mut chunk = [0u8; 64 * 1024];
            use std::io::{Read, Write};
            loop {
                let n = response
                    .read(&mut chunk)
                    .map_err(|e| format!("read chunk: {e}"))?;
                if n == 0 {
                    break;
                }
                file.write_all(&chunk[..n])
                    .map_err(|e| format!("write chunk: {e}"))?;
                downloaded += n as u64;
                // Throttle progress events to 4 Hz so we don't flood
                // the IPC channel with hundreds of events per second.
                if last_emit.elapsed() >= Duration::from_millis(250) {
                    last_emit = Instant::now();
                    let progress = match total {
                        Some(t) if t > 0 => ((downloaded as f64 / t as f64) * 100.0) as u8,
                        _ => 0,
                    };
                    let _ = app_handle.emit(
                        "stt://download-progress",
                        serde_json::json!({
                            "status": "downloading",
                            "modelId": &model_id,
                            "model": &model_file,
                            "progress": progress,
                            "downloaded": downloaded,
                            "total": total
                        }),
                    );
                }
            }
            // Atomic-ish rename so a crash mid-download never leaves a
            // partial `.bin` that whisper.cpp would gladly try to load.
            fs::rename(&tmp, &dest_clone).map_err(|e| {
                format!("rename {} -> {}: {e}", tmp.display(), dest_clone.display())
            })?;
            Ok(())
        });

        match join.join() {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => {
                let _ = self.app.emit(
                    "stt://download-progress",
                    serde_json::json!({
                        "status": "error",
                        "modelId": spec.id,
                        "model": spec.file_name,
                        "message": &msg,
                    }),
                );
                return Err(crate::Error::Recording(msg));
            }
            Err(_) => {
                return Err(crate::Error::Recording("download thread panicked".into()));
            }
        }

        // Promote the freshly installed model to active when nothing
        // was selected before. Avoids a second round-trip from the UI.
        if self.resolve_active_id().is_none() {
            self.write_active_marker(spec.id)?;
        }

        let _ = self.app.emit(
            "stt://download-progress",
            serde_json::json!({
                "status": "complete",
                "modelId": spec.id,
                "model": spec.file_name,
                "progress": 100
            }),
        );

        Ok(())
    }

    /// Deletes a model file and clears the active marker if the removed
    /// model was the active one. The freed disk is reflected in the
    /// next `list_models` response.
    pub fn remove_model(&self, id: String) -> crate::Result<()> {
        let spec = spec_by_id(&id).ok_or_else(|| crate::Error::UnknownModel(id.clone()))?;
        let path = self.model_path(spec);
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| crate::Error::Recording(format!("remove model: {e}")))?;
        }
        // If the removed model was active, clear / re-pick. Also drop
        // the in-memory context so the next session reloads cleanly.
        let was_active = self.resolve_active_id().as_deref() == Some(&id);
        if was_active {
            let _ = fs::remove_file(self.active_marker_path());
            let mut state = self.state.lock().unwrap();
            if state.loaded_model.as_deref() == Some(&id) {
                state.context = None;
                state.loaded_model = None;
            }
        }
        Ok(())
    }

    /// Sets which installed model `start_listening` should load next.
    /// Returns `ModelNotInstalled` when the requested model is missing
    /// so the UI can route the user to the install flow.
    pub fn set_active_model(&self, id: String) -> crate::Result<()> {
        let spec = spec_by_id(&id).ok_or_else(|| crate::Error::UnknownModel(id.clone()))?;
        if !self.model_path(spec).exists() {
            return Err(crate::Error::ModelNotInstalled(id));
        }
        self.write_active_marker(spec.id)?;
        // Force the next session to reload — cheap because mmap.
        let mut state = self.state.lock().unwrap();
        if state.loaded_model.as_deref() != Some(spec.id) {
            state.context = None;
            state.loaded_model = None;
        }
        Ok(())
    }

    /// Loads (and caches) the Whisper context for the active model.
    /// Errors with `ModelNotInstalled` when nothing is on disk yet.
    fn ensure_context(&self) -> crate::Result<Arc<WhisperContext>> {
        let active = self.resolve_active_id().ok_or_else(|| {
            crate::Error::ModelNotInstalled(
                "install a Whisper model from the voice settings first".into(),
            )
        })?;
        {
            let state = self.state.lock().unwrap();
            if state.loaded_model.as_deref() == Some(active.as_str()) {
                if let Some(ctx) = &state.context {
                    return Ok(ctx.clone());
                }
            }
        }

        let spec = spec_by_id(&active).ok_or_else(|| crate::Error::UnknownModel(active.clone()))?;
        let path = self.model_path(spec);
        let ctx = WhisperContext::new_with_params(
            path.to_string_lossy().as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| crate::Error::Recording(format!("load whisper model: {e}")))?;
        let ctx = Arc::new(ctx);

        let mut state = self.state.lock().unwrap();
        state.context = Some(ctx.clone());
        state.loaded_model = Some(active);
        Ok(ctx)
    }

    /// Builds (once) the cpal input stream that pushes mono i16 samples
    /// into `state.buffer` whenever `state.listening` is true. The
    /// stream is leaked on purpose so it survives across sessions —
    /// the first call is the slow one (microphone permission, device
    /// negotiation), every subsequent session reuses it.
    fn ensure_audio_stream(&self) -> crate::Result<()> {
        {
            let state = self.state.lock().unwrap();
            if state.stream_alive {
                return Ok(());
            }
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| crate::Error::Recording("no input device available".into()))?;
        let config = device
            .default_input_config()
            .map_err(|e| crate::Error::Recording(format!("input config: {e}")))?;

        let channels = config.channels() as usize;
        // cpal's `sample_rate()` returns the raw `u32` Hz value in this
        // version — no newtype wrapper to unpack.
        let device_rate = config.sample_rate() as f64;
        let sample_format = config.sample_format();
        // Cheap nearest-neighbour decimation. Whisper is robust enough
        // that high-quality resampling buys nothing here, especially at
        // the 16/44.1/48 kHz combinations we hit in practice.
        let stride = (device_rate / TARGET_SAMPLE_RATE as f64).max(1.0) as usize;

        let buffer = self.state.lock().unwrap().buffer.clone();
        let listening = self.state.lock().unwrap().listening.clone();

        let push = move |mono: Vec<i16>| {
            if !listening.load(Ordering::Relaxed) {
                return;
            }
            let mut buf = buffer.lock().unwrap();
            // Decimate-while-appending. Skipping inside the producer
            // avoids an extra alloc + iterator chain on the audio thread.
            for sample in mono.into_iter().step_by(stride) {
                buf.push(sample);
            }
        };

        let stream_config: cpal::StreamConfig = config.clone().into();
        let err_fn = |err| eprintln!("[tauri-plugin-stt] audio stream error: {err}");

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                let push = push.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _| {
                        let mono = downmix_f32(data, channels);
                        push(mono);
                    },
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let push = push.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _| {
                        let mono = downmix_i16(data, channels);
                        push(mono);
                    },
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let push = push.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _| {
                        let mono = downmix_u16(data, channels);
                        push(mono);
                    },
                    err_fn,
                    None,
                )
            }
            other => {
                return Err(crate::Error::Recording(format!(
                    "unsupported sample format: {other:?}"
                )));
            }
        }
        .map_err(|e| crate::Error::Recording(format!("build input stream: {e}")))?;

        stream
            .play()
            .map_err(|e| crate::Error::Recording(format!("play stream: {e}")))?;

        // Keep the stream alive without dragging it through the state
        // struct (cpal streams aren't `Send`). Leaking is the standard
        // pattern in cpal-based Tauri plugins.
        std::mem::forget(stream);

        let mut state = self.state.lock().unwrap();
        state.stream_alive = true;
        Ok(())
    }

    /// Runs Whisper over the captured buffer and emits the final
    /// transcript. Spawned on a worker thread because inference can
    /// take 100 ms – several seconds depending on model size and
    /// utterance length.
    fn transcribe_and_emit(&self, samples: Vec<i16>, language: Option<String>) {
        let app = self.app.clone();
        let state = self.state.clone();

        thread::spawn(move || {
            // Re-fetch the context inside the worker to keep the lock
            // duration short on the main path. If load fails we surface
            // it via the standard error event.
            let ctx = match state.lock().unwrap().context.clone() {
                Some(ctx) => ctx,
                None => {
                    let _ = app.emit(
                        "stt://error",
                        serde_json::json!({
                            "code": "NotAvailable",
                            "message": "Whisper context not initialised",
                        }),
                    );
                    return;
                }
            };

            // whisper-rs >= 0.14 takes a pre-allocated output slice so
            // callers can reuse buffers across calls. We allocate fresh
            // here because each utterance has a different length.
            let mut audio = vec![0.0_f32; samples.len()];
            if let Err(e) = convert_integer_to_float_audio(&samples, &mut audio) {
                let _ = app.emit(
                    "stt://error",
                    serde_json::json!({
                        "code": "Recording",
                        "message": format!("convert audio: {e}"),
                    }),
                );
                return;
            }
            // Whisper rejects clips shorter than ~100 ms with garbage
            // or silence — bail early instead of triggering a panic
            // inside whisper.cpp.
            if audio.len() < (TARGET_SAMPLE_RATE as usize / 10) {
                let _ = app.emit(
                    "stt://error",
                    serde_json::json!({
                        "code": "NoSpeech",
                        "message": "audio buffer too short to transcribe",
                    }),
                );
                return;
            }

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_translate(false);
            // Whisper expects a 2-letter ISO code (`en`, `pt`, …). Pass
            // `None`/auto-detect when we have no hint.
            let lang_short = language
                .as_deref()
                .and_then(|tag| tag.split(['-', '_']).next())
                .map(|s| s.to_string());
            if let Some(ref l) = lang_short {
                params.set_language(Some(l.as_str()));
            }
            // Single-threaded by default would leave most of the CPU
            // unused on multi-core machines. Use up to 4 threads —
            // beyond that whisper.cpp sees diminishing returns and we
            // want to leave headroom for the UI thread.
            let threads = num_cpus_capped(4);
            params.set_n_threads(threads);

            let mut whisper_state = match ctx.create_state() {
                Ok(s) => s,
                Err(e) => {
                    let _ = app.emit(
                        "stt://error",
                        serde_json::json!({
                            "code": "Recording",
                            "message": format!("create whisper state: {e}"),
                        }),
                    );
                    return;
                }
            };

            if let Err(e) = whisper_state.full(params, &audio) {
                let _ = app.emit(
                    "stt://error",
                    serde_json::json!({
                        "code": "RecognitionFailed",
                        "message": format!("whisper transcribe: {e}"),
                    }),
                );
                return;
            }

            let mut transcript = String::new();
            for segment in whisper_state.as_iter() {
                transcript.push_str(&segment.to_string());
            }
            let transcript = transcript.trim().to_string();

            let result = RecognitionResult {
                transcript,
                is_final: true,
                confidence: None,
            };
            let _ = app.emit("stt://result", &result);
            let _ = app.emit("plugin:stt:result", &result);
        });
    }

    pub fn start_listening(&self, config: ListenConfig) -> crate::Result<()> {
        // Load the active model up-front. Returns `ModelNotInstalled`
        // when nothing is on disk so the UI can route to the manager
        // instead of triggering an implicit multi-hundred-MB download.
        let _ = self.ensure_context()?;

        self.ensure_audio_stream()?;

        let language = config.language.clone();

        let mut state = self.state.lock().unwrap();
        if state.listening.load(Ordering::Relaxed) {
            return Err(crate::Error::Recording("already listening".into()));
        }
        state.buffer.lock().unwrap().clear();
        state.listening.store(true, Ordering::SeqCst);
        state.started_at = Some(Instant::now());
        state.language = language.clone();
        state.max_duration_ms = if config.max_duration > 0 {
            Some(config.max_duration as u64)
        } else {
            None
        };
        let max_ms = state.max_duration_ms;
        let listening_flag = state.listening.clone();
        drop(state);

        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Listening,
                is_available: true,
                language: language.clone(),
            },
        );

        // Auto-stop guard. Polls every 100 ms instead of `sleep(max_ms)`
        // so an early `stop_listening` short-circuits cleanly.
        if let Some(ms) = max_ms {
            let app = self.app.clone();
            let state = self.state.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_millis(ms);
                while Instant::now() < deadline {
                    if !listening_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                if listening_flag.load(Ordering::Relaxed) {
                    if let Ok(samples) = drain_and_stop(&state) {
                        let stt = Stt {
                            app: app.clone(),
                            state: state.clone(),
                        };
                        stt.transcribe_and_emit(samples, language.clone());
                        let _ = app.emit(
                            "plugin:stt:stateChange",
                            RecognitionStatus {
                                state: RecognitionState::Idle,
                                is_available: true,
                                language: language.clone(),
                            },
                        );
                    }
                }
            });
        }

        Ok(())
    }

    pub fn stop_listening(&self) -> crate::Result<()> {
        let samples = drain_and_stop(&self.state)?;
        let language = self.state.lock().unwrap().language.clone();
        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Processing,
                is_available: true,
                language: language.clone(),
            },
        );
        if !samples.is_empty() {
            self.transcribe_and_emit(samples, language.clone());
        }
        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Idle,
                is_available: true,
                language,
            },
        );
        Ok(())
    }

    pub fn is_available(&self) -> crate::Result<AvailabilityResponse> {
        // The engine is built in unconditionally when the `whisper`
        // feature is on. "Available" here means "ready to recognise" —
        // i.e. at least one model file is present on disk. The UI uses
        // this to gate the speaking exercise behind the install flow.
        let installed = self.resolve_active_id().is_some();
        Ok(AvailabilityResponse {
            available: installed,
            reason: if installed {
                None
            } else {
                Some("no Whisper model installed".into())
            },
        })
    }

    pub fn get_supported_languages(&self) -> crate::Result<SupportedLanguagesResponse> {
        let installed = self.resolve_active_id().is_some();
        let max = whisper_rs::get_lang_max_id();
        let mut languages = Vec::with_capacity((max + 1) as usize);
        for id in 0..=max {
            let (Some(code), Some(name)) = (
                whisper_rs::get_lang_str(id),
                whisper_rs::get_lang_str_full(id),
            ) else {
                continue;
            };
            languages.push(SupportedLanguage {
                code: code.to_string(),
                name: capitalise_first(name),
                installed: Some(installed),
            });
        }
        // Stable, locale-independent ordering so the UI can render the
        // same list on every machine without an extra sort step.
        languages.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(SupportedLanguagesResponse { languages })
    }

    pub fn check_permission(&self) -> crate::Result<PermissionResponse> {
        // Microphone access is mediated by the OS at stream creation
        // time; cpal triggers the prompt automatically. We optimistically
        // report "granted" and let any real denial surface as a stream
        // error on the next `start_listening`.
        Ok(PermissionResponse {
            microphone: PermissionStatus::Granted,
            speech_recognition: PermissionStatus::Granted,
        })
    }

    pub fn request_permission(&self) -> crate::Result<PermissionResponse> {
        self.check_permission()
    }
}

fn drain_and_stop(state: &Arc<Mutex<SttState>>) -> crate::Result<Vec<i16>> {
    let s = state.lock().unwrap();
    if !s.listening.load(Ordering::Relaxed) {
        return Ok(Vec::new());
    }
    s.listening.store(false, Ordering::SeqCst);
    let samples = std::mem::take(&mut *s.buffer.lock().unwrap());
    Ok(samples)
}

fn num_cpus_capped(cap: usize) -> std::os::raw::c_int {
    let avail = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    avail.min(cap).max(1) as std::os::raw::c_int
}

/// whisper.cpp returns language names in lowercase ("portuguese"). Do
/// the cheap title-case here once instead of forcing every JS consumer
/// to redo it on every render.
fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn downmix_f32(data: &[f32], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return data
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
    }
    data.chunks(channels)
        .map(|frame| {
            let avg = frame.iter().sum::<f32>() / channels as f32;
            (avg.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
        })
        .collect()
}

fn downmix_i16(data: &[i16], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

fn downmix_u16(data: &[u16], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return data.iter().map(|&s| (s as i32 - 32_768) as i16).collect();
    }
    data.chunks(channels)
        .map(|frame| {
            let avg = frame.iter().map(|&s| s as i32).sum::<i32>() / channels as i32;
            (avg - 32_768) as i16
        })
        .collect()
}
