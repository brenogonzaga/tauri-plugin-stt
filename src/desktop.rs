use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::{self, Cursor};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{plugin::PluginApi, AppHandle, Emitter, Manager, Runtime};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use vosk::{Model, Recognizer};

use crate::models::*;

/// Default Vosk model configuration
const DEFAULT_MODEL_NAME: &str = "vosk-model-en-us-0.42-gigaspeech";
const DEFAULT_MODEL_URL: &str =
    "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip";

/// Available Vosk models with their download URLs
/// Using high-accuracy models for better transcription quality
const AVAILABLE_MODELS: &[(&str, &str, &str)] = &[
    (
        "en-US",
        "vosk-model-en-us-0.42-gigaspeech",
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
    ),
    (
        "pt-BR",
        "vosk-model-pt-fb-v0.1.1-20220516_2113",
        "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip",
    ),
    (
        "es-ES",
        "vosk-model-es-0.42",
        "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
    ),
    (
        "fr-FR",
        "vosk-model-fr-0.22",
        "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip",
    ),
    (
        "de-DE",
        "vosk-model-de-0.21",
        "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
    ),
    (
        "ru-RU",
        "vosk-model-ru-0.42",
        "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
    ),
    (
        "zh-CN",
        "vosk-model-cn-0.22",
        "https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip",
    ),
    (
        "ja-JP",
        "vosk-model-ja-0.22",
        "https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip",
    ),
    (
        "it-IT",
        "vosk-model-it-0.22",
        "https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip",
    ),
];

use std::sync::atomic::{AtomicBool, Ordering};

/// Stop signal for the audio stream - when set to true, the audio callback will stop
static STOP_SIGNAL: AtomicBool = AtomicBool::new(false);

struct SttState {
    model: Option<Arc<Model>>,
    current_model_name: Option<String>,
    is_listening: bool,
    listen_start_time: Option<Instant>,
    max_duration_ms: Option<u64>,
}

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
) -> crate::Result<Stt<R>> {
    let state = Arc::new(Mutex::new(SttState {
        model: None,
        current_model_name: None,
        is_listening: false,
        listen_start_time: None,
        max_duration_ms: None,
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
    fn get_models_dir(&self) -> PathBuf {
        self.app
            .path()
            .app_data_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("vosk-models")
    }

    fn get_model_info_for_language(&self, language: &str) -> Option<(&'static str, &'static str)> {
        // First try exact match
        if let Some((_, name, url)) = AVAILABLE_MODELS
            .iter()
            .find(|(lang, _, _)| *lang == language)
        {
            return Some((*name, *url));
        }

        // If not found, try to match by language prefix (e.g., "pt" matches "pt-BR")
        if let Some(prefix) = language.split('-').next() {
            if let Some((_, name, url)) = AVAILABLE_MODELS
                .iter()
                .find(|(lang, _, _)| lang.split('-').next() == Some(prefix))
            {
                return Some((*name, *url));
            }
        }

        None
    }

    /// Download and extract a Vosk model in a separate thread to avoid tokio conflicts
    fn download_model(&self, model_name: &str, url: &str) -> crate::Result<PathBuf> {
        let models_dir = self.get_models_dir();
        fs::create_dir_all(&models_dir).map_err(|e| {
            crate::Error::Recording(format!("Failed to create models directory: {}", e))
        })?;

        let model_path = models_dir.join(model_name);

        // If already exists, return path
        if model_path.exists() {
            return Ok(model_path);
        }

        println!("Downloading model '{}' from {}", model_name, url);

        // Emit download start event
        let _ = self.app.emit(
            "stt://download-progress",
            serde_json::json!({
                "status": "downloading",
                "model": model_name,
                "progress": 0
            }),
        );

        // Download in a separate thread to avoid tokio runtime conflicts
        let url_owned = url.to_string();
        let model_name_owned = model_name.to_string();
        let app_handle = self.app.clone();

        let handle = std::thread::spawn(move || -> Result<Vec<u8>, String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(3000)) // Timeout total de 3000s
                .build()
                .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

            let response = client
                .get(&url_owned)
                .send()
                .map_err(|e| format!("Failed to download model from {}: {}", url_owned, e))?;

            let status = response.status();

            if !status.is_success() {
                return Err(format!(
                    "Failed to download model: HTTP {} - {}",
                    status,
                    response
                        .text()
                        .unwrap_or_else(|_| "Failed to get error details".to_string())
                ));
            }

            // Get content length if available
            let total_size = response.content_length();

            // Read bytes in chunks with progress tracking
            use std::io::Read;
            let mut reader = response;
            let mut buffer = Vec::new();
            let mut downloaded: usize = 0;
            let chunk_size = 64 * 1024; // 64KB chunks for better performance
            let mut chunk = vec![0u8; chunk_size];
            let mut last_progress_mb = 0;

            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        buffer.extend_from_slice(&chunk[..n]);
                        downloaded += n;

                        // Show progress every 5MB
                        let current_mb = downloaded / (5 * 1024 * 1024);
                        if current_mb > last_progress_mb {
                            last_progress_mb = current_mb;

                            if let Some(total) = total_size {
                                let progress = ((downloaded as f64 / total as f64) * 50.0) as u8;
                                print!(
                                    "\rProgress: {:.2} / {:.2} MB   ",
                                    downloaded as f64 / 1_048_576.0,
                                    total as f64 / 1_048_576.0
                                );
                                std::io::Write::flush(&mut std::io::stdout()).ok();

                                let _ = app_handle.emit(
                                    "stt://download-progress",
                                    serde_json::json!({
                                        "status": "downloading",
                                        "model": model_name_owned,
                                        "progress": progress
                                    }),
                                );
                            } else {
                                print!("\rProgress: {:.2} MB   ", downloaded as f64 / 1_048_576.0);
                                std::io::Write::flush(&mut std::io::stdout()).ok();
                            }
                        }
                    }
                    Err(e) => {
                        println!(); // New line after progress
                        return Err(format!("Failed to read chunk: {}", e));
                    }
                }
            }

            println!(); // New line after progress bar
            println!(
                "Download complete: {:.2} MB",
                downloaded as f64 / 1_048_576.0
            );

            // Emit extraction event
            let _ = app_handle.emit(
                "stt://download-progress",
                serde_json::json!({
                    "status": "extracting",
                    "model": model_name_owned,
                    "progress": 50
                }),
            );

            Ok(buffer)
        });

        // Wait for download to complete
        let bytes = handle
            .join()
            .map_err(|_| crate::Error::Recording("Download thread panicked".to_string()))?
            .map_err(crate::Error::Recording)?;

        println!("Extracting model...");

        // Extract the zip (this is fast enough to do on main thread)
        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| crate::Error::Recording(format!("Failed to open zip: {}", e)))?;

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| crate::Error::Recording(format!("Failed to read zip entry: {}", e)))?;

            let outpath = match file.enclosed_name() {
                Some(path) => models_dir.join(path),
                None => continue,
            };

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath).ok();
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p).ok();
                    }
                }
                let mut outfile = File::create(&outpath).map_err(|e| {
                    crate::Error::Recording(format!("Failed to create file: {}", e))
                })?;
                io::copy(&mut file, &mut outfile).map_err(|e| {
                    crate::Error::Recording(format!("Failed to extract file: {}", e))
                })?;
            }
        }

        // Emit completion event
        let _ = self.app.emit(
            "stt://download-progress",
            serde_json::json!({
                "status": "complete",
                "model": model_name,
                "progress": 100
            }),
        );

        Ok(model_path)
    }

    fn ensure_model(&self, language: Option<&str>) -> crate::Result<Arc<Model>> {
        let (model_name, model_url) = if let Some(lang) = language {
            match self.get_model_info_for_language(lang) {
                Some((name, url)) => (name, url),
                None => (DEFAULT_MODEL_NAME, DEFAULT_MODEL_URL),
            }
        } else {
            (DEFAULT_MODEL_NAME, DEFAULT_MODEL_URL)
        };

        let mut state = self.state.lock().unwrap();

        // Check if we already have this model loaded
        if let Some(current) = &state.current_model_name {
            if current == model_name {
                if let Some(model) = &state.model {
                    return Ok(model.clone());
                }
            }
        }

        // Drop existing model if switching
        state.model = None;
        state.current_model_name = None;

        drop(state);

        // Download model if needed
        let model_path = self.download_model(model_name, model_url)?;

        if !model_path.exists() {
            return Err(crate::Error::NotAvailable(format!(
                "Vosk model not found at {:?}",
                model_path
            )));
        }

        let model = Model::new(model_path.to_str().unwrap())
            .ok_or_else(|| crate::Error::Recording("Failed to load Vosk model".to_string()))?;

        let model = Arc::new(model);

        let mut state = self.state.lock().unwrap();
        state.model = Some(model.clone());
        state.current_model_name = Some(model_name.to_string());

        Ok(model)
    }

    pub fn start_listening(&self, config: ListenConfig) -> crate::Result<()> {
        let model = self.ensure_model(config.language.as_deref())?;

        let mut state = self.state.lock().unwrap();

        if state.is_listening {
            return Err(crate::Error::Recording("Already listening".to_string()));
        }

        // Store maxDuration config (in milliseconds)
        state.listen_start_time = Some(Instant::now());
        state.max_duration_ms = if config.max_duration > 0 {
            Some(config.max_duration as u64)
        } else {
            None
        };

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| crate::Error::Recording("No input device available".to_string()))?;

        let stream_config = device
            .default_input_config()
            .map_err(|e| crate::Error::Recording(format!("Failed to get input config: {}", e)))?;

        let channels = stream_config.channels() as usize;
        let sample_format = stream_config.sample_format();
        let device_sample_rate = stream_config.sample_rate().0 as f32;

        // Vosk expects 16kHz
        let target_sample_rate = 16000.0;
        let mut recognizer = Recognizer::new(&model, target_sample_rate)
            .ok_or_else(|| crate::Error::Recording("Failed to create recognizer".to_string()))?;

        recognizer.set_max_alternatives(config.max_alternatives.unwrap_or(1) as u16);
        recognizer.set_partial_words(config.interim_results);

        let app_handle = self.app.clone();
        let recognizer = Arc::new(Mutex::new(recognizer));
        let recognizer_clone = recognizer.clone();
        let interim_results = config.interim_results;
        let last_partial = Arc::new(Mutex::new(String::new()));

        // Accumulator buffer to collect audio samples
        let audio_buffer = Arc::new(Mutex::new(Vec::new()));

        // Simple resampling: skip samples if device rate > 16kHz
        let resample_step = (device_sample_rate / target_sample_rate) as usize;
        let resample_step = resample_step.max(1);

        // Reset stop signal before starting
        STOP_SIGNAL.store(false, Ordering::SeqCst);

        let process_audio = move |samples_i16: Vec<i16>| {
            // Check stop signal - if set, skip processing
            if STOP_SIGNAL.load(Ordering::SeqCst) {
                return;
            }

            // Accumulate samples in buffer
            let mut buffer = audio_buffer.lock().unwrap();
            buffer.extend_from_slice(&samples_i16);

            // Process when we have at least 0.1 seconds of audio after resampling
            // At 16kHz, 0.1s = 1600 samples, but we want at least that much
            let required_samples = (1600 * resample_step).max(3200);

            if buffer.len() < required_samples {
                return;
            }

            // Take all accumulated samples
            let samples_to_process: Vec<i16> = buffer.drain(..).collect();
            drop(buffer); // Release lock

            // Resample if needed
            let resampled: Vec<i16> = if resample_step > 1 {
                samples_to_process
                    .iter()
                    .step_by(resample_step)
                    .copied()
                    .collect()
            } else {
                samples_to_process
            };

            let mut rec = recognizer_clone.lock().unwrap();

            // Accept waveform returns Result<DecodingState, _>
            let result = rec.accept_waveform(&resampled);
            let is_final = matches!(result, Ok(vosk::DecodingState::Finalized));

            if is_final {
                let result = rec.result();
                let text = match result {
                    vosk::CompleteResult::Single(single) => single.text,
                    vosk::CompleteResult::Multiple(multiple) => multiple
                        .alternatives
                        .first()
                        .map(|alt| &alt.text)
                        .unwrap_or(&""),
                };

                if !text.is_empty() {
                    *last_partial.lock().unwrap() = String::new();

                    // Emit both event names for compatibility
                    let result = RecognitionResult {
                        transcript: text.to_string(),
                        is_final: true,
                        confidence: Some(1.0),
                    };
                    let _ = app_handle.emit("stt://result", &result);
                    let _ = app_handle.emit("plugin:stt:result", &result);
                }
            } else if interim_results {
                let partial = rec.partial_result();
                if !partial.partial.is_empty() {
                    let mut last = last_partial.lock().unwrap();
                    if *last != partial.partial {
                        *last = partial.partial.to_string();

                        // Emit both event names for compatibility
                        let result = RecognitionResult {
                            transcript: partial.partial.to_string(),
                            is_final: false,
                            confidence: None,
                        };
                        let _ = app_handle.emit("stt://result", &result);
                        let _ = app_handle.emit("plugin:stt:result", &result);
                    }
                }
            }
        };

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono_i16: Vec<i16> = if channels == 1 {
                        data.iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                            .collect()
                    } else {
                        data.chunks(channels)
                            .map(|frame| {
                                let avg = frame.iter().sum::<f32>() / channels as f32;
                                (avg.clamp(-1.0, 1.0) * 32767.0) as i16
                            })
                            .collect()
                    };
                    process_audio(mono_i16);
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mono_i16: Vec<i16> = if channels == 1 {
                        data.to_vec()
                    } else {
                        data.chunks(channels)
                            .map(|frame| {
                                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                                (sum / channels as i32) as i16
                            })
                            .collect()
                    };
                    process_audio(mono_i16);
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &stream_config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mono_i16: Vec<i16> = if channels == 1 {
                        data.iter().map(|&s| (s as i32 - 32768) as i16).collect()
                    } else {
                        data.chunks(channels)
                            .map(|frame| {
                                let avg =
                                    frame.iter().map(|&s| s as i32).sum::<i32>() / channels as i32;
                                (avg - 32768) as i16
                            })
                            .collect()
                    };
                    process_audio(mono_i16);
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                },
                None,
            ),
            _ => {
                return Err(crate::Error::Recording(format!(
                    "Unsupported sample format: {:?}",
                    sample_format
                )));
            }
        }
        .map_err(|e| crate::Error::Recording(format!("Failed to build stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| crate::Error::Recording(format!("Failed to start stream: {}", e)))?;

        // Reset stop signal
        STOP_SIGNAL.store(false, Ordering::SeqCst);

        state.is_listening = true;

        // Emit stateChange event with RecognitionStatus
        let _ = self.app.emit(
            "plugin:stt:stateChange",
            RecognitionStatus {
                state: RecognitionState::Listening,
                is_available: true,
                language: config.language.clone(),
            },
        );

        // Keep the stream alive using mem::forget
        // The stream callback checks STOP_SIGNAL and stops processing when it's set
        // This is intentional - the stream stays alive until app exit, but stops
        // processing audio when stop_listening is called
        // Note: This is a tradeoff - we can't properly drop the stream because
        // cpal::Stream is not Send+Sync, but the callback will stop processing
        std::mem::forget(stream);

        // Start maxDuration timer thread if configured (config.max_duration is in milliseconds)
        if config.max_duration > 0 {
            let max_ms = config.max_duration as u64;
            let app_handle_timer = self.app.clone();
            let state_clone = self.state.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(max_ms));

                // Check if still listening before triggering timeout
                let mut state = state_clone.lock().unwrap();
                if state.is_listening {
                    // Stop listening
                    STOP_SIGNAL.store(true, Ordering::SeqCst);
                    state.is_listening = false;
                    state.listen_start_time = None;
                    state.max_duration_ms = None;

                    // Emit events
                    let _ = app_handle_timer
                        .emit("stt://stateChange", serde_json::json!({ "state": "idle" }));
                    let _ = app_handle_timer.emit(
                        "stt://error",
                        serde_json::json!({
                            "error": "Maximum duration reached",
                            "code": -2
                        }),
                    );
                }
            });
        }

        Ok(())
    }

    pub fn stop_listening(&self) -> crate::Result<()> {
        let mut state = self.state.lock().unwrap();

        if !state.is_listening {
            return Ok(());
        }

        // Signal the stream thread to stop - this will cause the thread to exit
        // and drop the stream, which stops the audio capture
        STOP_SIGNAL.store(true, Ordering::SeqCst);

        state.is_listening = false;
        state.listen_start_time = None;
        state.max_duration_ms = None;

        Ok(())
    }

    pub fn is_available(&self) -> crate::Result<AvailabilityResponse> {
        Ok(AvailabilityResponse {
            available: true,
            reason: None,
        })
    }

    pub fn get_supported_languages(&self) -> crate::Result<SupportedLanguagesResponse> {
        let models_dir = self.get_models_dir();

        let languages: Vec<SupportedLanguage> = AVAILABLE_MODELS
            .iter()
            .map(|(code, model_name, _)| {
                let installed = models_dir.join(model_name).exists();
                SupportedLanguage {
                    code: code.to_string(),
                    name: get_language_display_name(code),
                    installed: Some(installed),
                }
            })
            .collect();

        Ok(SupportedLanguagesResponse { languages })
    }

    pub fn check_permission(&self) -> crate::Result<PermissionResponse> {
        Ok(PermissionResponse {
            microphone: PermissionStatus::Granted,
            speech_recognition: PermissionStatus::Granted,
        })
    }

    pub fn request_permission(&self) -> crate::Result<PermissionResponse> {
        Ok(PermissionResponse {
            microphone: PermissionStatus::Granted,
            speech_recognition: PermissionStatus::Granted,
        })
    }
}

fn get_language_display_name(code: &str) -> String {
    match code {
        "en-US" => "English (United States)".to_string(),
        "pt-BR" => "Portuguese (Brazil)".to_string(),
        "es-ES" => "Spanish (Spain)".to_string(),
        "fr-FR" => "French (France)".to_string(),
        "de-DE" => "German (Germany)".to_string(),
        "ru-RU" => "Russian (Russia)".to_string(),
        "zh-CN" => "Chinese (Simplified)".to_string(),
        "ja-JP" => "Japanese (Japan)".to_string(),
        "it-IT" => "Italian (Italy)".to_string(),
        _ => code.to_string(),
    }
}
