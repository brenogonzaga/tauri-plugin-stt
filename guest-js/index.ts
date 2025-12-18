import {
  invoke,
  PluginListener,
  addPluginListener,
} from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

export interface ListenConfig {
  /** Language code for recognition (e.g., "en-US", "pt-BR") */
  language?: string;
  /** Whether to return interim (partial) results */
  interimResults?: boolean;
  /** Whether to continue listening after getting a result */
  continuous?: boolean;
  /** Maximum duration to listen in milliseconds (0 = no limit) */
  maxDuration?: number;
  /** Use on-device recognition only (iOS 13+, no network required)
   * When true, recognition works offline but may be less accurate.
   * Falls back to server if on-device not available for the language.
   */
  onDevice?: boolean;
}

export type RecognitionState = "idle" | "listening" | "processing";

export interface RecognitionResult {
  transcript: string;
  isFinal: boolean;
  confidence?: number;
}

/**
 * Unified error codes for cross-platform consistency
 */
export type SttErrorCode =
  | "NONE"
  | "NOT_AVAILABLE"
  | "PERMISSION_DENIED"
  | "SPEECH_PERMISSION_DENIED"
  | "NETWORK_ERROR"
  | "AUDIO_ERROR"
  | "TIMEOUT"
  | "NO_SPEECH"
  | "LANGUAGE_NOT_SUPPORTED"
  | "CANCELLED"
  | "ALREADY_LISTENING"
  | "NOT_LISTENING"
  | "BUSY"
  | "UNKNOWN";

/**
 * Structured error event with code and message
 */
export interface SttError {
  /** Error code for programmatic handling */
  code: SttErrorCode;
  /** Human-readable error message */
  message: string;
  /** Platform-specific error details */
  details?: string;
}

/**
 * @deprecated Use SttError instead
 */
export interface RecognitionError {
  error: string;
  code?: string;
}

export interface StateChangeEvent {
  state: RecognitionState;
}

export interface SupportedLanguage {
  code: string;
  name: string;
  installed?: boolean;
}

export interface AvailabilityResponse {
  available: boolean;
  reason?: string;
}

export interface SupportedLanguagesResponse {
  languages: SupportedLanguage[];
}

export type PermissionStatus = "granted" | "denied" | "unknown";

export interface PermissionResponse {
  microphone: PermissionStatus;
  speechRecognition: PermissionStatus;
}

export async function startListening(config?: ListenConfig): Promise<void> {
  await invoke("plugin:stt|start_listening", { config: config || {} });
}

export async function stopListening(): Promise<void> {
  await invoke("plugin:stt|stop_listening");
}

export async function isAvailable(): Promise<AvailabilityResponse> {
  return await invoke("plugin:stt|is_available");
}

export async function getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
  return await invoke("plugin:stt|get_supported_languages");
}

export async function checkPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|check_permission");
}

export async function requestPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|request_permission");
}

/**
 * Listen for speech recognition results.
 * Uses channel-based communication on mobile, event system on desktop.
 */
export async function onResult(
  handler: (result: RecognitionResult) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<RecognitionResult>("stt", "result", handler);
  }

  const unlisten = await listen<RecognitionResult>(
    "plugin:stt:result",
    event => {
      handler(event.payload);
    }
  );
  return unlisten;
}

/**
 * Listen for state changes in the speech recognizer.
 */
export async function onStateChange(
  handler: (event: StateChangeEvent) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<StateChangeEvent>(
      "stt",
      "stateChange",
      handler
    );
  }

  const unlisten = await listen<StateChangeEvent>(
    "plugin:stt:stateChange",
    event => {
      handler(event.payload);
    }
  );
  return unlisten;
}

/**
 * Listen for speech recognition errors.
 */
export async function onError(
  handler: (error: SttError) => void
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<SttError>("stt", "error", handler);
  }

  return await listen<SttError>("plugin:stt:error", event => {
    handler(event.payload);
  });
}

/**
 * Detect if we're running on mobile platform
 * Mobile uses channel-based listeners, desktop uses event system
 */
function isMobilePlatform(): boolean {
  // Check for mobile-specific Tauri internals and Android WebView
  const w = window as any;

  // Check Tauri's internal platform detection first
  const platform = w.__TAURI_INTERNALS__?.plugins?.os?.platform;
  if (platform === "android" || platform === "ios") {
    return true;
  }

  // Check for Android WebView
  if (w.Android) {
    return true;
  }

  // Check for iOS-specific WebKit (but not macOS)
  // iOS has webkit.messageHandlers AND navigator.userAgent contains "iPhone" or "iPad"
  if (w.webkit?.messageHandlers) {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("iphone") || ua.includes("ipad") || ua.includes("ipod")) {
      return true;
    }
  }

  // Default to desktop
  return false;
}
