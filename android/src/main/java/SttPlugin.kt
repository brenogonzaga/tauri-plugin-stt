package io.affex.stt

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.Permission
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.util.Locale

@InvokeArg
class ListenConfig {
    var language: String? = null
    var interimResults: Boolean = false
    var continuous: Boolean = false
    var maxDuration: Int = 0
}

@TauriPlugin(
    permissions = [
        Permission(strings = [Manifest.permission.RECORD_AUDIO], alias = "microphone")
    ]
)
class SttPlugin(private val activity: Activity) : Plugin(activity) {

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var currentLanguage: String? = null
    
    // Pending permission request invoke to resolve after user responds
    private var pendingPermissionInvoke: Invoke? = null
    
    // maxDuration handling
    private var maxDurationHandler: Handler? = null
    private var maxDurationRunnable: Runnable? = null
    
    // Continuous mode restart handling
    private var consecutiveErrors = 0
    private var lastErrorTime = 0L
    private val restartHandler = Handler(Looper.getMainLooper())

    companion object {
        private const val TAG = "SttPlugin"
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val MAX_CONSECUTIVE_ERRORS = 3
        private const val RESTART_DELAY_MS = 500L
        private const val ERROR_RESET_TIME_MS = 30000L // 30 seconds
    }

    init {
        Log.d(TAG, "============================================")
        Log.d(TAG, "SttPlugin INIT")
        Log.d(TAG, "  Package: ${activity.packageName}")
        Log.d(TAG, "  Android SDK: ${Build.VERSION.SDK_INT}")
        Log.d(TAG, "  Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        
        activity.runOnUiThread {
            val available = SpeechRecognizer.isRecognitionAvailable(activity)
            Log.d(TAG, "  SpeechRecognizer available: $available")
            
            if (available) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(activity)
                Log.d(TAG, "  SpeechRecognizer created successfully")
            } else {
                Log.w(TAG, "  Speech recognition NOT available on device")
                Log.w(TAG, "  Hint: Install Google app from Play Store")
            }
        }
        Log.d(TAG, "============================================")
    }

    fun cleanup() {
        Log.d(TAG, "cleanup() CALLED")
        Log.d(TAG, "  isListening: $isListening")
        Log.d(TAG, "  consecutiveErrors: $consecutiveErrors")
        
        // Cancel max duration timer
        maxDurationRunnable?.let { maxDurationHandler?.removeCallbacks(it) }
        maxDurationHandler = null
        maxDurationRunnable = null
        Log.d(TAG, "  Max duration timer cancelled")
        
        // Cancel restart handler
        restartHandler.removeCallbacksAndMessages(null)
        Log.d(TAG, "  Restart handler cancelled")
        
        // Reset error tracking
        consecutiveErrors = 0
        lastErrorTime = 0L
        
        activity.runOnUiThread {
            speechRecognizer?.destroy()
            speechRecognizer = null
            Log.d(TAG, "  SpeechRecognizer destroyed")
        }
    }
    
    /**
     * Called when the plugin's activity is destroyed.
     * Ensures proper cleanup of speech recognizer resources.
     */
    override fun onDestroy() {
        Log.d(TAG, "onDestroy() CALLED")
        super.onDestroy()
        cleanup()
    }
    
    /**
     * Handle permission request results.
     * Uses Tauri's PermissionState system instead of deprecated onRequestPermissionsResult.
     */
    @Deprecated("Use PermissionState from Tauri instead")
    private fun handlePermissionResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            
            pendingPermissionInvoke?.let { invoke ->
                val result = JSObject()
                if (granted) {
                    result.put("microphone", "granted")
                    result.put("speechRecognition", "granted")
                } else {
                    val canRequest = ActivityCompat.shouldShowRequestPermissionRationale(
                        activity, Manifest.permission.RECORD_AUDIO
                    )
                    result.put("microphone", if (canRequest) "denied" else "permanently_denied")
                    result.put("speechRecognition", if (canRequest) "denied" else "permanently_denied")
                }
                invoke.resolve(result)
                pendingPermissionInvoke = null
            }
        }
    }

    @Command
    fun startListening(invoke: Invoke) {
        Log.d(TAG, "startListening called")
        val config = invoke.parseArgs(ListenConfig::class.java)
        Log.d(TAG, "Config parsed: language=${config.language}, interimResults=${config.interimResults}, continuous=${config.continuous}")

        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "Microphone permission not granted, requesting...")
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
            invoke.reject("Microphone permission required. Please grant permission and try again.")
            return
        }
        Log.d(TAG, "Microphone permission granted")

        val recognitionAvailable = SpeechRecognizer.isRecognitionAvailable(activity)
        Log.d(TAG, "Speech recognition available: $recognitionAvailable")
        if (!recognitionAvailable) {
            invoke.reject("Speech recognition not available on this device. Please install Google app.")
            return
        }

        if (isListening) {
            Log.w(TAG, "Already listening, rejecting")
            invoke.reject("Already listening")
            return
        }

        Log.d(TAG, "Preparing to start on UI thread...")
        activity.runOnUiThread {
            try {
                Log.d(TAG, "On UI thread, preparing SpeechRecognizer...")
                // Recreate speech recognizer if null
                if (speechRecognizer == null) {
                    Log.d(TAG, "SpeechRecognizer is null, creating new one...")
                    if (SpeechRecognizer.isRecognitionAvailable(activity)) {
                        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(activity)
                        Log.d(TAG, "SpeechRecognizer created successfully")
                    } else {
                        Log.e(TAG, "Speech recognition not available on UI thread check")
                        invoke.reject("Speech recognition not available")
                        return@runOnUiThread
                    }
                } else {
                    Log.d(TAG, "SpeechRecognizer already exists")
                }
                
                val intent = createRecognizerIntent(config)
                currentLanguage = config.language ?: Locale.getDefault().toLanguageTag()
                Log.d(TAG, "Starting speech recognition:")
                Log.d(TAG, "  - Package: ${activity.packageName}")
                Log.d(TAG, "  - Language: $currentLanguage")
                Log.d(TAG, "  - Interim results: ${config.interimResults}")
                Log.d(TAG, "  - Continuous: ${config.continuous}")
                Log.d(TAG, "  - Max duration: ${config.maxDuration}ms")

                speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        Log.d(TAG, "Ready for speech")
                        isListening = true
                        val event = JSObject()
                        event.put("state", "listening")
                        trigger("stateChange", event)
                    }

                    override fun onBeginningOfSpeech() {
                        Log.d(TAG, "Beginning of speech detected")
                        val event = JSObject()
                        event.put("state", "processing")
                        trigger("stateChange", event)
                    }

                    override fun onRmsChanged(rmsdB: Float) {}

                    override fun onBufferReceived(buffer: ByteArray?) {}

                    override fun onEndOfSpeech() {
                        Log.d(TAG, "End of speech detected")
                        val event = JSObject()
                        event.put("state", "processing")
                        trigger("stateChange", event)
                    }

                    override fun onError(error: Int) {
                        val errorMessage = getErrorMessage(error)
                        val currentTime = System.currentTimeMillis()
                        
                        // Reset consecutive errors if enough time passed
                        if (currentTime - lastErrorTime > ERROR_RESET_TIME_MS) {
                            Log.d(TAG, "Resetting error counter - enough time passed (${(currentTime - lastErrorTime) / 1000}s)")
                            consecutiveErrors = 0
                        }
                        lastErrorTime = currentTime
                        consecutiveErrors++
                        
                        Log.e(TAG, "Speech recognition error: $errorMessage (code: $error) [consecutive: $consecutiveErrors]")
                        
                        // Log additional details for ERROR_NO_MATCH
                        if (error == SpeechRecognizer.ERROR_NO_MATCH) {
                            Log.w(TAG, "ERROR_NO_MATCH details:")
                            Log.w(TAG, "  - Language: $currentLanguage")
                            Log.w(TAG, "  - Interim results enabled: ${config.interimResults}")
                            Log.w(TAG, "  - This means speech was detected but not recognized")
                            Log.w(TAG, "  - Possible causes: wrong language, unclear audio, background noise")
                        }
                        
                        // Log additional details for ERROR_CLIENT
                        if (error == SpeechRecognizer.ERROR_CLIENT) {
                            Log.e(TAG, "ERROR_CLIENT details:")
                            Log.e(TAG, "  - Package: ${activity.packageName}")
                            Log.e(TAG, "  - Language: $currentLanguage")
                            Log.e(TAG, "  - Recognition available: ${SpeechRecognizer.isRecognitionAvailable(activity)}")
                            Log.e(TAG, "  - Hint: This usually means audio input is not working. Try a physical device.")
                        }
                        
                        // Don't restart on ERROR_CLIENT or if too many consecutive errors
                        val shouldRestart = config.continuous && 
                            (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) &&
                            consecutiveErrors < MAX_CONSECUTIVE_ERRORS
                        
                        if (shouldRestart) {
                            Log.d(TAG, "Restarting recognition in ${RESTART_DELAY_MS}ms (attempt $consecutiveErrors/$MAX_CONSECUTIVE_ERRORS)")
                            // Add delay before restart to avoid overwhelming the service
                            restartHandler.postDelayed({
                                if (isListening && speechRecognizer != null) {
                                    Log.d(TAG, "Executing delayed restart")
                                    speechRecognizer?.startListening(intent)
                                }
                            }, RESTART_DELAY_MS)
                            return
                        }
                        
                        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                            Log.e(TAG, "Too many consecutive errors ($consecutiveErrors). Stopping continuous mode.")
                        }
                        
                        isListening = false
                        consecutiveErrors = 0
                        
                        val event = JSObject()
                        event.put("code", getErrorCode(error))
                        event.put("message", errorMessage)
                        event.put("details", "Android error code: $error")
                        trigger("error", event)

                        val stateEvent = JSObject()
                        stateEvent.put("state", "idle")
                        trigger("stateChange", stateEvent)
                    }

                    override fun onResults(results: Bundle?) {
                        // Save continuous state before setting isListening to false
                        val shouldContinue = config.continuous && isListening
                        isListening = false
                        
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val confidences = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)

                        Log.d(TAG, "onResults called - matches: ${matches?.size ?: 0}, shouldContinue: $shouldContinue")
                        if (!matches.isNullOrEmpty()) {
                            Log.d(TAG, "Final result: ${matches[0]}")
                            // Reset error counter on successful result
                            consecutiveErrors = 0
                            
                            val event = JSObject()
                            event.put("transcript", matches[0])
                            event.put("isFinal", true)
                            if (confidences != null && confidences.isNotEmpty()) {
                                event.put("confidence", confidences[0].toDouble())
                            }
                            trigger("result", event)
                        }

                        // Restart listening if in continuous mode
                        if (shouldContinue) {
                            Log.d(TAG, "Restarting listener for continuous mode (with delay)")
                            isListening = true
                            // Add small delay before restart
                            restartHandler.postDelayed({
                                if (isListening && speechRecognizer != null) {
                                    speechRecognizer?.startListening(intent)
                                }
                            }, RESTART_DELAY_MS)
                        } else {
                            val stateEvent = JSObject()
                            stateEvent.put("state", "idle")
                            trigger("stateChange", stateEvent)
                        }
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        if (config.interimResults) {
                            val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            Log.d(TAG, "onPartialResults called - matches: ${matches?.size ?: 0}")
                            if (!matches.isNullOrEmpty()) {
                                Log.d(TAG, "Partial result: ${matches[0]}")
                                val event = JSObject()
                                event.put("transcript", matches[0])
                                event.put("isFinal", false)
                                trigger("result", event)
                            }
                        }
                    }

                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })

                Log.d(TAG, "Starting listener with intent")
                speechRecognizer?.startListening(intent)
                Log.d(TAG, "Listener started")
                
                // Setup maxDuration timer if specified
                if (config.maxDuration > 0) {
                    Log.d(TAG, "Setting up maxDuration timer: ${config.maxDuration}ms")
                    maxDurationHandler = Handler(Looper.getMainLooper())
                    maxDurationRunnable = Runnable {
                        Log.d(TAG, "maxDuration reached, stopping recognition")
                        speechRecognizer?.stopListening()
                        speechRecognizer?.cancel()
                        isListening = false
                        
                        val event = JSObject()
                        event.put("state", "idle")
                        trigger("stateChange", event)
                        
                        val errorEvent = JSObject()
                        errorEvent.put("code", "TIMEOUT")
                        errorEvent.put("message", "Max duration reached")
                        errorEvent.put("details", "Recognition stopped after maxDuration limit")
                        trigger("error", errorEvent)
                    }
                    maxDurationHandler?.postDelayed(maxDurationRunnable!!, config.maxDuration.toLong())
                }
                
                invoke.resolve()

            } catch (e: Exception) {
                invoke.reject("Failed to start listening: ${e.message}")
            }
        }
    }

    @Command
    fun stopListening(invoke: Invoke) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "stopListening() CALLED")
        Log.d(TAG, "  isListening: $isListening")
        Log.d(TAG, "  consecutiveErrors: $consecutiveErrors")
        
        // Cancel max duration timer
        maxDurationRunnable?.let { maxDurationHandler?.removeCallbacks(it) }
        maxDurationHandler = null
        maxDurationRunnable = null
        Log.d(TAG, "  Max duration timer cancelled")
        
        // Cancel any pending restarts
        restartHandler.removeCallbacksAndMessages(null)
        consecutiveErrors = 0
        Log.d(TAG, "  Restart handler cancelled, errors reset")
        
        activity.runOnUiThread {
            try {
                Log.d(TAG, "  Stopping on UI thread...")
                speechRecognizer?.stopListening()
                speechRecognizer?.cancel()
                isListening = false
                Log.d(TAG, "  SpeechRecognizer stopped and cancelled")

                val event = JSObject()
                event.put("state", "idle")
                trigger("stateChange", event)
                
                invoke.resolve()
            } catch (e: Exception) {
                invoke.reject("Failed to stop listening: ${e.message}")
            }
        }
    }

    @Command
    fun isAvailable(invoke: Invoke) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "isAvailable() CALLED")
        
        val result = JSObject()
        val available = SpeechRecognizer.isRecognitionAvailable(activity)
        result.put("available", available)
        Log.d(TAG, "  Speech recognition available: $available")
        
        if (!available) {
            result.put("reason", "Speech recognition not available on this device. Please install Google app or enable voice input.")
            Log.w(TAG, "  Reason: Speech recognition not available")
        } else {
            // Check if Google app is installed (common cause of ERROR_CLIENT)
            try {
                val packageManager = activity.packageManager
                val googleAppPackage = "com.google.android.googlequicksearchbox"
                val googleAppInfo = packageManager.getPackageInfo(googleAppPackage, 0)
                result.put("googleAppInstalled", true)
                result.put("googleAppVersion", googleAppInfo.versionName)
                Log.d(TAG, "  Google app installed: ${googleAppInfo.versionName}")
            } catch (e: Exception) {
                result.put("googleAppInstalled", false)
                result.put("reason", "Google app not installed. Please install it from Play Store for speech recognition.")
                Log.w(TAG, "  Google app not installed: ${e.message}")
            }
        }
        
        Log.d(TAG, "============================================")
        invoke.resolve(result)
    }

    @Command
    fun getSupportedLanguages(invoke: Invoke) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "getSupportedLanguages() CALLED")
        
        val result = JSObject()
        val languagesArray = org.json.JSONArray()

        val locales = Locale.getAvailableLocales()
        val addedCodes = mutableSetOf<String>()
        Log.d(TAG, "  Total locales available: ${locales.size}")

        for (locale in locales) {
            val code = locale.toLanguageTag()
            if (code.contains("-") && !addedCodes.contains(code)) {
                addedCodes.add(code)
                val langObj = JSObject()
                langObj.put("code", code)
                langObj.put("name", locale.displayName)
                languagesArray.put(langObj)
            }
        }
        
        Log.d(TAG, "  Languages with regional codes: ${addedCodes.size}")

        result.put("languages", languagesArray)
        Log.d(TAG, "============================================")
        invoke.resolve(result)
    }

    @Command
    fun checkPermission(invoke: Invoke) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "checkPermission() CALLED")
        
        val result = JSObject()

        val hasPermission = ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val shouldShowRationale = ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.RECORD_AUDIO)
        
        Log.d(TAG, "  RECORD_AUDIO granted: $hasPermission")
        Log.d(TAG, "  Should show rationale: $shouldShowRationale")
        
        val micPermission = when {
            hasPermission -> "granted"
            shouldShowRationale -> "denied"
            else -> "unknown"
        }
        Log.d(TAG, "  Result: $micPermission")

        result.put("microphone", micPermission)
        result.put("speechRecognition", micPermission)

        Log.d(TAG, "============================================")
        invoke.resolve(result)
    }

    @Command
    fun requestPermission(invoke: Invoke) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "requestPermission() CALLED")
        
        val alreadyGranted = ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        Log.d(TAG, "  Already granted: $alreadyGranted")
        
        if (alreadyGranted) {
            val result = JSObject()
            result.put("microphone", "granted")
            result.put("speechRecognition", "granted")
            Log.d(TAG, "  Returning already granted")
            invoke.resolve(result)
            return
        }

        // Store invoke to resolve after user responds to permission dialog
        pendingPermissionInvoke = invoke
        Log.d(TAG, "  Requesting RECORD_AUDIO permission...")

        ActivityCompat.requestPermissions(
            activity,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE
        )
        Log.d(TAG, "  Permission dialog shown, waiting for result...")
        // Don't resolve here - wait for onRequestPermissionsResult callback
    }

    private fun createRecognizerIntent(config: ListenConfig): Intent {
        Log.d(TAG, "createRecognizerIntent() - Building intent")
        
        return Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )

            // CRITICAL: Set calling package to identify the app to Google services
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, activity.packageName)
            Log.d(TAG, "  CALLING_PACKAGE: ${activity.packageName}")
            
            val language = config.language ?: Locale.getDefault().toLanguageTag()
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, language)
            putExtra("android.speech.extra.EXTRA_ADDITIONAL_LANGUAGES", arrayOf(language))
            Log.d(TAG, "  LANGUAGE: $language")
            
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, config.interimResults)
            Log.d(TAG, "  PARTIAL_RESULTS: ${config.interimResults}")
            
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
            Log.d(TAG, "  MAX_RESULTS: 5")
            
            // Enable secure offline mode (prevents ERROR_CLIENT)
            putExtra(RecognizerIntent.EXTRA_SECURE, true)
            Log.d(TAG, "  SECURE: true")
            
            // Keep listening even after pause
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 3000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 3000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 1000L)
            Log.d(TAG, "  SILENCE_LENGTH: 3000ms, MINIMUM_LENGTH: 1000ms")
        }
    }

    private fun getErrorCode(errorCode: Int): String {
        return when (errorCode) {
            SpeechRecognizer.ERROR_AUDIO -> "AUDIO_ERROR"
            SpeechRecognizer.ERROR_CLIENT -> "UNKNOWN"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "PERMISSION_DENIED"
            SpeechRecognizer.ERROR_NETWORK -> "NETWORK_ERROR"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "TIMEOUT"
            SpeechRecognizer.ERROR_NO_MATCH -> "NO_SPEECH"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "BUSY"
            SpeechRecognizer.ERROR_SERVER -> "NETWORK_ERROR"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "TIMEOUT"
            else -> "UNKNOWN"
        }
    }

    private fun getErrorMessage(errorCode: Int): String {
        val message = when (errorCode) {
            SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
            SpeechRecognizer.ERROR_CLIENT -> "Client side error"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
            SpeechRecognizer.ERROR_NETWORK -> "Network error"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
            SpeechRecognizer.ERROR_NO_MATCH -> "No speech detected"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognition service busy"
            SpeechRecognizer.ERROR_SERVER -> "Server error"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input"
            else -> "Unknown error ($errorCode)"
        }
        Log.d(TAG, "getErrorMessage($errorCode) -> $message")
        return message
    }
}
