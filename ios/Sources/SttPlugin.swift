import AVFoundation
import Speech
import Tauri
import UIKit
import WebKit

/// Configuration for speech recognition
struct ListenConfig: Decodable {
    let language: String?
    let interimResults: Bool?
    let continuous: Bool?
    let maxDuration: Int?
    let onDevice: Bool?
    
    static var `default`: ListenConfig {
        return ListenConfig(language: nil, interimResults: true, continuous: false, maxDuration: nil, onDevice: nil)
    }
    
    init(language: String? = nil, interimResults: Bool? = true, continuous: Bool? = false, maxDuration: Int? = nil, onDevice: Bool? = nil) {
        self.language = language
        self.interimResults = interimResults
        self.continuous = continuous
        self.maxDuration = maxDuration
        self.onDevice = onDevice
    }
}

/// Tauri plugin for Speech-to-Text recognition on iOS
/// Uses Apple's Speech framework (SFSpeechRecognizer)
class SttPlugin: Plugin, SFSpeechRecognitionTaskDelegate {
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?
    private var isListening = false
    private var currentLanguage: String?
    private var currentConfig: ListenConfig?
    private var isManualStop = false
    private var wasListeningBeforeInterruption = false
    private var maxDurationTimer: DispatchWorkItem?
    
    override init() {
        super.init()
        NSLog("[SttPlugin] ============================================")
        NSLog("[SttPlugin] PLUGIN INIT")
        NSLog("[SttPlugin]   iOS Version: \(UIDevice.current.systemVersion)")
        NSLog("[SttPlugin]   Device: \(UIDevice.current.model)")
        audioEngine = AVAudioEngine()
        NSLog("[SttPlugin]   AudioEngine created")
        setupInterruptionHandling()
        NSLog("[SttPlugin] ============================================")
    }
    
    deinit {
        NSLog("[SttPlugin] deinit CALLED")
        NotificationCenter.default.removeObserver(self)
        maxDurationTimer?.cancel()
    }
    
    // MARK: - Audio Session Interruption Handling
    
    /// Setup observers for audio session interruptions (phone calls, Siri, etc.)
    private func setupInterruptionHandling() {
        NSLog("[SttPlugin] setupInterruptionHandling() CALLED")
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioSessionInterruption),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioRouteChange),
            name: AVAudioSession.routeChangeNotification,
            object: AVAudioSession.sharedInstance()
        )
        NSLog("[SttPlugin]   Observers registered")
    }
    
    /// Handles audio route changes (headphones plugged/unplugged, Bluetooth, etc.)
    @objc private func handleAudioRouteChange(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }
        
        switch reason {
        case .oldDeviceUnavailable:
            // Microphone device disconnected - stop recognition
            if isListening {
                NSLog("[SttPlugin] Recognition stopped due to audio route change (device unavailable)")
                stopRecognition()
                self.trigger("stateChange", data: ["state": "idle"] as JSObject)
            }
        case .newDeviceAvailable:
            NSLog("[SttPlugin] New audio device available")
        default:
            break
        }
    }
    
    /// Handle audio interruptions such as phone calls
    @objc private func handleAudioSessionInterruption(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }
        
        switch type {
        case .began:
            // Interruption began - stop recognition
            if isListening {
                wasListeningBeforeInterruption = true
                NSLog("[SttPlugin] Recognition interrupted, stopping...")
                stopRecognition()
                
                self.trigger("error", data: [
                    "code": "CANCELLED",
                    "message": "Recognition interrupted by system",
                    "details": "iOS audio session interruption"
                ] as JSObject)
                self.trigger("stateChange", data: ["state": "idle"] as JSObject)
            }
            
        case .ended:
            // Interruption ended - could resume but user should restart manually
            guard let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt else { return }
            let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
            
            if options.contains(.shouldResume) && wasListeningBeforeInterruption {
                NSLog("[SttPlugin] Interruption ended, user can restart recognition")
                // We don't auto-resume as recognition requires user interaction
            }
            wasListeningBeforeInterruption = false
            
        @unknown default:
            break
        }
    }
    
    // MARK: - SFSpeechRecognitionTaskDelegate
    
    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didHypothesizeTranscription transcription: SFTranscription) {
        let transcript = transcription.formattedString
        NSLog("[SttPlugin] didHypothesizeTranscription: \(transcript)")
        
        var confidence: Float?
        if let segment = transcription.segments.last {
            confidence = segment.confidence
        }
        
        var eventData: JSObject = [
            "transcript": transcript,
            "isFinal": false
        ]
        if let conf = confidence {
            eventData["confidence"] = conf
        }
        self.trigger("result", data: eventData)
    }
    
    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishRecognition recognitionResult: SFSpeechRecognitionResult) {
        let transcript = recognitionResult.bestTranscription.formattedString
        NSLog("[SttPlugin] didFinishRecognition: \(transcript)")
        
        var confidence: Float?
        if let segment = recognitionResult.bestTranscription.segments.last {
            confidence = segment.confidence
        }
        
        var eventData: JSObject = [
            "transcript": transcript,
            "isFinal": true
        ]
        if let conf = confidence {
            eventData["confidence"] = conf
        }
        self.trigger("result", data: eventData)
        self.trigger("stateChange", data: ["state": "idle"] as JSObject)
    }
    
    func speechRecognitionTaskWasCancelled(_ task: SFSpeechRecognitionTask) {
        NSLog("[SttPlugin] speechRecognitionTaskWasCancelled")
        stopRecognition()
        self.trigger("stateChange", data: ["state": "idle"] as JSObject)
    }
    
    func speechRecognitionTask(_ task: SFSpeechRecognitionTask, didFinishSuccessfully successfully: Bool) {
        NSLog("[SttPlugin] didFinishSuccessfully: \(successfully), isManualStop: \(isManualStop)")
        
        // Don't report error if user manually stopped recognition
        if !successfully && !isManualStop {
            self.trigger("error", data: [
                "code": "UNKNOWN",
                "message": "Recognition finished unsuccessfully",
                "details": "Speech recognition task failed"
            ] as JSObject)
        }
        
        stopRecognition()
        
        if let config = currentConfig, config.continuous ?? false, successfully {
            NSLog("[SttPlugin] Restarting in continuous mode...")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                do {
                    try self?.startRecognition(config: config)
                } catch {
                    NSLog("[SttPlugin] Failed to restart continuous recognition: \(error.localizedDescription)")
                    self?.trigger("error", data: [
                        "code": "UNKNOWN",
                        "message": "Failed to restart recognition",
                        "details": error.localizedDescription
                    ] as JSObject)
                }
            }
        }
    }
    
    // MARK: - Commands
    
    @objc public func startListening(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] startListening called")
        
        let args: ListenConfig
        do {
            args = try invoke.parseArgs(ListenConfig.self)
            NSLog("[SttPlugin] Args parsed: language=\(args.language ?? "nil"), interimResults=\(args.interimResults ?? true)")
        } catch {
            NSLog("[SttPlugin] Failed to parse args, using defaults: \(error)")
            args = ListenConfig.default
        }
        
        if isListening {
            NSLog("[SttPlugin] Already listening, rejecting")
            invoke.reject("Already listening")
            return
        }
        
        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        let micStatus = AVAudioSession.sharedInstance().recordPermission
        NSLog("[SttPlugin] Permissions - Speech: \(speechStatus.rawValue), Mic: \(micStatus.rawValue)")
        
        if speechStatus == .authorized && micStatus == .granted {
            NSLog("[SttPlugin] Permissions granted, starting...")
            startListeningWithConfig(args, invoke: invoke)
            return
        }
        
        if speechStatus == .denied || speechStatus == .restricted {
            NSLog("[SttPlugin] Speech permission denied or restricted")
            invoke.reject("Speech recognition permission denied. Please enable it in Settings.")
            return
        }
        
        if micStatus == .denied {
            NSLog("[SttPlugin] Microphone permission denied")
            invoke.reject("Microphone permission denied. Please enable it in Settings.")
            return
        }
        
        NSLog("[SttPlugin] Requesting permissions...")
        let group = DispatchGroup()
        var permissionError: String? = nil
        
        if speechStatus == .notDetermined {
            NSLog("[SttPlugin] Requesting speech authorization...")
            group.enter()
            SFSpeechRecognizer.requestAuthorization { status in
                NSLog("[SttPlugin] Speech authorization result: \(status.rawValue)")
                if status != .authorized {
                    permissionError = "Speech recognition permission not granted"
                }
                group.leave()
            }
        }
        
        if micStatus == .undetermined {
            NSLog("[SttPlugin] Requesting microphone permission...")
            group.enter()
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                NSLog("[SttPlugin] Microphone permission result: \(granted)")
                if !granted {
                    permissionError = "Microphone permission not granted"
                }
                group.leave()
            }
        }
        
        group.notify(queue: .main) { [weak self] in
            NSLog("[SttPlugin] Permission requests completed, error: \(permissionError ?? "none")")
            if let error = permissionError {
                invoke.reject(error)
                return
            }
            self?.startListeningWithConfig(args, invoke: invoke)
        }
    }
    
    private func startListeningWithConfig(_ args: ListenConfig, invoke: Invoke) {
        NSLog("[SttPlugin] startListeningWithConfig called")
        
        let locale: Locale
        if let language = args.language {
            locale = Locale(identifier: language)
            NSLog("[SttPlugin] Using specified locale: \(language)")
        } else {
            locale = Locale.current
            NSLog("[SttPlugin] Using current locale: \(locale.identifier)")
        }
        
        speechRecognizer = SFSpeechRecognizer(locale: locale)
        currentLanguage = locale.identifier
        
        guard let speechRecognizer = speechRecognizer else {
            NSLog("[SttPlugin] SFSpeechRecognizer is nil for locale: \(locale.identifier)")
            invoke.reject("Speech recognition not available for language: \(locale.identifier)")
            return
        }
        
        guard speechRecognizer.isAvailable else {
            NSLog("[SttPlugin] SFSpeechRecognizer not available for locale: \(locale.identifier)")
            invoke.reject("Speech recognition not available for language: \(locale.identifier)")
            return
        }
        
        NSLog("[SttPlugin] SFSpeechRecognizer available, starting recognition...")
        
        do {
            try startRecognition(config: args)
            NSLog("[SttPlugin] Recognition started successfully")
            invoke.resolve()
        } catch {
            NSLog("[SttPlugin] Failed to start recognition: \(error)")
            invoke.reject("Failed to start recognition: \(error.localizedDescription)")
        }
    }
    
    @objc public func stopListening(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] stopListening called - user requested stop")
        isManualStop = true
        stopRecognition()
        invoke.resolve()
    }
    
    @objc public func isAvailable(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] ============================================")
        NSLog("[SttPlugin] isAvailable() CALLED")
        
        let recognizer = SFSpeechRecognizer()
        let available = recognizer?.isAvailable ?? false
        NSLog("[SttPlugin]   SFSpeechRecognizer available: \(available)")
        
        if #available(iOS 13, *) {
            let supportsOnDevice = recognizer?.supportsOnDeviceRecognition ?? false
            NSLog("[SttPlugin]   Supports on-device recognition: \(supportsOnDevice)")
        }
        
        var result: JSObject = ["available": available]
        if !available {
            result["reason"] = "Speech recognition not available on this device"
            NSLog("[SttPlugin]   Reason: Not available")
        }
        
        invoke.resolve(result)
    }
    
    @objc public func getSupportedLanguages(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] getSupportedLanguages() CALLED")
        
        let supportedLocales = SFSpeechRecognizer.supportedLocales()
        NSLog("[SttPlugin]   Total supported locales: \(supportedLocales.count)")
        
        let languages = supportedLocales.map { locale -> [String: String] in
            return [
                "code": locale.identifier,
                "name": locale.localizedString(forIdentifier: locale.identifier) ?? locale.identifier
            ]
        }
        
        invoke.resolve(["languages": languages])
    }
    
    @objc public func checkPermission(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] checkPermission() CALLED")
        
        let micStatus: String
        let micRaw = AVAudioSession.sharedInstance().recordPermission
        switch micRaw {
        case .granted:
            micStatus = "granted"
        case .denied:
            micStatus = "denied"
        case .undetermined:
            micStatus = "unknown"
        @unknown default:
            micStatus = "unknown"
        }
        NSLog("[SttPlugin]   Microphone: \(micStatus) (raw: \(micRaw.rawValue))")
        
        let speechStatus: String
        let speechRaw = SFSpeechRecognizer.authorizationStatus()
        switch speechRaw {
        case .authorized:
            speechStatus = "granted"
        case .denied, .restricted:
            speechStatus = "denied"
        case .notDetermined:
            speechStatus = "unknown"
        @unknown default:
            speechStatus = "unknown"
        }
        NSLog("[SttPlugin]   Speech recognition: \(speechStatus) (raw: \(speechRaw.rawValue))")
        
        invoke.resolve([
            "microphone": micStatus,
            "speechRecognition": speechStatus
        ])
    }
    
    @objc public func requestPermission(_ invoke: Invoke) throws {
        NSLog("[SttPlugin] requestPermission() CALLED")
        
        let group = DispatchGroup()
        
        var micResult = "unknown"
        var speechResult = "unknown"
        
        group.enter()
        NSLog("[SttPlugin]   Requesting microphone permission...")
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            micResult = granted ? "granted" : "denied"
            NSLog("[SttPlugin]   Microphone result: \(micResult)")
            group.leave()
        }
        
        group.enter()
        NSLog("[SttPlugin]   Requesting speech recognition permission...")
        SFSpeechRecognizer.requestAuthorization { status in
            switch status {
            case .authorized:
                speechResult = "granted"
            case .denied, .restricted:
                speechResult = "denied"
            case .notDetermined:
                speechResult = "unknown"
            @unknown default:
                speechResult = "unknown"
            }
            NSLog("[SttPlugin]   Speech recognition result: \(speechResult)")
            group.leave()
        }
        
        group.notify(queue: .main) {
            NSLog("[SttPlugin]   Final results - mic: \(micResult), speech: \(speechResult)")
            invoke.resolve([
                "microphone": micResult,
                "speechRecognition": speechResult
            ])
        }
    }
    
    // MARK: - Private Methods
    
    private func startRecognition(config: ListenConfig) throws {
        NSLog("[SttPlugin] startRecognition called")
        
        // Reset manual stop flag when starting new recognition
        isManualStop = false
        
        recognitionTask?.cancel()
        recognitionTask = nil
        
        NSLog("[SttPlugin] Configuring audio session...")
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            NSLog("[SttPlugin] Audio session configured successfully")
        } catch {
            NSLog("[SttPlugin] Audio session configuration failed: \(error)")
            throw error
        }
        
        if audioEngine == nil {
            NSLog("[SttPlugin] Creating new audio engine")
            audioEngine = AVAudioEngine()
        }
        
        guard let audioEngine = audioEngine else {
            NSLog("[SttPlugin] Audio engine is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Audio engine not initialized"])
        }
        
        NSLog("[SttPlugin] Creating recognition request...")
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        
        guard let recognitionRequest = recognitionRequest else {
            NSLog("[SttPlugin] Recognition request is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create recognition request"])
        }
        
        recognitionRequest.shouldReportPartialResults = config.interimResults ?? true
        
        if #available(iOS 13, *) {
            let useOnDevice = config.onDevice ?? false
            if useOnDevice {
                // Check if on-device recognition is supported for this locale
                if speechRecognizer?.supportsOnDeviceRecognition == true {
                    recognitionRequest.requiresOnDeviceRecognition = true
                    NSLog("[SttPlugin] Using on-device recognition (offline)")
                } else {
                    NSLog("[SttPlugin] On-device recognition not available for this language, using server")
                    recognitionRequest.requiresOnDeviceRecognition = false
                }
            } else {
                recognitionRequest.requiresOnDeviceRecognition = false
            }
        }
        
        let inputNode = audioEngine.inputNode
        NSLog("[SttPlugin] Input node obtained")
        
        guard let recognizer = speechRecognizer else {
            NSLog("[SttPlugin] Speech recognizer is nil")
            throw NSError(domain: "SttPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Speech recognizer not available"])
        }
        
        currentConfig = config
        
        NSLog("[SttPlugin] Starting recognition task...")
        recognitionTask = recognizer.recognitionTask(with: recognitionRequest, delegate: self)
        
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        NSLog("[SttPlugin] Recording format: \(recordingFormat)")
        
        do {
            inputNode.removeTap(onBus: 0)
        } catch {
            NSLog("[SttPlugin] No existing tap to remove")
        }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        NSLog("[SttPlugin] Tap installed on input node")
        
        audioEngine.prepare()
        NSLog("[SttPlugin] Audio engine prepared")
        
        try audioEngine.start()
        NSLog("[SttPlugin] Audio engine started")
        
        isListening = true
        trigger("stateChange", data: ["state": "listening"] as JSObject)
        NSLog("[SttPlugin] Recognition started, isListening = true")
        
        // Setup maxDuration timer if configured
        if let maxDuration = config.maxDuration, maxDuration > 0 {
            NSLog("[SttPlugin] Setting up maxDuration timer: \(maxDuration)ms")
            maxDurationTimer?.cancel()
            
            let workItem = DispatchWorkItem { [weak self] in
                guard let self = self, self.isListening else { return }
                NSLog("[SttPlugin] maxDuration reached, stopping recognition")
                self.stopRecognition()
                self.trigger("stateChange", data: ["state": "idle"] as JSObject)
                self.trigger("error", data: [
                    "code": "TIMEOUT",
                    "message": "Maximum duration reached",
                    "details": "Recognition stopped after maxDuration limit"
                ] as JSObject)
            }
            maxDurationTimer = workItem
            DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(maxDuration), execute: workItem)
        }
    }
    
    private func stopRecognition() {
        NSLog("[SttPlugin] stopRecognition() CALLED")
        NSLog("[SttPlugin]   isListening: \(isListening)")
        NSLog("[SttPlugin]   isManualStop: \(isManualStop)")
        
        // Cancel maxDuration timer
        maxDurationTimer?.cancel()
        maxDurationTimer = nil
        NSLog("[SttPlugin]   Max duration timer cancelled")
        
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        NSLog("[SttPlugin]   Audio engine stopped, tap removed")
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        NSLog("[SttPlugin]   Recognition request ended")
        
        recognitionTask?.cancel()
        recognitionTask = nil
        NSLog("[SttPlugin]   Recognition task cancelled")
        
        isListening = false
        currentConfig = nil
        
        // Reset manual stop flag after cleanup
        isManualStop = false
        
        // Deactivate audio session - log errors but don't throw since we're in cleanup
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            NSLog("[SttPlugin]   Audio session deactivated successfully")
        } catch {
            NSLog("[SttPlugin]   Failed to deactivate audio session: \(error.localizedDescription)")
            // Don't throw here since deactivation failure is not critical during cleanup
        }
        NSLog("[SttPlugin]   stopRecognition() complete")
    }
}

@_cdecl("init_plugin_stt")
func initPlugin() -> Plugin {
    return SttPlugin()
}
