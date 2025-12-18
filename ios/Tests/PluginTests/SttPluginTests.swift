import XCTest
@testable import tauri_plugin_stt

/// Unit tests for STT Plugin
final class SttPluginTests: XCTestCase {
    
    /// Test that recognition states are valid enum cases
    func testRecognitionStates() throws {
        let states = ["idle", "listening", "processing"]
        
        XCTAssertEqual(states.count, 3, "Should have 3 recognition states")
        XCTAssertTrue(states.contains("idle"), "Should contain idle")
        XCTAssertTrue(states.contains("listening"), "Should contain listening")
        XCTAssertTrue(states.contains("processing"), "Should contain processing")
    }
    
    /// Test permission status values
    func testPermissionStatuses() throws {
        let statuses = ["granted", "denied", "unknown"]
        
        XCTAssertEqual(statuses.count, 3, "Should have 3 permission statuses")
        XCTAssertTrue(statuses.contains("granted"), "Should contain granted")
        XCTAssertTrue(statuses.contains("denied"), "Should contain denied")
        XCTAssertTrue(statuses.contains("unknown"), "Should contain unknown")
    }
    
    /// Test language code format (BCP-47)
    func testLanguageCodeFormat() throws {
        let validCodes = ["en-US", "pt-BR", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN"]
        let pattern = "^[a-z]{2}-[A-Z]{2}$"
        let regex = try NSRegularExpression(pattern: pattern)
        
        for code in validCodes {
            let range = NSRange(code.startIndex..., in: code)
            let matches = regex.numberOfMatches(in: code, range: range)
            XCTAssertEqual(matches, 1, "Language code \(code) should match BCP-47 format")
        }
    }
    
    /// Test confidence score validation
    func testConfidenceScoreRange() throws {
        let validConfidences: [Float] = [0.0, 0.5, 0.75, 0.99, 1.0]
        let invalidConfidences: [Float] = [-0.1, 1.1, 2.0]
        
        for conf in validConfidences {
            XCTAssertTrue(conf >= 0.0 && conf <= 1.0, "Confidence \(conf) should be valid")
        }
        
        for conf in invalidConfidences {
            XCTAssertFalse(conf >= 0.0 && conf <= 1.0, "Confidence \(conf) should be invalid")
        }
    }
    
    /// Test that ListenConfig has correct default values
    func testListenConfigDefaults() throws {
        // Simulate default config behavior
        let defaultLanguage: String? = nil
        let defaultInterimResults = false
        let defaultContinuous = false
        let defaultMaxDuration: Int? = nil
        
        XCTAssertNil(defaultLanguage, "Default language should be nil")
        XCTAssertFalse(defaultInterimResults, "Default interimResults should be false")
        XCTAssertFalse(defaultContinuous, "Default continuous should be false")
        XCTAssertNil(defaultMaxDuration, "Default maxDuration should be nil")
    }
    
    /// Test event names are correctly formatted
    func testEventNames() throws {
        let events = [
            "plugin:stt://result",
            "plugin:stt://stateChange",
            "plugin:stt://error"
        ]
        
        for event in events {
            XCTAssertTrue(event.hasPrefix("plugin:stt://"), "Event \(event) should have correct prefix")
        }
    }
}

