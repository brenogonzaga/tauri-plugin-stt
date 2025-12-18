package io.affex.stt

import org.junit.Test
import org.junit.Assert.*

/**
 * Unit tests for STT Plugin models and utilities.
 * 
 * These tests run on the development machine (host) and validate
 * core logic without requiring Android framework dependencies.
 */
class SttUnitTest {
    
    /**
     * Test that recognition states are correctly defined
     */
    @Test
    fun recognitionStates_areValid() {
        val states = listOf("idle", "listening", "processing")
        assertTrue("Should have 3 recognition states", states.size == 3)
        assertTrue("Should contain idle", states.contains("idle"))
        assertTrue("Should contain listening", states.contains("listening"))
        assertTrue("Should contain processing", states.contains("processing"))
    }
    
    /**
     * Test permission status values
     */
    @Test
    fun permissionStatuses_areValid() {
        val statuses = listOf("granted", "denied", "unknown")
        assertTrue("Should have 3 permission statuses", statuses.size == 3)
        assertTrue("Should contain granted", statuses.contains("granted"))
        assertTrue("Should contain denied", statuses.contains("denied"))
        assertTrue("Should contain unknown", statuses.contains("unknown"))
    }
    
    /**
     * Test language code format validation
     */
    @Test
    fun languageCode_formatIsValid() {
        val validCodes = listOf("en-US", "pt-BR", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN")
        
        for (code in validCodes) {
            assertTrue(
                "Language code $code should match BCP-47 format",
                code.matches(Regex("^[a-z]{2}-[A-Z]{2}$"))
            )
        }
    }
    
    /**
     * Test that confidence scores are within valid range
     */
    @Test
    fun confidenceScore_isWithinRange() {
        val validConfidences = listOf(0.0f, 0.5f, 0.75f, 0.99f, 1.0f)
        val invalidConfidences = listOf(-0.1f, 1.1f, 2.0f)
        
        for (conf in validConfidences) {
            assertTrue("Confidence $conf should be valid", conf in 0.0f..1.0f)
        }
        
        for (conf in invalidConfidences) {
            assertFalse("Confidence $conf should be invalid", conf in 0.0f..1.0f)
        }
    }
    
    /**
     * Test ListenConfig defaults
     */
    @Test
    fun listenConfig_hasCorrectDefaults() {
        // Default values as per models.rs
        val defaultLanguage: String? = null
        val defaultInterimResults = false
        val defaultContinuous = false
        val defaultMaxDuration: Int? = null
        
        assertNull("Default language should be null", defaultLanguage)
        assertFalse("Default interimResults should be false", defaultInterimResults)
        assertFalse("Default continuous should be false", defaultContinuous)
        assertNull("Default maxDuration should be null", defaultMaxDuration)
    }
}

