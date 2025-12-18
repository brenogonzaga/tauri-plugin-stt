package io.affex.stt

import android.Manifest
import android.content.pm.PackageManager
import android.speech.SpeechRecognizer
import androidx.core.content.ContextCompat
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented tests for STT Plugin.
 * 
 * These tests run on an Android device/emulator and validate
 * functionality that requires the Android framework.
 */
@RunWith(AndroidJUnit4::class)
class SttInstrumentedTest {
    
    /**
     * Test that the package name is correct
     */
    @Test
    fun packageName_isCorrect() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("io.affex.stt.test", appContext.packageName)
    }
    
    /**
     * Test SpeechRecognizer availability check
     */
    @Test
    fun speechRecognizer_isAvailable() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val isAvailable = SpeechRecognizer.isRecognitionAvailable(appContext)
        
        // This may be true or false depending on device
        // We just verify the check doesn't throw
        assertNotNull("isRecognitionAvailable should return a value", isAvailable)
    }
    
    /**
     * Test permission check for RECORD_AUDIO
     */
    @Test
    fun permission_canBeChecked() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val permission = ContextCompat.checkSelfPermission(
            appContext,
            Manifest.permission.RECORD_AUDIO
        )
        
        // Permission is either granted or denied
        assertTrue(
            "Permission should be granted or denied",
            permission == PackageManager.PERMISSION_GRANTED ||
            permission == PackageManager.PERMISSION_DENIED
        )
    }
    
    /**
     * Test SpeechRecognizer can be created
     */
    @Test
    fun speechRecognizer_canBeCreated() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        
        if (SpeechRecognizer.isRecognitionAvailable(appContext)) {
            val recognizer = SpeechRecognizer.createSpeechRecognizer(appContext)
            assertNotNull("SpeechRecognizer should be created", recognizer)
            recognizer.destroy()
        }
    }
}

