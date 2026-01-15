print("🔊 Voice implementation module loaded")
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator
from datetime import datetime
import json
import os

# Voice processing imports
try:
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel
    import pyttsx3
    import numpy as np
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Voice packages not installed: {e}")
    print("Install: pip install faster-whisper pyttsx3 sounddevice soundfile numpy")
    VOICE_AVAILABLE = False


# ============================================
# VOICE COMPONENTS
# ============================================

class VoiceProcessor:
    """Voice processor with silence detection"""
    
    def __init__(self):
        print("Loading Whisper...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ Whisper loaded")
        
        # TTS setup
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)
    
    def record_with_silence_detection(self, sample_rate=16000, silence_threshold=0.01, silence_duration=2.0):
        """
        Record until user stops speaking
        silence_threshold: Volume level to consider as silence
        silence_duration: Seconds of silence before stopping
        """
        print("\n🎤 Listening... (will stop when you finish speaking)")
        breakpoint()
        chunk_duration = 0.5  # Record in 0.5 second chunks
        chunks = []
        silence_chunks = 0
        max_silence_chunks = int(silence_duration / chunk_duration)
        
        while True:
            # Record a chunk
            chunk = sd.rec(
                int(chunk_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            chunks.append(chunk)
            
            # Check if this chunk is silent
            volume = np.abs(chunk).mean()
            
            if volume < silence_threshold:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    print("✓ Recording stopped (silence detected)")
                    break
            else:
                silence_chunks = 0  # Reset if sound detected
            
            # Safety limit: max 15 seconds
            if len(chunks) > 30:
                print("✓ Recording stopped (max duration)")
                break
        
        # Combine all chunks
        audio = np.concatenate(chunks)
        
        # Save temporarily
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio, sample_rate)
        
        return temp_file
    
    def speech_to_text(self, audio_file):
        """Convert speech to text"""
        print("🔄 Converting speech to text...")
        segments, info = self.whisper_model.transcribe(audio_file, beam_size=5)
        
        text = " ".join([segment.text for segment in segments]).strip()
        
        # Clean up
        import os
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        print(f"✓ You said: '{text}'")
        return text
    
    def text_to_speech(self, text):
        """Speak the response"""
        print(f"\n🔊 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()