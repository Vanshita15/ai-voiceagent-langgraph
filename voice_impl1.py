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
    """Handles voice input and output"""
    
    def __init__(self):
        if not VOICE_AVAILABLE:
            raise RuntimeError("Voice packages not installed")
        
        # Initialize Speech-to-Text (Whisper)
        print("Loading Whisper model (this may take a moment)...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ Whisper model loaded")
        
        # Initialize Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed
        self.tts_engine.setProperty('volume', 0.9)
        
        # Get available voices and set a pleasant one
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)  # Usually female voice
    
    def record_audio(self, duration=30, sample_rate=16000):
        """Record audio from microphone"""
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✓ Recording complete")
        
        # Save temporarily for Whisper
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio, sample_rate)
        return temp_file
    
    def speech_to_text(self, audio_file):
        """Convert speech to text using Whisper"""
        print("🔄 Converting speech to text...")
        segments, info = self.whisper_model.transcribe(audio_file, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text + " "
        
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        text = text.strip()
        print(f"✓ Transcribed: '{text}'")
        return text
    
    def text_to_speech(self, text):
        
        """Convert text to speech"""
        print("\n🔊 Speaking response...")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

