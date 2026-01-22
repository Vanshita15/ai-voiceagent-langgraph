print("🔊 Voice implementation module loaded")
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from datetime import datetime

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

    def _select_input_device(self):
        """Return a valid input device index for sounddevice, or raise if none exist."""
        try:
            default_device = sd.default.device
            if isinstance(default_device, (list, tuple)) and len(default_device) >= 1:
                default_input = default_device[0]
            else:
                default_input = default_device

            if isinstance(default_input, int) and default_input is not None and default_input >= 0:
                info = sd.query_devices(default_input)
                if info.get('max_input_channels', 0) > 0:
                    return default_input
        except Exception:
            pass

        try:
            devices = sd.query_devices()
        except Exception:
            raise RuntimeError("No working audio input device found. Please connect/enable a microphone and allow microphone access.")

        for idx, info in enumerate(devices):
            try:
                if info.get('max_input_channels', 0) > 0:
                    sd.check_input_settings(device=idx, channels=1)
                    return idx
            except Exception:
                continue

        raise RuntimeError("No working audio input device found. Please connect/enable a microphone and allow microphone access.")

    def record_audio(self, duration=30, sample_rate=16000):
        input_device = self._select_input_device()

        try:
            input_info = sd.query_devices(input_device)
            print(f"🎛️  Using input device {input_device}: {input_info.get('name', 'Unknown')}")
        except Exception:
            print(f"🎛️   error Using input device {input_device}")

        import sys
        import time
        import threading
        import tempfile
        import os

        stop_event = threading.Event()

        def _wait_for_enter():
            try:
                input()
                stop_event.set()
            except Exception:
                # Non-interactive environments may not allow stdin reads
                return

        if sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            print(f"\n🎤 Recording... (max {duration}s)\n↩️  Press ENTER to stop recording early")
            threading.Thread(target=_wait_for_enter, daemon=True).start()
        else:
            print(f"\n🎤 Recording... (max {duration}s)")

        frames = []
        start_time = time.monotonic()

        def callback(indata, frame_count, time_info, status):
            if status:
                print(status)
            frames.append(indata.copy())
            if stop_event.is_set() or (time.monotonic() - start_time) >= float(duration):
                raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=input_device,
            callback=callback,
        ):
            while not stop_event.is_set() and (time.monotonic() - start_time) < float(duration):
                sd.sleep(50)

        if len(frames) == 0:
            audio = np.zeros((0, 1), dtype=np.float32)
        else:
            audio = np.concatenate(frames, axis=0)

        fd, path = tempfile.mkstemp(prefix="temp_audio_", suffix=".wav")
        os.close(fd)
        sf.write(path, audio, sample_rate)
        return path
    
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

    def speech_to_text_from_file(self, audio_file, cleanup: bool = False):
        """Convert speech to text from an existing audio file.
        cleanup=False is useful for UIs that manage the uploaded file lifecycle.
        """
        print("🔄 Converting speech to text...")
        segments, info = self.whisper_model.transcribe(audio_file, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()

        if cleanup:
            import os
            if os.path.exists(audio_file):
                os.remove(audio_file)

        print(f"✓ You said: '{text}'")
        return text
    
    def text_to_speech(self, text):
        """Speak the response"""
        try:
            print(f"\n🔊 Assistant: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ TTS error: {e}")
    def text_to_speech_file(self, text):
        """Generate TTS audio to a temporary WAV file path (for web UI playback)."""
        import tempfile
        import os

        fd, path = tempfile.mkstemp(prefix="assistant_tts_", suffix=".wav")
        os.close(fd)

        self.tts_engine.save_to_file(text, path)
        self.tts_engine.runAndWait()
        return path