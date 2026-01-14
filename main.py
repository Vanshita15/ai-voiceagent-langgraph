print("🔥 MAIN.PY STARTED 🔥")
from agents1 import create_medical_assistant_graph
from voice_impl1 import VoiceProcessor,VOICE_AVAILABLE

class MedicalVoiceAgent:
    """Complete voice-enabled medical assistant"""
    
    def __init__(self):
        self.graph = create_medical_assistant_graph()
        self.voice_processor = VoiceProcessor()
        self.user_profile = {
            "medications": [],
            "conditions": []
        }
        
        print("\n" + "="*60)
        print("🏥 MEDICAL VOICE ASSISTANT INITIALIZED")
        print("="*60)
    
    def process_voice_input(self, duration=30):
        """Process voice input and return response"""
        # 1. Record audio
        audio_file = self.voice_processor.record_audio(duration)
        
        # 2. Convert to text
        user_input = self.voice_processor.speech_to_text(audio_file)
        
        if not user_input:
            return "I didn't catch that. Could you please repeat?"
        
        # 3. Process through LangGraph agent
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "user_profile": self.user_profile,
            "response": "",
            "next_action": ""
        }
        
        result = self.graph.invoke(initial_state)
        response = result["response"]
        
        # 4. Convert response to speech
        self.voice_processor.text_to_speech(response)
        
        return response
    
    def add_medication(self, medication):
        """Add medication to user profile"""
        self.user_profile["medications"].append(medication)
        print(f"✓ Added medication: {medication}")
    
    def add_condition(self, condition):
        """Add medical condition to user profile"""
        self.user_profile["conditions"].append(condition)
        print(f"✓ Added condition: {condition}")
    
    def run_interactive(self):
        """Run interactive voice session"""
        print("\n" + "="*60)
        print("🎙️  VOICE MODE ACTIVATED")
        print("="*60)
        print("\nCommands:")
        print("  - Press ENTER to speak")
        print("  - Type 'quit' to exit")
        print("  - Type 'profile' to see your profile")
        print("="*60)
        
        while True:
            command = input("\nPress ENTER to speak (or type command): ").strip().lower()
            
            if command == 'quit':
                print("\n🏥 Take care! Goodbye!")
                break
            
            if command == 'profile':
                print("\n📋 Your Profile:")
                print(f"  Medications: {self.user_profile['medications']}")
                print(f"  Conditions: {self.user_profile['conditions']}")
                continue
            
            if command:
                # Text input mode
                initial_state = {
                    "messages": [],
                    "user_input": command,
                    "intent": "",
                    "user_profile": self.user_profile,
                    "response": "",
                    "next_action": ""
                }
                result = self.graph.invoke(initial_state)
                print(f"\n🤖 {result['response']}")
            else:
                # Voice input mode
                try:
                    response = self.process_voice_input(duration=30)
                    print(f"\n🤖 Response: {response}")
                except KeyboardInterrupt:
                    print("\n\n🏥 Session ended. Take care!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    if not VOICE_AVAILABLE:
        print("\n❌ Voice packages not installed. Install them first:")
        print("pip install faster-whisper pyttsx3 sounddevice soundfile numpy")
        exit(1)
    try:
        # Create agent
        agent = MedicalVoiceAgent()
        
        # Optional: Add user profile info
        agent.add_condition("Type 2 Diabetes")
        agent.add_medication("Metformin 500mg - twice daily")
        
        # Run interactive voice mode
        agent.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\n🏥 Session ended. Take care!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
