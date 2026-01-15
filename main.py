from agents1 import create_conversational_graph
from voice_impl1 import VoiceProcessor


class ImprovedVoiceAgent:
    """Better voice agent with proper conversation flow"""
    
    def __init__(self):
        self.graph = create_conversational_graph()
        self.voice_processor = VoiceProcessor()
        self.conversation_stage = "greeting"
        self.user_profile = {
            "medications": ["Metformin 500mg - twice daily"],
            "conditions": ["Type 2 Diabetes"]
        }
        
        print("\n" + "="*60)
        print("🏥 MEDICAL VOICE ASSISTANT - READY")
        print("="*60)
    breakpoint()
    def start_conversation(self):
        """Start with greeting"""
        initial_state = {
            "conversation_stage": "greeting",
            "messages": [],
            "user_input": "",
            "intent": "",
            "user_profile": self.user_profile,
            "response": "",
            "context": {}
        }
        
        result = self.graph.invoke(initial_state)
        
        # Speak greeting
        self.voice_processor.text_to_speech(result["response"])
        
        return result["response"]
    
    def process_turn(self):
        """Process one conversation turn"""
        # 1. Listen to user
        audio_file = self.voice_processor.record_with_silence_detection()
        
        # 2. Convert to text
        user_input = self.voice_processor.speech_to_text(audio_file)
        
        if not user_input:
            self.voice_processor.text_to_speech("I didn't hear anything. Could you try again?")
            return
        
        # 3. Process through graph (skip greeting if not first time)
        state = {
            "conversation_stage": self.conversation_stage if self.conversation_stage != "greeting" else "asking_need",
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "user_profile": self.user_profile,
            "response": "",
            "context": {}
        }
        
        result = self.graph.invoke(state)
        
        # Update conversation stage
        self.conversation_stage = result.get("conversation_stage", "followup")
        
        # 4. Speak response
        self.voice_processor.text_to_speech(result["response"])
        
        return result["response"]
    
    def run_interactive(self):
        """Run the interactive session"""
        print("\n" + "="*60)
        print("🎙️  VOICE CONVERSATION MODE")
        print("="*60)
        print("\nCommands:")
        print("  - Press ENTER to speak")
        print("  - Type 'quit' to exit")
        print("="*60 + "\n")
        
        # Start with greeting
        self.start_conversation()
        
        while True:
            cmd = input("\nPress ENTER to speak (or 'quit'): ").strip().lower()
            
            if cmd == 'quit':
                goodbye = "Take care of your health! Goodbye!"
                self.voice_processor.text_to_speech(goodbye)
                break
            
            try:
                self.process_turn()
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                self.voice_processor.text_to_speech("Sorry, I had a problem. Let's try again.")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    try:
        agent = ImprovedVoiceAgent()
        agent.run_interactive()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()