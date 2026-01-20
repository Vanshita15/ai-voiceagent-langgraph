"""
IMPROVED MAIN.PY - Medical Voice Agent with Proper Flow

main_improved.py
"""

print("🔥 MEDICAL VOICE ASSISTANT STARTING... 🔥")

from agents_improved import create_conversational_graph, create_initial_state
from voice_impl1 import VoiceProcessor, VOICE_AVAILABLE


class MedicalVoiceAgent:
    """
    Enhanced Medical Voice Agent with Conversation Flow
    
    Flow:
    1. Greeting → Shows menu
    2. Listen to user choice
    3. Activate appropriate agent
    4. Return response
    5. Continue or end
    """
    
    def __init__(self):
        # Initialize graph
        self.graph = create_conversational_graph()
        
        # Initialize voice processor
        if VOICE_AVAILABLE:
            self.voice_processor = VoiceProcessor()
        else:
            self.voice_processor = None
        
        # User profile
        self.user_profile = {
            "medications": [],
            "conditions": []
        }
        
        # Conversation tracking
        self.is_first_interaction = True
        self.conversation_count = 0
        
        print("\n" + "="*60)
        print("🏥 MEDICAL VOICE ASSISTANT INITIALIZED")
        print("="*60)
        print("✅ Graph compiled")
        if VOICE_AVAILABLE:
            print("✅ Voice system ready")
        else:
            print("⚠️ Voice system not available (text mode only)")
        print("="*60)
    
    def start_session(self):
        """
        Start a new session with greeting
        Returns: greeting message
        """
        print("\n🎬 Starting new session...")
        
        # Create initial state for greeting
        state = create_initial_state(
            user_input="",
            user_profile=self.user_profile,
            is_first_message=True
        )
        
        # Run through graph to get greeting
        result = self.graph.invoke(state)
        greeting = result["response"]
        
        # Speak the greeting if voice available
        if self.voice_processor:
            self.voice_processor.text_to_speech(greeting)
        
        self.is_first_interaction = False
        return greeting
    
    def process_text_input(self, user_input):
        """
        Process text input through the agent
        
        Args:
            user_input: User's text message
            
        Returns:
            dict with response and intent
        """
        print(f"\n📝 Processing text: '{user_input}'")
        
        # Create state
        state = create_initial_state(
            user_input=user_input,
            user_profile=self.user_profile,
            is_first_message=self.is_first_interaction
        )
        
        # Process through graph
        result = self.graph.invoke(state)
        
        response = result["response"]
        intent = result.get("intent", "unknown")
        
        self.conversation_count += 1
        self.is_first_interaction = False
        
        print(f"🎯 Intent: {intent}")
        print(f"🤖 Response generated")
        
        return {
            "response": response,
            "intent": intent,
            "stage": result.get("stage", "complete")
        }
    
    def process_voice_input(self, duration=30):
        """
        Process voice input through the agent
        
        Args:
            duration: Maximum recording duration
            
        Returns:
            dict with response, transcription, and intent
        """
        if not self.voice_processor:
            return {
                "response": "Voice system not available. Please use text input.",
                "transcription": "",
                "intent": "error"
            }
        
        print(f"\n🎤 Starting voice input (max {duration}s)...")
        
        try:
            # 1. Record audio
            audio_file = self.voice_processor.record_audio(duration)
            
            # 2. Convert to text
            user_input = self.voice_processor.speech_to_text(audio_file)
            
            if not user_input or user_input.strip() == "":
                response = "I didn't catch that. Could you please repeat?"
                if self.voice_processor:
                    self.voice_processor.text_to_speech(response)
                
                return {
                    "response": response,
                    "transcription": "",
                    "intent": "unclear"
                }
            
            print(f"✓ Transcribed: '{user_input}'")
            
            # 3. Process through graph
            state = create_initial_state(
                user_input=user_input,
                user_profile=self.user_profile,
                is_first_message=self.is_first_interaction
            )
            
            result = self.graph.invoke(state)
            
            response = result["response"]
            intent = result.get("intent", "unknown")
            
            # 4. Speak the response
            self.voice_processor.text_to_speech(response)
            
            self.conversation_count += 1
            self.is_first_interaction = False
            
            return {
                "response": response,
                "transcription": user_input,
                "intent": intent,
                "stage": result.get("stage", "complete")
            }
        
        except Exception as e:
            print(f"❌ Voice processing error: {e}")
            error_response = "Sorry, I had trouble processing that. Please try again."
            
            if self.voice_processor:
                self.voice_processor.text_to_speech(error_response)
            
            return {
                "response": error_response,
                "transcription": "",
                "intent": "error"
            }
    
    def add_medication(self, medication):
        """Add medication to user profile"""
        self.user_profile["medications"].append(medication)
        print(f"✅ Added medication: {medication}")
    
    def add_condition(self, condition):
        """Add medical condition to user profile"""
        self.user_profile["conditions"].append(condition)
        print(f"✅ Added condition: {condition}")
    
    def get_profile(self):
        """Get current user profile"""
        return self.user_profile
    
    def reset_session(self):
        """Reset session state"""
        self.is_first_interaction = True
        self.conversation_count = 0
        print("🔄 Session reset")
    
    def run_interactive_cli(self):
        """
        Run interactive command-line interface
        """
        print("\n" + "="*60)
        print("🎙️  INTERACTIVE MODE")
        print("="*60)
        print("\nCommands:")
        print("  [ENTER]   - Start voice input")
        print("  [text]    - Send text message")
        print("  'profile' - View your profile")
        print("  'reset'   - Reset conversation")
        print("  'quit'    - Exit")
        print("="*60)
        
        # Start with greeting
        self.start_session()
        
        while True:
            try:
                user_cmd = input("\n💬 You (or press ENTER for voice): ").strip()
                
                # Handle commands
                if user_cmd.lower() == 'quit':
                    goodbye = "Take care of your health! Goodbye!"
                    print(f"\n🤖 {goodbye}")
                    if self.voice_processor:
                        self.voice_processor.text_to_speech(goodbye)
                    break
                
                elif user_cmd.lower() == 'profile':
                    print("\n📋 Your Medical Profile:")
                    print(f"  Conditions: {self.user_profile['conditions']}")
                    print(f"  Medications: {self.user_profile['medications']}")
                    continue
                
                elif user_cmd.lower() == 'reset':
                    self.reset_session()
                    self.start_session()
                    continue
                
                elif user_cmd == "":
                    # Voice input
                    if not VOICE_AVAILABLE:
                        print("❌ Voice not available. Please type your message.")
                        continue
                    
                    result = self.process_voice_input(duration=30)
                    print(f"\n🎤 You said: {result['transcription']}")
                    print(f"🤖 Response: {result['response']}")
                
                else:
                    # Text input
                    result = self.process_text_input(user_cmd)
                    print(f"\n🤖 {result['response']}")
                    
                    # Optionally speak in text mode too
                    if self.voice_processor:
                        self.voice_processor.text_to_speech(result['response'])
            
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                msg = str(e).lower()
                if "no working audio input device" in msg or "error querying device" in msg:
                    self.voice_processor.text_to_speech(
                        "I can't find a working microphone input. Please connect or enable a microphone and check Windows microphone privacy permissions, then try again."
                    )
                else:
                    self.voice_processor.text_to_speech("Sorry, I had a problem. Let's try again.")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 MEDICAL VOICE ASSISTANT")
    print("="*60)
    
    if not VOICE_AVAILABLE:
        print("\n⚠️  Voice packages not fully installed")
        print("For voice features, install:")
        print("pip install faster-whisper pyttsx3 sounddevice soundfile numpy")
        print("\nContinuing in TEXT MODE ONLY...")
        print("="*60)
    
    try:
        # Create agent
        agent = MedicalVoiceAgent()
        
        # Optional: Pre-load user profile
        agent.add_condition("Type 2 Diabetes")
        agent.add_medication("Metformin 500mg - twice daily")
        
        # Run interactive mode
        agent.run_interactive_cli()
    
    except KeyboardInterrupt:
        print("\n\n👋 Session ended. Take care!")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


"""
🎯 USAGE EXAMPLES:

1. VOICE MODE:
   You: [Press ENTER]
   🎤 Recording...
   You: "I have a headache"
   🤖 "I understand headaches can be uncomfortable..."

2. TEXT MODE:
   You: I have a headache
   🤖 "I understand headaches can be uncomfortable..."

3. COMMANDS:
   You: profile
   📋 Shows your medical profile
   
   You: reset
   🔄 Starts new conversation with greeting
   
   You: quit
   👋 Exits gracefully

CONVERSATION FLOW:

Session Start:
└─ 🎤 "Hello! I'm your Medical Voice Assistant..."
   └─ Shows menu (1. Symptoms, 2. Medications, 3. General, 4. Emergency)

User Response:
└─ "I have a headache"
   └─ 🧠 Detects intent: symptom_check
      └─ 🩺 Activates Symptom Agent
         └─ 🤖 Returns analysis
            └─ 🔊 Speaks response

Next Interaction:
└─ Can continue conversation or end
"""