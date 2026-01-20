"""
FIXED MAIN.PY - Interactive Menu System

main.py
"""

print("🔥 MEDICAL VOICE ASSISTANT STARTING... 🔥")

from agents1 import create_conversational_graph, create_initial_state, handle_user_choice
from voice_impl1 import VoiceProcessor, VOICE_AVAILABLE


class MedicalVoiceAgent:
    """
    Medical Voice Agent with Interactive Menu
    
    Flow:
    1. Show greeting + menu → WAIT
    2. User selects service → Process choice
    3. Agent asks for details if needed → WAIT
    4. User provides info → Agent responds
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
            "medications": ["Metformin 500mg - twice daily"],
            "conditions": ["Type 2 Diabetes"]
        }
        
        # Conversation tracking
        self.current_stage = "greeting"  # greeting, waiting_for_choice, waiting_for_details, complete
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
    
    def show_greeting(self):
        """
        Show initial greeting and menu
        Returns: greeting message
        """
        print("\n🎬 Showing menu...")
        
        # Create state for greeting
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
        
        # Update stage
        self.current_stage = result.get("stage", "waiting_for_choice")
        
        return greeting
    
    def process_user_choice(self, user_input, input_type="text"):
        """
        Process user's menu choice
        
        Args:
            user_input: User's choice (number or text)
            input_type: "text" or "voice"
            
        Returns:
            dict with response, intent, and stage
        """
        print(f"\n📝 Processing user choice: '{user_input}'")
        
        # Create state
        state = create_initial_state(
            user_input=user_input,
            user_profile=self.user_profile,
            is_first_message=False
        )
        state["stage"] = "waiting_for_choice"
        
        # First, classify the choice
        choice_state = handle_user_choice(state)
        intent = choice_state.get("intent", "unclear")
        
        print(f"🎯 Intent detected: {intent}")
        
        # Now route to appropriate agent
        # Create a separate graph invocation starting from choice_handler
        from langgraph.graph import StateGraph, END
        from agents1 import (
            ConversationState,
            symptom_agent,
            medication_agent,
            health_advisor_agent,
            emergency_agent,
            unclear_handler,
            route_from_choice,
            route_after_agent,
        )
        
        # Build mini-graph for this turn
        workflow = StateGraph(ConversationState)
        workflow.add_node("choice_handler", handle_user_choice)
        workflow.add_node("symptom", symptom_agent)
        workflow.add_node("medication", medication_agent)
        workflow.add_node("health", health_advisor_agent)
        workflow.add_node("emergency", emergency_agent)
        workflow.add_node("unclear", unclear_handler)
        
        workflow.set_entry_point("choice_handler")
        
        workflow.add_conditional_edges(
            "choice_handler",
            route_from_choice,
            {
                "symptom": "symptom",
                "medication": "medication",
                "health": "health",
                "emergency": "emergency",
                "unclear": "unclear"
            }
        )
        
        for node in ["symptom", "medication", "health", "emergency", "unclear"]:
            workflow.add_conditional_edges(node, route_after_agent, {"end": END})
        
        mini_graph = workflow.compile()
        
        # Run through graph
        result = mini_graph.invoke(state)
        
        response = result["response"]
        new_stage = result.get("stage", "complete")
        
        # Update internal state
        self.current_stage = new_stage
        self.conversation_count += 1
        
        print(f"🤖 Stage: {new_stage}")
        
        # Speak response whenever voice is available (demo-friendly)
        if self.voice_processor:
            self.voice_processor.text_to_speech(response)
        
        return {
            "response": response,
            "intent": intent,
            "stage": new_stage
        }
    
    def process_text_input(self, user_input):
        """
        Process text input through the agent
        
        Args:
            user_input: User's text message
            
        Returns:
            dict with response and intent
        """
        # If we're at greeting stage, show menu first
        if self.current_stage == "greeting":
            greeting = self.show_greeting()
            # Return greeting and wait for next input
            return {
                "response": greeting,
                "intent": "greeting",
                "stage": "waiting_for_choice"
            }
        
        # Otherwise, process the choice/input
        return self.process_user_choice(user_input, input_type="text")
    
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
                "intent": "error",
                "stage": "error"
            }
        
        # If we're at greeting stage, show menu first
        if self.current_stage == "greeting":
            greeting = self.show_greeting()
            # Now wait for voice input
            print("\n🎤 Now listening for your choice...")
        
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
                    "intent": "unclear",
                    "stage": self.current_stage
                }
            
            print(f"✓ Transcribed: '{user_input}'")
            
            # 3. Process the input
            result = self.process_user_choice(user_input, input_type="voice")
            
            return {
                "response": result["response"],
                "transcription": user_input,
                "intent": result["intent"],
                "stage": result["stage"]
            }
        
        except Exception as e:
            print(f"❌ Voice processing error: {e}")
            error_response = "Sorry, I had trouble processing that. Please try again."
            
            if self.voice_processor:
                self.voice_processor.text_to_speech(error_response)
            
            return {
                "response": error_response,
                "transcription": "",
                "intent": "error",
                "stage": "error"
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
        """Reset session to greeting"""
        self.current_stage = "greeting"
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
        greeting = self.show_greeting()
        print(f"\n🤖 {greeting}")
        
        while True:
            try:
                user_cmd = input("\n💬 Your choice (or press ENTER for voice): ").strip()
                
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
                    greeting = self.show_greeting()
                    print(f"\n🤖 {greeting}")
                    continue
                
                elif user_cmd == "":
                    # Voice input
                    if not VOICE_AVAILABLE:
                        print("❌ Voice not available. Please type your message.")
                        continue
                    
                    result = self.process_voice_input(duration=30)
                    
                    if result["transcription"]:
                        print(f"\n🎤 You said: {result['transcription']}")
                    print(f"\n🤖 Response: {result['response']}")
                    
                    # If waiting for more details, continue loop
                    if result["stage"] == "waiting_for_details":
                        print("\n💡 Agent is waiting for more information...")
                
                else:
                    # Text input
                    result = self.process_text_input(user_cmd)
                    print(f"\n🤖 Response: {result['response']}")
                    
                    # If waiting for more details, continue loop
                    if result["stage"] == "waiting_for_details":
                        print("\n💡 Agent is waiting for more information...")
            
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


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
        
        # Run interactive mode
        agent.run_interactive_cli()
    
    except KeyboardInterrupt:
        print("\n\n👋 Session ended. Take care!")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


"""
🎯 NEW FLOW EXAMPLE:

========================
SESSION START:
========================

Bot: 🔊 "Hello! I'm your Medical Voice Assistant.

I can help you with:

1️⃣ SYMPTOM CHECK - Analyze your health symptoms
2️⃣ MEDICATION HELP - Manage your medications
3️⃣ HEALTH ADVICE - General health tips
4️⃣ EMERGENCY - Urgent medical guidance

Please tell me which service you need."

[BOT WAITS - doesn't record automatically]

========================
USER SELECTS OPTION:
========================

User types: "1"
OR
User speaks: "I need symptom check"

Bot: 🔊 "Great! I'll help with your symptoms. 
     Please describe what you're experiencing."

[BOT WAITS again for details]

========================
USER PROVIDES DETAILS:
========================

User: "I have a headache and feel tired"

Bot: 🔊 "I understand headaches can be uncomfortable.
     This could be from stress or dehydration.
     Try resting and drinking water.
     If it persists, see a doctor.
     
     Remember, this is not a diagnosis."

========================
SESSION COMPLETE
========================

User can type 'reset' to start over
Or 'quit' to exit
"""