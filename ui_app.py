"""
Enhanced Medical Voice Assistant UI - Updated for New Flow
Beautiful interface with proper conversation stages
"""

from fastapi import FastAPI
import gradio as gr
from main_improved import MedicalVoiceAgent
from voice_impl1 import VOICE_AVAILABLE

# Create FastAPI app
app = FastAPI(title="Medical Voice Assistant")

# Global agent instance
agent = None

# Session state
session_state = {
    "initialized": False,
    "greeting_shown": False,
    "total_interactions": 0,
    "current_stage": "idle"
}


def initialize_agent():
    """Initialize agent on first use"""
    global agent
    if agent is None and VOICE_AVAILABLE:
        agent = MedicalVoiceAgent()
        # Pre-load sample profile
        agent.add_condition("Type 2 Diabetes")
        agent.add_medication("Metformin 500mg - twice daily")
    return agent


def get_greeting():
    """Get initial greeting"""
    global agent, session_state
    
    if not VOICE_AVAILABLE:
        return [[None, "❌ Voice system not available. Please install required packages."]]
    
    initialize_agent()
    
    if not session_state["greeting_shown"]:
        greeting = agent.start_session()
        session_state["greeting_shown"] = True
        session_state["current_stage"] = "menu"
        return [[None, greeting]]
    
    return []


def get_status_info():
    """Get system status"""
    if not VOICE_AVAILABLE:
        return """
### 🤖 Status: ❌ Offline

**Issue:** Voice packages not installed

**Install:**
```bash
pip install faster-whisper pyttsx3 sounddevice soundfile numpy
```
""", "❌ Offline"
    
    status = f"""
### 🤖 Status: ✅ Online

**Components:**
- 🧠 LLM: ✅ Ollama (llama3.1)
- 🎤 STT: ✅ Whisper (base)
- 🔊 TTS: ✅ pyttsx3

**Session:**
- 💬 Interactions: {session_state['total_interactions']}
- 📍 Stage: {session_state['current_stage']}
"""
    
    return status, "✅ Online"


def format_profile(conditions, medications):
    """Format profile display"""
    cond_list = "\n".join([f"  • {c}" for c in conditions]) if conditions else "  None"
    med_list = "\n".join([f"  • {m}" for m in medications]) if medications else "  None"
    
    return f"""
**🏥 Medical Conditions:**
{cond_list}

**💊 Medications:**
{med_list}
"""


def handle_text_message(message, history, conditions, medications):
    """Handle text input"""
    global agent, session_state
    
    if not message or not message.strip():
        return history, "", get_status_info()[0], format_profile(conditions, medications)
    
    if not VOICE_AVAILABLE:
        history = history + [[message, "❌ System not available"]]
        return history, "", get_status_info()[0], format_profile(conditions, medications)
    
    initialize_agent()
    
    # Update stage
    session_state["current_stage"] = "processing"
    
    # Process through agent
    result = agent.process_text_input(message)
    
    # Update session
    session_state["total_interactions"] += 1
    session_state["current_stage"] = result.get("stage", "complete")
    
    # Update history
    history = history + [[message, result["response"]]]
    
    return history, "", get_status_info()[0], format_profile(conditions, medications)


def handle_voice_input(history, conditions, medications):
    """Handle voice input"""
    global agent, session_state
    
    if not VOICE_AVAILABLE:
        history = history + [["🎤 Voice", "❌ Voice system not available"]]
        return history, get_status_info()[0], format_profile(conditions, medications)
    
    initialize_agent()
    
    # Update stage
    session_state["current_stage"] = "recording"
    
    try:
        # Process voice
        result = agent.process_voice_input(duration=30)
        
        # Update session
        session_state["total_interactions"] += 1
        session_state["current_stage"] = result.get("stage", "complete")
        
        # Update history with transcription and response
        transcription = result.get("transcription", "Voice input")
        history = history + [[f"🎤 You: {transcription}", result["response"]]]
    
    except Exception as e:
        history = history + [["🎤 Error", f"Voice processing failed: {str(e)}"]]
        session_state["current_stage"] = "error"
    
    return history, get_status_info()[0], format_profile(conditions, medications)


def add_profile_item(condition, medication, conditions, medications):
    """Add to profile"""
    global agent
    
    if not VOICE_AVAILABLE:
        return conditions, medications, format_profile(conditions, medications), "❌ System not available"
    
    initialize_agent()
    
    updated_cond = list(conditions or [])
    updated_meds = list(medications or [])
    
    if condition and condition.strip():
        agent.add_condition(condition)
        updated_cond.append(condition)
    
    if medication and medication.strip():
        agent.add_medication(medication)
        updated_meds.append(medication)
    
    return updated_cond, updated_meds, format_profile(updated_cond, updated_meds), "✅ Profile updated"


def reset_conversation(conditions, medications):
    """Reset conversation to greeting"""
    global agent, session_state
    
    if not VOICE_AVAILABLE:
        return [], get_status_info()[0]
    
    initialize_agent()
    
    # Reset session state
    session_state["greeting_shown"] = False
    session_state["total_interactions"] = 0
    session_state["current_stage"] = "idle"
    
    # Reset agent
    agent.reset_session()
    
    # Get new greeting
    greeting_history = get_greeting()
    
    return greeting_history, get_status_info()[0]


# ============================================
# BUILD UI
# ============================================

custom_css = """
.gradio-container {
    max-width: 1400px !important;
}

.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.status-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.profile-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.chat-card {
    border: 3px solid #667eea;
    border-radius: 12px;
    padding: 15px;
    background: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.voice-btn {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    padding: 15px 30px !important;
    border-radius: 30px !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
}

.voice-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.3) !important;
}

.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: bold !important;
}

.reset-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
}
"""

def create_ui():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Medical Voice Assistant") as demo:
        
        # Header
        with gr.Row(elem_classes="header-gradient"):
            gr.Markdown("""
# 🏥 Medical Voice Assistant
## AI-Powered Health Companion with LangGraph

**Intelligent conversation flow:** Greeting → Menu → Agent Selection → Response

Get personalized help with symptoms, medications, and health questions using advanced AI agents.
""")
        
        # Main layout
        with gr.Row():
            # Left: Chat
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                
                chat = gr.Chatbot(
                    value=[],
                    height=500,
                    show_label=False,
                    elem_classes="chat-card"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your health question or press ENTER for voice...",
                        show_label=False,
                        scale=4
                    )
                    send = gr.Button("Send 📤", scale=1, elem_classes="send-btn")
                
                with gr.Row():
                    voice = gr.Button(
                        "🎤 Voice Input (30s)",
                        size="lg",
                        elem_classes="voice-btn"
                    )
                    reset = gr.Button(
                        "🔄 Reset Conversation",
                        size="lg",
                        elem_classes="reset-btn"
                    )
            
            # Right: Status & Profile
            with gr.Column(scale=1):
                # Status
                with gr.Group(elem_classes="status-card"):
                    gr.Markdown("### 🤖 System Status")
                    status = gr.Markdown(
                        get_status_info()[0],
                        show_label=False
                    )
                    refresh = gr.Button("🔄 Refresh", size="sm")
                
                # Profile
                with gr.Group(elem_classes="profile-card"):
                    gr.Markdown("### 👤 Your Profile")
                    profile_display = gr.Markdown(
                        format_profile([], []),
                        show_label=False
                    )
                    
                    with gr.Accordion("➕ Manage Profile", open=False):
                        cond_input = gr.Textbox(
                            label="Add Condition",
                            placeholder="e.g., Hypertension"
                        )
                        med_input = gr.Textbox(
                            label="Add Medication",
                            placeholder="e.g., Aspirin 81mg - daily"
                        )
                        add_btn = gr.Button("Add", size="sm")
                        profile_msg = gr.Textbox(
                            show_label=False,
                            interactive=False
                        )
        
        # Info section
        with gr.Accordion("ℹ️ How It Works", open=False):
            gr.Markdown("""
### 🧠 Conversation Flow

**Stage 1: Greeting**
- System welcomes you
- Shows menu of services
- Waits for your choice

**Stage 2: Intent Detection**
- Analyzes your input
- Fast keyword matching
- LLM fallback for unclear cases

**Stage 3: Agent Selection**
- Routes to specialized agent:
  - 🩺 **Symptom Analyzer** - Health concerns
  - 💊 **Medication Manager** - Prescriptions & reminders
  - 🏃 **Health Advisor** - General wellness
  - 🚨 **Emergency Handler** - Urgent situations

**Stage 4: Response**
- Agent processes your query
- Generates personalized response
- Speaks back (if voice mode)

### 🎯 Features

✅ **Natural Conversation** - Flows like talking to a real assistant  
✅ **Multi-Agent System** - Specialized experts for each domain  
✅ **Voice Enabled** - Speak and listen naturally  
✅ **Context Aware** - Remembers your medical profile  
✅ **Privacy First** - All processing happens locally  
""")
        
        # State
        conditions = gr.State([])
        medications = gr.State([])
        
        # Events
        
        # Text input
        msg.submit(
            handle_text_message,
            inputs=[msg, chat, conditions, medications],
            outputs=[chat, msg, status, profile_display]
        )
        
        send.click(
            handle_text_message,
            inputs=[msg, chat, conditions, medications],
            outputs=[chat, msg, status, profile_display]
        )
        
        # Voice input
        voice.click(
            handle_voice_input,
            inputs=[chat, conditions, medications],
            outputs=[chat, status, profile_display]
        )
        
        # Reset
        reset.click(
            reset_conversation,
            inputs=[conditions, medications],
            outputs=[chat, status]
        )
        
        # Refresh status
        refresh.click(
            lambda: get_status_info()[0],
            outputs=[status]
        )
        
        # Profile management
        add_btn.click(
            add_profile_item,
            inputs=[cond_input, med_input, conditions, medications],
            outputs=[conditions, medications, profile_display, profile_msg]
        )
        
        # Initialize with greeting
        demo.load(
            get_greeting,
            outputs=[chat]
        )
        
        demo.load(
            lambda: get_status_info()[0],
            outputs=[status]
        )
    
    return demo


# ============================================
# Launch
# ============================================

demo = create_ui()

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 MEDICAL VOICE ASSISTANT UI")
    print("="*60)
    print("📍 URL: http://localhost:8000")
    print("="*60)
    print("\n✨ Features:")
    print("  ✅ Conversational flow (Greeting → Menu → Help)")
    print("  ✅ Voice & text input")
    print("  ✅ Multi-agent routing")
    print("  ✅ Real-time status")
    print("  ✅ Profile management")
    print("="*60 + "\n")
    
    uvicorn.run("ui_app:app", host="0.0.0.0", port=8000, reload=True)