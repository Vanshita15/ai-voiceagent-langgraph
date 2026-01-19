"""
IMPROVED MEDICAL VOICE AGENT - PROPER CONVERSATION FLOW
Complete redesign with stages: Greeting → Ask → Process → Help

agents_improved.py
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

# LLM import
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1", temperature=0.7)
    print("✓ Using Ollama (llama3.1)")
except ImportError:
    print("❌ Install: pip install langchain-ollama")
    exit(1)


# ============================================
# ENHANCED STATE WITH CONVERSATION STAGES
# ============================================

class ConversationState(TypedDict):
    """Enhanced state tracking conversation flow"""
    # Core fields
    messages: Annotated[list, operator.add]
    user_input: str
    response: str
    
    # Conversation management
    stage: str  # greeting, menu, processing, helping, complete
    intent: str  # symptom_check, medication, general_health, emergency
    
    # User data
    user_profile: dict
    
    # Flow control
    next_action: str
    conversation_history: list


# ============================================
# STAGE 1: GREETING NODE
# ============================================

def greeting_node(state: ConversationState) -> ConversationState:
    """
    First interaction - greet user warmly
    """
    greeting_text = """Hello! I'm your Medical Voice Assistant. 

I can help you with:
1️⃣ Analyzing your symptoms
2️⃣ Managing your medications  
3️⃣ General health questions
4️⃣ Emergency guidance

What would you like help with today?"""
    
    state["response"] = greeting_text
    state["stage"] = "menu"  # Move to menu stage
    state["messages"].append({
        "role": "assistant", 
        "content": greeting_text
    })
    
    print("🤖 Stage: GREETING → MENU")
    return state


# ============================================
# STAGE 2: MENU SELECTION NODE
# ============================================

def menu_selection_node(state: ConversationState) -> ConversationState:
    """
    Listen to user choice and classify intent
    """
    user_input = state["user_input"].lower()
    
    # Simple keyword matching first (faster)
    if any(word in user_input for word in ["symptom", "pain", "ache", "sick", "feel", "hurt", "headache", "fever", "cough"]):
        state["intent"] = "symptom_check"
        state["stage"] = "processing"
    
    elif any(word in user_input for word in ["medicine", "medication", "pill", "drug", "forgot", "dose", "prescription"]):
        state["intent"] = "medication_reminder"
        state["stage"] = "processing"
    
    elif any(word in user_input for word in ["health", "food", "diet", "exercise", "sleep", "wellness", "tips"]):
        state["intent"] = "general_health"
        state["stage"] = "processing"
    
    elif any(word in user_input for word in ["emergency", "urgent", "chest pain", "can't breathe", "bleeding"]):
        state["intent"] = "emergency"
        state["stage"] = "processing"
    
    else:
        # Use LLM for unclear cases
        classification_prompt = f"""Classify user intent into ONE category:

Categories:
- symptom_check: User mentions symptoms, pain, sickness
- medication_reminder: User asks about medication, pills, prescriptions
- general_health: General health questions, diet, exercise
- emergency: Urgent medical situation

User said: "{state['user_input']}"

Respond with ONLY the category name."""
        
        try:
            intent = llm.invoke(classification_prompt).content.strip().lower()
            
            if "symptom" in intent:
                state["intent"] = "symptom_check"
            elif "medication" in intent:
                state["intent"] = "medication_reminder"
            elif "general" in intent:
                state["intent"] = "general_health"
            elif "emergency" in intent:
                state["intent"] = "emergency"
            else:
                state["intent"] = "unclear"
        except:
            state["intent"] = "unclear"
        
        state["stage"] = "processing"
    
    state["messages"].append({
        "role": "system",
        "content": f"Intent detected: {state['intent']}"
    })
    
    print(f"🧠 Intent: {state['intent']} | Stage: MENU → PROCESSING")
    return state


# ============================================
# STAGE 3: AGENT HANDLERS
# ============================================

def symptom_agent(state: ConversationState) -> ConversationState:
    """Handle symptom analysis"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    conditions = user_profile.get("conditions", [])
    context = f"Patient's known conditions: {', '.join(conditions)}" if conditions else "No known conditions"
    
    prompt = f"""You are a caring medical assistant analyzing symptoms.

{context}

Patient says: "{user_input}"

Respond conversationally in 3-4 sentences:
1. Acknowledge their concern with empathy
2. Explain possible common causes
3. Suggest self-care if appropriate
4. Advise when to see a doctor

IMPORTANT: Keep response SHORT and CONVERSATIONAL for voice output.
End with: "Remember, this is not a diagnosis. Please consult a doctor if symptoms worsen."

Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["stage"] = "complete"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I'm having trouble analyzing that. Could you describe your symptoms again?"
        state["stage"] = "menu"
    
    print("🩺 Symptom Agent activated")
    return state


def medication_agent(state: ConversationState) -> ConversationState:
    """Handle medication management"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    meds = user_profile.get("medications", [])
    meds_context = f"Current medications: {', '.join(meds)}" if meds else "No medications on record"
    
    prompt = f"""You are a supportive medication management assistant.

{meds_context}

Patient says: "{user_input}"

Respond warmly in 2-3 sentences:
1. Acknowledge their situation
2. Provide helpful medication advice
3. Encourage adherence

Keep SHORT and SUPPORTIVE for voice.

Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["stage"] = "complete"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I understand medication management can be challenging. How can I help you specifically?"
        state["stage"] = "menu"
    
    print("💊 Medication Agent activated")
    return state


def health_advisor_agent(state: ConversationState) -> ConversationState:
    """Handle general health questions"""
    user_input = state["user_input"]
    
    prompt = f"""You are a knowledgeable health advisor.

Question: "{user_input}"

Provide a brief, practical answer in 2-3 sentences:
1. Answer their question directly
2. Give actionable tips
3. Keep it simple

SHORT and CLEAR for voice.

Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["stage"] = "complete"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "That's a great question! Could you be more specific about what you'd like to know?"
        state["stage"] = "menu"
    
    print("🏃 Health Advisor Agent activated")
    return state


def emergency_agent(state: ConversationState) -> ConversationState:
    """Handle emergency situations"""
    
    response = """⚠️ EMERGENCY ALERT ⚠️

This sounds like a medical emergency!

PLEASE DO THIS NOW:
1. Call 911 (or your local emergency number) IMMEDIATELY
2. Do NOT drive yourself
3. Stay calm and wait for help

This requires urgent medical attention. Please get help right away!"""
    
    state["response"] = response
    state["stage"] = "complete"
    
    print("🚨 EMERGENCY AGENT ACTIVATED!")
    return state


def unclear_handler(state: ConversationState) -> ConversationState:
    """Handle unclear intent"""
    
    response = """I didn't quite understand that. 

Could you tell me which area you need help with?

1️⃣ Symptoms or health concerns
2️⃣ Medication help
3️⃣ General health questions
4️⃣ Emergency situation

What would you like?"""
    
    state["response"] = response
    state["stage"] = "menu"  # Back to menu
    
    print("❓ Unclear intent - returning to menu")
    return state


# ============================================
# ROUTING LOGIC
# ============================================

def route_from_greeting(state: ConversationState) -> Literal["menu_selection"]:
    """After greeting, always go to menu"""
    return "menu_selection"


def route_from_menu(state: ConversationState) -> Literal["symptom", "medication", "health", "emergency", "unclear"]:
    """Route based on detected intent"""
    intent = state.get("intent", "unclear")
    
    if intent == "symptom_check":
        return "symptom"
    elif intent == "medication_reminder":
        return "medication"
    elif intent == "general_health":
        return "health"
    elif intent == "emergency":
        return "emergency"
    else:
        return "unclear"


def should_continue(state: ConversationState) -> Literal["menu_selection", "end"]:
    """Decide if conversation should continue"""
    stage = state.get("stage", "complete")
    
    # If we're back at menu stage, continue conversation
    if stage == "menu":
        return "menu_selection"
    
    # Otherwise end this turn
    return "end"


# ============================================
# BUILD THE GRAPH
# ============================================

def create_conversational_graph():
    """Build LangGraph with proper conversation flow"""
    
    workflow = StateGraph(ConversationState)
    
    # Add all nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("menu_selection", menu_selection_node)
    workflow.add_node("symptom", symptom_agent)
    workflow.add_node("medication", medication_agent)
    workflow.add_node("health", health_advisor_agent)
    workflow.add_node("emergency", emergency_agent)
    workflow.add_node("unclear", unclear_handler)
    
    # Set entry point
    workflow.set_entry_point("greeting")
    
    # Greeting always goes to menu
    workflow.add_edge("greeting", "menu_selection")
    
    # From menu, route to appropriate agent
    workflow.add_conditional_edges(
        "menu_selection",
        route_from_menu,
        {
            "symptom": "symptom",
            "medication": "medication",
            "health": "health",
            "emergency": "emergency",
            "unclear": "unclear"
        }
    )
    
    # All agents can either continue or end
    for agent in ["symptom", "medication", "health", "emergency"]:
        workflow.add_conditional_edges(
            agent,
            should_continue,
            {
                "menu_selection": "menu_selection",
                "end": END
            }
        )
    
    # Unclear goes back to menu
    workflow.add_conditional_edges(
        "unclear",
        should_continue,
        {
            "menu_selection": "menu_selection",
            "end": END
        }
    )
    
    return workflow.compile()


# ============================================
# HELPER FUNCTION FOR INITIAL STATE
# ============================================

def create_initial_state(user_input="", user_profile=None, is_first_message=True):
    """Create initial state for graph"""
    return {
        "messages": [],
        "user_input": user_input,
        "response": "",
        "stage": "greeting" if is_first_message else "menu",
        "intent": "",
        "user_profile": user_profile or {"medications": [], "conditions": []},
        "next_action": "",
        "conversation_history": []
    }


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING CONVERSATIONAL FLOW")
    print("="*60)
    
    graph = create_conversational_graph()
    
    # Test 1: Initial greeting
    print("\n📝 Test 1: Initial Greeting")
    print("-" * 60)
    state = create_initial_state(is_first_message=True)
    result = graph.invoke(state)
    print(f"Assistant: {result['response']}")
    
    # Test 2: Symptom query
    print("\n📝 Test 2: Symptom Query")
    print("-" * 60)
    state = create_initial_state(
        user_input="I have a headache and feel tired",
        is_first_message=False
    )
    result = graph.invoke(state)
    print(f"Detected Intent: {result['intent']}")
    print(f"Assistant: {result['response']}")
    
    # Test 3: Medication query
    print("\n📝 Test 3: Medication Query")
    print("-" * 60)
    state = create_initial_state(
        user_input="I forgot to take my medicine today",
        user_profile={"medications": ["Metformin 500mg"], "conditions": []},
        is_first_message=False
    )
    result = graph.invoke(state)
    print(f"Detected Intent: {result['intent']}")
    print(f"Assistant: {result['response']}")


"""
🎯 CONVERSATION FLOW EXPLANATION:

STAGE 1: GREETING
├─ User enters/starts
├─ greeting_node() activates
├─ Shows menu with options
└─ Stage: greeting → menu

STAGE 2: MENU SELECTION
├─ User speaks their choice
├─ menu_selection_node() classifies intent
├─ Fast keyword matching + LLM fallback
└─ Stage: menu → processing

STAGE 3: AGENT ACTIVATION
├─ Router sends to appropriate agent:
│  ├─ symptom_agent() → Health analysis
│  ├─ medication_agent() → Med management
│  ├─ health_advisor_agent() → General advice
│  └─ emergency_agent() → Urgent care
└─ Stage: processing → complete

STAGE 4: COMPLETION
├─ Agent returns response
├─ Can loop back to menu if needed
└─ Stage: complete → [end or menu]

VISUAL FLOW:
User → Greeting → Menu → Intent Detection → Agent Selection → Response → [Loop or End]

KEY FEATURES:
✅ Always greets first
✅ Shows clear menu options
✅ Fast intent detection (keywords + LLM)
✅ Specialized agents for each task
✅ Can continue conversation
✅ Emergency detection
"""