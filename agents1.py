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
# STATE DEFINITION
# ============================================

class ConversationState(TypedDict):
    """State tracking conversation flow"""
    messages: Annotated[list, operator.add]
    user_input: str
    response: str
    stage: str  # greeting, waiting_for_choice, processing, complete
    intent: str  # symptom_check, medication, general_health, emergency
    user_profile: dict
    next_action: str
    conversation_history: list


# ============================================
# GREETING NODE - Shows Menu
# ============================================

def greeting_node(state: ConversationState) -> ConversationState:
    """
    Shows greeting and menu - WAITS for user choice
    """
    greeting_text = """Hello! I'm your Medical Voice Assistant. 

I can help you with:

1️⃣ SYMPTOM CHECK - Analyze your health symptoms
2️⃣ MEDICATION HELP - Manage your medications and reminders
3️⃣ HEALTH ADVICE - General health and wellness tips
4️⃣ EMERGENCY - Urgent medical guidance

Please tell me which service you need (say 1, 2, 3, 4 or describe what you want)."""
    
    state["response"] = greeting_text
    state["stage"] = "waiting_for_choice"  # IMPORTANT: Wait for choice
    state["messages"].append({
        "role": "assistant", 
        "content": greeting_text
    })
    
    print("🤖 Stage: GREETING → WAITING FOR USER CHOICE")
    return state


# ============================================
# CHOICE HANDLER - Processes User Selection
# ============================================

def handle_user_choice(state: ConversationState) -> ConversationState:
    """
    Listens to user's choice and classifies intent
    """
    user_input = state["user_input"].lower().strip()
    
    print(f"🎯 User said: '{user_input}'")
    
    # Check for number choices first
    if "1" in user_input or "first" in user_input or "one" in user_input:
        state["intent"] = "symptom_check"
        state["stage"] = "processing"
        print("✅ User selected: SYMPTOM CHECK (Option 1)")
    
    elif "2" in user_input or "second" in user_input or "two" in user_input:
        state["intent"] = "medication_reminder"
        state["stage"] = "processing"
        print("✅ User selected: MEDICATION HELP (Option 2)")
    
    elif "3" in user_input or "third" in user_input or "three" in user_input:
        state["intent"] = "general_health"
        state["stage"] = "processing"
        print("✅ User selected: HEALTH ADVICE (Option 3)")
    
    elif "4" in user_input or "fourth" in user_input or "four" in user_input or "emergency" in user_input:
        state["intent"] = "emergency"
        state["stage"] = "processing"
        print("✅ User selected: EMERGENCY (Option 4)")
    
    # Keyword matching for natural language
    elif any(word in user_input for word in ["symptom", "pain", "ache", "sick", "feel", "hurt", "headache", "fever", "cough", "tired"]):
        state["intent"] = "symptom_check"
        state["stage"] = "processing"
        print("✅ Detected: SYMPTOM CHECK (keyword match)")
    
    elif any(word in user_input for word in ["medicine", "medication", "pill", "drug", "forgot", "dose", "prescription", "metformin"]):
        state["intent"] = "medication_reminder"
        state["stage"] = "processing"
        print("✅ Detected: MEDICATION HELP (keyword match)")
    
    elif any(word in user_input for word in ["health", "food", "diet", "exercise", "sleep", "wellness", "tips", "eat", "nutrition"]):
        state["intent"] = "general_health"
        state["stage"] = "processing"
        print("✅ Detected: HEALTH ADVICE (keyword match)")
    
    else:
        # If still unclear, ask LLM
        classification_prompt = f"""User is choosing a medical service. Classify into ONE category:

    Options shown to user:
    1. symptom_check - Analyzing symptoms
    2. medication_reminder - Medication management
    3. general_health - General health advice
    4. emergency - Urgent situations

    User said: "{state['user_input']}"

    Respond with ONLY the category name (symptom_check, medication_reminder, general_health, or emergency)."""
        
        try:
            intent = llm.invoke(classification_prompt).content.strip().lower()
            
            if "symptom" in intent:
                state["intent"] = "symptom_check"
            elif "medication" in intent:
                state["intent"] = "medication_reminder"
            elif "general" in intent or "health" in intent:
                state["intent"] = "general_health"
            elif "emergency" in intent:
                state["intent"] = "emergency"
            else:
                state["intent"] = "unclear"
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            state["intent"] = "unclear"
        
        state["stage"] = "processing"
        print(f"✅ LLM classified: {state['intent']}")
    
    state["messages"].append({
        "role": "system",
        "content": f"Intent: {state['intent']}"
    })
    
    return state


# ============================================
# AGENT NODES
# ============================================

def symptom_agent(state: ConversationState) -> ConversationState:
    """Symptom analysis agent"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    # Check if user just selected option or has actual symptoms
    if any(x in user_input.lower() for x in ["1", "symptom", "first", "one"]) and len(user_input.split()) < 5:
        # User just selected the option, ask for symptoms
        state["response"] = "Great! I'll help you with your symptoms. Please describe what you're experiencing - what symptoms do you have?"
        state["stage"] = "waiting_for_details"
        print("🩺 Symptom Agent: Asking for details...")
        return state
    
    # User has provided symptoms
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
        print("🩺 Symptom Agent: Analysis complete")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I'm having trouble analyzing that. Could you describe your symptoms again?"
        state["stage"] = "waiting_for_details"
    
    return state


def medication_agent(state: ConversationState) -> ConversationState:
    """Medication management agent"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    # Check if user just selected option
    if any(x in user_input.lower() for x in ["2", "medication", "second", "two"]) and len(user_input.split()) < 5:
        state["response"] = "I can help you with your medications! What do you need help with? (e.g., 'I forgot to take my medicine' or 'When should I take my medication?')"
        state["stage"] = "waiting_for_details"
        print("💊 Medication Agent: Asking for details...")
        return state
    
    # User has provided medication query
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
        print("💊 Medication Agent: Response complete")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I understand medication management can be challenging. How can I help you specifically?"
        state["stage"] = "waiting_for_details"
    
    return state


def health_advisor_agent(state: ConversationState) -> ConversationState:
    """General health advisor agent"""
    user_input = state["user_input"]
    
    # Check if user just selected option
    if any(x in user_input.lower() for x in ["3", "health", "third", "three", "advice"]) and len(user_input.split()) < 5:
        state["response"] = "I'd be happy to provide health advice! What would you like to know about? (e.g., 'healthy foods', 'exercise tips', 'better sleep')"
        state["stage"] = "waiting_for_details"
        print("🏃 Health Advisor: Asking for details...")
        return state
    
    # User has provided health question
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
        print("🏃 Health Advisor: Response complete")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "That's a great question! Could you be more specific about what you'd like to know?"
        state["stage"] = "waiting_for_details"
    
    return state


def emergency_agent(state: ConversationState) -> ConversationState:
    """Emergency handler"""
    response = """🚨 EMERGENCY ALERT!

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
    """Handle unclear responses"""
    response = """I didn't quite understand. Please choose one of these options:

1️⃣ SYMPTOM CHECK - If you're feeling unwell
2️⃣ MEDICATION HELP - For medication questions
3️⃣ HEALTH ADVICE - For general health tips
4️⃣ EMERGENCY - If it's urgent

You can say the number (like "1") or describe what you need."""
    
    state["response"] = response
    state["stage"] = "waiting_for_choice"
    print("❓ Unclear - showing menu again")
    return state


# ============================================
# ROUTING FUNCTIONS
# ============================================

def route_from_greeting(state: ConversationState) -> Literal["end"]:
    """After greeting, END and wait for user input"""
    return "end"


def route_from_choice(state: ConversationState) -> Literal["symptom", "medication", "health", "emergency", "unclear"]:
    """Route based on user's choice"""
    intent = state.get("intent", "unclear")
    
    routing = {
        "symptom_check": "symptom",
        "medication_reminder": "medication",
        "general_health": "health",
        "emergency": "emergency"
    }
    
    return routing.get(intent, "unclear")


def route_after_agent(state: ConversationState) -> Literal["end"]:
    """After agent completes, end turn"""
    return "end"


# ============================================
# BUILD GRAPH
# ============================================

def create_conversational_graph():
    """Build the conversation flow graph"""
    
    workflow = StateGraph(ConversationState)
    
    # Add all nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("choice_handler", handle_user_choice)
    workflow.add_node("symptom", symptom_agent)
    workflow.add_node("medication", medication_agent)
    workflow.add_node("health", health_advisor_agent)
    workflow.add_node("emergency", emergency_agent)
    workflow.add_node("unclear", unclear_handler)
    
    # Set entry point
    workflow.set_entry_point("greeting")
    
    # Greeting → END (shows menu, waits for user)
    workflow.add_conditional_edges(
        "greeting",
        route_from_greeting,
        {"end": END}
    )
    
    # Choice handler → Routes to agents
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
    
    # All agents → END
    workflow.add_conditional_edges("symptom", route_after_agent, {"end": END})
    workflow.add_conditional_edges("medication", route_after_agent, {"end": END})
    workflow.add_conditional_edges("health", route_after_agent, {"end": END})
    workflow.add_conditional_edges("emergency", route_after_agent, {"end": END})
    workflow.add_conditional_edges("unclear", route_after_agent, {"end": END})
    
    return workflow.compile()


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_initial_state(user_input="", user_profile=None, is_first_message=True):
    """Create initial state"""
    return {
        "messages": [],
        "user_input": user_input,
        "response": "",
        "stage": "greeting" if is_first_message else "waiting_for_choice",
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
    print("TESTING INTERACTIVE MENU FLOW")
    print("="*60)
    
    graph = create_conversational_graph()
    
    # Test 1: Greeting
    print("\n📝 Test 1: Greeting (shows menu)")
    print("-" * 60)
    state = create_initial_state(is_first_message=True)
    result = graph.invoke(state)
    print(f"Bot: {result['response']}")
    print(f"Stage: {result['stage']}")
    
    # Test 2: User selects option 1
    print("\n📝 Test 2: User selects '1' (symptoms)")
    print("-" * 60)
    state = create_initial_state(user_input="1", is_first_message=False)
    state["stage"] = "waiting_for_choice"
    
    # First invoke choice_handler
    from langgraph.graph import START
    workflow = StateGraph(ConversationState)
    workflow.add_node("choice_handler", handle_user_choice)
    workflow.add_node("symptom", symptom_agent)
    workflow.set_entry_point("choice_handler")
    workflow.add_conditional_edges(
        "choice_handler",
        route_from_choice,
        {"symptom": "symptom", "medication": "medication", "health": "health", "emergency": "emergency", "unclear": "unclear"}
    )
    workflow.add_conditional_edges("symptom", route_after_agent, {"end": END})
    
    test_graph = workflow.compile()
    result = test_graph.invoke(state)
    print(f"Bot: {result['response']}")
    print(f"Intent: {result['intent']}")


"""
🎯 NEW FLOW:

SESSION START:
User opens app/website
    ↓
Bot shows greeting + menu
    ↓
Bot WAITS (doesn't record yet!)
    ↓
User selects option (1, 2, 3, 4)
    ↓
Specific agent activates
    ↓
Agent asks for details if needed
    ↓
User provides details
    ↓
Agent responds
    ↓
Session complete

EXAMPLE CONVERSATION:

Bot: "Hello! I can help with:
     1. Symptoms
     2. Medications
     3. Health advice
     4. Emergency
     
     What do you need?"

User: "1" or "I need symptom check"

Bot: "Great! Please describe your symptoms."

User: "I have a headache"

Bot: "I understand headaches can be..."

KEY FEATURES:
✅ Menu shows FIRST
✅ Bot WAITS for user choice
✅ Agent activates ONLY after choice
✅ Can handle both numbers and natural language
✅ Asks for details when needed
"""