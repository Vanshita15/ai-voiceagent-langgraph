from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator
from datetime import datetime
import json
import os
from voice_impl1 import VoiceProcessor


# LLM import
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1", temperature=0.7)
    print("✓ Using Ollama (llama3.1)")
except ImportError:
    print("❌ Install: pip install langchain-ollama")
    print("Then run: ollama pull llama3.1")
    exit(1)


class MedicalAssistantState(TypedDict):
    """Enhanced state with conversation tracking"""
    conversation_stage: str  # greeting, asking_need, helping, followup
    messages: Annotated[list, operator.add]
    user_input: str
    intent: str
    user_profile: dict
    response: str
    context: dict 

def greeting_node(state: MedicalAssistantState) -> MedicalAssistantState:
    """
    NODE 1: Greet the user properly
    """
    stage = state.get("conversation_stage", "greeting")
    
    if stage == "greeting":
        # First interaction
        greeting = """Hello! I'm your medical assistant. I'm here to help you with:
        1. Understanding your symptoms
        2. Managing your medications
        3. General health questions

        What can I help you with today?"""
        
        state["response"] = greeting
        state["conversation_stage"] = "asking_need"
        
    return state

def understand_need_node(state: MedicalAssistantState) -> MedicalAssistantState:
    """
    NODE 2: Understand what user actually wants
    Uses LLM to properly classify intent
    """
    user_input = state["user_input"].lower()
    
    # FIX: If no input (start of conversation), don't classify
    if not user_input or not user_input.strip():
        state["intent"] = "waiting"
        return state
    
    # Better intent classification with examples
    classification_prompt = f"""You are an intent classifier for a medical assistant.

    Classify the user's message into EXACTLY ONE category:

    Categories:
    1. symptom_check - User mentions: headache, fever, pain, cough, tired, sick, feeling unwell, hurt, ache
    2. medication_reminder - User mentions: medicine, medication, forgot, take pills, diabetes medicine, prescription
    3. general_health - User asks: what foods, how to, healthy, exercise, diet, sleep, wellness
    4. emergency - User mentions: chest pain, can't breathe, severe pain, bleeding heavily, unconscious
    5. greeting - User says: hi, hello, hey, good morning, what can you do
    6. unclear - Cannot determine intent

    User message: "{state['user_input']}"

    Think step by step:
    - What is the user asking about?
    - Which category fits best?

    Respond with ONLY the category name (symptom_check, medication_reminder, general_health, emergency, greeting, or unclear)"""
    
    try:
        intent_response = llm.invoke(classification_prompt).content.strip().lower()
        
        # Extract just the category
        if "symptom" in intent_response:
            intent = "symptom_check"
        elif "medication" in intent_response or "reminder" in intent_response:
            intent = "medication_reminder"
        elif "general" in intent_response:
            intent = "general_health"
        elif "emergency" in intent_response:
            intent = "emergency"
        elif "greeting" in intent_response:
            intent = "greeting"
        else:
            intent = "unclear"
        
        state["intent"] = intent
        print(f"🧠 Detected intent: {intent}")
        
    except Exception as e:
        print(f"⚠️ Intent classification error: {e}")
        state["intent"] = "unclear"
    
    return state


def handle_symptom_check(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle symptom-related queries - conversational style"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    conditions = user_profile.get("conditions", [])
    context = f"Patient has: {', '.join(conditions)}" if conditions else "No known conditions"
    
    # Conversational prompt
    prompt = f"""You are a caring medical assistant having a conversation with a patient.

Context: {context}

Patient says: "{user_input}"

Respond naturally and conversationally (2-3 short sentences):
1. Acknowledge their concern with empathy
2. Briefly explain possible causes
3. Suggest when to see a doctor

Keep it SHORT and CONVERSATIONAL like you're talking to them in person.
End with: "This is not a diagnosis. Please consult a doctor if symptoms persist."

Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["conversation_stage"] = "followup"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I'm having trouble processing that. Could you describe your symptoms again?"
    
    return state


def handle_medication_reminder(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle medication queries - conversational"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    meds = user_profile.get("medications", [])
    meds_text = f"You're taking: {', '.join(meds)}" if meds else "No medications on record"
    
    prompt = f"""You are a supportive medication assistant talking to a patient.

    {meds_text}

    Patient says: "{user_input}"

    Respond warmly and conversationally (2-3 sentences):
    1. Acknowledge their situation
    2. Provide helpful advice or reminder
    3. Be encouraging about medication adherence

    Keep it SHORT and FRIENDLY.

    Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["conversation_stage"] = "followup"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "I understand medication can be challenging. Can you tell me more about what you need help with?"
    
    return state


def handle_general_health(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle general health questions - conversational"""
    user_input = state["user_input"]
    
    prompt = f"""You are a friendly health advisor.

Question: "{user_input}"

Give a brief, conversational answer (2-3 sentences):
1. Answer their question directly
2. Give 1-2 practical tips
3. Keep it simple and actionable

SHORT and CONVERSATIONAL.

Your response:"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        state["response"] = response
        state["conversation_stage"] = "followup"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        state["response"] = "That's a great health question! Could you be more specific about what you'd like to know?"
    
    return state


def handle_emergency(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle emergency - immediate and clear"""
    state["response"] = """This sounds like an emergency! Please call 911 or your emergency number RIGHT NOW. Do not wait. Get immediate medical help."""
    state["conversation_stage"] = "emergency"
    return state


def handle_unclear(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle when we don't understand"""
    state["response"] = """I didn't quite catch that. Could you tell me:
- Are you experiencing symptoms?
- Do you need medication help?
- Or do you have a general health question?"""
    state["conversation_stage"] = "asking_need"
    return state


def route_to_handler(state: MedicalAssistantState) -> str:
    """Router with better logic"""
    intent = state.get("intent", "unclear")
    stage = state.get("conversation_stage", "greeting")
    
    # If first time, greet
    if stage == "greeting":
        return "greeting"
    
    # Route based on intent
    if intent == "emergency":
        return "emergency"
    elif intent == "symptom_check":
        return "symptom"
    elif intent == "medication_reminder":
        return "medication"
    elif intent == "general_health":
        return "general"
    elif intent == "greeting":
        return "greeting"
    elif intent == "waiting":
        return "end"
    else:
        return "unclear"


# ============================================
# BUILD IMPROVED GRAPH
# ============================================

def create_conversational_graph():
    """Build the LangGraph with proper flow"""
    workflow = StateGraph(MedicalAssistantState)
    
    # Add all nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("understand", understand_need_node)
    workflow.add_node("symptom", handle_symptom_check)
    workflow.add_node("medication", handle_medication_reminder)
    workflow.add_node("general", handle_general_health)
    workflow.add_node("emergency", handle_emergency)
    workflow.add_node("unclear", handle_unclear)
    
    # Entry point
    workflow.set_entry_point("greeting")
    
    # From greeting, always go to understand
    workflow.add_edge("greeting", "understand")
    
    # From understand, route based on intent
    workflow.add_conditional_edges(
        "understand",
        route_to_handler,
        {
            "greeting": "greeting",
            "symptom": "symptom",
            "medication": "medication",
            "general": "general",
            "emergency": "emergency",
            "general": "general",
            "emergency": "emergency",
            "unclear": "unclear",
            "end": END
        }
    )
    
    # All handlers end
    for node in ["symptom", "medication", "general", "emergency", "unclear"]:
        workflow.add_edge(node, END)
    
    return workflow.compile()