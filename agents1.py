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
    messages: Annotated[list, operator.add]
    user_input: str
    intent: str
    user_profile: dict
    response: str
    next_action: str


def classify_intent(state: MedicalAssistantState) -> MedicalAssistantState:
    """Classify user intent"""
    user_input = state["user_input"]
    
    classification_prompt = f"""Classify into ONE category:
    - symptom_check
    - medication_reminder
    - general_health
    - emergency
    
    User: {user_input}
    
    Respond with ONLY the category."""
    
    intent = llm.invoke(classification_prompt).content.strip().lower()
    state["intent"] = intent
    state["messages"].append({"role": "system", "content": f"Intent: {intent}"})
    
    return state


def handle_symptom_check(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle symptom analysis"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    conditions = user_profile.get("conditions", [])
    context = f"Known conditions: {', '.join(conditions)}" if conditions else ""
    
    prompt = f"""You are a medical assistant. Be concise for voice output (2-3 sentences max).

{context}

User: {user_input}

Provide:
1. Acknowledge symptoms
2. Possible causes
3. When to see doctor

Add: "This is not medical advice. Consult a healthcare professional."

Keep response SHORT for voice."""
    
    response = llm.invoke(prompt).content
    state["response"] = response
    state["next_action"] = "complete"
    
    return state


def handle_medication_reminder(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle medication reminders"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    
    meds = user_profile.get("medications", [])
    
    prompt = f"""Medication assistant. Be concise for voice (2-3 sentences).

Current meds: {meds if meds else "None"}

User: {user_input}

Provide helpful, supportive response about medication management.

Keep SHORT for voice."""
    
    response = llm.invoke(prompt).content
    state["response"] = response
    state["next_action"] = "complete"
    
    return state


def handle_general_health(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle general health questions"""
    user_input = state["user_input"]
    
    prompt = f"""Health assistant. Be concise for voice (2-3 sentences).

User: {user_input}

Provide brief, helpful health advice.

Keep SHORT for voice."""
    
    response = llm.invoke(prompt).content
    state["response"] = response
    state["next_action"] = "complete"
    
    return state


def handle_emergency(state: MedicalAssistantState) -> MedicalAssistantState:
    """Handle emergencies"""
    state["response"] = """Emergency alert! Call 911 immediately or your local emergency number. This may require immediate medical attention. Do not wait. Get help now."""
    state["next_action"] = "complete"
    return state


def route_to_handler(state: MedicalAssistantState) -> Literal["symptom_check", "medication_reminder", "general_health", "emergency"]:
    """Route to appropriate handler"""
    intent = state["intent"]
    
    if "emergency" in intent:
        return "emergency"
    elif "symptom" in intent:
        return "symptom_check"
    elif "medication" in intent:
        return "medication_reminder"
    else:
        return "general_health"


def create_medical_assistant_graph():
    """Create LangGraph workflow"""
    workflow = StateGraph(MedicalAssistantState)
    
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("symptom_check", handle_symptom_check)
    workflow.add_node("medication_reminder", handle_medication_reminder)
    workflow.add_node("general_health", handle_general_health)
    workflow.add_node("emergency", handle_emergency)
    
    workflow.set_entry_point("classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_to_handler,
        {
            "symptom_check": "symptom_check",
            "medication_reminder": "medication_reminder",
            "general_health": "general_health",
            "emergency": "emergency"
        }
    )
    
    workflow.add_edge("symptom_check", END)
    workflow.add_edge("medication_reminder", END)
    workflow.add_edge("general_health", END)
    workflow.add_edge("emergency", END)
    
    return workflow.compile()
