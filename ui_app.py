"""Enhanced Medical Voice Assistant UI (Streamlit)."""

import os
import tempfile

import streamlit as st

from agents1 import (
    ConversationState,
    create_conversational_graph,
    create_initial_state,
    emergency_agent,
    handle_user_choice,
    health_advisor_agent,
    medication_agent,
    route_after_agent,
    route_from_choice,
    symptom_agent,
    unclear_handler,
)
from langgraph.graph import END, StateGraph
from voice_impl1 import VOICE_AVAILABLE, VoiceProcessor


def _build_choice_graph():
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
            "unclear": "unclear",
        },
    )
    for node in ["symptom", "medication", "health", "emergency", "unclear"]:
        workflow.add_conditional_edges(node, route_after_agent, {"end": END})
    return workflow.compile()


@st.cache_resource
def _resources():
    greeting_graph = create_conversational_graph()
    choice_graph = _build_choice_graph()
    voice = VoiceProcessor()
    return greeting_graph, choice_graph, voice


def _init_session():
    if "phase" not in st.session_state:
        st.session_state.phase = "init"  # init -> greeted -> responded
    if "stage" not in st.session_state:
        st.session_state.stage = "greeting"  # greeting, waiting_for_choice, waiting_for_details, complete
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "last_tts_path" not in st.session_state:
        st.session_state.last_tts_path = None

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role: user|assistant, content: str)]


def _step_html():
    phase = st.session_state.phase
    greeted = phase in {"greeted", "responded"}
    responded = phase == "responded"

    def badge(done: bool):
        return "<span class='badge badge-done'>DONE</span>" if done else "<span class='badge badge-todo'>PENDING</span>"

    return f"""
    <div class='steps'>
      <div class='step'><div class='step-title'>1. Greeting</div>{badge(greeted)}</div>
      <div class='step'><div class='step-title'>2. Audio / Text Input</div>{badge(responded)}</div>
      <div class='step'><div class='step-title'>3. Speech-to-Text (Whisper)</div>{badge(responded)}</div>
      <div class='step'><div class='step-title'>4. Intent + Response (LangGraph)</div>{badge(responded)}</div>
      <div class='step'><div class='step-title'>5. Text-to-Speech</div>{badge(responded)}</div>
    </div>
    """


def _profile():
    return {
        "medications": ["Metformin 500mg - twice daily"],
        "conditions": ["Type 2 Diabetes"],
    }


def _run_greeting():
    greeting_graph, _, voice = _resources()

    state = create_initial_state(user_input="", user_profile=_profile(), is_first_message=True)
    result = greeting_graph.invoke(state)

    response = result.get("response", "")
    st.session_state.phase = "greeted"
    st.session_state.stage = result.get("stage", "waiting_for_choice")
    st.session_state.last_response = response
    st.session_state.last_tts_path = voice.text_to_speech_file(response) if response else None
    st.session_state.messages.append({"role": "assistant", "content": response})


def _run_turn(user_text: str, audio_file):
    _, choice_graph, voice = _resources()

    if st.session_state.phase == "init":
        st.warning("Click 'Start Greeting' first.")
        return

    text = (user_text or "").strip()
    if not text and audio_file is None:
        st.warning("Provide audio (upload) or type text, then click Run.")
        return

    if not text and audio_file is not None:
        suffix = os.path.splitext(audio_file.name)[1] or ".wav"
        fd, path = tempfile.mkstemp(prefix="ui_audio_", suffix=suffix)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(audio_file.getbuffer())
        text = voice.speech_to_text_from_file(path, cleanup=True)

    state = create_initial_state(user_input=text, user_profile=_profile(), is_first_message=False)
    state["stage"] = "waiting_for_choice"

    result = choice_graph.invoke(state)

    response = result.get("response", "")
    intent = result.get("intent", "")
    stage = result.get("stage", "complete")

    st.session_state.phase = "responded"
    st.session_state.stage = stage
    st.session_state.last_transcript = text
    st.session_state.last_intent = intent
    st.session_state.last_response = response
    st.session_state.last_tts_path = voice.text_to_speech_file(response) if response else None

    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.messages.append({"role": "assistant", "content": response})


def _reset():
    st.session_state.phase = "init"
    st.session_state.stage = "greeting"
    st.session_state.chat = []
    st.session_state.last_transcript = ""
    st.session_state.last_intent = ""
    st.session_state.last_response = ""
    st.session_state.last_tts_path = None
    st.session_state.messages = []


def main():
    st.set_page_config(page_title="Medical Voice Assistant", page_icon="🏥", layout="wide")
    _init_session()

    st.markdown(
        """
        <style>
          .hero{border:1px solid rgba(15,23,42,.10);background:linear-gradient(180deg, rgba(59,130,246,.10), rgba(99,102,241,.06));padding:18px;border-radius:16px;margin-bottom:14px}
          .steps{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
          .step{border:1px solid rgba(15,23,42,.10);background:rgba(255,255,255,.75);padding:10px;border-radius:14px}
          .step-title{font-size:12px;font-weight:600;color:rgba(15,23,42,.85);margin-bottom:8px}
          .badge{font-size:11px;padding:4px 8px;border-radius:999px;border:1px solid rgba(15,23,42,.12);display:inline-block}
          .badge-todo{background:rgba(148,163,184,.18);color:rgba(15,23,42,.75)}
          .badge-done{background:rgba(16,185,129,.18);color:rgba(6,95,70,.95);border-color:rgba(16,185,129,.25)}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='hero'>
          <h2 style='margin:0'>Medical Voice Assistant (Streamlit)</h2>
          <div style='margin-top:6px;color:rgba(15,23,42,.75)'>Greeting → Choice → Details → Response (with browser-playable TTS)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not VOICE_AVAILABLE:
        st.error("Voice system not available. Install voice packages from requriments.txt")
        return

    with st.sidebar:
        st.subheader("Flow")
        st.markdown(_step_html(), unsafe_allow_html=True)
        st.divider()
        st.subheader("Debug")
        st.text_input("Stage", value=st.session_state.stage, disabled=True)
        st.text_input("Intent", value=st.session_state.last_intent, disabled=True)
        st.text_area("Transcript", value=st.session_state.last_transcript, height=90, disabled=True)
        st.text_area("Response", value=st.session_state.last_response, height=180, disabled=True)
        if st.session_state.last_tts_path and os.path.exists(st.session_state.last_tts_path):
            with open(st.session_state.last_tts_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")

        st.divider()
        cols = st.columns(2)
        with cols[0]:
            if st.button("Start Greeting", use_container_width=True):
                _run_greeting()
                st.rerun()
        with cols[1]:
            if st.button("Reset", use_container_width=True):
                _reset()
                st.rerun()

        st.subheader("Send Audio")
        audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], key="audio_uploader")
        if st.button("Send Audio", use_container_width=True):
            _run_turn("", audio_file)
            st.rerun()

    # Main chat area
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type a message...")
    if prompt:
        _run_turn(prompt, None)
        st.rerun()


def _escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()