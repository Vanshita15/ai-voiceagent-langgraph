import gradio as gr

from agents1 import create_conversational_graph
from voice_impl1 import VoiceProcessor


def _new_ui_state():
    return {
        "phase": "init",  # init -> greeted -> responded
        "conversation_stage": "greeting",
        "last_transcript": "",
        "last_intent": "",
        "last_response": "",
        "last_tts_path": None,
    }


def _render_steps(state: dict) -> str:
    phase = state.get("phase", "init")

    def badge(done: bool, active: bool) -> str:
        if active:
            return "<span class='badge badge-active'>IN PROGRESS</span>"
        if done:
            return "<span class='badge badge-done'>DONE</span>"
        return "<span class='badge badge-todo'>PENDING</span>"

    greeted = phase in {"greeted", "responded"}
    responded = phase == "responded"

    html = """
    <div class='steps'>
      <div class='step'>
        <div class='step-title'>1. Greeting</div>
        <div class='step-badge'>%s</div>
      </div>
      <div class='step'>
        <div class='step-title'>2. Audio Input</div>
        <div class='step-badge'>%s</div>
      </div>
      <div class='step'>
        <div class='step-title'>3. Speech-to-Text (Whisper)</div>
        <div class='step-badge'>%s</div>
      </div>
      <div class='step'>
        <div class='step-title'>4. Intent + Response (LangGraph)</div>
        <div class='step-badge'>%s</div>
      </div>
      <div class='step'>
        <div class='step-title'>5. Text-to-Speech</div>
        <div class='step-badge'>%s</div>
      </div>
    </div>
    """ % (
        badge(greeted, False),
        badge(responded, False),
        badge(responded, False),
        badge(responded, False),
        badge(responded, False),
    )

    return html


def _build_initial_graph_state(conversation_stage: str, user_input: str):
    return {
        "conversation_stage": conversation_stage,
        "messages": [],
        "user_input": user_input,
        "intent": "",
        "user_profile": {
            "medications": ["Metformin 500mg - twice daily"],
            "conditions": ["Type 2 Diabetes"],
        },
        "response": "",
        "context": {},
    }


def _safe_str(v) -> str:
    return "" if v is None else str(v)


def build_app():
    graph = create_conversational_graph()
    voice = VoiceProcessor()

    def reset():
        state = _new_ui_state()
        return (
            state,
            _render_steps(state),
            "",
            "",
            "",
            None,
            "",
        )

    def start_greeting(state: dict):
        state = state or _new_ui_state()

        if state.get("phase") != "init":
            state["phase"] = "greeted"
            return (
                state,
                _render_steps(state),
                "Greeting already completed. You can proceed with audio.",
                _safe_str(state.get("last_transcript")),
                _safe_str(state.get("last_intent")),
                _safe_str(state.get("last_response")),
                state.get("last_tts_path"),
            )

        initial = _build_initial_graph_state("greeting", "")
        result = graph.invoke(initial)

        response = result.get("response", "")
        tts_path = voice.text_to_speech_file(response) if response else None

        state["phase"] = "greeted"
        state["conversation_stage"] = result.get("conversation_stage", "asking_need")
        state["last_response"] = response
        state["last_tts_path"] = tts_path

        return (
            state,
            _render_steps(state),
            response,
            "",
            result.get("intent", ""),
            response,
            tts_path,
        )

    def run_turn(audio_path, typed_text: str, state: dict):
        state = state or _new_ui_state()

        if state.get("phase") == "init":
            msg = "Please click 'Start Greeting' first."
            return (
                state,
                _render_steps(state),
                msg,
                "",
                "",
                "",
                None,
            )

        user_text = (typed_text or "").strip()
        if not user_text:
            if not audio_path:
                msg = "Provide audio (record/upload) or type a message, then click 'Run Turn'."
                return (
                    state,
                    _render_steps(state),
                    msg,
                    "",
                    "",
                    "",
                    None,
                )

            user_text = voice.speech_to_text_from_file(audio_path, cleanup=False)

        stage = state.get("conversation_stage", "asking_need")
        if stage == "greeting":
            stage = "asking_need"

        graph_state = _build_initial_graph_state(stage, user_text)
        result = graph.invoke(graph_state)

        response = result.get("response", "")
        tts_path = voice.text_to_speech_file(response) if response else None

        state["phase"] = "responded"
        state["conversation_stage"] = result.get("conversation_stage", "followup")
        state["last_transcript"] = user_text
        state["last_intent"] = result.get("intent", "")
        state["last_response"] = response
        state["last_tts_path"] = tts_path

        return (
            state,
            _render_steps(state),
            "Turn completed.",
            user_text,
            result.get("intent", ""),
            response,
            tts_path,
        )

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
    )

    css = """
    .app-wrap { max-width: 1100px; margin: 0 auto; }
    .hero {
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: linear-gradient(180deg, rgba(59, 130, 246, 0.10), rgba(99, 102, 241, 0.06));
      padding: 18px 18px;
      border-radius: 16px;
    }
    .hero h1 { margin: 0; font-size: 20px; }
    .hero p { margin: 6px 0 0 0; color: rgba(15, 23, 42, 0.75); }

    .card {
      border: 1px solid rgba(15, 23, 42, 0.10) !important;
      border-radius: 16px !important;
      background: rgba(255, 255, 255, 0.85) !important;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
    }

    .steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
    .step { padding: 10px 10px; border-radius: 14px; border: 1px solid rgba(15, 23, 42, 0.08); background: rgba(255,255,255,0.7); }
    .step-title { font-size: 12px; color: rgba(15, 23, 42, 0.85); font-weight: 600; margin-bottom: 8px; }

    .badge { font-size: 11px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(15, 23, 42, 0.12); }
    .badge-todo { background: rgba(148, 163, 184, 0.18); color: rgba(15, 23, 42, 0.75); }
    .badge-active { background: rgba(59, 130, 246, 0.18); color: rgba(30, 64, 175, 0.95); border-color: rgba(59, 130, 246, 0.25); }
    .badge-done { background: rgba(16, 185, 129, 0.18); color: rgba(6, 95, 70, 0.95); border-color: rgba(16, 185, 129, 0.25); }

    @media (max-width: 980px) {
      .steps { grid-template-columns: 1fr; }
    }
    """

    with gr.Blocks(theme=theme, css=css, title="Medical Voice Assistant Demo") as demo:
        with gr.Column(elem_classes=["app-wrap"]):
            gr.HTML(
                """
                <div class='hero'>
                  <h1>Medical Voice Assistant (LangGraph + Whisper + TTS)</h1>
                  <p>Demo flow is enforced in sequence. Use a light theme UI suitable for showcasing.</p>
                </div>
                """
            )

            ui_state = gr.State(_new_ui_state())

            with gr.Row():
                with gr.Column(scale=2):
                    step_view = gr.HTML(_render_steps(_new_ui_state()))

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Controls")
                        with gr.Row():
                            btn_greet = gr.Button("Start Greeting", variant="primary")
                            btn_run = gr.Button("Run Turn", variant="secondary")
                            btn_reset = gr.Button("Reset")

                        status = gr.Textbox(label="Status", value="", interactive=False)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Input")
                        audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Record or Upload Audio",
                        )
                        typed = gr.Textbox(
                            label="Or type your message (optional)",
                            placeholder="e.g., I have a headache since morning",
                        )

                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Outputs")
                        greeting_out = gr.Textbox(label="Assistant Greeting / Latest Assistant Message", lines=4, interactive=False)
                        transcript_out = gr.Textbox(label="Transcript", lines=2, interactive=False)
                        intent_out = gr.Textbox(label="Detected Intent", interactive=False)
                        response_out = gr.Textbox(label="Assistant Response", lines=5, interactive=False)
                        tts_audio = gr.Audio(label="Assistant Voice (TTS)", type="filepath")

            btn_reset.click(
                fn=reset,
                inputs=[],
                outputs=[ui_state, step_view, status, greeting_out, intent_out, response_out, tts_audio],
            )

            btn_greet.click(
                fn=start_greeting,
                inputs=[ui_state],
                outputs=[ui_state, step_view, greeting_out, transcript_out, intent_out, response_out, tts_audio],
            )

            btn_run.click(
                fn=run_turn,
                inputs=[audio, typed, ui_state],
                outputs=[ui_state, step_view, status, transcript_out, intent_out, response_out, tts_audio],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
