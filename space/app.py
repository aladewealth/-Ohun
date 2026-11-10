"""Hugging Face Space entry (Gradio SDK). Serves the Ohùn REST API (/v1/*) and a Gradio demo at /."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "server")]

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ohun_api", os.path.join(HERE, "server", "app.py"))
api = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(api)

def speak(text, voice, speed):
    v = api.VOICES[voice]
    eng = api._engine(v["engine"])
    if eng is None:
        raise gr.Error("This voice is still training.")
    text = text[: api.DEMO_MAX_CHARS]
    if v["engine"] == "yo":
        sr, audio = eng.synthesize(text, speaker=v["speaker"] or 0, speed=speed)
    else:
        sr, audio = eng.synthesize(text, voice=v["kokoro"], speed=speed)
    return sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16)

with gr.Blocks(title="Ohùn Voice") as demo:
    gr.Markdown("# Ohùn\nVoices that get the tones right. Yoruba, Spanish, French, English. REST API at `/v1/tts`, docs at `/docs`.")
    choices = [(v["label"], k) for k, v in api.VOICES.items()]
    t = gr.Textbox(value="Hola, soy Ohùn. Puedo leer cualquier texto en voz alta.", lines=3, label="Text")
    v = gr.Dropdown(choices=choices, value="es-female-1", label="Voice")
    s = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
    out = gr.Audio(label="Audio")
    gr.Button("Speak").click(speak, [t, v, s], out, api_name="tts")

app = gr.mount_gradio_app(api.app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
