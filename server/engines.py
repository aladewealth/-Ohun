"""Speech engines behind the Ohùn API.

- PiperPhonemeVoice: our own Yoruba VITS voice (ONNX, CPU). Text -> rule-based G2P -> phoneme ids -> audio.
- KokoroVoice: Kokoro-82M (Apache-2.0) for English, Spanish and French. No training needed.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unicodedata
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from voice.yoruba_g2p import to_ipa  # noqa: E402

BOS, EOS, PAD = "^", "$", "_"


class PiperPhonemeVoice:
    def __init__(self, onnx_path: str, config_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.config = json.load(open(config_path, encoding="utf-8"))
        self.id_map = self.config["phoneme_id_map"]
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.speakers = self.config.get("speaker_id_map", {}) or {}
        inf = self.config.get("inference", {})
        self.scales = [inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0), inf.get("noise_w", 0.8)]

    def ids(self, phonemes: list[str]) -> list[int]:
        out = list(self.id_map[BOS]) + list(self.id_map[PAD])
        for p in phonemes:
            if p in self.id_map:
                out += self.id_map[p] + self.id_map[PAD]
        return out + list(self.id_map[EOS])

    def synthesize(self, text: str, speaker: int = 0, speed: float = 1.0):
        chunks = []
        for sent in _sentences(text):
            phonemes = list(unicodedata.normalize("NFD", to_ipa(sent)))
            ids = np.array([self.ids(phonemes)], dtype=np.int64)
            args = {
                "input": ids,
                "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                "scales": np.array([self.scales[0], self.scales[1] / speed, self.scales[2]], dtype=np.float32),
            }
            if len(self.speakers) > 1:
                args["sid"] = np.array([speaker], dtype=np.int64)
            audio = self.session.run(None, args)[0].squeeze()
            chunks += [audio, np.zeros(int(self.sample_rate * 0.25), dtype=np.float32)]
        return self.sample_rate, np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)


class KokoroVoice:
    LANG = {"en": "a", "es": "e", "fr": "f"}

    def __init__(self, lang: str):
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code=self.LANG[lang], repo_id="hexgrad/Kokoro-82M")
        self.sample_rate = 24000

    def synthesize(self, text: str, voice: str, speed: float = 1.0):
        parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                 for _, _, a in self.pipe(text, voice=voice, speed=speed)]
        return self.sample_rate, np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def _sentences(text: str, max_len: int = 220):
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!\n" and len(buf.strip()) > 1:
            yield buf.strip(); buf = ""
        elif len(buf) > max_len and ch in ", ":
            yield buf.strip(); buf = ""
    if buf.strip():
        yield buf.strip()


def to_wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate); w.writeframes(pcm.tobytes())
    return buf.getvalue()
