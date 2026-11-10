"""Ohùn voice API. Runs on a free Hugging Face Space (CPU).

POST /v1/tts      {"text": "...", "voice": "yo-female-1", "speed": 1.0}  -> audio/wav
GET  /v1/voices   list of voices and whether each is live
GET  /health

Auth: header `Authorization: Bearer <key>`; keys in env OHUN_API_KEYS (comma separated).
Requests without a key are allowed for the public demo, capped at DEMO_MAX_CHARS characters.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from engines import KokoroVoice, PiperPhonemeVoice, to_wav_bytes

DEMO_MAX_CHARS = int(os.getenv("DEMO_MAX_CHARS", "300"))
PAID_MAX_CHARS = int(os.getenv("PAID_MAX_CHARS", "5000"))
DEMO_PER_MIN = int(os.getenv("DEMO_PER_MIN", "6"))
API_KEYS = {k.strip() for k in os.getenv("OHUN_API_KEYS", "").split(",") if k.strip()}
YO_REPO = os.getenv("OHUN_YO_REPO", "")  # e.g. aladewealth/ohun-yoruba on Hugging Face

VOICES = {
    "yo-female-1": {"lang": "yo", "label": "Yoruba · Adùnní (female)", "engine": "yo", "speaker": 0},
    "yo-male-1":   {"lang": "yo", "label": "Yoruba · Tọ́lá (male)", "engine": "yo", "speaker": None},
    "es-female-1": {"lang": "es", "label": "Spanish · Dora (female)", "engine": "es", "kokoro": "ef_dora"},
    "es-male-1":   {"lang": "es", "label": "Spanish · Alex (male)", "engine": "es", "kokoro": "em_alex"},
    "fr-female-1": {"lang": "fr", "label": "French · Siwis (female)", "engine": "fr", "kokoro": "ff_siwis"},
    "en-female-1": {"lang": "en", "label": "English · Heart (female)", "engine": "en", "kokoro": "af_heart"},
}

_engines: dict = {}
_lock = threading.Lock()


def _engine(name: str):
    with _lock:
        if name in _engines:
            return _engines[name]
        if name == "yo":
            onnx, cfg = "models/ohun_yo.onnx", "models/ohun_yo.onnx.json"
            if not os.path.exists(onnx) and YO_REPO:
                from huggingface_hub import hf_hub_download
                onnx = hf_hub_download(YO_REPO, "ohun_yo.onnx")
                cfg = hf_hub_download(YO_REPO, "ohun_yo.onnx.json")
            if not os.path.exists(onnx):
                return None
            eng = PiperPhonemeVoice(onnx, cfg)
            # pick the first male speaker id for the male voice
            males = [i for s, i in sorted(eng.speakers.items(), key=lambda x: x[1]) if s.startswith("m")]
            females = [i for s, i in sorted(eng.speakers.items(), key=lambda x: x[1]) if s.startswith("f")]
            VOICES["yo-female-1"]["speaker"] = int(os.getenv("YO_FEMALE_SID", females[0] if females else 0))
            VOICES["yo-male-1"]["speaker"] = int(os.getenv("YO_MALE_SID", males[0] if males else 0))
        else:
            eng = KokoroVoice(name)
        _engines[name] = eng
        return eng


app = FastAPI(title="Ohùn Voice API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_hits: dict[str, deque] = defaultdict(deque)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "yo-female-1"
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/voices")
def voices():
    yo_live = os.path.exists("models/ohun_yo.onnx") or bool(YO_REPO)
    return [{"id": k, "language": v["lang"], "label": v["label"],
             "live": yo_live if v["engine"] == "yo" else True} for k, v in VOICES.items()]


@app.post("/v1/tts")
async def tts(request: Request):
    """Body is JSON. Content-Type text/plain is accepted too, so browsers can call it without a CORS preflight."""
    import json as _json
    from pydantic import ValidationError
    try:
        req = TTSRequest(**_json.loads(await request.body() or b"{}"))
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, str(e)[:300])
    authorization = request.headers.get("authorization")
    key = (authorization or "").removeprefix("Bearer ").strip()
    paid = key in API_KEYS
    if key and not paid:
        raise HTTPException(401, "invalid API key")
    if not paid:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
        q, now = _hits[ip], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= DEMO_PER_MIN:
            raise HTTPException(429, "demo limit reached, get an API key at ohun.vercel.app")
        q.append(now)
    limit = PAID_MAX_CHARS if paid else DEMO_MAX_CHARS
    if len(req.text) > limit:
        raise HTTPException(413, f"text over {limit} characters" + ("" if paid else " (demo limit)"))
    v = VOICES.get(req.voice)
    if not v:
        raise HTTPException(404, "unknown voice")
    from starlette.concurrency import run_in_threadpool as _rt
    eng = await _rt(_engine, v["engine"])
    if eng is None:
        raise HTTPException(503, "this voice is still training")
    from starlette.concurrency import run_in_threadpool
    if v["engine"] == "yo":
        sr, audio = await run_in_threadpool(eng.synthesize, req.text, speaker=v["speaker"] or 0, speed=req.speed)
    else:
        sr, audio = await run_in_threadpool(eng.synthesize, req.text, voice=v["kokoro"], speed=req.speed)
    return Response(to_wav_bytes(sr, audio), media_type="audio/wav")
