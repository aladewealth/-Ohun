# Ohùn

Text-to-speech API by Atoms Intelligence Labs.

**Status (Nov 2026):** Spanish, French and English are live through Kokoro-82M. The Yoruba voice is our own and is **not trained yet**: the front end and training recipe are done, the weights come from the first Kaggle run. Hausa and Igbo are planned, with no training data chosen yet.

- Live API and demo: https://huggingface.co/spaces/aladewealth/ohun-api
- Open Yoruba model (card, G2P, recipe; weights after training): https://huggingface.co/aladewealth/ohun-yoruba

## API

Base URL: `https://aladewealth-ohun-api.hf.space/api`

| Method | Path | What |
|---|---|---|
| POST | `/v1/tts` | `{"text": "...", "voice": "es-female-1", "speed": 1.0}` → 24 kHz (Kokoro) or 22.05 kHz (Yoruba) mono WAV |
| GET | `/v1/voices` | Voices and whether each is live |
| GET | `/health` | `{"ok": true}` |

Interactive reference: `/api/docs`.

```bash
curl -X POST https://aladewealth-ohun-api.hf.space/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour, je suis Ohùn.","voice":"fr-female-1"}' -o out.wav
```

Voices: `es-female-1`, `es-male-1`, `fr-female-1`, `en-female-1` (live); `yo-female-1`, `yo-male-1` (return 503 until the Yoruba weights are published).

Without an API key: 300 characters per request, 6 requests a minute per IP. With a key (`Authorization: Bearer <key>`, keys set in the `OHUN_API_KEYS` secret): 5,000 characters. No paid plans or key sign-up exist yet. The body may also be sent as `text/plain`, so browsers can call it without a CORS preflight.

The free Space sleeps when idle; the first request after a sleep, and the first request per language, is slower while models load.

## Repo layout

| Part | What | Runs on |
|---|---|---|
| `voice/yoruba_g2p.py` | Rule-based Yoruba text → IPA with tones (high, mid, low) and nasal vowels, mapped to Piper's default phoneme ids | anywhere |
| `training/` | OpenSLR 86 prep and a Kaggle notebook: fine-tunes a Piper (VITS) voice from the public-domain LJSpeech medium checkpoint, exports ONNX, uploads it to `aladewealth/ohun-yoruba` | Kaggle GPU T4 x2 |
| `server/` | FastAPI app (the REST API) and inference engines | |
| `space/` | Hugging Face Space entry: Gradio demo at `/`, the FastAPI app mounted at `/api` | HF Space, Gradio SDK, free ZeroGPU hardware (inference runs on CPU) |
| `web/` | Landing page with live demo | Vercel (not deployed yet) |
| `Dockerfile` | For running the API in Docker elsewhere (HF Docker Spaces are a paid option) | any Docker host |

## Yoruba model

- Architecture: VITS via Piper, exported to ONNX. Medium Piper voices have roughly 15–20M parameters (estimate).
- Data: OpenSLR 86 (Google), CC BY-SA 4.0, about 3,600 lines from about 36 speakers; transcripts not all hand-checked.
- Training plan: about 1,000 epochs across 2–3 Kaggle sessions (estimate, untested).
- Needs diacritized input; unmarked text is read with mid tones.

## Deploy

1. **Train Yoruba:** open `training/train_yoruba_kaggle.ipynb` on Kaggle (GPU T4 x2, Internet on, secret `HF_TOKEN` with write access). The last cell publishes the weights.
2. **API:** copy `space/*`, `voice/` and `server/` into the Space repo root. Space variable `OHUN_YO_REPO=aladewealth/ohun-yoruba`; optional secret `OHUN_API_KEYS`. Restart the Space after weights are published.
3. **Site:** Vercel → import this repo → Root Directory `web`, no build step.

## Licenses

- Code: Apache-2.0. Training calls piper1-gpl (GPL-3.0) as an external tool; no GPL code ships in the API.
- Yoruba weights: CC BY-SA 4.0 (derived from OpenSLR 86, © Google).
- Kokoro-82M: Apache-2.0.
