# Ohùn

Text-to-speech API for Yoruba (our own tone-aware voice), Spanish, French and English. Hausa and Igbo next.

| Part | What | Runs on (free) |
|---|---|---|
| `voice/yoruba_g2p.py` | Rule-based Yoruba text → IPA with tones (high 3, mid 2, low 1) and nasal vowels | anywhere |
| `training/` | OpenSLR 86 prep + Kaggle notebook: fine-tunes a Piper (VITS) voice from the public-domain LJSpeech checkpoint | Kaggle T4 x2 |
| `server/` | FastAPI: `/v1/tts`, `/v1/voices`, API keys, demo rate limit. Yoruba via ONNX, es/fr/en via Kokoro-82M | Hugging Face Space (CPU) |
| `web/` | Landing page with live demo and pricing | Vercel |

## Ship order
1. **Train Yoruba** – open `training/train_yoruba_kaggle.ipynb` on Kaggle (GPU T4 x2, Internet on). ~1,000 epochs over 2–3 sessions. Export `ohun_yo.onnx` + `.onnx.json` and upload them to a Hugging Face model repo (license CC BY-SA 4.0, because the data is ShareAlike).
2. **Deploy API** – create a Docker Space, push this repo with `server/SPACE_README.md` as the Space `README.md`. Secrets: `OHUN_API_KEYS`, `OHUN_YO_REPO`. Spanish/French/English work before Yoruba finishes.
3. **Deploy site** – Vercel → import repo → Root Directory `web`, no build step. Set `window.OHUN_API` (or edit the `API` constant) to the Space URL.

## Licenses
- Code: Apache-2.0. Training uses piper1-gpl (GPL-3.0) as an external tool; nothing GPL ships in the API.
- Yoruba voice weights: CC BY-SA 4.0 (derived from OpenSLR 86, © Google). Attribution is on the site footer.
- Kokoro-82M: Apache-2.0.
