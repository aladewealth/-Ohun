# Hugging Face Space (Docker, free CPU). Serves the Ohùn API on port 7860.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng git && rm -rf /var/lib/apt/lists/*
RUN useradd -m user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r server/requirements.txt
COPY --chown=user . .
WORKDIR /home/user/app/server
# warm Kokoro weights at build time so the first request is fast
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='e', repo_id='hexgrad/Kokoro-82M')" || true
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
