# Harper Voice Services - Multi-Model Pod
# Whisper STT + Chatterbox Multilingual + Chatterbox Turbo
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for STT
RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    ctranslate2==4.4.0

# Install Chatterbox TTS (includes both Multilingual and Turbo)
RUN pip install --no-cache-dir chatterbox-tts

# Install server dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    websockets==13.1 \
    python-multipart==0.0.12 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2 \
    orjson==3.10.7 \
    numpy==1.26.4 \
    librosa==0.10.2 \
    opuslib==3.0.1 \
    loguru==0.7.2

# Pre-download models at build time to reduce cold start
# This bakes models into the image (~10GB total)
RUN python3 -c "from faster_whisper import WhisperModel; print('Downloading Whisper...'); WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"

RUN python3 -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('Downloading Chatterbox Multilingual...'); ChatterboxMultilingualTTS.from_pretrained(device='cpu')"

RUN python3 -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('Downloading Chatterbox Turbo...'); ChatterboxTurboTTS.from_pretrained(device='cpu')"

# Create app directory
WORKDIR /app

# Copy server code
COPY runpod_server.py /app/
COPY utils/ /app/utils/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "-u", "runpod_server.py"]
