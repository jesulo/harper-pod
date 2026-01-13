# Harper Pod

**Harper Pod** is a REST API for Speech-to-Text (STT) and Text-to-Speech (TTS) with voice cloning capabilities. It provides OpenAI-compatible endpoints powered by Whisper and Chatterbox models.

## Features

- 🎤 **Whisper STT**: Fast transcription using `faster-whisper` (large-v3)
- 🔊 **Chatterbox TTS**: Multilingual text-to-speech with voice cloning
- 🎭 **Voice Management**: Upload, list, and delete custom voices
- 🌍 **Multi-language Support**: 23+ languages (ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh)
- 📦 **Docker Ready**: Production-ready images with models baked in
- 🚀 **Streaming Audio**: Real-time WAV streaming for TTS

## API Endpoints

### Health Check
\`\`\`bash
GET /health
\`\`\`

### Speech-to-Text (STT)
\`\`\`bash
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

Parameters:
- file: Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
- model: whisper-1 (default)
- language: Language code (optional)
- response_format: json (default), text, srt, verbose_json, vtt
\`\`\`

### Text-to-Speech (TTS)
\`\`\`bash
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "Text to synthesize",
  "voice": "default",
  "language": "en",
  "response_format": "mp3"
}
\`\`\`

### TTS Streaming
\`\`\`bash
POST /v1/audio/speech/stream
Content-Type: application/json

{
  "input": "Text to synthesize",
  "voice": "default",
  "language": "en"
}
\`\`\`

### Voice Management

#### Upload Voice
\`\`\`bash
POST /v1/voices
Content-Type: multipart/form-data

Parameters:
- file: Audio file (wav, mp3, flac, m4a, ogg)
- name: Unique voice name
\`\`\`

#### List Voices
\`\`\`bash
GET /v1/voices
\`\`\`

#### Delete Voice
\`\`\`bash
DELETE /v1/voices/{name}
\`\`\`

## Quick Start

### Using Docker (Recommended)

\`\`\`bash
# Clone repository
git clone https://github.com/jesulo/harper-pod.git
cd harper-pod

# Build and run with Docker Compose
docker compose -f docker-compose.prod.yml up -d

# Wait for models to load (~2-3 minutes)
# Check health
curl http://localhost:8007/health
\`\`\`

### Manual Installation

#### Requirements
- CUDA-compatible GPU (24GB+ VRAM recommended)
- Python 3.10+
- CUDA 12.1+
- ffmpeg

#### Setup
\`\`\`bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
cd server
pip install -r requirements.txt

# Run server
python api_server.py
\`\`\`

## Usage Examples

### Transcribe Audio
\`\`\`bash
curl -X POST http://localhost:8007/v1/audio/transcriptions \\
  -F "file=@audio.mp3" \\
  -F "language=en"
\`\`\`

### Generate Speech
\`\`\`bash
curl -X POST http://localhost:8007/v1/audio/speech/stream \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, world!", "language": "en"}' \\
  -o output.wav
\`\`\`

### Upload Custom Voice
\`\`\`bash
curl -X POST http://localhost:8007/v1/voices \\
  -F "file=@my_voice.wav" \\
  -F "name=myvoice"
\`\`\`

### Generate with Custom Voice
\`\`\`bash
curl -X POST http://localhost:8007/v1/audio/speech/stream \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello with my voice!", "voice": "myvoice", "language": "en"}' \\
  -o output_custom.wav
\`\`\`

## Configuration

### Environment Variables

\`\`\`bash
# Server port (default: 8007)
PORT=8007

# Whisper model (default: large-v3)
WHISPER_MODEL=large-v3

# Chatterbox model path
CHATTERBOX_MODEL_PATH=./chatterbox_infer/models
\`\`\`

## Docker Images

### Build Production Image
\`\`\`bash
docker build -f Dockerfile.full -t speech-pod:latest .
\`\`\`

This creates a production-ready image (~16GB) with:
- PyTorch 2.5.1 + CUDA 12.1
- Whisper large-v3 model
- Chatterbox multilingual TTS model

### Push to Docker Hub
\`\`\`bash
docker tag speech-pod:latest your-username/speech-pod:latest
docker push your-username/speech-pod:latest
\`\`\`

## Architecture

\`\`\`
harper-pod/
├── server/
│   ├── api_server.py          # Main FastAPI application
│   ├── audio_samples/         # Default voice samples
│   ├── voices/                # User-uploaded voices
│   ├── chatterbox_infer/      # Chatterbox TTS engine
│   ├── stt/                   # Whisper STT implementations
│   ├── tts/                   # TTS factory and utilities
│   └── utils/                 # Helper functions
├── Dockerfile.api             # Development Dockerfile
├── Dockerfile.full            # Production Dockerfile (with models)
└── docker-compose.prod.yml    # Production Docker Compose
\`\`\`

## Models

### Whisper (STT)
- **Model**: \`large-v3\` via \`faster-whisper\`
- **Languages**: 99+ languages
- **Speed**: ~0.05x Real-Time Factor on GPU
- **Download**: Automatic on first run

### Chatterbox (TTS)
- **Model**: \`ResembleAI/chatterbox\` (multilingual)
- **Languages**: 23 languages
- **Voice Cloning**: Supports custom voice upload
- **Download**: Automatic on first run (~3.2GB)

## Performance

### Benchmarks (RTX 3090)
- **STT**: ~1.2s for 27s audio (RTF 0.046x)
- **TTS**: ~5s for 20-word sentence
- **Streaming TTS**: First chunk in <500ms

### Resource Usage
- **GPU Memory**: ~10GB (Whisper + Chatterbox loaded)
- **Disk Space**: ~20GB (models + dependencies)

## API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8007/docs
- **ReDoc**: http://localhost:8007/redoc

## Supported Audio Formats

### Input (STT)
- mp3, mp4, mpeg, mpga, m4a, wav, webm

### Output (TTS)
- wav (default for streaming)
- mp3, opus (for non-streaming)

### Voice Upload
- wav, mp3, flac, m4a, ogg

## Troubleshooting

### Out of Memory Error
- Reduce \`max_cap\` in \`server/chatterbox_infer/mtl_tts.py\`
- Use a smaller Whisper model: \`medium\` or \`small\`

### Slow Generation
- Check GPU is being used: \`nvidia-smi\`
- Ensure CUDA drivers are up to date

### Model Download Fails
- Check internet connection
- Manually download models from Hugging Face

## Credits

Built on top of:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Chatterbox by ResembleAI](https://huggingface.co/ResembleAI/chatterbox)
- [PyTorch](https://pytorch.org/)

## License

MIT License

## Contact

Questions or feedback? Open an issue on GitHub.

---

**Harper Pod** - REST API for STT + TTS with Voice Cloning
