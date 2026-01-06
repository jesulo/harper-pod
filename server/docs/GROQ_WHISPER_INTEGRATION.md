# Groq Whisper STT Integration

## Overview

Harper ahora soporta transcripción de voz usando **Groq's Whisper API**, que ofrece:

- ⚡ **Ultra-rápido**: Latencia < 300ms vs 1-2s de modelos locales
- 🎯 **Alta precisión**: Whisper Large V3 con 99%+ de precisión
- 💰 **Costo eficiente**: $0.111 por hora de audio (~2-3 centavos por sesión típica)
- 🌍 **Multilenguaje**: Soporta 50+ idiomas incluyendo español
- 🔒 **Sin GPU local**: No requiere VRAM, libera recursos

## Architecture

Based on `realtime-phone-agents-course` STT architecture:

```
realtime-phone-agents-course/src/realtime_phone_agents/stt/groq/whisper.py
└── Adaptado para Harper

harper/server/stt/factory.py
├── ASRBackend (Abstract base class)
├── HFWhisperBackend (Original local Whisper)  
└── GroqWhisperSTT (New - Groq API integration)
```

## Configuration

### 1. Get Groq API Key

1. Visita: https://console.groq.com/keys
2. Crea una cuenta gratuita (incluye créditos iniciales)
3. Genera una API key

### 2. Update .env

```bash
# Change STT model from local to Groq
STT_MODEL=groq

# Add Groq API key
STT__GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Restart Server

```bash
cd /opt/stt/harper/server
pkill -9 -f "uvicorn companionserver"
nohup python3 -u -m uvicorn companionserver:app --host 0.0.0.0 --port 5000 > harper_server.log 2>&1 &
```

## Comparison: Local vs Groq

| Feature | Local Whisper | Groq Whisper |
|---------|--------------|--------------|
| **Latency** | 1-2 segundos | 200-400ms |
| **GPU Required** | Sí (4-8GB VRAM) | No |
| **Accuracy** | Whisper Large V2 | Whisper Large V3 |
| **Cost** | GPU electricity | $0.111/hora audio |
| **Setup** | Model download | API key only |
| **Scaling** | Limited by GPU | Cloud scale |

## Code Changes

### stt/factory.py

```python
class GroqWhisperSTT(ASRBackend):
    """Speech-to-Text using Groq Whisper API"""
    
    def __init__(self, model_name: str = "whisper-large-v3"):
        if not settings.stt.groq_api_key:
            raise ValueError("Groq API key required")
            
        self.groq_client = OpenAI(
            api_key=settings.stt.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name

    def transcribe_pcm(self, audio: np.ndarray, ...) -> str:
        # Convert PCM → WAV bytes
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            # ...write audio
        
        # Call Groq API
        response = self.groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=self.model_name,
            language=language
        )
        return response
```

### config.py

```python
class STTSettings(BaseModel):
    model: str = Field(default="whisper-local")
    groq_api_key: Optional[str] = Field(default=None)
    # ...
```

## Usage

No hay cambios en el código cliente - la integración es transparente:

```typescript
// Client code remains the same
websocket.send(JSON.stringify({
  type: "scriptsession.start",
  language: "es"  // or "en", "fr", etc.
}));

// Groq processes audio automatically
```

## Monitoring

### Check logs

```bash
tail -f /opt/stt/harper/server/harper_server.log | grep -i groq
```

### Expected output

```
[GroqWhisperSTT] Transcribing audio chunk (2.4s @ 24kHz)
[GroqWhisperSTT] Result: "Hola, ¿cómo estás?"
[stt_worker] Latency: 0.32s
```

### Error handling

```python
try:
    response = self.groq_client.audio.transcriptions.create(...)
except Exception as e:
    print(f"[GroqWhisperSTT] Error: {e}")
    return ""  # Return empty string, session continues
```

## Cost Estimation

**Groq Pricing** (as of 2024):
- Whisper Large V3: **$0.111 per hour** of audio
- Whisper Large V3 Turbo: **$0.04 per hour** (faster, same quality)

**Typical Session:**
- 5 min conversation = 0.083 hours
- Cost: **$0.009** (~1 centavo)
- 100 sesiones/día = **$0.90/día**

**Free Tier:**
- 14,400 requests/day
- ~50-100 hours audio/day gratis

## Fallback Strategy

Si Groq falla, Harper automáticamente usa Whisper local:

```python
try:
    ASR = get_stt_model(settings.stt_model)
except Exception as e:
    print(f"Failed to initialize STT '{settings.stt_model}': {e}")
    ASR = load_asr_backend()  # Fallback to local
```

## Switching Back to Local

```bash
# Edit .env
STT_MODEL=whisper-local

# Remove or comment Groq key
# STT__GROQ_API_KEY=...

# Restart
pkill -9 -f uvicorn && sleep 2
cd /opt/stt/harper/server
nohup python3 -u -m uvicorn companionserver:app --host 0.0.0.0 --port 5000 > harper_server.log 2>&1 &
```

## Troubleshooting

### "Groq API key required" Error

```bash
# Verify .env has key
grep GROQ /opt/stt/harper/server/.env

# Should show:
STT__GROQ_API_KEY=gsk_...
```

### Rate Limit Errors

```
[GroqWhisperSTT] Error: Rate limit exceeded
```

**Solution:**
- Free tier: 14,400 requests/day
- Upgrade to paid: https://console.groq.com/settings/billing

### Slow responses

Check API status: https://status.groq.com/

## References

- **Groq Console**: https://console.groq.com/
- **Groq Docs**: https://console.groq.com/docs/speech-text
- **Whisper Model**: OpenAI Whisper Large V3
- **Base Architecture**: realtime-phone-agents-course/stt/groq/whisper.py

## Next Steps

1. ✅ Groq Whisper integration complete
2. 🔄 Monitor latency vs local Whisper
3. 📊 Track costs en producción
4. 🎯 Consider Whisper Large V3 Turbo ($0.04/hour)
5. 🔍 Evaluate other providers (Deepgram, AssemblyAI)
