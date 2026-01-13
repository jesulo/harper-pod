# Harper RunPod Architecture

## 🎯 Objetivo

Mover Whisper STT y Chatterbox TTS de local a RunPod con:
- **Streaming real** para ambos servicios
- **Auto-scaling inteligente**: warm standby + creación on-demand
- **Bajo costo**: pods detenidos cuando no hay clientes

---

## 📊 Análisis Actual

### Whisper Local (STT)
```python
# stt/asr.py - NO es streaming
def transcribe_pcm(self, pcm_bytes, sample_rate, channels, language):
    audio = _pcm_int16_to_f32_mono(pcm_bytes, channels)  
    audio = _ensure_sr(audio, sample_rate, WHISPER_SR)
    out = self.pipe(audio)  # ← Procesa BATCH completo
    return text.strip()  # ← Devuelve texto completo
```

**Problema:** Whisper procesa audio acumulado de golpe, NO genera transcripciones parciales.

**Solución:** Usar `faster-whisper` con VAD chunks (similar a realtime-phone).

### Chatterbox Local (TTS)
```python
# tts/chatter_infer.py - SÍ es streaming
async for evt in tts_model.generate_stream(text, ...):  # ← STREAMING
    audio_chunk = evt.get("audio")
    opus_frames = opus_enc.encode(audio_chunk)
    await websocket.send(opus_frames)  # ← Envía 20ms frames
```

**Problema:** Solo funciona local con GPU, no en RunPod serverless.

**Solución:** Adaptar a WebSocket API persistente en RunPod pod.

---

## 🏗️ Arquitectura Propuesta

### Opción A: 1 Pod Multi-Servicio (Recomendado)

```
┌─────────────────────────────────────────┐
│  RunPod Pod: Harper Voice Services      │
│  GPU: RTX 4090 (24GB VRAM)              │
├─────────────────────────────────────────┤
│  ┌────────────────────────────────┐     │
│  │ FastAPI Server (Port 8000)     │     │
│  ├────────────────────────────────┤     │
│  │ /ws/stt - Whisper (8GB VRAM)   │     │
│  │ /ws/tts - Chatterbox (12GB)    │     │
│  │ /health - Healthcheck          │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
```

**Pros:**
- ✅ 1 solo pod = costos reducidos (~$0.80/hora RTX 4090)
- ✅ Compartir GPU eficientemente
- ✅ Warm standby más económico

**Contras:**
- ❌ Si un servicio falla, afecta ambos
- ❌ Menos escalable (no puede separar cargas)

### Opción B: 2 Pods Separados

```
Pod 1: Whisper STT       Pod 2: Chatterbox TTS
RTX 3090 (8GB)           RTX 4090 (24GB)
$0.50/hora               $0.80/hora
```

**Pros:**
- ✅ Escalado independiente
- ✅ Fallas aisladas
- ✅ Optimización de hardware por servicio

**Contras:**
- ❌ Costo doble (~$1.30/hora)
- ❌ Complejidad de gestión

**Decisión:** **Opción A** (1 pod) para MVP, migrar a Opción B si escala.

---

## 🔄 Sistema de Auto-Scaling

### Estados de Pod

```
┌──────────┐  create    ┌──────────┐  start   ┌───────────┐
│ STOPPED  │────────────>│  READY   │─────────>│  RUNNING  │
│ (Warm)   │            │ (Standby)│          │ (Serving) │
└──────────┘            └──────────┘          └───────────┘
     ↑                       ↑                       │
     │                       │        stop           │
     └───────────────────────┴───────────────────────┘
           (siempre mantener 1 STOPPED)
```

### Flujo de Operación

**1. Estado Inicial (Sin clientes):**
```
Pod 1: STOPPED (warm standby)
Costo: $0/hora (solo storage ~$2/mes)
```

**2. Cliente 1 se conecta:**
```
1. Detectar nueva sesión WebSocket
2. start_pod("pod_1") → RUNNING (~30-60s)
3. create_pod("pod_2") → STOPPED (nuevo standby)
4. Cliente usa Pod 1
```

**3. Cliente 1 activo + Cliente 2 se conecta:**
```
Pod 1: RUNNING (Cliente 1)
Pod 2: STOPPED → start_pod("pod_2") (~30-60s)
create_pod("pod_3") → STOPPED
```

**4. Cliente 1 desconecta:**
```
stop_pod("pod_1") después de 5 min idle
Pod 1: RUNNING → STOPPED
Mantener al menos 1 STOPPED siempre
```

### Tiempos de Inicio

| Estado | Tiempo | Costo |
|--------|--------|-------|
| **Cold Start** (crear desde 0) | 3-5 min | $0 durante build |
| **Warm Start** (STOPPED → RUNNING) | 30-60 seg | $0 mientras STOPPED |
| **Hot** (ya RUNNING) | 0 seg | $0.80/hora |

**Estrategia:** Siempre mantener **1 pod STOPPED** = warm start de 30-60s (aceptable).

---

## 📦 Implementación de Servicios

### 1. Dockerfile RunPod

```dockerfile
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget

# Install Whisper (faster-whisper)
RUN pip install faster-whisper ctranslate2

# Install Chatterbox
RUN pip install --no-deps chatterbox-tts

# Install server dependencies
COPY requirements-runpod.txt /requirements.txt
RUN pip install -r /requirements.txt

# Pre-download models (baked in image)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cuda')"
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cuda')"

WORKDIR /app
COPY runpod_server.py /app/
COPY utils/ /app/utils/

EXPOSE 8000

CMD ["python", "runpod_server.py"]
```

### 2. FastAPI Server (runpod_server.py)

```python
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from chatterbox.tts import ChatterboxTTS
import asyncio
import numpy as np

app = FastAPI()

# Load models at startup (baked in pod)
whisper_model = None
tts_model = None

@app.on_event("startup")
def load_models():
    global whisper_model, tts_model
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    tts_model = ChatterboxTTS.from_pretrained(device="cuda")
    print("✅ Models loaded and ready")

@app.websocket("/ws/stt")
async def stt_endpoint(websocket: WebSocket):
    """Whisper STT streaming"""
    await websocket.accept()
    audio_buffer = b""
    
    while True:
        data = await websocket.receive_bytes()
        audio_buffer += data
        
        # Transcribe when buffer > 3 seconds
        if len(audio_buffer) >= 48000 * 2:  # 3s @ 16kHz int16
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = whisper_model.transcribe(audio_np, beam_size=1)
            
            text = " ".join([seg.text for seg in segments])
            await websocket.send_json({"text": text, "is_final": False})
            audio_buffer = b""

@app.websocket("/ws/tts")
async def tts_endpoint(websocket: WebSocket):
    """Chatterbox TTS streaming"""
    await websocket.accept()
    opus_enc = OpusEncoder(sr=24000)
    seq = 0
    
    while True:
        message = await websocket.receive_json()
        text = message["text"]
        voice_audio = load_voice(message.get("voice_id", "default"))
        
        # Stream audio chunks
        async for audio_chunk in tts_model.generate_stream(
            text, 
            audio_prompt=voice_audio,
            chunk_size=32
        ):
            audio = audio_chunk["audio"].cpu().numpy()[0]
            opus_frames = opus_enc.encode(audio)
            
            for frame in opus_frames:
                packed = pack_frame(seq, int(time.time() * 1e6), frame)
                await websocket.send_bytes(packed)
                seq += 1

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "whisper": whisper_model is not None,
        "tts": tts_model is not None
    }
```

### 3. Pod Management (scripts/runpod/harper_pod_manager.py)

```python
import runpod
from realtime_phone_agents.config import settings

class HarperPodManager:
    def __init__(self, api_key: str):
        runpod.api_key = api_key
        self.active_pods = []
        self.standby_pod = None
    
    def create_standby_pod(self):
        """Create a STOPPED pod for warm standby"""
        pod = runpod.create_pod(
            name="harper-voice-standby",
            image_name="your-docker-user/harper-voice-services:latest",
            gpu_type_id="NVIDIA GeForce RTX 4090",
            cloud_type="SECURE",
            gpu_count=1,
            volume_in_gb=30,
            ports="8000/http",
            env={
                "MODEL_CACHE": "/workspace/models"
            }
        )
        
        # Stop immediately for warm standby
        runpod.stop_pod(pod["id"])
        self.standby_pod = pod["id"]
        return pod["id"]
    
    def start_pod(self, pod_id: str):
        """Activate warm standby pod"""
        runpod.resume_pod(pod_id)
        # Wait for healthcheck
        wait_for_pod_ready(pod_id, timeout=60)
        self.active_pods.append(pod_id)
        return get_pod_url(pod_id)
    
    def stop_pod(self, pod_id: str, delay_minutes: int = 5):
        """Stop pod after idle period"""
        asyncio.create_task(self._delayed_stop(pod_id, delay_minutes))
    
    async def _delayed_stop(self, pod_id: str, delay: int):
        await asyncio.sleep(delay * 60)
        if not is_pod_active(pod_id):
            runpod.stop_pod(pod_id)
            self.active_pods.remove(pod_id)
    
    def ensure_standby(self):
        """Always maintain 1 stopped pod"""
        if self.standby_pod is None:
            self.standby_pod = self.create_standby_pod()
```

---

## 🔌 Integración con Harper

### Modificar `companionserver.py`

```python
from scripts.runpod.harper_pod_manager import HarperPodManager

pod_manager = HarperPodManager(api_key=settings.runpod_api_key)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    
    # Start pod when client connects
    if settings.use_runpod:
        pod_url = await pod_manager.start_pod(pod_manager.standby_pod)
        sess.stt_endpoint = f"{pod_url}/ws/stt"
        sess.tts_endpoint = f"{pod_url}/ws/tts"
        
        # Create new standby
        pod_manager.ensure_standby()
    
    # ... rest of session logic
    
    # Stop pod when client disconnects
    @ws.on_disconnect
    async def cleanup():
        if settings.use_runpod:
            pod_manager.stop_pod(sess.pod_id, delay_minutes=5)
```

---

## 💰 Análisis de Costos

### Escenario 1: 10 clientes/día, 30 min/sesión

```
Total tiempo activo: 10 * 0.5h = 5 horas/día
Costo RTX 4090: $0.80/hora * 5h = $4/día = $120/mes
Warm standby storage: $2/mes
Total: ~$122/mes
```

### Escenario 2: 100 clientes/día, 15 min/sesión

```
Total tiempo activo: 100 * 0.25h = 25 horas/día
Pods paralelos necesarios: ~3-5 (depende concurrencia)
Costo: $0.80 * 25h * 30 días = $600/mes
Standby storage: $10/mes
Total: ~$610/mes
```

### Comparación: Local vs RunPod

| Concepto | Local (GPU propia) | RunPod |
|----------|-------------------|--------|
| **Hardware inicial** | $1,500 (RTX 4090) | $0 |
| **Electricidad/mes** | $50-100 | $0 |
| **Mantenimiento** | Tu tiempo | $0 |
| **Escalado** | Imposible | Automático |
| **Costo/mes (10 clientes)** | $50 + amortización | $122 |
| **Costo/mes (100 clientes)** | Same (bottleneck) | $610 |

**Ventaja RunPod:** Paga solo por uso real, escala según demanda.

---

## 🚀 Plan de Implementación

### Fase 1: Setup Básico
1. ✅ Analizar arquitectura actual
2. ⏳ Crear Dockerfile con Whisper + Chatterbox
3. ⏳ Implementar FastAPI server con WebSocket
4. ⏳ Test local del servidor

### Fase 2: RunPod Integration
5. ⏳ Build y push imagen Docker
6. ⏳ Crear primer pod en RunPod
7. ⏳ Implementar pod manager (create/start/stop)
8. ⏳ Test conexión Harper → RunPod

### Fase 3: Auto-Scaling
9. ⏳ Sistema warm standby
10. ⏳ Detección de sesiones activas
11. ⏳ Auto-stop después de idle
12. ⏳ Monitoring y alertas

### Fase 4: Optimización
13. ⏳ Fine-tuning de tiempos (idle timeout)
14. ⏳ Cache de voces pre-cargadas
15. ⏳ Métricas de uso y costos
16. ⏳ Fallback a local si RunPod falla

---

## 📝 Archivos a Crear

```
harper-clean/
├── docker/
│   └── runpod/
│       ├── Dockerfile
│       ├── requirements-runpod.txt
│       └── runpod_server.py
├── scripts/
│   └── runpod/
│       ├── __init__.py
│       ├── harper_pod_manager.py
│       ├── create_harper_pod.py
│       └── monitor_pods.py
├── server/
│   ├── stt/
│   │   └── runpod_whisper.py  # Cliente WebSocket STT
│   └── tts/
│       └── runpod_chatterbox.py  # Cliente WebSocket TTS
└── .env (añadir):
    RUNPOD__API_KEY=...
    RUNPOD__USE_RUNPOD=true
    RUNPOD__GPU_TYPE=NVIDIA GeForce RTX 4090
```

---

## ⚠️ Consideraciones

### Whisper NO es verdadero streaming
- Faster-whisper transcribe chunks completos (3-5s)
- NO genera transcripciones parciales palabra por palabra
- Alternativa: Usar Groq Whisper API (verdadero streaming)

### Latencia total esperada
- Warm start: 30-60s (pod activation)
- Primera respuesta TTS: 200-500ms (después de activación)
- STT latency: 1-3s por chunk

### Fallback strategy
Si RunPod falla:
1. Intentar otro pod en standby
2. Si todos fallan → fallback a local
3. Alertar al admin

---

¿Procedo con la implementación de los archivos?
