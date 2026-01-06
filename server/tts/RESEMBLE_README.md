# Resemble AI Streaming TTS Integration

Integración de la API de streaming de **Resemble AI** (modelo Chatterbox) con Harper.

## 📚 Documentación Oficial

- **API Reference**: https://docs.resemble.ai/streaming
- **Get API Key**: https://app.resemble.ai/account/api
- **Manage Voices**: https://app.resemble.ai/voices

## 🔑 Autenticación

Todas las solicitudes requieren un Bearer token:

```bash
Authorization: Bearer <YOUR_API_KEY>
```

Obtén tu API key desde el panel de Resemble AI: https://app.resemble.ai/account/api

## 🎙️ Voces Disponibles

Resemble AI **no usa nombres de voces estáticos** (como "Alloy" o "Tara"). En su lugar, utiliza **UUIDs**.

### Opciones:

1. **Voces del Sistema** (stock voices de Resemble)
2. **Clonación de Voz** (sube un clip de 10 segundos)
3. **Voice Design** (crea voces sintéticas únicas)

### Listar Voces Disponibles

```bash
curl -H "Authorization: Bearer <YOUR_API_KEY>" \
     https://api.resemble.ai/v1/voices
```

Respuesta (ejemplo):
```json
[
  {
    "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "Sarah - Professional",
    "created_at": "2024-01-15T10:30:00Z"
  },
  {
    "uuid": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "name": "John - Conversational",
    "created_at": "2024-01-20T14:15:00Z"
  }
]
```

## 🚀 Endpoints

### HTTP Streaming (Recomendado)

**URL**: `https://f.cluster.resemble.ai/stream`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Ventajas**:
- No requiere plan Business
- Pay-as-you-go
- Ideal para baja/media concurrencia
- Latencia ~200ms

### WebSocket Streaming (Business+)

**URL**: `wss://websocket.cluster.resemble.ai/stream`

**Ventajas**:
- Ultra baja latencia
- Comunicación bidireccional
- Ideal para alta concurrencia

**Desventaja**: Requiere plan Business o Enterprise

## 📦 Payload Estructura (HTTP Streaming)

```json
{
  "voice_uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "data": "Texto a sintetizar (máximo 2000 caracteres)",
  "project_uuid": "opcional-para-organizar-clips",
  "precision": "PCM_16",
  "sample_rate": 48000,
  "use_hd": false
}
```

### Parámetros

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `voice_uuid` | string | ✅ Sí | UUID de la voz a usar |
| `data` | string | ✅ Sí | Texto o SSML (max 2000 chars) |
| `project_uuid` | string | ❌ No | UUID del proyecto (organización) |
| `precision` | enum | ❌ No | `MULAW`, `PCM_16`, `PCM_24`, `PCM_32` (default) |
| `sample_rate` | integer | ❌ No | 16000, 22050, 44100, 48000 Hz |
| `use_hd` | boolean | ❌ No | Modo alta definición (aumenta latencia) |

## 🔄 Formato de Respuesta

La API responde con un **stream de audio WAV fragmentado** (`Transfer-Encoding: chunked`).

- **Inicio**: Respuesta casi instantánea (~200ms)
- **Procesamiento**: Stream binario continuo de audio PCM
- **Finalización**: Stream se cierra automáticamente

### Estructura WAV

```
[44 bytes WAV header] + [PCM audio data...]
```

El header incluye:
- Sample rate
- Channels (mono)
- Bit depth (16-bit por defecto)
- Timestamps

## 🐍 Configuración en Harper

### 1. Configurar `.env`

```bash
# TTS Model Selection
TTS_MODEL=resemble

# Resemble AI Configuration
TTS__RESEMBLE_API_KEY=your_api_key_here
TTS__RESEMBLE_VOICE_UUID=a1b2c3d4-e5f6-7890-abcd-ef1234567890
TTS__RESEMBLE_PROJECT_UUID=  # Opcional
TTS__RESEMBLE_PRECISION=PCM_16
TTS__RESEMBLE_USE_HD=false
TTS__SAMPLE_RATE=24000
```

### 2. Obtener Voice UUID

```bash
# Listar voces disponibles
curl -H "Authorization: Bearer sk_abc123..." \
     https://api.resemble.ai/v1/voices | jq
```

Copia el `uuid` de la voz que quieras usar.

### 3. Iniciar Harper

```bash
cd /opt/stt/harper/server
python companionserver.py
```

El TTS factory cargará automáticamente `ResembleStreamingModel`.

## 💡 Ejemplo de Uso (Python)

### Streaming Simple

```python
import asyncio
from tts.resemble_streaming import ResembleStreamingAPI

async def main():
    api = ResembleStreamingAPI(
        api_key="sk_abc123...",
        voice_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    )
    
    text = "Hola, esto es una prueba de Resemble AI streaming."
    
    async for audio_chunk in api.stream_audio(text):
        print(f"Received {len(audio_chunk)} audio samples")
        # Procesar/reproducir audio chunk

asyncio.run(main())
```

### Integración con Harper Factory

```python
from tts.factory import get_tts_model

# Cargar modelo desde configuración
tts = get_tts_model("resemble")

# Generar audio
async for audio_chunk in tts.stream_speech("Hello world"):
    # Audio chunk es numpy array (PCM int16)
    process_audio(audio_chunk)
```

## ⚠️ Manejo de Errores

| Código | Error | Causa Probable |
|--------|-------|----------------|
| 400 | Bad Request | Parámetros faltantes o texto >2000 chars |
| 401 | Unauthorized | API token inválido o expirado |
| 404 | Not Found | `voice_uuid` no existe |
| 429 | Too Many Requests | Límite de cuota excedido |
| 500 | Internal Error | Problema en servidores de Resemble |

### Ejemplo de Manejo

```python
try:
    async for chunk in api.stream_audio(text):
        yield chunk
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 404:
        print("Voice UUID not found")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    else:
        print(f"HTTP error: {e}")
```

## 🎯 Ventajas vs. Local Chatterbox

| Aspecto | Resemble AI (Cloud) | Local Chatterbox |
|---------|---------------------|------------------|
| **Infraestructura** | Sin GPU necesaria | Requiere GPU potente |
| **Costo** | Pay-per-use | Costo fijo GPU |
| **Latencia** | ~200ms | <50ms |
| **Escalabilidad** | ✅ Automática | ⚠️ Manual |
| **Mantenimiento** | ✅ Zero | ⚠️ Updates manuales |
| **Voces** | Clonación ilimitada | Limitado a modelo |

## 🔧 Troubleshooting

### Error: "voice_uuid required"

```bash
# Solución: Configurar UUID en .env
TTS__RESEMBLE_VOICE_UUID=your-uuid-here
```

### Error: "Invalid API key"

```bash
# Verificar API key
curl -H "Authorization: Bearer sk_abc123..." \
     https://api.resemble.ai/v1/voices
```

### Audio con clicks/artefactos

```bash
# Probar modo HD (aumenta latencia pero mejora calidad)
TTS__RESEMBLE_USE_HD=true
```

### Latencia alta

```bash
# Desactivar HD mode
TTS__RESEMBLE_USE_HD=false

# Usar precision más baja
TTS__RESEMBLE_PRECISION=PCM_16
```

## 📚 Referencias

- [Resemble AI Streaming Docs](https://docs.resemble.ai/streaming)
- [Resemble AI API Reference](https://docs.resemble.ai/api-reference)
- [Voice Cloning Guide](https://docs.resemble.ai/voice-cloning)
- [Harper Factory Pattern](../README.md#tts-factory)

## 🆘 Soporte

- **Resemble AI Support**: support@resemble.ai
- **Harper Issues**: https://github.com/jesulo/harper/issues
