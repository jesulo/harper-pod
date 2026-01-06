# Harper TTS: Comparación de Backends

Este documento compara las 3 opciones de TTS disponibles en Harper.

## 📊 Comparación General

| Característica | Local Chatterbox | Resemble AI | Together AI |
|----------------|------------------|-------------|-------------|
| **Deployment** | Local (GPU) | Cloud API | Cloud API |
| **Infraestructura** | GPU potente requerida | Sin requisitos | Sin requisitos |
| **Modelo** | Chatterbox (local) | Chatterbox (Resemble) | Orpheus-3B |
| **Latencia** | <50ms | ~200ms | ~300ms |
| **Costo Setup** | Alto (GPU) | Ninguno | Ninguno |
| **Costo Operación** | Fijo (electricidad) | Pay-per-use | Pay-per-use |
| **Escalabilidad** | Limitada | Ilimitada | Ilimitada |
| **Calidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Voces** | 1 voz (entrenada) | Clonación ilimitada | Voces predefinidas |
| **Personalización** | ⚙️ Alta | ⚙️ Media | ⚙️ Baja |

## 1️⃣ Local Chatterbox (Original Harper)

### ✅ Ventajas

- **Ultra baja latencia** (<50ms)
- **Control total** sobre el modelo
- **Parámetros personalizables**:
  - `exaggeration=0.68` (énfasis emocional)
  - `cfg_weight=0.32` (calidad vs. variabilidad)
  - `temperature=1.0` (aleatoriedad)
- **Sin límites de uso**
- **Privacidad total** (datos no salen del servidor)

### ❌ Desventajas

- **Requiere GPU potente** (8GB+ VRAM)
- **Costo inicial alto** (hardware)
- **Mantenimiento manual** (updates, bugs)
- **Escalabilidad limitada** (1 GPU = 1 sesión concurrente típicamente)
- **Una sola voz** (requiere reentrenar para cambiar)

### 🎯 Ideal Para

- **Aplicaciones con GPU dedicada**
- **Baja concurrencia** (1-5 usuarios simultáneos)
- **Requisitos de privacidad estrictos**
- **Latencia crítica** (<100ms requerida)

### 🔧 Configuración

```bash
TTS_MODEL=chatterbox
```

---

## 2️⃣ Resemble AI Streaming (Chatterbox Cloud)

### ✅ Ventajas

- **Sin GPU necesaria**
- **Misma calidad que local** (modelo Chatterbox oficial)
- **Clonación de voz ilimitada** (10 segundos de audio)
- **Voice Design** (crear voces sintéticas únicas)
- **Escalabilidad automática**
- **Mantenimiento zero**
- **Latencia aceptable** (~200ms)
- **Pay-per-use** (sin costo idle)

### ❌ Desventajas

- **Requiere internet**
- **Latencia mayor que local** (~200ms vs <50ms)
- **Costo por minuto generado** (~$0.006/min)
- **Menos control** sobre parámetros internos
- **Datos pasan por servidores de Resemble**

### 🎯 Ideal Para

- **Producción sin GPU**
- **Media/alta concurrencia** (10-100+ usuarios)
- **Múltiples voces** (clonación)
- **Prototipado rápido**
- **Escalabilidad importante**

### 🔧 Configuración

```bash
TTS_MODEL=resemble
TTS__RESEMBLE_API_KEY=sk_abc123...
TTS__RESEMBLE_VOICE_UUID=a1b2c3d4...
TTS__RESEMBLE_PRECISION=PCM_16
TTS__RESEMBLE_USE_HD=false
```

### 💰 Costos Estimados

- **Pay-as-you-go**: ~$0.006/minuto
- **100 minutos/mes**: ~$0.60
- **1000 minutos/mes**: ~$6.00

---

## 3️⃣ Together AI

### ✅ Ventajas

- **Sin GPU necesaria**
- **Modelo Orpheus-3B** (optimizado para latencia)
- **Escalabilidad automática**
- **Voces predefinidas** (no requiere clonación)
- **Pay-per-use**
- **API simple**

### ❌ Desventajas

- **Calidad inferior a Chatterbox**
- **Menos opciones de voces**
- **Latencia mayor** (~300ms)
- **Sin clonación de voz**
- **Menos control sobre parámetros**

### 🎯 Ideal Para

- **Aplicaciones no críticas**
- **Prototipado inicial**
- **Presupuesto ajustado**
- **Simplicidad sobre calidad**

### 🔧 Configuración

```bash
TTS_MODEL=together
TTS__TOGETHER_API_KEY=your_key_here
TTS__VOICE=tara
```

---

## 📈 Casos de Uso Recomendados

### Escenario 1: Asistente de Voz Personal (1-5 usuarios)

**Recomendación**: Local Chatterbox o Resemble AI

| Opción | Pros | Contras |
|--------|------|---------|
| Local | Ultra baja latencia, sin costos recurrentes | Requiere GPU |
| Resemble | Sin GPU, fácil setup | ~$5-10/mes por usuario activo |

**Decisión**: Si ya tenés GPU → Local. Si no → Resemble.

---

### Escenario 2: Call Center (100+ usuarios concurrentes)

**Recomendación**: Resemble AI

- Escalabilidad automática
- Sin límite de concurrencia
- Pay-per-use (no pagás idle)
- Clonación de voz para múltiples agentes

**Costo estimado**: ~$50-100/mes por 1000 minutos

---

### Escenario 3: Prototipo/MVP

**Recomendación**: Resemble AI o Together AI

- Setup inmediato (sin GPU)
- Pay-as-you-go (sin compromisos)
- Fácil de cambiar después

---

### Escenario 4: Máxima Privacidad (Datos Sensibles)

**Recomendación**: Local Chatterbox

- Datos **nunca** salen del servidor
- Control total sobre infraestructura
- Cumplimiento HIPAA/GDPR más sencillo

---

## 🔄 Migración Entre Backends

Harper usa un **factory pattern**, por lo que cambiar de backend es trivial:

```bash
# De local a Resemble
TTS_MODEL=resemble

# De Resemble a Together
TTS_MODEL=together

# Volver a local
TTS_MODEL=chatterbox
```

No requiere cambios de código.

---

## 🧪 Testing de Backends

### Test Resemble AI

```bash
cd /opt/stt/harper/server/tts
export RESEMBLE_API_KEY=sk_abc123...
export RESEMBLE_VOICE_UUID=a1b2c3d4...
python test_resemble.py
```

### Test Together AI

```bash
cd /opt/stt/harper/server
export TTS__TOGETHER_API_KEY=your_key
TTS_MODEL=together python companionserver.py
```

---

## 💡 Recomendaciones Finales

### Para Desarrollo

1. **Empezar con Resemble AI** (setup rápido, sin GPU)
2. **Clonar tu voz** (10 segundos de audio)
3. **Probar latencia** en tu red

### Para Producción (Low Volume)

- **<10 usuarios concurrentes** → Local Chatterbox (si tenés GPU)
- **10-100 usuarios** → Resemble AI
- **100+ usuarios** → Resemble AI (definitivo)

### Para Producción (High Volume)

- **Siempre Resemble AI** (escalabilidad infinita)
- Considerar plan Enterprise si >100k minutos/mes

### Para Aplicaciones Offline

- **Local Chatterbox** (única opción sin internet)

---

## 📚 Referencias

- [Resemble AI Documentation](./RESEMBLE_README.md)
- [Harper Factory Pattern](../README.md)
- [Chatterbox Local Setup](../../docs/local_tts.md)
- [Together AI Docs](https://docs.together.ai/reference/audio-speech)
