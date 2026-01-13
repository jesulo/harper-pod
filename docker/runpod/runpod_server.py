"""
Harper Voice Services - RunPod Server
Multi-Model Support:
- Whisper large-v3-turbo (STT)
- Chatterbox Multilingual (TTS - 23+ languages)
- Chatterbox Turbo (TTS - English, ultra-low latency)
"""
import asyncio
import time
import struct
from typing import Optional, Literal
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import opuslib

# Model imports
from faster_whisper import WhisperModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Opus frame packing (compatible with Harper client)
MAGIC = b'\xA1\x51'

def pack_frame(seq: int, ts_usec: int, payload: bytes, is_final: bool = False) -> bytes:
    """Pack audio frame with header for Harper WebSocket protocol"""
    flags = 1 if is_final else 0
    header = MAGIC + struct.pack('<BBIQI', flags, 0, seq, ts_usec, len(payload))
    return header + payload

class OpusEncoder:
    """Opus encoder with carry buffer for seamless streaming"""
    def __init__(self, sr: int = 24000, channels: int = 1, bitrate: int = 32000, frame_ms: int = 20):
        self.sr = sr
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_size = sr * frame_ms // 1000  # 480 samples @ 24kHz
        self.encoder = opuslib.Encoder(sr, channels, opuslib.APPLICATION_VOIP)
        self.encoder.bitrate = bitrate
        self._carry = np.empty(0, dtype=np.float32)
    
    def encode(self, pcm_f32: np.ndarray) -> list[bytes]:
        """Encode PCM float32 to Opus packets"""
        # Prepend carry buffer
        if self._carry.size > 0:
            pcm_f32 = np.concatenate([self._carry, pcm_f32])
        
        frames = []
        n = len(pcm_f32)
        offset = 0
        
        # Encode only complete frames
        while offset + self.frame_size <= n:
            frame_f32 = pcm_f32[offset:offset + self.frame_size]
            # Convert to int16 for Opus encoder
            frame_i16 = (np.clip(frame_f32, -1.0, 1.0) * 32767).astype(np.int16)
            packet = self.encoder.encode(frame_i16.tobytes(), self.frame_size)
            frames.append(packet)
            offset += self.frame_size
        
        # Save leftover for next call
        self._carry = pcm_f32[offset:] if offset < n else np.empty(0, dtype=np.float32)
        
        return frames

# Global models
whisper_model: Optional[WhisperModel] = None
tts_multilingual: Optional[ChatterboxMultilingualTTS] = None
tts_turbo: Optional[ChatterboxTurboTTS] = None

app = FastAPI(title="Harper Voice Services", version="1.0.0")

@app.on_event("startup")
async def load_models():
    """Load all models to GPU at startup"""
    global whisper_model, tts_multilingual, tts_turbo
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Loading models on device: {device}")
    
    try:
        # Load Whisper STT
        logger.info("📥 Loading Whisper large-v3-turbo...")
        whisper_model = WhisperModel(
            "large-v3-turbo",
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        logger.success("✅ Whisper loaded")
        
        # Load Chatterbox Multilingual
        logger.info("📥 Loading Chatterbox Multilingual...")
        tts_multilingual = ChatterboxMultilingualTTS.from_pretrained(device=device)
        logger.success("✅ Chatterbox Multilingual loaded")
        
        # Load Chatterbox Turbo
        logger.info("📥 Loading Chatterbox Turbo...")
        tts_turbo = ChatterboxTurboTTS.from_pretrained(device=device)
        logger.success("✅ Chatterbox Turbo loaded")
        
        logger.success("🎉 All models ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "whisper": whisper_model is not None,
            "tts_multilingual": tts_multilingual is not None,
            "tts_turbo": tts_turbo is not None
        },
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/models")
async def list_models():
    """List available models and their capabilities"""
    return {
        "stt": {
            "whisper-large-v3-turbo": {
                "languages": "96+ languages",
                "streaming": False,
                "description": "Fast and accurate STT"
            }
        },
        "tts": {
            "chatterbox-multilingual": {
                "languages": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", 
                             "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", 
                             "sw", "tr", "zh"],
                "streaming": True,
                "description": "Multi-language TTS with voice cloning"
            },
            "chatterbox-turbo": {
                "languages": ["en"],
                "streaming": True,
                "features": ["[laugh]", "[chuckle]", "[cough]", "[sigh]"],
                "description": "Ultra-low latency English TTS with paralinguistic tags"
            }
        }
    }

@app.websocket("/ws/stt")
async def stt_endpoint(websocket: WebSocket):
    """
    Whisper STT WebSocket endpoint
    Receives: Raw audio bytes (int16 PCM)
    Sends: JSON with transcription
    """
    await websocket.accept()
    logger.info("🎤 STT WebSocket connected")
    
    audio_buffer = b""
    sample_rate = 16000  # Whisper expects 16kHz
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer += data
            
            # Transcribe when buffer >= 3 seconds
            min_samples = sample_rate * 3 * 2  # 3s @ 16kHz int16
            if len(audio_buffer) >= min_samples:
                # Convert to float32 numpy array
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe
                segments, info = whisper_model.transcribe(
                    audio_np,
                    beam_size=1,
                    language=None  # Auto-detect
                )
                
                text = " ".join([seg.text for seg in segments]).strip()
                
                if text:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": text,
                        "language": info.language,
                        "is_final": False
                    })
                    logger.info(f"📝 Transcribed: {text}")
                
                # Clear buffer
                audio_buffer = b""
                
    except WebSocketDisconnect:
        logger.info("🎤 STT WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ STT error: {e}")
        await websocket.close(code=1011, reason=str(e))

@app.websocket("/ws/tts")
async def tts_endpoint(websocket: WebSocket):
    """
    Multi-Model TTS WebSocket endpoint
    Receives: JSON with text and model selection
    Sends: Opus audio frames
    
    Message format:
    {
        "text": "Hello world",
        "model": "turbo" | "multilingual",
        "language": "en" (for multilingual),
        "voice_audio": null (optional base64 audio for cloning)
    }
    """
    await websocket.accept()
    logger.info("🎵 TTS WebSocket connected")
    
    opus_enc = OpusEncoder(sr=24000, channels=1, bitrate=32000)
    seq = 0
    session_start = time.time()
    
    try:
        while True:
            message = await websocket.receive_json()
            
            text = message.get("text", "")
            model_choice = message.get("model", "turbo")  # Default to turbo
            language = message.get("language", "en")
            voice_audio_b64 = message.get("voice_audio")
            
            if not text:
                continue
            
            logger.info(f"🎵 TTS request: '{text[:50]}...' (model={model_choice}, lang={language})")
            
            # Select model
            if model_choice == "turbo":
                if language != "en":
                    await websocket.send_json({
                        "type": "error",
                        "message": "Turbo model only supports English"
                    })
                    continue
                model = tts_turbo
            elif model_choice == "multilingual":
                model = tts_multilingual
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown model: {model_choice}"
                })
                continue
            
            # Generate audio (streaming)
            try:
                # Use streaming generation
                if model_choice == "turbo":
                    # Turbo uses generate_stream
                    audio_generator = model.generate_stream(
                        text,
                        audio_prompt_path=None,  # TODO: Support voice cloning
                        chunk_size=32
                    )
                else:
                    # Multilingual uses generate_stream
                    audio_generator = model.generate_stream(
                        text,
                        audio_prompt_path=None,  # TODO: Support voice cloning
                        language_id=language,
                        chunk_size=32
                    )
                
                # Stream audio chunks
                async for event in audio_generator:
                    audio_chunk = event.get("audio")
                    
                    if audio_chunk is None:
                        if event.get("type") == "eos":
                            # Send final frame
                            timestamp = int((time.time() - session_start) * 1e6)
                            final_frame = pack_frame(seq, timestamp, b"", is_final=True)
                            await websocket.send_bytes(final_frame)
                            seq += 1
                        continue
                    
                    # Convert tensor to numpy
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy()
                        if audio_np.ndim == 2:
                            audio_np = audio_np[0]  # Take first channel
                    else:
                        audio_np = audio_chunk
                    
                    # Encode to Opus frames
                    opus_frames = opus_enc.encode(audio_np.astype(np.float32))
                    
                    # Send each frame
                    for frame in opus_frames:
                        timestamp = int((time.time() - session_start) * 1e6)
                        packed = pack_frame(seq, timestamp, frame, is_final=False)
                        await websocket.send_bytes(packed)
                        seq += 1
                
                logger.success(f"✅ TTS completed (sent {seq} frames)")
                
            except Exception as e:
                logger.error(f"❌ TTS generation error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        logger.info("🎵 TTS WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        await websocket.close(code=1011, reason=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
