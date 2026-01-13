"""
Simple API Server for Harper Pod
Provides REST API endpoints for:
- Whisper STT (local)
- Chatterbox TTS (local)
- Voice management (CRUD)
"""
import os
import io
import base64
import tempfile
import shutil
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Voice library configuration
VOICES_DIR = Path("./voices")
VOICES_METADATA_FILE = VOICES_DIR / "voices.json"
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DEFAULT_VOICE_PATH = "./audio_samples/output_full.wav"

# Initialize FastAPI
app = FastAPI(
    title="Harper Pod API",
    description="Local Whisper STT and Chatterbox TTS APIs",
    version="1.0.0"
)

# Global model holders
whisper_model = None
chatterbox_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
CHATTERBOX_MODEL_PATH = os.getenv("CHATTERBOX_MODEL_PATH", "./chatterbox_infer/models")

# Pydantic models for request/response
class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None

class TTSRequest(BaseModel):
    input: str
    model: str = "chatterbox"
    voice: str = "default"
    language: str = "en"
    speed: float = 1.0
    response_format: str = "mp3"

class VoiceInfo(BaseModel):
    name: str
    filename: str
    created_at: str
    size_bytes: int

class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]
    default_voice: str

class VoiceCreateResponse(BaseModel):
    success: bool
    name: str
    filename: str
    message: str

def load_whisper():
    """Load Whisper model for STT"""
    global whisper_model
    if whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
            whisper_model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=device,
                compute_type="float16" if device == "cuda" else "int8"
            )
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            # Fallback to transformers
            import whisper
            whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
            print("Using openai-whisper fallback")
    return whisper_model

def load_chatterbox():
    """Load Chatterbox TTS model"""
    global chatterbox_model
    if chatterbox_model is None:
        try:
            from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
            print("Loading Chatterbox Multilingual TTS model from HuggingFace...")
            chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            print("Chatterbox model loaded successfully")
        except Exception as e:
            print(f"Error loading Chatterbox: {e}")
            raise
    return chatterbox_model

def _load_voices_metadata() -> dict:
    """Load voices metadata from JSON file"""
    if VOICES_METADATA_FILE.exists():
        with open(VOICES_METADATA_FILE, "r") as f:
            return json.load(f)
    return {"voices": {}, "default": "default"}

def _save_voices_metadata(metadata: dict):
    """Save voices metadata to JSON file"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOICES_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def _get_voice_path(voice_name: str) -> str:
    """Get the file path for a voice by name"""
    if voice_name == "default":
        return DEFAULT_VOICE_PATH
    
    metadata = _load_voices_metadata()
    if voice_name in metadata["voices"]:
        voice_path = VOICES_DIR / metadata["voices"][voice_name]["filename"]
        if voice_path.exists():
            return str(voice_path)
    
    # Fallback to default if voice not found
    print(f"Voice '{voice_name}' not found, using default")
    return DEFAULT_VOICE_PATH

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 60)
    print("Starting Harper Pod API Server")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create voices directory
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Voices directory: {VOICES_DIR.absolute()}")
    
    # Load models
    load_whisper()
    load_chatterbox()
    
    print("All models loaded. Server ready!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    metadata = _load_voices_metadata()
    return {
        "status": "healthy",
        "device": device,
        "whisper_loaded": whisper_model is not None,
        "chatterbox_loaded": chatterbox_model is not None,
        "voices_count": len(metadata.get("voices", {}))
    }

# =============================================================================
# VOICE MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/v1/voices", response_model=VoiceCreateResponse)
async def create_voice(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """
    Create a new voice from an audio file.
    
    Args:
        file: Audio file (wav, mp3, flac, m4a, ogg)
        name: Unique name for the voice
    
    Returns:
        VoiceCreateResponse with success status
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {SUPPORTED_AUDIO_FORMATS}"
        )
    
    # Sanitize voice name (remove special characters)
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    
    # Check if voice already exists
    metadata = _load_voices_metadata()
    if safe_name in metadata["voices"]:
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{safe_name}' already exists. Delete it first or use a different name."
        )
    
    # Create voices directory if needed
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the audio file
    filename = f"{safe_name}{file_ext}"
    voice_path = VOICES_DIR / filename
    
    try:
        content = await file.read()
        with open(voice_path, "wb") as f:
            f.write(content)
        
        # Update metadata
        metadata["voices"][safe_name] = {
            "filename": filename,
            "created_at": datetime.now().isoformat(),
            "size_bytes": len(content)
        }
        _save_voices_metadata(metadata)
        
        return VoiceCreateResponse(
            success=True,
            name=safe_name,
            filename=filename,
            message=f"Voice '{safe_name}' created successfully"
        )
    except Exception as e:
        # Clean up on error
        if voice_path.exists():
            voice_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error saving voice: {str(e)}")

@app.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices():
    """
    List all available voices.
    
    Returns:
        VoiceListResponse with list of voices
    """
    metadata = _load_voices_metadata()
    voices = []
    
    for name, info in metadata.get("voices", {}).items():
        voice_path = VOICES_DIR / info["filename"]
        if voice_path.exists():
            voices.append(VoiceInfo(
                name=name,
                filename=info["filename"],
                created_at=info.get("created_at", "unknown"),
                size_bytes=info.get("size_bytes", voice_path.stat().st_size)
            ))
    
    return VoiceListResponse(
        voices=voices,
        default_voice=metadata.get("default", "default")
    )

@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """
    Delete a voice by name.
    
    Args:
        name: Name of the voice to delete
    
    Returns:
        Success message
    """
    if name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default voice")
    
    metadata = _load_voices_metadata()
    
    if name not in metadata["voices"]:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    
    # Delete the audio file
    voice_info = metadata["voices"][name]
    voice_path = VOICES_DIR / voice_info["filename"]
    
    try:
        if voice_path.exists():
            voice_path.unlink()
        
        # Update metadata
        del metadata["voices"][name]
        
        # Reset default if deleted voice was default
        if metadata.get("default") == name:
            metadata["default"] = "default"
        
        _save_voices_metadata(metadata)
        
        return {
            "success": True,
            "message": f"Voice '{name}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting voice: {str(e)}")

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """
    OpenAI-compatible Whisper transcription endpoint
    
    Args:
        file: Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
        model: Model ID (ignored, uses local Whisper)
        language: Language code (e.g., 'en', 'es', 'fr')
        prompt: Optional text to guide the model's style
        response_format: json, text, srt, verbose_json, vtt
        temperature: Sampling temperature (0-1)
    
    Returns:
        Transcription result in requested format
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and transcribe
        model = load_whisper()
        
        # Use faster-whisper API
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature
        )
        
        # Collect all segments
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)
        
        full_text = " ".join(text_segments).strip()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Format response
        if response_format == "text":
            return Response(content=full_text, media_type="text/plain")
        elif response_format == "json":
            return {
                "text": full_text,
                "language": info.language if hasattr(info, 'language') else language,
                "duration": info.duration if hasattr(info, 'duration') else None
            }
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": info.language if hasattr(info, 'language') else language,
                "duration": info.duration if hasattr(info, 'duration') else None,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    }
                    for i, seg in enumerate(segments)
                ]
            }
        else:
            return {"text": full_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint using Chatterbox
    
    Args:
        request: TTSRequest with text, voice, language, speed, etc.
    
    Returns:
        Audio file in requested format
    """
    try:
        model = load_chatterbox()
        
        # Map language code to language_id for Chatterbox
        # Supported: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
        # Chatterbox expects the language code string, not a numeric ID
        
        # Get voice path based on request.voice parameter
        audio_prompt_path = _get_voice_path(request.voice)
        print(f"[TTS] Using voice: {request.voice} -> {audio_prompt_path}")
        
        # Generate audio using Chatterbox with voice cloning
        audio_array = model.generate(
            text=request.input,
            language_id=request.language.lower(),
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.68,
            cfg_weight=0.55,
            temperature=0.75,
            repetition_penalty=1.3,
            min_p=0.02,
            top_p=0.9
        )
        
        # Convert to the requested format
        # Chatterbox outputs 24kHz audio
        sample_rate = 24000
        
        # Convert numpy array to audio bytes
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # Add batch dimension
        
        # Save to buffer
        buffer = io.BytesIO()
        
        if request.response_format == "mp3":
            # Convert to mp3 using torchaudio
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate,
                format="mp3"
            )
            media_type = "audio/mpeg"
        elif request.response_format == "wav":
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate,
                format="wav"
            )
            media_type = "audio/wav"
        elif request.response_format == "opus":
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate,
                format="opus"
            )
            media_type = "audio/opus"
        else:
            # Default to wav
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate,
                format="wav"
            )
            media_type = "audio/wav"
        
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/v1/audio/speech/stream")
async def generate_speech_stream(request: TTSRequest, req: Request):
    """
    Generate speech from text using Chatterbox TTS with streaming
    Compatible with OpenAI TTS API but returns audio in chunks
    Cancels generation if client disconnects
    
    Returns:
        Streaming audio response in WAV format
    """
    try:
        model = load_chatterbox()
        
        # Get voice path based on request.voice parameter
        audio_prompt_path = _get_voice_path(request.voice)
        print(f"[TTS Stream] Using voice: {request.voice} -> {audio_prompt_path}")
        
        async def audio_generator():
            """Generate audio chunks from TTS stream - TRUE STREAMING"""
            # Same constants as harper-clean
            TRIM = 3600
            OVERLAP = int(0.04 * 24000)
            sr = 24000
            prev_audio_end_at = 0
            
            # Send WAV header for streaming (uses max size - actual audio will be smaller)
            wav_header = _create_wav_header(channels=1, sample_rate=sr, bits_per_sample=16)
            yield wav_header
            
            total_samples = 0
            
            async for chunk_data in model.generate_stream(
                text=request.input,
                language_id=request.language.lower(),
                audio_prompt_path=audio_prompt_path,
                chunk_size=32,
                exaggeration=0.68,
                cfg_weight=0.55,
                temperature=0.75,
                repetition_penalty=1.3,
                min_p=0.02,
                top_p=0.9,
                # max_cache_len uses default from t3no.py (3000)
            ):
                # Check if client disconnected - cancel generation
                if await req.is_disconnected():
                    print(f"[TTS STREAM] Client disconnected, cancelling generation")
                    break
                
                if chunk_data["type"] == "chunk":
                    # Get accumulated audio (torch.Tensor) - same as harper-clean
                    wav: torch.Tensor = chunk_data["audio"]
                    
                    if wav is None or wav.numel() == 0:
                        continue
                    
                    # Ensure correct shape (channels, samples)
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    
                    total_audio_length = wav.shape[-1]
                    
                    # Skip if no new audio
                    if total_audio_length - prev_audio_end_at <= 0:
                        continue
                    
                    # Extract NEW part - exactly like harper-clean
                    if total_audio_length < TRIM + OVERLAP + int(sr*0.8):
                        new_part = wav[:, prev_audio_end_at:]
                    else:
                        # Apply fade with overlap
                        fade_out_ms = 20
                        fade_samples = int(24000 * fade_out_ms / 1000)
                        fade_samples_ip = int(24000 * 15 / 1000)
                        new_part = wav[:, max(prev_audio_end_at-fade_samples, 0):-TRIM].clone()
                        
                        # Fade curves
                        fade_curve = torch.tensor(np.linspace(1, 0, fade_samples), device=new_part.device)
                        fade_curve_up = torch.tensor(np.linspace(0, 1, fade_samples_ip), device=new_part.device)
                        
                        # Apply fades
                        new_part[:, -fade_samples:] = new_part[:, -fade_samples:] * fade_curve
                        new_part[:, :fade_samples_ip] = new_part[:, :fade_samples_ip] * fade_curve_up
                    
                    # Update position
                    prev_audio_end_at = total_audio_length - TRIM
                    
                    # Convert to PCM int16 and yield immediately (TRUE STREAMING)
                    audio_np = new_part.cpu().numpy()
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    total_samples += audio_int16.shape[-1]
                    
                    # Yield raw PCM bytes
                    yield audio_int16.tobytes()
                    
                elif chunk_data["type"] == "eos":
                    break
        
        def _create_wav_header(channels, sample_rate, bits_per_sample, data_size=0xFFFFFFFF):
            """Create WAV header for streaming (uses max size by default)"""
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            # Use a large, but not max, value for chunk size to avoid overflow issues
            chunk_size = 36 + data_size if data_size != 0xFFFFFFFF else 0x7FFFFFFF - 36
            
            header = bytearray()
            header.extend(b'RIFF')
            header.extend(chunk_size.to_bytes(4, 'little'))
            header.extend(b'WAVE')
            header.extend(b'fmt ')
            header.extend((16).to_bytes(4, 'little'))
            header.extend((1).to_bytes(2, 'little'))  # PCM
            header.extend(channels.to_bytes(2, 'little'))
            header.extend(sample_rate.to_bytes(4, 'little'))
            header.extend(byte_rate.to_bytes(4, 'little'))
            header.extend(block_align.to_bytes(2, 'little'))
            header.extend(bits_per_sample.to_bytes(2, 'little'))
            header.extend(b'data')
            header.extend(data_size.to_bytes(4, 'little'))
            return bytes(header)
        
        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_stream.wav",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS streaming error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Harper Pod API",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "transcription": "/v1/audio/transcriptions",
            "tts": "/v1/audio/speech",
            "tts_stream": "/v1/audio/speech/stream",
            "voices_list": "/v1/voices [GET]",
            "voices_create": "/v1/voices [POST]",
            "voices_delete": "/v1/voices/{name} [DELETE]",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8007))
    uvicorn.run(app, host="0.0.0.0", port=port)
