"""
STT Factory - Modular Speech-to-Text backend selection
Based on realtime-phone-agents-course architecture
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from openai import OpenAI
from config import settings


class ASRBackend(ABC):
    """Abstract base class for STT backends"""
    
    @abstractmethod
    def transcribe_pcm(self, audio: np.ndarray, sample_rate: int, channels: int, language: str = "en") -> str:
        """Transcribe PCM audio to text"""
        pass


class HFWhisperBackend(ASRBackend):
    """Original Harper Whisper backend - maintains compatibility"""
    
    def __init__(self):
        # Import original Harper STT
        from stt.asr import load_asr_backend
        self._backend = load_asr_backend()
    
    def transcribe_pcm(self, audio: np.ndarray, sample_rate: int, channels: int, language: str = "en") -> str:
        return self._backend.transcribe_pcm(audio, sample_rate, channels, language)


class GroqWhisperSTT(ASRBackend):
    """Speech-to-Text using Groq Whisper API
    
    Supports:
    - whisper-large-v3-turbo: Faster, 216x real-time, 12% WER
    - whisper-large-v3: More accurate, 189x real-time, 10.3% WER
    """
    
    def __init__(self, model_name: str = "whisper-large-v3-turbo"):
        if not settings.stt.groq_api_key:
            raise ValueError("Groq API key required. Set STT__GROQ_API_KEY in .env")
            
        self.groq_client = OpenAI(
            api_key=settings.stt.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name
        print(f"[GroqWhisperSTT] Initialized with model: {model_name}")

    def transcribe_pcm(self, audio, sample_rate: int, channels: int, language: str = "en") -> str:
        """Convert PCM audio to text using Groq
        
        Args:
            audio: Can be np.ndarray or bytes (PCM int16)
            sample_rate: Audio sample rate
            channels: Number of audio channels
            language: Language code
        """
        try:
            import io
            import wave
            
            # Handle both bytes and numpy array input
            if isinstance(audio, bytes):
                # Audio already in bytes (PCM int16) - convert to numpy for processing
                audio_np = np.frombuffer(audio, dtype=np.int16)
            elif isinstance(audio, np.ndarray):
                # Ensure audio is int16
                if audio.dtype != np.int16:
                    audio_np = (audio * 32767).astype(np.int16)
                else:
                    audio_np = audio
            else:
                print(f"[GroqWhisperSTT] Unsupported audio type: {type(audio)}")
                return ""
            
            # Create wav bytes (Groq will downsample to 16KHz mono automatically)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_np.tobytes())
            
            audio_bytes = buffer.getvalue()
            
            # Ensure language is in ISO-639-1 format (e.g., 'es', 'en')
            # Groq recommends temperature=0 for transcriptions
            response = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_bytes),
                model=self.model_name,
                response_format="text",
                language=language if language not in ["auto", None, ""] else None,
                temperature=0.0
            )
            return response
            
        except Exception as e:
            print(f"[GroqWhisperSTT] Error: {e}")
            import traceback
            traceback.print_exc()
            return ""


def get_stt_model(model_name: str = "whisper-local", **kwargs) -> ASRBackend:
    """
    Get STT model based on configuration
    
    Args:
        model_name: STT model type ('whisper-local', 'groq')
        **kwargs: Additional model-specific arguments
    
    Returns:
        ASRBackend instance
    
    Raises:
        ValueError: If model_name is invalid
    """
    if model_name == "whisper-local":
        return HFWhisperBackend()
    elif model_name == "groq":
        return GroqWhisperSTT(**kwargs)
    else:
        raise ValueError(f"Invalid STT model: {model_name}. Available: whisper-local, groq")