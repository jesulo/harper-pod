"""
STT Model Factory - Dynamic Speech-to-Text backend selection
"""
from typing import Optional
import os
from .base import ASRBackend
from .asr import HFWhisperBackend


class GroqWhisperSTT(ASRBackend):
    """Groq Whisper API implementation"""
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required for Groq STT. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int, channels: int, language: Optional[str] = None) -> str:
        """Transcribe PCM audio using Groq API"""
        if not pcm_bytes:
            return ""
        
        try:
            # Convert PCM to WAV format for API
            import io
            import wave
            import numpy as np
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)
            
            wav_buffer.seek(0)
            
            # Send to Groq API
            response = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer.getvalue()),
                model=self.model,
                response_format="text"
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Groq STT error: {e}")
            return ""


def get_stt_model(model_name: str = "whisper-local", **kwargs) -> ASRBackend:
    """
    STT Model Factory - Get STT backend by name
    
    Available models:
    - "whisper-local": Local HuggingFace Whisper (default)
    - "groq": Groq API Whisper (fast, requires GROQ_API_KEY)
    
    Args:
        model_name: Name of the STT model to use
        **kwargs: Additional arguments for model initialization
        
    Returns:
        ASRBackend instance
    """
    if model_name == "whisper-local":
        model_id = kwargs.get("model_id", "openai/whisper-large-v3-turbo")
        device = kwargs.get("device", None)
        return HFWhisperBackend(model_id=model_id, device=device)
        
    elif model_name == "groq":
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model", "whisper-large-v3")
        return GroqWhisperSTT(api_key=api_key, model=model)
        
    else:
        available = ["whisper-local", "groq"]
        raise ValueError(f"Invalid STT model: {model_name}. Available: {available}")