"""
TTS Factory - Modular Text-to-Speech backend selection  
Based on realtime-phone-agents-course architecture
"""
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Generator
import numpy as np
import httpx
from config import settings


class BaseTTSModel(ABC):
    """Abstract base class for TTS models"""
    
    @abstractmethod
    async def stream_speech(self, text: str, voice: str = "tara", sample_rate: int = 24000) -> AsyncGenerator[np.ndarray, None]:
        """Stream speech audio from text"""
        pass


class ChatterboxTTSWrapper(BaseTTSModel):
    """Wrapper for original Harper Chatterbox TTS - maintains compatibility"""
    
    def __init__(self):
        # This will use the original Harper TTS system
        pass
    
    async def stream_speech(self, text: str, voice: str = "tara", sample_rate: int = 24000) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream speech using original chatterbox system
        NOTE: This is a wrapper - actual implementation would integrate with chatter_infer
        """
        # Placeholder - in real implementation this would interface with chatter_infer
        yield np.array([0], dtype=np.int16)


class ResembleStreamingModel(BaseTTSModel):
    """Text-to-Speech using Resemble AI Streaming API (Chatterbox model)"""
    
    def __init__(self):
        print("[TTS Factory] Initializing ResembleStreamingModel...")
        from tts.resemble_streaming import ResembleStreamingWrapper
        try:
            self.wrapper = ResembleStreamingWrapper()
            print("[TTS Factory] ✅ ResembleStreamingModel initialized successfully")
        except Exception as e:
            print(f"[TTS Factory] ❌ Failed to initialize ResembleStreamingModel: {e}")
            raise
    
    async def stream_speech(self, text: str, voice: str = "tara", sample_rate: int = 24000) -> AsyncGenerator[np.ndarray, None]:
        """Stream speech using Resemble AI streaming API"""
        print(f"[TTS Factory] ResembleStreamingModel.stream_speech called with text: '{text[:50]}...'")
        async for chunk in self.wrapper.stream_speech(text, voice, sample_rate):
            yield chunk


class TogetherTTSModel(BaseTTSModel):
    """Text-to-Speech using Together AI API"""
    
    MIN_CHUNK_SIZE = 1024  # 512 samples (about 21ms at 24kHz)
    
    def __init__(self):
        if not settings.tts.together_api_key:
            raise ValueError("Together AI API key required. Set TTS__TOGETHER_API_KEY in .env")
        
        self.api_key = settings.tts.together_api_key
        self.api_url = "https://api.together.xyz/v1"
        self.model = "canopylabs/orpheus-3b-0.1-ft"
        
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _stream_audio_sync(self, text: str, voice: str, sample_rate: int) -> Generator[np.ndarray, None, None]:
        """Stream audio synchronously from Together AI API"""
        speech_url = f"{self.api_url}/audio/speech"
        
        payload = {
            "model": self.model,
            "input": text.strip(),
            "voice": voice,
            "stream": True,
            "response_format": "raw",
            "response_encoding": "pcm_s16le",  # 16-bit PCM
            "sample_rate": sample_rate,
        }
        
        pcm_buffer = b""
        
        try:
            with httpx.Client(
                timeout=httpx.Timeout(300.0, connect=10.0),
                headers=self._get_headers(),
            ) as client:
                with client.stream("POST", speech_url, json=payload) as response:
                    response.raise_for_status()
                    
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        
                        pcm_buffer += chunk
                        
                        # Send complete 2-byte aligned chunks (int16 = 2 bytes per sample)
                        if len(pcm_buffer) >= self.MIN_CHUNK_SIZE:
                            complete_samples = len(pcm_buffer) // 2
                            if complete_samples > 0:
                                complete_bytes = complete_samples * 2
                                
                                audio_chunk = np.frombuffer(
                                    pcm_buffer[:complete_bytes], dtype=np.int16
                                )
                                
                                yield audio_chunk
                                pcm_buffer = pcm_buffer[complete_bytes:]
                    
                    # Flush remaining buffer
                    if pcm_buffer:
                        complete_samples = len(pcm_buffer) // 2
                        if complete_samples > 0:
                            complete_bytes = complete_samples * 2
                            
                            audio_chunk = np.frombuffer(
                                pcm_buffer[:complete_bytes], dtype=np.int16
                            )
                            yield audio_chunk
                            
        except httpx.HTTPStatusError as e:
            print(f"[TogetherTTS] API error: {e.response.status_code}")
            raise
        except Exception as e:
            print(f"[TogetherTTS] Error: {e}")
            raise
    
    async def stream_speech(self, text: str, voice: str = "tara", sample_rate: int = 24000) -> AsyncGenerator[np.ndarray, None]:
        """Stream speech from Together AI API"""
        loop = asyncio.get_running_loop()
        
        # Run sync streaming in thread to avoid blocking
        def run_sync():
            return list(self._stream_audio_sync(text, voice, sample_rate))
        
        try:
            chunks = await loop.run_in_executor(None, run_sync)
            for chunk in chunks:
                yield chunk
        except Exception as e:
            print(f"[TogetherTTS] Stream error: {e}")
            raise


def get_tts_model(model_name: str = "chatterbox", **kwargs) -> BaseTTSModel:
    """
    Get TTS model based on configuration
    
    Args:
        model_name: TTS model type ('chatterbox', 'resemble', 'together')
        **kwargs: Additional model-specific arguments
    
    Returns:
        BaseTTSModel instance
    
    Raises:
        ValueError: If model_name is invalid
    
    Supported models:
    - chatterbox: Original Harper Chatterbox TTS (local deployment)
    - resemble: Resemble AI Streaming API (Chatterbox model, cloud-based)
    - together: Together AI TTS (cloud-based, requires API key)
    """
    if model_name == "chatterbox":
        return ChatterboxTTSWrapper()
    elif model_name == "resemble":
        return ResembleStreamingModel()
    elif model_name == "together":
        return TogetherTTSModel()
    else:
        raise ValueError(
            f"Invalid TTS model: {model_name}. "
            f"Available: chatterbox, resemble, together"
        )