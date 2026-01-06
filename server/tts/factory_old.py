"""
TTS Model Factory - Dynamic Text-to-Speech backend selection
"""
from typing import Optional, Generator, Tuple
import os
import numpy as np
from numpy.typing import NDArray


class BaseTTSModel:
    """Base class for TTS models"""
    def generate_stream(self, *args, **kwargs):
        """Generate streaming audio"""
        raise NotImplementedError
        
    def synthesize_speech(self, *args, **kwargs):
        """Synthesize complete speech"""
        raise NotImplementedError


class ChatterboxTTSWrapper(BaseTTSModel):
    """Wrapper for existing Chatterbox implementation"""
    def __init__(self):
        from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        
    def generate_stream(self, *args, **kwargs):
        """Delegate to existing Chatterbox implementation"""
        return self.model.generate_stream(*args, **kwargs)
        
    def synthesize_speech(self, *args, **kwargs):
        """Delegate to existing Chatterbox implementation"""  
        return self.model.synthesize_speech(*args, **kwargs)


class TogetherTTSModel(BaseTTSModel):
    """Together AI TTS API implementation"""
    def __init__(self, api_key: Optional[str] = None, model: str = "canopylabs/orpheus-3b-0.1-ft"):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable required")
            
        self.model = model
        self.api_url = "https://api.together.xyz/v1/text-to-speech"
        
    def generate_stream(
        self, 
        text: str, 
        voice: str = "tara",
        sample_rate: int = 24000,
        **kwargs
    ) -> Generator[dict, None, None]:
        """Generate streaming TTS via Together AI"""
        try:
            import httpx
            import json
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": text,
                "voice": voice,
                "stream": True,
                "format": "pcm",
                "sample_rate": sample_rate
            }
            
            with httpx.stream("POST", self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                for chunk in response.iter_bytes():
                    if chunk:
                        # Convert raw PCM to numpy array
                        audio_data = np.frombuffer(chunk, dtype=np.int16)
                        if audio_data.size > 0:
                            # Convert to float32 and reshape for compatibility
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            audio_tensor = audio_float.reshape(1, -1)  # (1, samples)
                            
                            yield {
                                "type": "audio", 
                                "audio": audio_tensor,
                                "sample_rate": sample_rate
                            }
                            
        except Exception as e:
            print(f"Together TTS error: {e}")
            yield {"type": "error", "message": str(e)}
            
        # Signal end of stream
        yield {"type": "eos", "audio": None}


def get_tts_model(model_name: str = "chatterbox", **kwargs) -> BaseTTSModel:
    """
    TTS Model Factory - Get TTS backend by name
    
    Available models:
    - "chatterbox": Local Chatterbox Multilingual TTS (default)
    - "together": Together AI API TTS (requires TOGETHER_API_KEY)
    
    Args:
        model_name: Name of the TTS model to use
        **kwargs: Additional arguments for model initialization
        
    Returns:
        BaseTTSModel instance
    """
    if model_name == "chatterbox":
        return ChatterboxTTSWrapper()
        
    elif model_name == "together":
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model", "canopylabs/orpheus-3b-0.1-ft")
        return TogetherTTSModel(api_key=api_key, model=model)
        
    else:
        available = ["chatterbox", "together"]
        raise ValueError(f"Invalid TTS model: {model_name}. Available: {available}")