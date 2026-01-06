"""
Resemble AI Streaming TTS Backend
Based on official documentation: https://docs.resemble.ai/streaming
Powered by Chatterbox model for ultra-low latency voice synthesis
"""
import asyncio
import httpx
import numpy as np
import io
import wave
from typing import AsyncGenerator, Literal, Optional
from config import settings


class ResembleStreamingAPI:
    """
    Text-to-Speech using Resemble AI Streaming API (Chatterbox model)
    Official endpoint: https://f.cluster.resemble.ai/stream
    """
    
    def __init__(
        self, 
        api_key: str,
        voice_uuid: Optional[str] = None,
        project_uuid: Optional[str] = None
    ):
        """
        Initialize Resemble AI Streaming client
        
        Args:
            api_key: Resemble AI API token (from https://app.resemble.ai/account/api)
            voice_uuid: UUID of the voice to use (get from /voices endpoint)
            project_uuid: Optional project UUID for organization
        """
        self.api_key = api_key
        self.voice_uuid = voice_uuid
        self.project_uuid = project_uuid
        self.base_url = "https://f.cluster.resemble.ai"
        self.api_base = "https://api.resemble.ai/v1"
    
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests"""
    async def stream_audio(
        self,
        text: str,
        voice_uuid: Optional[str] = None,
        precision: Literal["MULAW", "PCM_16", "PCM_24", "PCM_32"] = "PCM_16",
        sample_rate: int = 48000,
        use_hd: bool = False
    ) -> AsyncGenerator[np.ndarray, None]:
        """_weight: float = 0.4,
        temperature: float = 0.9,
        streaming_chunk_size: int = 200,
        streaming_strategy: Literal["sentence", "paragraph", "fixed", "word"] = "sentence"
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream raw WAV audio chunks from /audio/speech/stream
        
        Args:
            text: Text to convert to speech
            exaggeration: Expression level (0.0-1.0)
            cfg_weight: Delivery/rhythm control
            temperature: Synthesis randomness
            streaming_chunk_size: Characters per chunk
            streaming_strategy: Chunking strategy
            
        Yields:
            Audio chunks as numpy arrays (int16)
        """
        url = f"{self.base_url}/audio/speech/stream"
        
        payload = {
            "input": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "streaming_chunk_size": streaming_chunk_size,
            "streaming_strategy": streaming_strategy
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()
                
                wav_buffer = b""
                
                # Stream raw WAV data
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    
                    wav_buffer += chunk
                    
                    # Process complete samples (int16 = 2 bytes)
                    while len(wav_buffer) >= 1024:  # Min chunk size
                        complete_samples = len(wav_buffer) // 2
                        complete_bytes = (complete_samples // 256) * 512  # Align to 256 samples
                        
                        if complete_bytes > 0:
                            audio_chunk = np.frombuffer(
                                wav_buffer[:complete_bytes],
                                dtype=np.int16
                            )
                            yield audio_chunk
                            wav_buffer = wav_buffer[complete_bytes:]
                        else:
                            break
                
                # Flush remaining buffer
                if wav_buffer:
                    complete_samples = len(wav_buffer) // 2
                    if complete_samples > 0:
                        complete_bytes = complete_samples * 2
                        audio_chunk = np.frombuffer(
                            wav_buffer[:complete_bytes],
                            dtype=np.int16
                        )
                        yield audio_chunk
    
    async def stream_sse(
        self,
        text: str,
        exaggeration: float = 0.7,
        cfg_weight: float = 0.4,
        temperature: float = 0.9,
        streaming_chunk_size: int = 200,
        streaming_strategy: Literal["sentence", "paragraph", "fixed", "word"] = "sentence"
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio via SSE (Server-Sent Events) from /audio/speech
        OpenAI-compatible format
        
        Args:
            text: Text to convert to speech
            exaggeration: Expression level
            cfg_weight: Delivery control
            temperature: Synthesis randomness
            streaming_chunk_size: Characters per chunk
            streaming_strategy: Chunking strategy
            
        Yields:
            Audio chunks as numpy arrays (int16)
        """
        url = f"{self.base_url}/audio/speech"
        
        payload = {
            "input": text,
            "stream_format": "sse",
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "streaming_chunk_size": streaming_chunk_size,
            "streaming_strategy": streaming_strategy
        }
        
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                # Parse SSE events
                async for line in response.aiter_lines():
                    if not line or line.startswith(':'):
                        continue
                    
                    if line.startswith('data: '):
                        import json
                        event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                        
                        # Handle different event types
                        if event_data.get('type') == 'speech.audio.delta':
                            # Decode base64 audio
                            audio_b64 = event_data.get('audio', '')
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                                yield audio_chunk
                        
                        elif event_data.get('type') == 'speech.audio.done':
                            # Stream complete
                            print(f"[ChatterboxSSE] Stream complete. Usage: {event_data.get('usage')}")
                            break
    
    async def get_available_voices(self) -> list[dict]:
        """
        Get list of available voices from /voices endpoint
        
        Returns:
            List of voice dictionaries with metadata
        """
        url = f"{self.base_url}/voices"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response.json()


# Integration with Harper's factory system
class ChatterboxStreamingWrapper:
    """Wrapper to integrate Chatterbox Streaming API with Harper's TTS factory"""
    
    def __init__(self, use_sse: bool = False):
        """
        Initialize wrapper
        
        Args:
            use_sse: Use SSE streaming (True) or raw WAV streaming (False)
        """
        base_url = settings.chatterbox_api_url or "http://localhost:4123"
        api_key = None  # Add CHATTERBOX_API_KEY to settings if needed
        
        self.api = ChatterboxStreamingAPI(base_url=base_url, api_key=api_key)
        self.use_sse = use_sse
    
    async def stream_speech(
        self, 
        text: str, 
        voice: str = None,  # Not used - voices managed via separate endpoint
        sample_rate: int = 24000
    ) -> AsyncGenerator[np.ndarray, None]:
        """Stream speech compatible with Harper's factory interface"""
        
        # Get Chatterbox-specific parameters from settings
        exaggeration = settings.tts.exaggeration
        cfg_weight = settings.tts.cfg_weight
        temperature = settings.tts.temperature
        streaming_chunk_size = settings.tts.streaming_chunk_size
        streaming_strategy = settings.tts.streaming_strategy
        
        if self.use_sse:
            # Use SSE streaming (OpenAI-compatible)
            async for chunk in self.api.stream_sse(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                streaming_chunk_size=streaming_chunk_size,
                streaming_strategy=streaming_strategy
            ):
                yield chunk
        else:
            # Use raw WAV streaming (default)
            async for chunk in self.api.stream_raw_audio(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                streaming_chunk_size=streaming_chunk_size,
                streaming_strategy=streaming_strategy
            ):
                yield chunk


# Example usage
async def test_chatterbox_streaming():
    """Test the Chatterbox Streaming API implementation"""
    api = ChatterboxStreamingAPI()
    
    text = "Hello, this is a test of the Chatterbox streaming API."
    
    print("Testing raw WAV streaming...")
    async for audio_chunk in api.stream_raw_audio(text):
        print(f"Received audio chunk: {audio_chunk.shape} samples")
    
    print("\nTesting SSE streaming...")
    async for audio_chunk in api.stream_sse(text):
        print(f"Received SSE audio chunk: {audio_chunk.shape} samples")
    
    print("\nFetching available voices...")
    voices = await api.get_available_voices()
    print(f"Available voices: {voices}")


if __name__ == "__main__":
    asyncio.run(test_chatterbox_streaming())