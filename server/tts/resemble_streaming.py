"""
Resemble AI Streaming TTS Backend
Based on official documentation: https://docs.resemble.ai/streaming
Powered by Chatterbox model for ultra-low latency voice synthesis
"""
import asyncio
import httpx
import numpy as np
from typing import AsyncGenerator, Literal, Optional
from loguru import logger
from config import settings


class ResembleStreamingAPI:
    """
    Text-to-Speech using Resemble AI Streaming API (Chatterbox model)
    
    Official HTTP endpoint: https://f.cluster.resemble.ai/stream
    Official WebSocket: wss://websocket.cluster.resemble.ai/stream (Business+ plan)
    
    Authentication: Bearer token from https://app.resemble.ai/account/api
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
        self.stream_url = "https://f.cluster.resemble.ai/stream"
        self.api_base = "https://api.resemble.ai/v1"
    
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def stream_audio(
        self,
        text: str,
        voice_uuid: Optional[str] = None,
        precision: Literal["MULAW", "PCM_16", "PCM_24", "PCM_32"] = "PCM_16",
        sample_rate: int = 48000,
        use_hd: bool = False
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio from Resemble AI HTTP endpoint
        
        Official endpoint: POST https://f.cluster.resemble.ai/stream
        Response: Chunked WAV audio stream (Transfer-Encoding: chunked)
        
        Args:
            text: Text to synthesize (max 2000 characters)
            voice_uuid: Voice UUID (uses self.voice_uuid if not provided)
            precision: Audio precision (MULAW, PCM_16, PCM_24, PCM_32)
            sample_rate: Sample rate in Hz (16000, 22050, 44100, 48000)
            use_hd: Use high-definition model (increases latency)
        
        Yields:
            PCM int16 audio chunks as numpy arrays
        """
        print(f"[ResembleAPI] stream_audio called")
        
        # Use provided voice_uuid or fallback to instance default
        v_uuid = voice_uuid or self.voice_uuid
        if not v_uuid:
            raise ValueError("voice_uuid required (set in __init__ or pass as parameter)")
        
        payload = {
            "voice_uuid": v_uuid,
            "data": text[:2000],  # Max 2000 chars per Resemble docs
            "precision": precision,
            "sample_rate": sample_rate,
            "use_hd": use_hd
        }
        
        # Add project_uuid if configured
        if self.project_uuid:
            payload["project_uuid"] = self.project_uuid
        
        print(f"[ResembleAPI] Payload: voice={v_uuid}, text_len={len(text)}, sr={sample_rate}, hd={use_hd}")
        print(f"[ResembleAPI] Endpoint: {self.stream_url}")
        
        # Following Together AI pattern: larger chunks for cleaner streaming
        MIN_CHUNK_SIZE = 2048  # 1024 samples at 16-bit = ~46ms at 22050Hz
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                print(f"[ResembleAPI] Sending POST request...")
                async with client.stream(
                    "POST",
                    self.stream_url,
                    json=payload,
                    headers=self._get_headers()
                ) as response:
                    # Check status BEFORE streaming
                    response.raise_for_status()
                    
                    # Stream WAV audio chunks - KOKORO PATTERN: Process each chunk immediately
                    skip_header = True
                    chunk_num = 0
                    
                    # Stream chunks: Use 38400 bytes (400ms at 48kHz) - same as Kokoro
                    # 19200 samples * 2 bytes = 38400 bytes (400ms at 48kHz)
                    # This matches Kokoro's chunk_size for consistent behavior
                    async for chunk in response.aiter_bytes(chunk_size=38400):
                        if not chunk:
                            continue
                        
                        # Skip WAV header on first chunk (44 bytes)
                        if skip_header and chunk.startswith(b"RIFF"):
                            print(f"[ResembleAPI] Skipping WAV header (44 bytes), chunk size: {len(chunk)} bytes")
                            chunk = chunk[44:]
                            skip_header = False
                        
                        # KOKORO PATTERN: Yield chunk immediately without buffering
                        # Ensure chunk is 2-byte aligned (int16)
                        if len(chunk) % 2 != 0:
                            # Keep last byte for potential next chunk (though shouldn't happen with API chunks)
                            chunk = chunk[:-1]
                        
                        if len(chunk) > 0:
                            chunk_num += 1
                            # Convert bytes to PCM int16 numpy array
                            audio = np.frombuffer(chunk, dtype=np.int16)
                            
                            if chunk_num <= 3:  # Log first 3 chunks
                                print(f"[ResembleAPI] Yielding chunk {chunk_num}: {len(audio)} samples ({len(chunk)} bytes)")
                            
                            yield audio
                    
                    # No buffer to flush - everything sent immediately
                    print(f"[ResembleAPI] Stream complete, yielded {chunk_num} chunks")
                
        except httpx.HTTPStatusError as e:
            # Don't try to read response.text on streaming responses
            error_msg = f"❌ Resemble AI API error: HTTP {e.response.status_code}"
            logger.error(error_msg)
            print(f"[ResembleAPI] {error_msg}")
            raise
        except Exception as e:
            print(f"[ResembleAPI] ❌ Error: {e}")
            raise
    
    async def get_available_voices(self) -> list[dict]:
        """
        Get list of available voices
        
        Endpoint: GET https://api.resemble.ai/v1/voices
        Returns list of voices with UUIDs
        
        Returns:
            List of voice dictionaries with fields:
            - uuid: Voice unique identifier
            - name: Voice display name
            - created_at: Creation timestamp
            - etc.
        """
        url = f"{self.api_base}/voices"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response.json()


# Integration with Harper's factory system
class ResembleStreamingWrapper:
    """Wrapper to integrate Resemble AI Streaming with Harper's TTS factory"""
    
    def __init__(self):
        """
        Initialize Resemble streaming wrapper
        Reads configuration from settings
        """
        print("[ResembleAI] Initializing ResembleStreamingWrapper...")
        
        # Get API key from settings
        api_key = settings.tts.resemble_api_key
        print(f"[ResembleAI] API Key found: {api_key[:20] if api_key else 'None'}...")
        if not api_key:
            raise ValueError(
                "Resemble AI API key required. Set TTS__RESEMBLE_API_KEY in .env\n"
                "Get your key from: https://app.resemble.ai/account/api"
            )
        
        # Get voice UUID from settings
        voice_uuid = settings.tts.resemble_voice_uuid
        print(f"[ResembleAI] Voice UUID: {voice_uuid}")
        if not voice_uuid:
            raise ValueError(
                "Resemble voice UUID required. Set TTS__RESEMBLE_VOICE_UUID in .env\n"
                "List voices with: curl -H 'Authorization: Bearer <KEY>' "
                "https://api.resemble.ai/v1/voices"
            )
        
        # Optional project UUID
        project_uuid = getattr(settings.tts, 'resemble_project_uuid', None)
        print(f"[ResembleAI] Project UUID: {project_uuid}")
        
        self.client = ResembleStreamingAPI(
            api_key=api_key,
            voice_uuid=voice_uuid,
            project_uuid=project_uuid
        )
        print("[ResembleAI] ✅ Initialization complete")
    
    async def stream_speech(
        self, 
        text: str, 
        voice: str = None,  # Ignored - uses voice_uuid from settings
        sample_rate: int = 24000
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream speech compatible with Harper's factory interface
        
        Args:
            text: Text to synthesize
            voice: Ignored (Resemble uses voice_uuid)
            sample_rate: Target sample rate (will be resampled if needed)
        
        Yields:
            PCM int16 audio chunks
        """
        print(f"[ResembleAI] stream_speech called")
        print(f"[ResembleAI] Text: '{text[:100]}...' ({len(text)} chars)")
        print(f"[ResembleAI] Requested sample rate: {sample_rate}")
        
        # Get precision and HD settings from config
        precision = getattr(settings.tts, 'resemble_precision', 'PCM_16')
        use_hd = getattr(settings.tts, 'resemble_use_hd', False)
        print(f"[ResembleAI] Precision: {precision}, HD: {use_hd}")
        
        # Resemble supports: 16000, 22050, 44100, 48000
        # For 24kHz target, use 48000 (higher quality, will downsample to 24k)
        # Using higher sample rate and downsampling is cleaner than upsampling from 22050
        # Request the target sample rate directly - NO resampling
        # Similar to Together AI pattern: API returns exactly what we request
        resemble_rate = sample_rate
        
        print(f"[ResembleAI] Resemble rate: {resemble_rate} (no resampling)")
        print(f"[ResembleAI] Calling Resemble API...")
        
        chunk_count = 0
        try:
            async for chunk in self.client.stream_audio(
                text=text,
                precision=precision,
                sample_rate=resemble_rate,
                use_hd=use_hd
            ):
                chunk_count += 1
                if chunk_count <= 3:
                    print(f"[ResembleAI] Chunk {chunk_count}: {len(chunk)} samples")
                yield chunk
            
            print(f"[ResembleAI] ✅ Streaming complete. Total chunks: {chunk_count}")
        except Exception as e:
            print(f"[ResembleAI] ❌ Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            raise


# Example usage
async def test_resemble_streaming():
    """Test the Resemble AI Streaming API implementation"""
    import os
    
    api_key = os.getenv("RESEMBLE_API_KEY")
    voice_uuid = os.getenv("RESEMBLE_VOICE_UUID")
    
    if not api_key or not voice_uuid:
        print("Set RESEMBLE_API_KEY and RESEMBLE_VOICE_UUID environment variables")
        return
    
    api = ResembleStreamingAPI(api_key=api_key, voice_uuid=voice_uuid)
    
    text = "Hello, this is a test of the Resemble AI streaming API powered by Chatterbox."
    
    print("Testing HTTP streaming...")
    chunk_count = 0
    async for audio_chunk in api.stream_audio(text):
        chunk_count += 1
        print(f"Received audio chunk {chunk_count}: {audio_chunk.shape} samples")
    
    print(f"\nTotal chunks received: {chunk_count}")
    
    print("\nFetching available voices...")
    voices = await api.get_available_voices()
    print(f"Found {len(voices)} voices:")
    for voice in voices[:3]:  # Show first 3
        print(f"  - {voice.get('name', 'Unknown')}: {voice.get('uuid', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(test_resemble_streaming())
