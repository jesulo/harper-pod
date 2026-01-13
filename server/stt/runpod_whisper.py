"""
RunPod Whisper STT Client
Conecta con el endpoint WebSocket de Whisper en RunPod
"""
import asyncio
import websockets
import json
import numpy as np
from typing import Optional, Callable, AsyncGenerator
from loguru import logger


class RunPodWhisperClient:
    """Cliente para Whisper STT en RunPod"""
    
    def __init__(self, pod_url: str):
        """
        Args:
            pod_url: Base URL del pod (ej: https://xxxxx-8000.proxy.runpod.net)
        """
        self.pod_url = pod_url
        self.ws_url = pod_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/stt"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        
    async def connect(self):
        """Conecta al WebSocket de STT"""
        try:
            logger.info(f"🔌 Connecting to RunPod Whisper: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self._running = True
            logger.success(f"✅ Connected to RunPod Whisper")
        except Exception as e:
            logger.error(f"❌ Failed to connect to RunPod Whisper: {e}")
            raise
    
    async def disconnect(self):
        """Desconecta del WebSocket"""
        self._running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Disconnected from RunPod Whisper")
    
    async def transcribe_stream(
        self, 
        audio_generator: AsyncGenerator[bytes, None],
        callback: Optional[Callable[[str, str, bool], None]] = None
    ):
        """
        Transcribe audio stream from generator
        
        Args:
            audio_generator: Async generator que produce bytes de audio PCM int16 @ 16kHz
            callback: Función llamada con (text, language, is_final) para cada transcripción
        """
        if not self.websocket or not self._running:
            raise RuntimeError("Not connected to RunPod Whisper")
        
        try:
            # Task para enviar audio
            send_task = asyncio.create_task(self._send_audio(audio_generator))
            
            # Task para recibir transcripciones
            receive_task = asyncio.create_task(self._receive_transcriptions(callback))
            
            # Esperar ambos tasks
            await asyncio.gather(send_task, receive_task)
            
        except Exception as e:
            logger.error(f"❌ Error in transcription stream: {e}")
            raise
    
    async def _send_audio(self, audio_generator: AsyncGenerator[bytes, None]):
        """Envía audio al WebSocket"""
        try:
            async for audio_chunk in audio_generator:
                if not self._running:
                    break
                
                # Enviar bytes directamente (PCM int16)
                await self.websocket.send(audio_chunk)
                
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")
            self._running = False
    
    async def _receive_transcriptions(self, callback: Optional[Callable]):
        """Recibe transcripciones del WebSocket"""
        try:
            while self._running:
                message = await self.websocket.recv()
                
                # Parse JSON
                data = json.loads(message)
                
                if data.get("type") == "transcription":
                    text = data.get("text", "")
                    language = data.get("language", "unknown")
                    is_final = data.get("is_final", False)
                    
                    logger.debug(f"📝 Transcription: {text} (lang={language}, final={is_final})")
                    
                    if callback:
                        callback(text, language, is_final)
                
                elif data.get("type") == "error":
                    logger.error(f"❌ RunPod Whisper error: {data.get('message')}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ RunPod Whisper connection closed")
            self._running = False
        except Exception as e:
            logger.error(f"❌ Error receiving transcriptions: {e}")
            self._running = False


class BufferedWhisperClient:
    """
    Cliente Whisper con buffer acumulativo
    Compatible con la interfaz de Harper (transcribe_pcm)
    """
    
    def __init__(self, pod_url: str, sample_rate: int = 16000):
        """
        Args:
            pod_url: Base URL del pod RunPod
            sample_rate: Sample rate del audio (debe ser 16000 para Whisper)
        """
        self.client = RunPodWhisperClient(pod_url)
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.last_transcription = ""
        self.last_language = "unknown"
        
    async def connect(self):
        """Conecta al servidor RunPod"""
        await self.client.connect()
    
    async def disconnect(self):
        """Desconecta del servidor"""
        await self.client.disconnect()
    
    def transcribe_pcm(self, audio_data: bytes) -> str:
        """
        Transcribe audio PCM (sincrónico, compatible con Harper)
        
        Args:
            audio_data: Bytes de audio PCM int16 @ 16kHz
        
        Returns:
            Texto transcrito
        """
        # Esta función necesita ser llamada desde un contexto async
        # pero Harper espera una función sincrónica
        # Solución: usar asyncio.run() o mantener un event loop
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._transcribe_async(audio_data))
    
    async def _transcribe_async(self, audio_data: bytes) -> str:
        """Versión async de transcribe_pcm"""
        
        # Agregar al buffer
        self.buffer.extend(audio_data)
        
        # Si tenemos suficiente audio (3 segundos = 16000 * 2 bytes * 3)
        min_bytes = self.sample_rate * 2 * 3
        
        if len(self.buffer) >= min_bytes:
            # Enviar buffer completo
            await self.client.websocket.send(bytes(self.buffer))
            
            # Recibir transcripción
            message = await self.client.websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "transcription":
                self.last_transcription = data.get("text", "")
                self.last_language = data.get("language", "unknown")
                
                # Limpiar buffer si es transcripción final
                if data.get("is_final", False):
                    self.buffer.clear()
                
                return self.last_transcription
        
        # Si no hay suficiente audio, retornar última transcripción
        return self.last_transcription
