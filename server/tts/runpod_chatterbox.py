"""
RunPod Chatterbox TTS Client
Conecta con el endpoint WebSocket de TTS en RunPod
Soporta Chatterbox Turbo y Multilingual
"""
import asyncio
import websockets
import json
from typing import Optional, AsyncGenerator, Literal
from loguru import logger


class RunPodChatterboxClient:
    """Cliente para Chatterbox TTS en RunPod"""
    
    def __init__(
        self, 
        pod_url: str,
        model: Literal["turbo", "multilingual"] = "turbo"
    ):
        """
        Args:
            pod_url: Base URL del pod (ej: https://xxxxx-8000.proxy.runpod.net)
            model: Modelo a usar ("turbo" para inglés + tags, "multilingual" para 23 idiomas)
        """
        self.pod_url = pod_url
        self.model = model
        self.ws_url = pod_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/tts"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        
        logger.info(f"🎵 RunPod Chatterbox client initialized (model={model})")
    
    async def connect(self):
        """Conecta al WebSocket de TTS"""
        try:
            logger.info(f"🔌 Connecting to RunPod Chatterbox: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self._running = True
            logger.success(f"✅ Connected to RunPod Chatterbox ({self.model})")
        except Exception as e:
            logger.error(f"❌ Failed to connect to RunPod Chatterbox: {e}")
            raise
    
    async def disconnect(self):
        """Desconecta del WebSocket"""
        self._running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Disconnected from RunPod Chatterbox")
    
    async def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        voice_audio: Optional[bytes] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Sintetiza texto a audio en streaming
        
        Args:
            text: Texto a sintetizar (puede incluir tags como [laugh] con Turbo)
            language: Código de idioma (solo para multilingual)
            voice_audio: Audio de referencia para clonación (opcional)
        
        Yields:
            Frames de audio Opus (con pack_frame header compatible con Harper)
        """
        if not self.websocket or not self._running:
            raise RuntimeError("Not connected to RunPod Chatterbox")
        
        try:
            # Preparar mensaje
            request = {
                "text": text,
                "model": self.model,
                "language": language,
                "voice_audio": None  # TODO: Implementar clonación de voz si es necesario
            }
            
            # Enviar request
            await self.websocket.send(json.dumps(request))
            logger.debug(f"📤 Sent TTS request: {text[:50]}... (model={self.model})")
            
            # Recibir frames
            frame_count = 0
            while self._running:
                message = await self.websocket.recv()
                
                # Si es JSON, es un mensaje de control
                if isinstance(message, str):
                    data = json.loads(message)
                    
                    if data.get("type") == "complete":
                        logger.debug(f"✅ TTS generation complete ({frame_count} frames)")
                        break
                    
                    elif data.get("type") == "error":
                        error_msg = data.get("message", "Unknown error")
                        logger.error(f"❌ RunPod TTS error: {error_msg}")
                        raise RuntimeError(f"TTS error: {error_msg}")
                
                # Si es bytes, es un frame de audio
                elif isinstance(message, bytes):
                    frame_count += 1
                    yield message  # Ya viene con pack_frame header
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ RunPod Chatterbox connection closed")
            self._running = False
        except Exception as e:
            logger.error(f"❌ Error in TTS stream: {e}")
            raise


class AdaptiveRunPodChatterbox:
    """
    Cliente adaptativo que elige automáticamente Turbo o Multilingual
    según el idioma detectado
    """
    
    def __init__(self, pod_url: str):
        """
        Args:
            pod_url: Base URL del pod RunPod
        """
        self.pod_url = pod_url
        self.turbo_client = RunPodChatterboxClient(pod_url, model="turbo")
        self.multilingual_client = RunPodChatterboxClient(pod_url, model="multilingual")
        
        # Idiomas soportados por Turbo (solo inglés)
        self.turbo_languages = {"en"}
        
        # Idiomas soportados por Multilingual
        self.multilingual_languages = {
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh"
        }
    
    async def connect(self):
        """Conecta ambos clientes"""
        await asyncio.gather(
            self.turbo_client.connect(),
            self.multilingual_client.connect()
        )
    
    async def disconnect(self):
        """Desconecta ambos clientes"""
        await asyncio.gather(
            self.turbo_client.disconnect(),
            self.multilingual_client.disconnect()
        )
    
    def _should_use_turbo(self, text: str, language: str) -> bool:
        """
        Determina si usar Turbo o Multilingual
        
        Criterios:
        - Usar Turbo si: idioma es inglés Y texto contiene tags paralinguísticos
        - Usar Multilingual si: idioma no es inglés O no hay tags
        """
        # Si no es inglés, usar Multilingual
        if language not in self.turbo_languages:
            return False
        
        # Si es inglés pero tiene tags, usar Turbo
        paralinguistic_tags = ["[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]"]
        has_tags = any(tag in text for tag in paralinguistic_tags)
        
        return has_tags
    
    async def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        voice_audio: Optional[bytes] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Sintetiza texto usando el modelo apropiado
        
        Args:
            text: Texto a sintetizar
            language: Código de idioma ISO 639-1
            voice_audio: Audio de referencia (opcional)
        
        Yields:
            Frames de audio Opus
        """
        # Decidir qué cliente usar
        use_turbo = self._should_use_turbo(text, language)
        
        if use_turbo:
            logger.info(f"🚀 Using Turbo TTS (detected tags in English text)")
            client = self.turbo_client
        else:
            logger.info(f"🌍 Using Multilingual TTS (language={language})")
            client = self.multilingual_client
        
        # Generar audio
        async for frame in client.synthesize_stream(text, language, voice_audio):
            yield frame
