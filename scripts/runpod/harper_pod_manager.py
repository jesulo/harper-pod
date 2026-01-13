"""
Harper RunPod Manager
Gestiona pods de RunPod con warm standby automático
"""
import runpod
import asyncio
import httpx
import time
from typing import Optional, Dict, List
from loguru import logger
from pydantic import BaseModel


class PodInfo(BaseModel):
    """Información de un pod"""
    id: str
    name: str
    status: str  # STOPPED, RUNNING, STARTING
    url: Optional[str] = None
    gpu_type: str
    created_at: float


class HarperPodManager:
    """
    Gestor de pods RunPod con estrategia warm standby
    Mantiene siempre al menos 1 pod STOPPED listo para activación rápida
    """
    
    def __init__(
        self,
        api_key: str,
        image_name: str = "your-dockerhub-user/harper-voice-services:latest",
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        volume_gb: int = 40
    ):
        runpod.api_key = api_key
        self.image_name = image_name
        self.gpu_type = gpu_type
        self.volume_gb = volume_gb
        
        # Track pods
        self.active_pods: Dict[str, PodInfo] = {}
        self.standby_pod: Optional[PodInfo] = None
        
        logger.info(f"🚀 Harper Pod Manager initialized")
        logger.info(f"   Image: {image_name}")
        logger.info(f"   GPU: {gpu_type}")
    
    def create_pod(self, name: str = None, start_immediately: bool = False) -> str:
        """
        Crea un nuevo pod en RunPod
        
        Args:
            name: Nombre del pod (auto-generado si None)
            start_immediately: Si True, deja el pod RUNNING; si False, lo detiene (warm standby)
        
        Returns:
            Pod ID
        """
        if name is None:
            name = f"harper-voice-{int(time.time())}"
        
        logger.info(f"📦 Creating pod: {name}")
        
        try:
            pod = runpod.create_pod(
                name=name,
                image_name=self.image_name,
                gpu_type_id=self.gpu_type,
                cloud_type="SECURE",
                gpu_count=1,
                volume_in_gb=self.volume_gb,
                volume_mount_path="/workspace",
                ports="8000/http",
                env={
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
                }
            )
            
            pod_id = pod["id"]
            logger.success(f"✅ Pod created: {pod_id}")
            
            # Si no queremos que inicie, lo detenemos inmediatamente
            if not start_immediately:
                logger.info(f"⏸️  Stopping pod for warm standby...")
                runpod.stop_pod(pod_id)
                status = "STOPPED"
                url = None
            else:
                status = "RUNNING"
                url = self._get_pod_url(pod_id)
            
            # Guardar info
            pod_info = PodInfo(
                id=pod_id,
                name=name,
                status=status,
                url=url,
                gpu_type=self.gpu_type,
                created_at=time.time()
            )
            
            if status == "STOPPED":
                self.standby_pod = pod_info
            else:
                self.active_pods[pod_id] = pod_info
            
            return pod_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create pod: {e}")
            raise
    
    async def start_pod(self, pod_id: str, timeout: int = 120) -> str:
        """
        Activa un pod detenido (warm start)
        
        Args:
            pod_id: ID del pod a activar
            timeout: Tiempo máximo de espera en segundos
        
        Returns:
            URL del pod activado
        """
        logger.info(f"▶️  Starting pod: {pod_id}")
        
        try:
            # Resume el pod
            runpod.resume_pod(pod_id)
            
            # Esperar a que esté listo
            url = await self._wait_for_pod_ready(pod_id, timeout)
            
            # Actualizar tracking
            if self.standby_pod and self.standby_pod.id == pod_id:
                self.standby_pod.status = "RUNNING"
                self.standby_pod.url = url
                self.active_pods[pod_id] = self.standby_pod
                self.standby_pod = None
            
            logger.success(f"✅ Pod ready: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Failed to start pod: {e}")
            raise
    
    async def stop_pod(self, pod_id: str, delay_minutes: int = 0):
        """
        Detiene un pod activo
        
        Args:
            pod_id: ID del pod a detener
            delay_minutes: Minutos a esperar antes de detener (para evitar stop/start frecuente)
        """
        if delay_minutes > 0:
            logger.info(f"⏰ Scheduling pod stop in {delay_minutes} minutes: {pod_id}")
            await asyncio.sleep(delay_minutes * 60)
        
        logger.info(f"⏸️  Stopping pod: {pod_id}")
        
        try:
            runpod.stop_pod(pod_id)
            
            # Actualizar tracking
            if pod_id in self.active_pods:
                pod_info = self.active_pods[pod_id]
                pod_info.status = "STOPPED"
                pod_info.url = None
                
                # Si no hay standby, usar este como standby
                if self.standby_pod is None:
                    self.standby_pod = pod_info
                
                del self.active_pods[pod_id]
            
            logger.success(f"✅ Pod stopped: {pod_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop pod: {e}")
    
    def terminate_pod(self, pod_id: str):
        """
        Elimina completamente un pod (irreversible)
        """
        logger.warning(f"🗑️  Terminating pod: {pod_id}")
        
        try:
            runpod.terminate_pod(pod_id)
            
            # Limpiar tracking
            if pod_id in self.active_pods:
                del self.active_pods[pod_id]
            if self.standby_pod and self.standby_pod.id == pod_id:
                self.standby_pod = None
            
            logger.success(f"✅ Pod terminated: {pod_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to terminate pod: {e}")
    
    def ensure_standby(self) -> str:
        """
        Asegura que siempre haya al menos 1 pod en standby
        
        Returns:
            Pod ID del standby (existente o nuevo)
        """
        if self.standby_pod is None:
            logger.info("🔄 No standby pod available, creating one...")
            pod_id = self.create_pod(start_immediately=False)
            return pod_id
        
        logger.info(f"✅ Standby pod already exists: {self.standby_pod.id}")
        return self.standby_pod.id
    
    def get_pod_url(self, pod_id: str) -> Optional[str]:
        """Obtiene la URL de un pod activo"""
        if pod_id in self.active_pods:
            return self.active_pods[pod_id].url
        return self._get_pod_url(pod_id)
    
    def list_pods(self) -> Dict[str, List[PodInfo]]:
        """Lista todos los pods gestionados"""
        return {
            "active": list(self.active_pods.values()),
            "standby": [self.standby_pod] if self.standby_pod else []
        }
    
    async def _wait_for_pod_ready(self, pod_id: str, timeout: int = 120) -> str:
        """Espera a que el pod esté listo y responda health check"""
        url = self._get_pod_url(pod_id)
        health_url = f"{url}/health"
        
        start_time = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            while (time.time() - start_time) < timeout:
                try:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy":
                            logger.success(f"✅ Pod health check passed")
                            return url
                except Exception:
                    pass
                
                await asyncio.sleep(5)
                logger.info(f"⏳ Waiting for pod to be ready... ({int(time.time() - start_time)}s)")
        
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")
    
    def _get_pod_url(self, pod_id: str) -> str:
        """Construye la URL del pod"""
        return f"https://{pod_id}-8000.proxy.runpod.net"


# Singleton instance
_pod_manager: Optional[HarperPodManager] = None

def get_pod_manager(
    api_key: str = None,
    image_name: str = None,
    gpu_type: str = None
) -> HarperPodManager:
    """Get or create global pod manager instance"""
    global _pod_manager
    
    if _pod_manager is None:
        if api_key is None:
            raise ValueError("api_key required for first initialization")
        
        _pod_manager = HarperPodManager(
            api_key=api_key,
            image_name=image_name or "your-dockerhub-user/harper-voice-services:latest",
            gpu_type=gpu_type or "NVIDIA GeForce RTX 4090"
        )
    
    return _pod_manager
