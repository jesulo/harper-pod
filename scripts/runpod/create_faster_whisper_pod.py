"""
Create a RunPod instance for Faster Whisper STT server.
This uses the Speaches API server with Faster Whisper model pre-loaded.
"""
import os
import sys
from pathlib import Path

try:
    import runpod
except ImportError:
    print("Error: runpod package not installed.")
    print("Install with: pip install runpod")
    sys.exit(1)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY not found in environment variables")
    print("Add it to your .env file")
    sys.exit(1)

runpod.api_key = RUNPOD_API_KEY

# Configuration
GPU_TYPE = os.getenv("RUNPOD_FASTER_WHISPER_GPU_TYPE", "NVIDIA GeForce RTX 4090")
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "Systran/faster-whisper-large-v3")

print("=" * 60)
print("Creating Faster Whisper STT Pod on RunPod")
print("=" * 60)
print(f"GPU Type: {GPU_TYPE}")
print(f"Model: {FASTER_WHISPER_MODEL}")
print()

try:
    pod = runpod.create_pod(
        name="Harper Faster Whisper Server",
        image_name="theneuralmaze/faster-whisper-server:latest",
        gpu_type_id=GPU_TYPE,
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=20,
        volume_mount_path="/workspace",
        ports="8000/http",
        env={
            "DEFAULT_MODEL": FASTER_WHISPER_MODEL,
            "COMPUTE_TYPE": "int8",
        },
    )

    pod_id = pod.get("id")
    pod_url = f"https://{pod_id}-8000.proxy.runpod.net"

    print("✅ Pod created successfully!")
    print()
    print("=" * 60)
    print("Pod Details:")
    print("=" * 60)
    print(f"Pod ID: {pod_id}")
    print(f"Pod URL: {pod_url}")
    print()
    print("=" * 60)
    print("Add the following to your .env file:")
    print("=" * 60)
    print(f"FASTER_WHISPER_API_URL={pod_url}")
    print("=" * 60)
    print()
    print("Note: It may take a few minutes for the pod to be ready.")
    print("You can check the status in your RunPod dashboard.")

except Exception as e:
    print(f"❌ Error creating pod: {e}")
    sys.exit(1)
