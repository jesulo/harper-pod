"""
Create a RunPod instance for Orpheus TTS server.
This uses llama.cpp server with Orpheus 3B model pre-loaded.
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
GPU_TYPE = os.getenv("RUNPOD_ORPHEUS_GPU_TYPE", "NVIDIA GeForce RTX 5090")
ORPHEUS_MODEL = os.getenv("ORPHEUS_MODEL", "PkmX/orpheus-3b-0.1-ft-Q8_0-GGUF")

print("=" * 60)
print("Creating Orpheus TTS Pod on RunPod")
print("=" * 60)
print(f"GPU Type: {GPU_TYPE}")
print(f"Model: {ORPHEUS_MODEL}")
print()

try:
    pod = runpod.create_pod(
        name="Harper Orpheus TTS Server",
        image_name="theneuralmaze/orpheus-llamacpp-server:latest",
        gpu_type_id=GPU_TYPE,
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=20,
        volume_mount_path="/workspace",
        ports="8080/http",
    )

    pod_id = pod.get("id")
    pod_url = f"https://{pod_id}-8080.proxy.runpod.net"

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
    print(f"ORPHEUS_API_URL={pod_url}")
    print("=" * 60)
    print()
    print("Note: It may take a few minutes for the pod to be ready.")
    print("You can check the status in your RunPod dashboard.")

except Exception as e:
    print(f"❌ Error creating pod: {e}")
    sys.exit(1)
