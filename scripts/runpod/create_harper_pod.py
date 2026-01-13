"""
Create a RunPod instance for Harper server (main application).
This deploys the complete Harper voice agent system.
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
DOCKER_IMAGE = os.getenv("HARPER_DOCKER_IMAGE", "your-dockerhub-user/harper-server:latest")
CPU_INSTANCE = os.getenv("RUNPOD_CPU_INSTANCE", "cpu5c-2-4")

print("=" * 60)
print("Creating Harper Server Pod on RunPod")
print("=" * 60)
print(f"Docker Image: {DOCKER_IMAGE}")
print(f"CPU Instance: {CPU_INSTANCE}")
print()
print("⚠️  Make sure you have:")
print("  1. Built and pushed your Docker image to Docker Hub")
print("  2. Created Faster Whisper and Orpheus pods (if using RunPod models)")
print("  3. Updated FASTER_WHISPER_API_URL and ORPHEUS_API_URL in your .env")
print()

# Collect environment variables from .env
env_vars = {}
required_vars = [
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
]

optional_vars = [
    "FASTER_WHISPER_API_URL",
    "ORPHEUS_API_URL",
    "STT_PROVIDER",
    "TTS_PROVIDER",
]

# Add required vars
for var in required_vars:
    value = os.getenv(var)
    if not value:
        print(f"Warning: {var} not found in environment")
    else:
        env_vars[var] = value

# Add optional vars
for var in optional_vars:
    value = os.getenv(var)
    if value:
        env_vars[var] = value

try:
    pod = runpod.create_pod(
        name="Harper Server",
        image_name=DOCKER_IMAGE,
        cloud_type="SECURE",
        volume_in_gb=20,
        volume_mount_path="/workspace",
        instance_id=CPU_INSTANCE,
        ports="8000/http,443/tcp,80/tcp,22/tcp",
        env=env_vars,
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
    print("Harper Server is ready!")
    print(f"Access your server at: {pod_url}")
    print("=" * 60)
    print()
    print("Note: It may take a few minutes for the pod to be ready.")
    print("You can check the status in your RunPod dashboard.")

except Exception as e:
    print(f"❌ Error creating pod: {e}")
    sys.exit(1)
