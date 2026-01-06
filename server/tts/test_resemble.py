#!/usr/bin/env python3
"""
Test script for Resemble AI Streaming TTS integration
Usage: python test_resemble.py
"""
import asyncio
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.resemble_streaming import ResembleStreamingAPI


async def test_list_voices():
    """Test listing available voices"""
    print("=" * 60)
    print("TEST 1: Listing Available Voices")
    print("=" * 60)
    
    api_key = os.getenv("RESEMBLE_API_KEY") or os.getenv("TTS__RESEMBLE_API_KEY")
    if not api_key:
        print("❌ RESEMBLE_API_KEY not set")
        print("   Set it with: export RESEMBLE_API_KEY=sk_your_key_here")
        return False
    
    api = ResembleStreamingAPI(api_key=api_key)
    
    try:
        voices = await api.get_available_voices()
        print(f"✅ Found {len(voices)} voices:\n")
        
        for i, voice in enumerate(voices[:5], 1):  # Show first 5
            print(f"{i}. {voice.get('name', 'Unknown')}")
            print(f"   UUID: {voice.get('uuid', 'N/A')}")
            print(f"   Created: {voice.get('created_at', 'N/A')}")
            print()
        
        if len(voices) > 5:
            print(f"   ... and {len(voices) - 5} more voices")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_streaming():
    """Test audio streaming"""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Audio")
    print("=" * 60)
    
    api_key = os.getenv("RESEMBLE_API_KEY") or os.getenv("TTS__RESEMBLE_API_KEY")
    voice_uuid = os.getenv("RESEMBLE_VOICE_UUID") or os.getenv("TTS__RESEMBLE_VOICE_UUID")
    
    if not api_key:
        print("❌ RESEMBLE_API_KEY not set")
        return False
    
    if not voice_uuid:
        print("❌ RESEMBLE_VOICE_UUID not set")
        print("   Run test_list_voices first to get a voice UUID")
        print("   Then: export RESEMBLE_VOICE_UUID=your_voice_uuid")
        return False
    
    api = ResembleStreamingAPI(api_key=api_key, voice_uuid=voice_uuid)
    
    text = "Hello, this is a test of the Resemble AI streaming API powered by Chatterbox."
    
    print(f"Text: {text}")
    print(f"Voice UUID: {voice_uuid[:8]}...")
    print("\nStreaming audio chunks...")
    
    try:
        chunk_count = 0
        total_samples = 0
        
        async for audio_chunk in api.stream_audio(text):
            chunk_count += 1
            samples = len(audio_chunk)
            total_samples += samples
            print(f"  Chunk {chunk_count}: {samples} samples ({samples/48000:.2f}s)")
        
        print(f"\n✅ Streaming complete!")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Total samples: {total_samples}")
        print(f"   Duration: {total_samples/48000:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_factory_integration():
    """Test Harper factory integration"""
    print("\n" + "=" * 60)
    print("TEST 3: Harper Factory Integration")
    print("=" * 60)
    
    # Check if .env is configured
    from config import settings
    
    if not settings.tts.resemble_api_key:
        print("❌ TTS__RESEMBLE_API_KEY not set in .env")
        return False
    
    if not settings.tts.resemble_voice_uuid:
        print("❌ TTS__RESEMBLE_VOICE_UUID not set in .env")
        return False
    
    print(f"TTS Model: {settings.tts_model}")
    print(f"Voice UUID: {settings.tts.resemble_voice_uuid[:8]}...")
    print(f"Precision: {settings.tts.resemble_precision}")
    print(f"HD Mode: {settings.tts.resemble_use_hd}")
    
    # Import factory
    from tts.factory import get_tts_model
    
    try:
        tts = get_tts_model("resemble")
        print("\n✅ Factory loaded ResembleStreamingModel")
        
        # Test speech generation
        text = "Testing Harper factory integration."
        print(f"\nGenerating speech: '{text}'")
        
        chunk_count = 0
        async for chunk in tts.stream_speech(text):
            chunk_count += 1
            print(f"  Chunk {chunk_count}: {len(chunk)} samples")
        
        print(f"\n✅ Factory integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🧪 Resemble AI Streaming TTS Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: List voices
    results.append(("List Voices", await test_list_voices()))
    
    # Test 2: Streaming
    results.append(("Audio Streaming", await test_streaming()))
    
    # Test 3: Factory integration
    results.append(("Factory Integration", await test_factory_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    return all(passed for _, passed in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
