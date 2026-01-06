"""
Adaptive TTS Streaming - Works with both factory pattern and original chatterbox
"""
import asyncio
import numpy as np
import time
import struct
import opuslib
import librosa
from typing import Optional
from app.session import Session
from utils.utils import dprint
from config import settings
import orjson as json

def jdumps(o): return json.dumps(o).decode()

# Opus frame packing (same as chatterbox)
MAGIC = b'\xA1\x51'

def pack_frame(seq, ts_usec, payload: bytes, is_final=False):
    flags = 1 if is_final else 0
    header = MAGIC + struct.pack('<BBIQI', flags, 0, seq, ts_usec, len(payload))
    return header + payload

class OpusEncoder:
    """Opus encoder for streaming audio with carry buffer"""
    def __init__(self, sr=24000, channels=1, bitrate=32000, frame_ms=20):
        self.sr = sr
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_size = int(sr * frame_ms / 1000)  # samples per frame
        self.encoder = opuslib.Encoder(sr, channels, opuslib.APPLICATION_VOIP)
        self.encoder.bitrate = bitrate
        self._carry = np.empty(0, dtype=np.float32)  # Keep carry as float32 to avoid rounding errors
    
    def encode(self, pcm_f32: np.ndarray) -> list[bytes]:
        """Encode PCM float32 to Opus packets with carry buffer"""
        # Prepend carry buffer from previous call (keep as float32)
        if self._carry.size > 0:
            pcm_f32 = np.concatenate([self._carry, pcm_f32])
        
        frames = []
        n = len(pcm_f32)
        offset = 0
        
        # Encode only complete frames
        while offset + self.frame_size <= n:
            frame_f32 = pcm_f32[offset:offset + self.frame_size]
            # Convert to int16 only at encoding time to minimize rounding errors
            frame_i16 = (np.clip(frame_f32, -1.0, 1.0) * 32767).astype(np.int16)
            packet = self.encoder.encode(frame_i16.tobytes(), self.frame_size)
            frames.append(packet)
            offset += self.frame_size
        
        # Save leftover samples for next call (keep as float32)
        self._carry = pcm_f32[offset:] if offset < n else np.empty(0, dtype=np.float32)
        
        return frames


async def adaptive_tts_streamer(sess: Session):
    """
    Adaptive TTS streamer that uses factory pattern if available.
    NO FALLBACK to chatterbox when remote TTS fails.
    """
    if hasattr(sess, 'tts_model') and sess.tts_model is not None:
        # Use factory pattern TTS
        await factory_tts_streamer(sess)
    else:
        # NO FALLBACK: If TTS model is not available, don't use local Chatterbox
        print("[TTS] ⚠️ TTS model not available, NO fallback to local Chatterbox")
        print("[TTS] Streamer will not process any TTS requests")
        # Keep the task alive but idle
        while sess.running:
            await asyncio.sleep(1)


async def factory_tts_streamer(sess: Session):
    """
    TTS streamer using factory pattern with Opus encoding (same as chatterbox)
    """
    tts_model = sess.tts_model
    print(f"[TTS Factory] factory_tts_streamer started, model: {tts_model}")
    
    # CRITICAL: Opus encoder must use 24kHz (same as chatterbox)
    opus_sr = 24000
    opus_enc = OpusEncoder(sr=opus_sr, channels=1, bitrate=64000)
    seq = 0
    session_t0 = time.time()
    
    # Send audio metadata (same as chatterbox)
    sess.out_q.put_nowait(jdumps({
        "type": "tts_audio_meta",
        "format": "opus",
    }))
    print(f"[TTS Factory] Sent tts_audio_meta (format: opus, sr: {opus_sr})")
    
    try:
        while sess.running:
            try:
                # Wait for text input
                print(f"[TTS Factory] Waiting for text from tts_in_q...")
                text = await asyncio.wait_for(sess.tts_in_q.get(), timeout=1.0)
                print(f"[TTS Factory] ===== RECEIVED FROM QUEUE =====")
                print(f"[TTS Factory] Type: {type(text)}")
                if isinstance(text, tuple):
                    print(f"[TTS Factory] Tuple content: {text}")
                else:
                    print(f"[TTS Factory] Text length: {len(text) if text else 0} chars")
                    print(f"[TTS Factory] Full text: {repr(text)}")
                print(f"[TTS Factory] =================================")
                
                # Handle special messages (silence markers, etc.)
                if isinstance(text, tuple):
                    msg_type = text[0]
                    if msg_type == "__silence__":
                        duration = text[1]
                        print(f"[TTS Factory] Silence marker: {duration}s - generating silence audio")
                        # Generate silence audio frames
                        num_samples = int(duration * opus_sr)
                        silence_audio = np.zeros(num_samples, dtype=np.float32)
                        
                        # Encode silence to Opus and send frames
                        opus_frames = opus_enc.encode(silence_audio)
                        now = time.time()
                        base_ts = int((now - session_t0) * 1e6)
                        dt = int(opus_enc.frame_ms * 1000)
                        
                        for k, opus_packet in enumerate(opus_frames):
                            ts_usec = base_ts + k * dt
                            binary_frame = pack_frame(seq, ts_usec, opus_packet, is_final=False)
                            sess.out_q.put_nowait(binary_frame)
                            seq += 1
                        
                        print(f"[TTS Factory] Sent {len(opus_frames)} silence frames ({duration}s)")
                    continue
                
                if text is None or (isinstance(text, str) and text.strip() == ""):
                    print(f"[TTS Factory] ⚠️ Skipping empty text")
                    continue
                
                # Check if this is a continuation (more content coming)
                is_continuation = False
                text_to_process = text
                if "<cont>" in text:
                    is_continuation = True
                    text_to_process = text.replace("<cont>", "")
                    print(f"[TTS Factory] 🔄 Continuation detected (<cont> marker found)")
                    print(f"[TTS Factory] Removed <cont> marker, new text: {repr(text_to_process)}")
                    
                print(f"[TTS Factory] 🎤 Processing text for TTS: {repr(text_to_process[:100])}...")
                print(f"[TTS Factory] Full text to synthesize: {repr(text_to_process)}")
                print(f"[TTS Factory] Is continuation: {is_continuation}")
                
                # Generate audio using factory TTS model
                chunk_count = 0
                frame_count = 0
                requested_sr = opus_sr  # 24000 Hz
                
                # Together AI pattern: NO RESAMPLING, NO complex buffering
                # API returns at requested rate, encode directly to Opus
                
                async for audio_chunk in tts_model.stream_speech(
                    text=text_to_process,
                    voice=settings.tts.voice,
                    sample_rate=requested_sr
                ):
                    if not sess.running:
                        break
                    
                    chunk_count += 1
                    
                    # Convert to int16 numpy array (Together AI pattern)
                    if isinstance(audio_chunk, np.ndarray):
                        if audio_chunk.dtype == np.int16:
                            pcm_i16 = audio_chunk
                        elif audio_chunk.dtype == np.float32:
                            pcm_i16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                        else:
                            pcm_i16 = audio_chunk.astype(np.int16)
                    else:
                        pcm_i16 = np.frombuffer(audio_chunk, dtype=np.int16)
                    
                    if chunk_count <= 3:
                        print(f"[TTS Factory] Chunk {chunk_count}: {len(pcm_i16)} samples")
                    
                    # Convert int16 to float32 for Opus (correct normalization)
                    chunk_audio_f32 = pcm_i16.astype(np.float32) / 32767.0
                    
                    # Together AI pattern: Encode directly, OpusEncoder handles carry internally
                    opus_frames = opus_enc.encode(chunk_audio_f32)
                    
                    # Send each Opus frame with proper timing
                    now = time.time()
                    base_ts = int((now - session_t0) * 1e6)
                    dt = int(opus_enc.frame_ms * 1000)
                    
                    for k, opus_packet in enumerate(opus_frames):
                        ts_usec = base_ts + k * dt
                        binary_frame = pack_frame(seq, ts_usec, opus_packet, is_final=False)
                        sess.out_q.put_nowait(binary_frame)
                        seq += 1
                        frame_count += 1
                    
                    if chunk_count % 20 == 0:
                        print(f"[TTS Factory] Progress: {chunk_count} chunks, {frame_count} frames")
                
                # Flush any remaining samples in OpusEncoder
                if opus_enc._carry.size > 0:
                    print(f"[TTS Factory] Flushing {opus_enc._carry.size} samples")
                    # Pad with zeros to complete frame
                    frame_size = opus_enc.frame_size
                    padding_needed = frame_size - opus_enc._carry.size
                    if padding_needed > 0:
                        padded = np.concatenate([opus_enc._carry, np.zeros(padding_needed, dtype=np.float32)])
                        final_frames = opus_enc.encode(padded)
                        now = time.time()
                        base_ts = int((now - session_t0) * 1e6)
                        dt = int(opus_enc.frame_ms * 1000)
                        for k, opus_packet in enumerate(final_frames):
                            ts_usec = base_ts + k * dt
                            binary_frame = pack_frame(seq, ts_usec, opus_packet, is_final=False)
                            sess.out_q.put_nowait(binary_frame)
                            seq += 1
                            frame_count += 1
                
                # Send final frame ONLY if this is not a continuation
                if not is_continuation:
                    print(f"[TTS Factory] 🏁 Sending final frame (is_final=True)")
                    ts_usec = int((time.time() - session_t0) * 1e6)
                    final_frame = pack_frame(seq, ts_usec, b"", is_final=True)
                    sess.out_q.put_nowait(final_frame)
                    seq += 1
                else:
                    print(f"[TTS Factory] ⏭️  Skipping final frame (continuation expected)")
                
                print(f"[TTS Factory] ✅ Audio streaming complete. Chunks: {chunk_count}, Frames: {frame_count}")
                
                # Mark task as done
                sess.tts_in_q.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                print(f"[TTS Factory] ❌ Error processing TTS: {e}")
                traceback.print_exc()
                
                # Notify frontend about TTS error
                try:
                    error_msg = f"Error en TTS: {str(e)[:100]}"
                    sess.out_q.put_nowait(jdumps({
                        "type": "error",
                        "message": error_msg,
                        "error_type": "tts_streaming_failed"
                    }))
                    print(f"[TTS Factory] 📤 Sent error notification to frontend")
                except Exception as notify_err:
                    print(f"[TTS Factory] ⚠️ Failed to notify frontend: {notify_err}")
                
                # Mark task as done even on error
                try:
                    sess.tts_in_q.task_done()
                except:
                    pass
                
    except Exception as e:
        import traceback
        print(f"[TTS Factory] ❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("[TTS Factory] Streamer stopped")


# Export the adaptive streamer as the main interface
tts_streamer = adaptive_tts_streamer