import torch
import asyncio
import threading
import inspect
from app.session import Session
from llm.openai import chat_followup
from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
import librosa
from librosa.util import normalize
from utils.process import pcm16_b64
from utils.constants import COMMON_STARTERS, DEFAULT_VOICE_PATH, DEFAULT_KOREAN_VOICE_PATH, FOLLOWUP_SILENCE_DELAY
from concurrent.futures import ThreadPoolExecutor
import orjson as json
import time
import torchaudio
import re
import random
import opuslib
import numpy as np
import struct

MAGIC = b'\xA1\x51'
CHUNK_SIZE = 32

def pack_frame(seq, ts_usec, payload: bytes, is_final=False):
    flags = 1 if is_final else 0
    header = MAGIC + struct.pack('<BBIQI', flags, 0, seq, ts_usec, len(payload))
    return header + payload

class OpusEnc:
    def __init__(self, sr=24000, channels=1, bitrate=32000):
        self.sr, self.ch = sr, channels
        self.frame_ms = 20
        self.frame_size = sr * self.frame_ms // 1000  # 480 @ 24k
        self.enc = opuslib.Encoder(sr, channels, opuslib.APPLICATION_AUDIO)
        try:
            self.enc.bitrate = bitrate
        except Exception:
            self.enc.set_bitrate(bitrate)
        self._carry = np.empty(0, dtype=np.int16)  # <--- NEW

    def encode(self, pcm_f32: np.ndarray) -> list[bytes]:
        # float32 -> int16
        pcm_i16 = np.clip(pcm_f32, -1.0, 1.0)
        pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16, copy=False)

        # prepend carry
        if self._carry.size:
            pcm_i16 = np.concatenate([self._carry, pcm_i16], axis=0)

        frames = []
        n = pcm_i16.shape[0]
        i = 0
        # consume only full frames
        while i + self.frame_size <= n:
            chunk_i16 = pcm_i16[i:i + self.frame_size]
            pkt = self.enc.encode(chunk_i16.tobytes(), self.frame_size)
            frames.append(pkt)
            i += self.frame_size

        # save leftover as next-call carry
        self._carry = pcm_i16[i:] if i < n else np.empty(0, dtype=np.int16)
        return frames

ENC_EXEC = ThreadPoolExecutor(max_workers=6)

tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

def jdumps(o): return json.dumps(o).decode()

@torch.inference_mode()
async def chatter_streamer(sess: Session):
    try:
        loop = asyncio.get_running_loop()
        sr = 24000
        TRIM = 3600
        # OVERLAP = int(0.01 * sr)
        OVERLAP = int(0.04 * sr)

        sess.tts_buffer_sr = sr
        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)

        opus_enc = OpusEnc(sr=sr, channels=1, bitrate=32000)
        seq_ref = {"seq": 0}
        session_t0 = time.time()
        
        sess.out_q.put_nowait(jdumps({
            "type": "tts_audio_meta",
            "format": "opus",
        }))

        def put_binary(out_q, b: bytes):
            out_q.put_nowait(b)
        
        async def emit_opus_frames(
            out_q: asyncio.Queue,
            enc: OpusEnc,
            pcm: torch.Tensor,
            sr: int,
            seq_ref: dict,
            is_final: bool = False,
            t0: float | None = None,
        ):
            """
            Encode pcm to 20ms Opus frames and push to out_q.
            """
            if pcm is None or pcm.numel() == 0:
                # no payload end frame
                ts_usec = int(((time.time()) - (t0 or 0.0)) * 1e6) if t0 else 0
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, b"", is_final=is_final))
                seq_ref["seq"] += 1
                return
        
            if pcm.dim() == 2:
                pcm = pcm.squeeze(0)
            pcm = pcm.detach().cpu().to(torch.float32).clamp_(-1.0, 1.0)
            sess.tts_pcm_buffer = np.concatenate([sess.tts_pcm_buffer, pcm.numpy()], axis=0)
        
            pcm_f32 = pcm.numpy().astype(np.float32, copy=False)
        
            frames = enc.encode(pcm_f32)

            now = time.time()
            if seq_ref["seq"] == 0:
                print(f"[TTS 📡 SEND] ⏰ [{now:.3f}] Sending FIRST Opus frame to client (seq=0)")
            base_ts = int(((now) - (t0 or now)) * 1e6)  # 배치 시작 기준
            dt = int(enc.frame_ms * 1000)  

            for k, pkt in enumerate(frames):
                ts_usec = base_ts + k * dt
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, pkt, is_final=False))
                seq_ref["seq"] += 1
            
            if is_final:
                ts_usec = int((time.time() - (t0 or now)) * 1e6)
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, b"", is_final=True))
                seq_ref["seq"] += 1

        def start_tts_producer_in_thread(text_chunk: str, ref_audio, out_q: asyncio.Queue):
            stop_evt = sess.tts_stop_event
            text_chunk = re.sub(r"\\n", " ... ", text_chunk)
            text_chunk = re.sub(r"\n", " ... ", text_chunk)

            async def produce():
                try:
                    gen_start = time.time()
                    print(f"[TTS 🎤 GEN] ⏰ [{gen_start:.3f}] Starting TTS generation for: '{text_chunk[:100]}'")
                    print(f"[TTS 📊 PARAMS] chunk_size={CHUNK_SIZE}, exaggeration=0.9, cfg_weight=0.8, temperature=0.7, repetition_penalty=1.3, min_p=0.02, top_p=0.9, language={sess.language}")
                    agen = tts_model.generate_stream(
                        text_chunk,
                        audio_prompt_path=ref_audio,
                        language_id=sess.language,
                        chunk_size=CHUNK_SIZE, # This should be adjusted for realtime factor
                        exaggeration=0.68,
                        cfg_weight=0.55,
                        temperature=0.75,
                        repetition_penalty=1.3,
                        min_p=0.02,
                        top_p=0.9
                    )
                    first_chunk_received = False
                    if inspect.isasyncgen(agen):
                        async for evt in agen:
                            if not first_chunk_received:
                                first_chunk_time = time.time()
                                print(f"[TTS 🎤 FIRST] ⏰ [{first_chunk_time:.3f}] First audio chunk received (+{(first_chunk_time-gen_start)*1000:.1f}ms from gen start)")
                                first_chunk_received = True
                            if stop_evt.is_set():
                                print("\n\nstop event\n\n")
                                break
                            audio = evt.get("audio", None)
                            if audio is None and evt["type"] == "eos":
                                loop.call_soon_threadsafe(out_q.put_nowait, ("chunk_eos", audio))
                            else:
                                loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))
                    else:
                        for evt in agen:
                            if stop_evt.is_set():
                                print("\n\nstop event\n\n")
                                break
                            audio = evt.get("audio", None)
                            if audio is None and evt["type"] == "eos":
                                loop.call_soon_threadsafe(out_q.put_nowait, ("chunk_eos", audio))
                            else:
                                loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))
                    print("Come here")
                    loop.call_soon_threadsafe(out_q.put_nowait, ("eos", None))
                except Exception as e:
                    print("Error as e ", e)
                    loop.call_soon_threadsafe(out_q.put_nowait, ("error", str(e)))

            def thread_target():
                asyncio.run(produce())

            t = threading.Thread(target=thread_target, daemon=True)
            t.start()

        async def consume_loop():
            while sess.running:
                t_before_get = time.time()
                print(f"[TTS 🔄] ⏰ [{t_before_get:.3f}] Waiting for text from queue (queue size: {sess.tts_in_q.qsize()})...")
                text_chunk = await sess.tts_in_q.get()
                t_got_text = time.time()
                print(f"[TTS ⏱️ START] ⏰ [{t_got_text:.3f}] +{(t_got_text-t_before_get)*1000:.1f}ms - 📥 Received from queue: '{text_chunk[:100] if isinstance(text_chunk, str) else text_chunk}'")
                tts_start_time = time.time()
                sample_used = False
                sample_send_time = None
                
                if not text_chunk or sess.tts_stop_event.is_set():
                    print(f"[chatter_streamer] ⏰ [{time.time():.3f}] TTS stop event is set or empty text: {text_chunk}")
                    continue
                
                is_last = True
                if "<cont>" in text_chunk:
                    is_last=False
                    text_chunk = re.sub('<cont>', '', text_chunk)
                
                if isinstance(text_chunk, tuple) and len(text_chunk) == 2 and text_chunk[0] == "__silence__":
                    try:
                        silence_sec = float(text_chunk[1]) * 0.4
                        if silence_sec > 0:
                            num_samples = int(silence_sec * sr)
                            silence_wav = torch.zeros(1, num_samples, dtype=torch.float32)
                            await emit_opus_frames(sess.out_q, opus_enc, silence_wav, sr, seq_ref, is_final=False, t0=session_t0)
                            print(f"[silence] sent {silence_sec:.2f}s (opus)")
                    except Exception as e:
                        print(f"[silence] error: {e}")
                    continue


                # === Prepare reference audio ===
                if hasattr(sess, "ref_audios") and not getattr(sess, "ref_audios").empty():
                    ref_audio = sess.ref_audios.get()
                    sess.ref_audios.put(ref_audio)
                    ref_audio = normalize(ref_audio[-int(16000 * 15):])
                else:
                    ref_audio = DEFAULT_VOICE_PATH if sess.language != 'ko' else DEFAULT_KOREAN_VOICE_PATH

                cancel_silence_nudge(sess)
                
                # === Thread → Main loop chunk queue ===
                tts_chunk_q: asyncio.Queue = asyncio.Queue(maxsize=6)
                print(f"[TTS] ⏰ [{time.time():.3f}] Created TTS chunk queue")

                # === (Add) Detect starter → Send sample wav ===
                stripped = text_chunk.lstrip()
                matched = None
                
                for s in COMMON_STARTERS:
                    if stripped.startswith(s):
                        matched = s
                        break
                if matched is not None:
                    token = matched.lower().strip().replace(".", "").replace(",", "")
                    idx = random.choice([0, 1, 2])
                    sample_path = f"./audio_samples/{token}_{idx}.wav"
                    sample_load_start = time.time()
                    print(f"[TTS 🎵 SAMPLE] Using pregenerated sample: {sample_path} for '{matched}'")
                    try:
                        # Use librosa instead of torchaudio (no torchcodec dependency)
                        wav_np, sr_file = librosa.load(sample_path, sr=None, mono=False)
                        
                        # Convert to torch tensor and ensure correct shape
                        if wav_np.ndim == 1:
                            wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
                        else:
                            wav = torch.from_numpy(wav_np)  # (ch, T)
                            if wav.shape[0] > 1:
                                wav = wav.mean(dim=0, keepdim=True)  # mono
                        
                        # Resample if needed
                        if sr_file != sr:
                            wav_np_resampled = librosa.resample(wav.squeeze(0).numpy(), orig_sr=sr_file, target_sr=sr)
                            wav = torch.from_numpy(wav_np_resampled).unsqueeze(0)
                        
                        # Send to Opus
                        asyncio.create_task(emit_opus_frames(sess.out_q, opus_enc, wav, sr, seq_ref, is_final=False, t0=session_t0))
                        sample_send_time = time.time()
                        sample_duration = wav.shape[-1]/sr
                        sample_load_time = sample_send_time - sample_load_start
                        time_to_first_audio = sample_send_time - tts_start_time
                        sample_used = True
                        print(f"[TTS 🎵 SAMPLE] ✅ Sent pregenerated sample ({sample_duration:.2f}s audio) in {sample_load_time:.3f}s (total latency: {time_to_first_audio:.3f}s)")
                    except Exception as e:
                        print(f"[starter] ⚠️ Failed to load '{sample_path}': {e}")
                    text_chunk = re.sub(matched, '', text_chunk[:10]) + text_chunk[10:]
                
                prev_audio_end_at = 0
                last_tail_audio: torch.Tensor | None = None
                start_time = time.time()
                
                print(f"[TTS 🎤 GENERATE] ⏰ [{time.time():.3f}] Starting TTS generation for: '{text_chunk[:80]}...'")
                start_tts_producer_in_thread(text_chunk, ref_audio, tts_chunk_q)
                print(f"[TTS 🎤 GENERATE] ⏰ [{time.time():.3f}] TTS producer thread started")
                start_sending_at = 0
                total_audio_seconds = 0

                allaudios = torch.zeros(1, 0).to('cuda')
                ii=0
                while True:
                    try:
                        if sess.tts_stop_event.is_set():
                            try:
                                while True:
                                    _ = tts_chunk_q.get_nowait()
                                    tts_chunk_q.task_done()
                            except asyncio.QueueEmpty:
                                pass
                            break
                        evt_type, payload = await tts_chunk_q.get()
                    except Exception as e:
                        print("Event error ", e)

                    if evt_type == "chunk":
                        ii += 1
                        try:
                            wav: torch.Tensor = payload  # (ch, T)
                            if wav is None or wav.numel() == 0:
                                await asyncio.sleep(0)
                                continue
                            
                            total_audio_length = wav.shape[-1]
                            if total_audio_length - prev_audio_end_at <= 0:
                                await asyncio.sleep(0)
                                continue

                            if total_audio_length < TRIM + OVERLAP + int(sr*0.8):
                                new_part = wav[:, prev_audio_end_at:]
                                trimmed_part = None
                            else:
                                new_part = wav[:, prev_audio_end_at:-TRIM]
                                trimmed_part = wav[:, -TRIM:]
                            # print("new_part : ", new_part.shape)

                            # fade_out_ms = 50
                            # fade_samples = int(24000 * fade_out_ms / 1000)
                            # fade_curve = torch.tensor(np.linspace(1, 0, fade_samples), device=new_part.device)
                            # new_part[:, -fade_samples:] *= fade_curve

                            fade_out_ms = 20
                            fade_samples = int(24000 * fade_out_ms / 1000)
                            fade_samples_ip = int(24000 * 15 / 1000)
                            # new_part = wav[:, max(prev_audio_end_at, 0):-TRIM].clone()  # 반드시 clone() 붙이기
                            new_part = wav[:, max(prev_audio_end_at-fade_samples, 0):-TRIM].clone()  # 반드시 clone() 붙이기
                            fade_curve = torch.tensor(np.linspace(1, 0, fade_samples), device=new_part.device)
                            fade_curve_up = torch.tensor(np.linspace(0, 1, fade_samples_ip), device=new_part.device)
                            
                            new_part[:, -fade_samples:] = new_part[:, -fade_samples:] * fade_curve
                            new_part[:, :fade_samples_ip] = new_part[:, :fade_samples_ip] * fade_curve_up

                            out_chunk = new_part

                            t_before_emit = time.time()
                            await emit_opus_frames(sess.out_q, opus_enc, out_chunk, sr, seq_ref, is_final=False, t0=session_t0)
                            t_after_emit = time.time()
                            
                            prev_audio_end_at = total_audio_length - TRIM
                            
                            allaudios = torch.cat([allaudios, out_chunk], dim=-1)
                            if start_sending_at == 0:
                                start_sending_at = time.time()
                                print(f"[TTS 🎵 FIRST] ⏰ [{start_sending_at:.3f}] First audio sent! Latency: {(start_sending_at-tts_start_time):.3f}s, emit time: {(t_after_emit-t_before_emit)*1000:.1f}ms")
                            total_audio_seconds += (total_audio_length-prev_audio_end_at)/24000
                            
                            await asyncio.sleep(0)
                        except Exception as e:
                            print("Erorr : ", e)

                    if evt_type == "eos":
                        print
                        if not is_last:
                            break

                        total_audio_seconds = sess.tts_pcm_buffer.shape[-1]/24000
                        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)
                        await emit_opus_frames(sess.out_q, opus_enc, None, sr, seq_ref, is_final=True, t0=session_t0)

                        taken = time.time() - start_sending_at
                        remaining_until_audio_end = total_audio_seconds - taken + 1
                        total_tts_time = time.time() - tts_start_time
                        
                        # Calculate time to first audio (sample vs TTS generation)
                        if sample_used:
                            tts_gen_start = sample_send_time
                            time_to_first_audio_tts = start_sending_at - tts_start_time if start_sending_at > 0 else 0
                            print(f"[TTS ⏱️ COMPLETE] ✅ Used SAMPLE for first audio")
                            print(f"  📊 Sample latency: {sample_send_time - tts_start_time:.3f}s")
                            print(f"  📊 TTS generation started at: +{tts_gen_start - tts_start_time:.3f}s")
                            if start_sending_at > 0:
                                print(f"  📊 First TTS audio at: +{time_to_first_audio_tts:.3f}s")
                                print(f"  💡 Sample saved: {time_to_first_audio_tts - (sample_send_time - tts_start_time):.3f}s vs full TTS")
                        else:
                            time_to_first_audio_notts = start_sending_at - tts_start_time if start_sending_at > 0 else 0
                            print(f"[TTS ⏱️ COMPLETE] ⚠️ NO sample used, pure TTS")
                            print(f"  📊 Time to first audio: {time_to_first_audio_notts:.3f}s")
                        
                        print(f"  📊 Total: Taken={taken:.3f}s, Audio={total_audio_seconds:.3f}s, Remain={remaining_until_audio_end:.3f}s, Total TTS time={total_tts_time:.3f}s")
                        
                        schedule_silence_nudge(sess, delay=FOLLOWUP_SILENCE_DELAY, remain=remaining_until_audio_end)
                        
                        break

                    elif evt_type == "error":
                        print("🥊 [chatter_streamer] TTS error:", payload)
                        sess.out_q.put_nowait(jdumps({"type": "tts_error", "message": payload}))
                        break

        await consume_loop()

    except asyncio.CancelledError:
        raise
    except Exception as e:
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        pass

async def proactive_say(sess: Session):
    print("[proactive_say] 🤖 Generating silence nudge message...")
    loop = asyncio.get_running_loop()

    def run_blocking():
        return chat_followup(
            prev_scripts=sess.transcripts[-10:],
            prev_answers=sess.outputs[-10:],
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time,
        )

    try:
        output = await loop.run_in_executor(None, run_blocking)
        tuples = (output.get("text", "") or "")
        print("[proactive_say] 📝 Generated text:", tuples, "\n")
        if tuples[0] is None or tuples[0] == '':
            print("[proactive_say] ⚠️ Empty text, skipping")
            return
        if tuples[1] == 'wait':
            print("[proactive_say] ⏸️ Received 'wait' signal, skipping")
            return
        text = tuples[0]
        
        sess.answer = text.strip()
        sess.outputs[-1] = sess.outputs[-1] + " (User silence for six seconds) " + text
        print(f"[proactive_say] ✅ Will send text to TTS: '{text}'")

    except Exception as e:
        print("[proactive_say] ❌ NUDGE ERROR:", e)
        return

    t_start = time.time()
    print(f"[proactive_say] ⏰ [{t_start:.3f}] 📡 Sending tts_audio_meta to frontend")        
    sess.out_q.put_nowait(jdumps({
        "type": "tts_audio_meta",
        "format": "opus",
    }))
    t_after_meta = time.time()
    print(f"[proactive_say] ⏰ [{t_after_meta:.3f}] +{(t_after_meta-t_start)*1000:.1f}ms - 📤 Sending text to TTS queue (length={len(text)}): '{text[:50]}...'")
    print(f"[proactive_say] 🔍 TTS queue current size: {sess.tts_in_q.qsize()}")
    loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, text)
    t_after_tts_queue = time.time()
    print(f"[proactive_say] ⏰ [{t_after_tts_queue:.3f}] +{(t_after_tts_queue-t_start)*1000:.1f}ms - ✅ Text sent to TTS queue")
    print(f"[proactive_say] 📡 Sending speaking message to frontend")
    loop.call_soon_threadsafe(
        sess.out_q.put_nowait,
        jdumps({"type": "speaking", "script": "", "text": text, "is_final": True})
    )
    t_end = time.time()
    print(f"[proactive_say] ⏰ [{t_end:.3f}] +{(t_end-t_start)*1000:.1f}ms - ✅ All messages sent, total time: {(t_end-t_start)*1000:.1f}ms")

def cancel_silence_nudge(sess: Session):
    """Cancel silence nudge task"""
    task = getattr(sess, "silence_nudge_task", None)
    if task and not task.done():
        task.cancel()
    setattr(sess, "silence_nudge_task", None)

def schedule_silence_nudge(sess: Session, delay: float = 5.0, remain: float = 1.0):
    cancel_silence_nudge(sess)
    print(f"[schedule_silence_nudge] ⏰ Scheduling nudge: delay={delay}s, remain={remain}s")

    async def waiter():
        try:
            await asyncio.sleep(remain)
            if getattr(sess, "current_audio_state", "none") != "none":
                print("[silence_nudge] ⚠️ Audio still playing, skipping nudge")
                return
            print(f"[silence_nudge] ⏳ Waiting {delay}s for user response...")
            await asyncio.sleep(delay)
            if getattr(sess, "current_audio_state", "none") == "none":
                print("[silence_nudge] 🎯 Silence detected, triggering proactive_say")
                await proactive_say(sess)
            else:
                print("[silence_nudge] ⚠️ User responded, cancelling nudge")
        except asyncio.CancelledError:
            print("[silence_nudge] ❌ Nudge cancelled")
            pass

    # Register new timer
    if random.random() > 0.5:
        print("[schedule_silence_nudge] ✅ Registered nudge task (50% chance)")
        sess.silence_nudge_task = asyncio.create_task(waiter())
    else:
        print("[schedule_silence_nudge] ⏭️ Skipped nudge task (50% chance)")

