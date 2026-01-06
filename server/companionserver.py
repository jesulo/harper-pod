import os
import socket
import multiprocessing as mp
import asyncio
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import orjson as json
import time
from websockets.asyncio.client import connect as ws_connect
import numpy as np
import librosa
import torch
import threading
import re

from stt.asr import load_asr_backend
from stt.factory import get_stt_model
from tts.factory import get_tts_model
from stt.vad import check_audio_state
from utils.process import process_data_to_audio
from app.session import Session
from app.session_control import teardown_session, outbound_sender
from utils.text_process import text_pr
from utils.process import get_volume, pcm16_b64
from utils.utils import dprint, lprint
from llm.conversation import conversation_worker, answer_greeting

# Lazy imports for TTS to avoid loading unused models
# from tts.chatter_infer import chatter_streamer, cancel_silence_nudge
from tts.adaptive_streamer import tts_streamer
from utils.opus import ensure_opus_decoder, decode_opus_float, parse_frame
from third.smart_turn.inference import predict_endpoint
from config import settings
from utils.constants import DEFAULT_VOICE_PATH

INPUT_SAMPLE_RATE = settings.input_sample_rate
WHISPER_SR = 16000

def jdumps(o):
    return json.dumps(o).decode()

# Helper function for dynamic import of chatterbox-specific functions
def cancel_silence_nudge_safe(sess):
    """Safely cancel silence nudge - only imports chatterbox if needed"""
    if settings.tts_model == "chatterbox":
        from tts.chatter_infer import cancel_silence_nudge
        cancel_silence_nudge(sess)
    # For other TTS models, nudge cancellation is handled in their own streamers

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI()

ASR = None
TTS = None
LLM = None

sessions: Dict[int, Session] = {}


@app.on_event("startup")
def init_models():
    """Initialize global runtime switches and preload optional assets."""
    # Multiprocessing / env knobs must be set ASAP
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    # Python multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set by another import path
        pass

    # Warmup STT backend if configured
    global ASR
    if settings.stt_model and settings.stt_model.lower() not in ["none", "disabled", ""]:
        # Remote STT services (groq, openai) don't need local model loading
        if settings.stt_model.lower() in ["groq", "openai"]:
            print(f"[STARTUP] STT configurado para usar servicio remoto: {settings.stt_model.upper()}")
            print(f"[STARTUP] ✅ No se carga modelo local STT (usando API {settings.stt_model})")
            ASR = get_stt_model(settings.stt_model)
        else:
            print(f"[STARTUP] Inicializando modelo STT local (STT_MODEL={settings.stt_model})...")
            try:
                ASR = get_stt_model(settings.stt_model)
                print(f"[STARTUP] ✅ Modelo STT local cargado: {settings.stt_model}")
            except Exception as e:
                print(f"[STARTUP] ⚠️ Error al cargar modelo STT: {e}")
                ASR = None
    else:
        print(f"[STARTUP] STT deshabilitado (STT_MODEL={settings.stt_model or 'not set'})")

    # Warmup Chatterbox TTS ONLY if configured and enabled
    global TTS
    if settings.tts_model and settings.tts_model.lower() not in ["none", "disabled", ""]:
        # Remote TTS services (resemble, openai) don't need local model loading
        if settings.tts_model.lower() in ["resemble", "openai"]:
            print(f"[STARTUP] TTS configurado para usar servicio remoto: {settings.tts_model.upper()}")
            print(f"[STARTUP] ✅ No se carga modelo local TTS (usando API {settings.tts_model})")
            TTS = None  # Remote services don't need pre-loaded model
        elif settings.tts_model == "chatterbox":
            print(f"[STARTUP] Cargando modelo TTS local Chatterbox...")
            try:
                from tts.chatter_infer import tts_model
                print(f"[STARTUP] ✅ Modelo Chatterbox local cargado: {type(tts_model).__name__}")
                TTS = "chatterbox_warmed"
            except Exception as e:
                print(f"[STARTUP] ⚠️ Error al cargar Chatterbox: {e}")
                TTS = None
        else:
            print(f"[STARTUP] Cargando TTS: {settings.tts_model}")
            TTS = None
    else:
        print(f"[STARTUP] TTS deshabilitado (TTS_MODEL={settings.tts_model or 'not set'})")

    # If you need to warm up any LLM backends, do it here (lazy by default)
    # global LLM


async def transcribe_pcm_generic(audios, sample_rate: int, channels: int, language: str) -> str:
    if not audios:
        return ""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: ASR.transcribe_pcm(audios, sample_rate, channels, language=language)
    )


async def stt_worker(sess: Session, in_q: asyncio.Queue, out_q: asyncio.Queue):
    try:
        while True:
            pcm_bytes = await in_q.get()  # wait independently of the main loop
            try:
                sttstart = time.time()
                print(f"[stt_worker] Processing {len(pcm_bytes)} bytes, queue size: {in_q.qsize()}")
                text = await transcribe_pcm_generic(
                    audios=pcm_bytes,
                    sample_rate=WHISPER_SR,
                    channels=sess.input_channels,
                    language=sess.language,
                )
                stt_duration = time.time() - sttstart
                print(f"[stt_worker] Transcription took {stt_duration:.3f}s, result: '{text[:100]}'")
                if text:
                    await out_q.put({"type": "delta", "text": text})
                else:
                    print(f"[stt_worker] ⚠️ Empty transcription result")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[stt_worker] ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                in_q.task_done()
    finally:
        # drain to avoid pending tasks on shutdown
        while not in_q.empty():
            try:
                in_q.get_nowait()
                in_q.task_done()
            except Exception:
                break


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    print(f"[WS] New WebSocket connection attempt from {ws.client}")
    await ws.accept()
    print(f"[WS] WebSocket connection accepted")
    sess = Session(input_sr=INPUT_SAMPLE_RATE, input_channels=1)
    sessions[id(ws)] = sess

    sess.sender_task = asyncio.create_task(outbound_sender(sess, ws))

    try:
        while True:
            msg = await ws.receive()
            # print(f"[WS] Received message type: {type(msg)}, keys: {msg.keys() if isinstance(msg, dict) else 'N/A'}")
            if msg.get("text") is not None:
                # print(f"[WS] Received text message: {msg['text'][:200]}...")
                try:
                    data = json.loads(msg["text"])
                    # print(f"[WS] Parsed JSON message type: {data.get('type')}")
                except json.JSONDecodeError:
                    print(f"[WS] JSON decode error")
                    await ws.send_text(jdumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                t = data.get("type")
                # print(f"[WS] Processing message type: {t}")

                if t == "ping":
                    await ws.send_text(
                        jdumps(
                            {
                                "type": "pong",
                                "t0": data.get("t0"),
                                "server_now": int(time.time() * 1000),
                            }
                        )
                    )
                    continue

                if t == "scriptsession.setvoice":
                    inputprompt = data.get("prompt")
                    print("[scriptsession.setvoice] : ", inputprompt)
                    if inputprompt:
                        sess.prompt = inputprompt

                if t == "scriptsession.clonevoice":
                    if data.get("voice") is not None:
                        voice = data.get("voice")
                        audio = process_data_to_audio(voice, input_sample_rate=24000, whisper_sr=WHISPER_SR)
                        sess.ref_audios.put(audio)

                # 1) Session start
                if t == "scriptsession.start":
                    global ASR, TTS
                    lprint("Start ", data)

                    # current time (ms) if provided by client
                    if data.get("time") is not None:
                        sess.current_time = data.get("time")

                    requested_lang = data.get("language", "en").strip()
                    
                    # ========== HEALTHCHECK: Verify external APIs BEFORE starting session ==========
                    print("[Healthcheck] 🏥 Verificando APIs externas antes de iniciar sesión...")
                    healthcheck_failed = False
                    error_messages = []
                    
                    # Check STT service (Groq, OpenAI, etc.)
                    if settings.stt_model and settings.stt_model.lower() in ["groq", "openai"]:
                        try:
                            print(f"[Healthcheck] Verificando STT ({settings.stt_model.upper()})...")
                            if settings.stt_model == "groq":
                                from groq import Groq
                                groq_client = Groq(api_key=settings.stt.groq_api_key)
                                # Simple API check - list models
                                models = groq_client.models.list()
                                print(f"[Healthcheck] ✅ Groq API OK")
                            elif settings.stt_model == "openai":
                                from openai import OpenAI
                                openai_client = OpenAI(api_key=settings.openai_api_key)
                                # Simple API check
                                models = openai_client.models.list()
                                print(f"[Healthcheck] ✅ OpenAI STT API OK")
                        except Exception as e:
                            error_msg = f"STT ({settings.stt_model}) API no disponible: {str(e)[:100]}"
                            print(f"[Healthcheck] ❌ {error_msg}")
                            error_messages.append(error_msg)
                            healthcheck_failed = True
                    
                    # Check TTS service (Resemble, OpenAI, etc.)
                    if settings.tts_model and settings.tts_model.lower() in ["resemble", "openai"]:
                        try:
                            print(f"[Healthcheck] Verificando TTS ({settings.tts_model.upper()})...")
                            if settings.tts_model == "resemble":
                                import httpx
                                # Verify Resemble API key with a simple request
                                async with httpx.AsyncClient(timeout=5.0) as client:
                                    # Test endpoint to verify credentials
                                    headers = {
                                        "Authorization": f"Bearer {settings.tts.resemble_api_key}",
                                        "Content-Type": "application/json"
                                    }
                                    # Simple health check - try to access voices endpoint
                                    response = await client.get(
                                        "https://app.resemble.ai/api/v2/voices",
                                        headers=headers
                                    )
                                    if response.status_code == 401:
                                        raise Exception("API key inválida o expirada (401)")
                                    elif response.status_code >= 400:
                                        raise Exception(f"HTTP {response.status_code}")
                                    print(f"[Healthcheck] ✅ Resemble API OK")
                            elif settings.tts_model == "openai":
                                from openai import OpenAI
                                openai_client = OpenAI(api_key=settings.openai_api_key)
                                models = openai_client.models.list()
                                print(f"[Healthcheck] ✅ OpenAI TTS API OK")
                        except Exception as e:
                            error_msg = f"TTS ({settings.tts_model}) API no disponible: {str(e)[:100]}"
                            print(f"[Healthcheck] ❌ {error_msg}")
                            error_messages.append(error_msg)
                            healthcheck_failed = True
                    
                    # If healthcheck failed, abort session start
                    if healthcheck_failed:
                        full_error = "\\n".join(error_messages)
                        print(f"[Healthcheck] ⛔ Healthcheck fallido. No se puede iniciar la sesión.")
                        await ws.send_text(jdumps({
                            "type": "error",
                            "message": f"No se puede iniciar la sesión. Servicios no disponibles:\\n{full_error}",
                            "error_type": "api_healthcheck_failed",
                            "details": error_messages
                        }))
                        continue  # Don't start session
                    
                    print("[Healthcheck] ✅ Todos los servicios están disponibles")
                    # ========== END HEALTHCHECK ==========
                    
                    # Use pre-loaded STT from startup (or load now if not done)
                    if ASR is None and settings.stt_model and settings.stt_model.lower() not in ["none", "disabled", ""]:
                        try:
                            print(f"[STT] Loading model on-demand: {settings.stt_model}")
                            ASR = get_stt_model(settings.stt_model)
                            print(f"[STT] ✅ Model loaded: {settings.stt_model}")
                        except Exception as e:
                            print(f"[STT] ❌ Failed to initialize '{settings.stt_model}': {e}")
                            ASR = None
                    
                    # Use pre-loaded TTS from startup (or load now if not done)
                    if TTS is None and settings.tts_model and settings.tts_model.lower() not in ["none", "disabled", ""]:
                        if settings.tts_model == "chatterbox":
                            print("[TTS] Using pre-loaded Chatterbox model")
                            TTS = "chatterbox_warmed"
                        else:
                            try:
                                print(f"[TTS] Loading model on-demand: {settings.tts_model}")
                                TTS = get_tts_model(settings.tts_model)
                                sess.tts_model = TTS
                                print(f"[TTS] ✅ Model loaded: {settings.tts_model}")
                            except Exception as e:
                                error_msg = f"Error al inicializar TTS {settings.tts_model}: {str(e)}"
                                print(f"[TTS] ❌ {error_msg}")
                                await ws.send_text(jdumps({
                                    "type": "error",
                                    "message": error_msg,
                                    "error_type": "tts_initialization_failed"
                                }))
                                TTS = None
                                sess.tts_model = None
                    
                    if ASR:
                        print(f"[Session] Using STT: {type(ASR).__name__ if hasattr(ASR, '__name__') else type(ASR).__class__.__name__}")
                    if TTS:
                        print(f"[Session] Using TTS: {TTS if isinstance(TTS, str) else type(TTS).__name__}")
                            
                    sess.language = requested_lang

                    sess.name = data.get("name", "hojin")

                    # Start STT workers ONLY if STT is enabled
                    if ASR is not None:
                        if sess.stt_task is None:
                            print(f"[Session] Starting STT worker (ASR={type(ASR).__name__})")
                            sess.stt_task = asyncio.create_task(stt_worker(sess, sess.stt_in_q, sess.stt_out_q))

                        if sess.stt_out_consumer_task is None:
                            print("[Session] Starting STT output consumer")
                            sess.stt_out_consumer_task = asyncio.create_task(stt_out_consumer(sess))
                        
                        print("[Session] ✅ STT workers initialized")
                    else:
                        print("[Session] ⚠️ STT workers skipped - STT not initialized")

                    # Start TTS workers ONLY if TTS is enabled AND successfully loaded
                    if sess.tts_model is not None and sess.tts_task is None:
                        try:
                            # Use chatter_streamer for chatterbox model, tts_streamer for others
                            if settings.tts_model == "chatterbox":
                                from tts.chatter_infer import chatter_streamer
                                sess.tts_task = asyncio.create_task(chatter_streamer(sess))
                            else:
                                sess.tts_task = asyncio.create_task(tts_streamer(sess))
                            sess.conversation_task = asyncio.create_task(conversation_worker(sess))
                            sess.is_use_filler = data.get("use_filler", False)

                            await ws.send_text(jdumps({"type": "scriptsession.started"}))
                            
                            # Small delay to ensure tts_task is waiting on queue
                            await asyncio.sleep(0.1)
                            print("[Session] TTS and conversation workers started, sending greeting...")
                        except Exception as e:
                            dprint("TTS connection error ", e)
                    elif sess.tts_model is None:
                        print("[Session] ⚠️ TTS workers skipped - TTS initialization failed")
                        print("[Session] ⛔ Session cannot continue without TTS")
                        # Don't send started event if TTS failed - session is not usable
                        await ws.send_text(jdumps({
                            "type": "error",
                            "message": "No se pudo inicializar TTS. La sesión no puede continuar.",
                            "error_type": "session_start_failed"
                        }))
                        continue
                    else:
                        await ws.send_text(jdumps({"type": "warn", "message": "already started"}))

                    # Send greeting ONLY if TTS is available
                    if sess.tts_model is not None:
                        await answer_greeting(sess)
                        print("[Session] Greeting sent to TTS queue")
                    else:
                        print("[Session] ⛔ Greeting skipped - TTS not available")

                elif t == "input_audio_buffer.append":
                    # print(f"[AUDIO] Received audio buffer message")
                    try:
                        aud = data.get("audio")
                        # print(f"[AUDIO] Audio data length: {len(aud) if aud else 0}")
                        if aud:
                            audio = process_data_to_audio(
                                aud, input_sample_rate=INPUT_SAMPLE_RATE, whisper_sr=WHISPER_SR
                            )
                            # print(f"[AUDIO] Processed audio shape: {audio.shape if audio is not None else 'None'}")
                            if audio is None:
                                dprint("[NO AUDIO]")
                                continue

                            vad_event = check_audio_state(audio)
                            
                            # Debug: mostrar nivel de audio y estado VAD
                            audio_level = get_volume(audio.astype(np.float32, copy=False))[1]
                            if vad_event or audio_level > 0.01:
                                print(f"[VAD] event={vad_event}, audio_level={audio_level:.4f}, state={sess.current_audio_state}")

                            if sess.current_audio_state != "start":
                                sess.pre_roll.append(audio)
                                if vad_event == "start":
                                    energy_ok = get_volume(
                                        np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                                    )[1] > 0.02
                                    if not energy_ok:
                                        continue

                                    sess.current_audio_state = "start"
                                    cancel_silence_nudge_safe(sess)
                                    await interrupt_output(sess, reason="start speaking")
                                    if len(sess.pre_roll) > 0:
                                        sess.audios = (np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))
                                    else:
                                        sess.audios = audio.astype(np.float32, copy=False)

                                    print("[Voice Start]")
                                    sess.pre_roll.clear()
                                    sess.buf_count = 0
                                continue

                            sess.audios = np.concatenate([sess.audios, audio])
                            sess.buf_count += 1

                            if vad_event == "end" and sess.transcript != "": # immediately stop
                                # check by using smart-turn-detection-v3 from pipecat
                                print("[Voice End] - ", sess.transcript)
                                audio = sess.audios[:WHISPER_SR * 12]
                                turn_result = predict_endpoint(audio)
                                
                                print("Turn result : ", turn_result, "\n")
                                if turn_result['probability'] < 0.01:
                                    score = turn_result['probability']
                                    delay = 4.0 * (1 - score)
                                    # We can ignore when vad evenet is end but smart turn detection is not.
                                    # But what if it is wrong?
                                    # Set timer and trigger turn end after 1 second.
                                    if getattr(sess, "pending_turn_task", None):
                                        sess.pending_turn_task.cancel()
                                        sess.pending_turn_task = None
                            
                                    # 2) 새로 1초짜리 타이머 걸기
                                    sess.pending_turn_task = asyncio.create_task(
                                        delayed_force_turn_end(sess, delay=delay, script=sess.transcript.strip())
                                    )
                                    continue
                                
                                # Esperar a que se procesen todos los chunks pendientes en la cola STT
                                pending_items = sess.stt_in_q.qsize()
                                if pending_items > 0:
                                    print(f"[Voice End] Waiting for {pending_items} pending STT chunks to process...")
                                    await sess.stt_in_q.join()  # Wait for all items to be processed
                                    print(f"[Voice End] All STT chunks processed")
                                
                                sess.current_audio_state = "none"
                                await sess.out_q.put(
                                    jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True})
                                )
                                
                                sess.audios = np.empty(0, dtype=np.float32)
                                sess.end_scripting_time = time.time() % 1000

                                sess.answer_q.put_nowait(sess.transcript)
                                sess.transcript = ""
                                continue

                            if sess.buf_count % 8 == 7 and sess.current_audio_state == "start":
                                sess.audios = sess.audios[-WHISPER_SR * 20 :]
                                pcm_bytes = (
                                    np.clip(sess.audios, -1.0, 1.0) * 32767.0
                                ).astype(np.int16).tobytes()

                                print(f"[STT] Sending audio to queue: {len(pcm_bytes)} bytes")
                                # Non-blocking queue push (drop oldest on overflow)
                                try:
                                    sess.stt_in_q.put_nowait(pcm_bytes)
                                    print(f"[STT] Audio queued successfully")
                                except asyncio.QueueFull:
                                    print("[STT] Queue full, dropping oldest")
                                    try:
                                        _ = sess.stt_in_q.get_nowait()
                                        sess.stt_in_q.task_done()
                                    except asyncio.QueueEmpty:
                                        pass
                                    try:
                                        sess.stt_in_q.put_nowait(pcm_bytes)
                                    except asyncio.QueueFull:
                                        pass
                                sess.buf_count = 0

                    except Exception as e:
                        dprint("Error : ", e)

                elif t == "input_audio_buffer.commit":
                    lprint("input_audio_buffer.commit - ", sess.transcript)
                    if sess.transcript is not None and sess.transcript != "":
                        await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript, "is_final": True}))

                    if sess.transcript is not None and sess.transcript != "":
                        sess.answer_q.put_nowait(sess.transcript)
                        sess.transcript = ""

                    sess.current_audio_state = "none"
                    sess.audios = np.empty(0, dtype=np.float32)

                elif t == "session.close":
                    await ws.send_text(
                        jdumps(
                            {
                                "type": "session.close",
                                "payload": {"status": "closed successfully"},
                                "connected_time": time.time() - sess.connection_start_time,
                                "llm_cached_token_count": sess.llm_cached_token_count,
                                "llm_input_token_count": sess.llm_input_token_count,
                                "llm_output_token_count": sess.llm_output_token_count,
                            }
                        )
                    )
                    break

                else:
                    # other message types
                    pass

            elif msg.get("bytes") is not None:
                buf: bytes = msg["bytes"]

                frm = parse_frame(buf)
                if not frm:
                    await ws.send_text(jdumps({"type": "binary_ack", "payload": {"received_bytes": len(buf)}}))
                    continue
            
                if frm["is_config"]:
                    continue

                dec = ensure_opus_decoder(sess, sr=INPUT_SAMPLE_RATE, ch=1)
                try:
                    audio = decode_opus_float(frm["payload"], dec, sr=INPUT_SAMPLE_RATE)  # np.float32, mono
                except Exception as e:
                    dprint("[Opus decode error]", e)
                    continue

                if len(sess.bufs) == 0:
                    sess.bufs.append(audio)
                    continue
                else:
                    sess.bufs.append(audio)
                    audio = np.concatenate(sess.bufs, axis=0).astype(np.float32)
                    sess.bufs = []

                vad_event = check_audio_state(audio)

                if sess.current_audio_state != "start":
                    sess.pre_roll.append(audio)
                    if vad_event == "start":
                        energy_enough = get_volume(np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False))[1] > 0.02
                        if not energy_enough:
                            continue

                        sess.current_audio_state = "start"
                        cancel_silence_nudge_safe(sess)
                        await interrupt_output(sess, reason="start speaking")

                        if len(sess.pre_roll) > 0:
                            sess.audios = np.concatenate(list(sess.pre_roll) + [audio]).astype(np.float32, copy=False)
                        else:
                            sess.audios = audio.astype(np.float32, copy=False)
                        print("[Voice Start]")
                        sess.pre_roll.clear()
                        sess.buf_count = 0
                    continue

                sess.audios = np.concatenate([sess.audios, audio]).astype(np.float32, copy=False)
                sess.buf_count += 1

                if vad_event == "end" and sess.transcript != "":
                    print("[Voice End] - ", sess.transcript)
                    await sess.out_q.put(jdumps({"type": "transcript", "text": sess.transcript.strip(), "is_final": True}))
                    sess.current_audio_state = "none"
                    sess.audios = np.empty(0, dtype=np.float32)
                    sess.end_scripting_time = time.time() % 1000
                    sess.answer_q.put_nowait(sess.transcript) # answer_q will be used as input for llm response
                    sess.transcript = ""
                    continue

                if sess.buf_count % 5 == 4 and sess.current_audio_state == "start":
                    sess.audios = sess.audios[-WHISPER_SR * 20:]
                    pcm_bytes = (np.clip(sess.audios, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    try:
                        sess.stt_in_q.put_nowait(pcm_bytes)
                    except asyncio.QueueFull:
                        _ = sess.stt_in_q.get_nowait()
                        sess.stt_in_q.task_done()
                        sess.stt_in_q.put_nowait(pcm_bytes)
                    sess.buf_count = 0
                
    except WebSocketDisconnect:
        pass
    finally:
        await teardown_session(sess)
        sessions.pop(id(ws), None)


async def stt_out_consumer(sess: Session):
    while sess.running:
        msg = await sess.stt_out_q.get()
        try:
            new_text = (msg or {}).get("text", "") or ""
            if sess.current_audio_state != "none":
                sess.transcript = text_pr(sess.transcript, new_text)
                await sess.out_q.put(jdumps({"type": "delta", "text": sess.transcript, "is_final": False}))
        finally:
            sess.stt_out_q.task_done()


def _trim_last_one_words(s: str) -> str:
    words = re.findall(r"\S+", s)
    if len(words) <= 1:
        return ""
    return " ".join(words[:-1])

async def _transcribe_tts_buffer(sess: Session) -> str:
    buf = getattr(sess, "tts_pcm_buffer", np.empty(0, dtype=np.float32))
    sr  = getattr(sess, "tts_buffer_sr", 24000)

    print("buf size : ", buf.size)
    if buf.size < 4800:
        return ""
    # float32 [-1,1] → int16 bytes
    pcm_i16 = np.clip(buf, -1.0, 1.0)
    pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16, copy=False).tobytes()
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: ASR.transcribe_pcm(pcm_i16, sr, 1, language=sess.language)
        )
        return text or ""
    except Exception as e:
        print("[interrupt_output] ASR error:", e)
        return ""
    finally:
        # 사용 후 즉시 비움
        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)


async def interrupt_output(sess: Session, reason: str = "start speaking"):
    print(f"[interrupt_output] 🛑 Interrupting playback - reason: {reason}")
    st = time.time()

    now = time.time()
    if now - getattr(sess, "last_interrupt_ts", 0) < 1.0:
        print("[interrupt_output] ⚠️ Skipping - already interrupted within 1s")
        return
    sess.last_interrupt_ts = now

    try:
        sess.out_q.put_nowait(jdumps({"type": "tts_stop", "reason": reason}))
        print(f"[interrupt_output] ✅ Sent tts_stop message to frontend")
    except Exception as e:
        print(f"[interrupt_output] ❌ Failed to send tts_stop: {e}")
        pass

    try:
        sess.tts_stop_event.set()
        print(f"[interrupt_output] ✅ Set tts_stop_event")
    except Exception as e:
        print(f"[interrupt_output] ❌ Failed to set tts_stop_event: {e}")
        pass

    for task_name in ("tts_task", "conversation_task", "silence_nudge_task"):
        task = getattr(sess, task_name, None)
        if task and not task.done():
            task.cancel()

            try:
                await asyncio.wait_for(task, timeout=0.2)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                setattr(sess, task_name, None)

    # Drain pending queues
    def drain_aio_queue(q: asyncio.Queue):
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except asyncio.QueueEmpty:
            pass

    for q in (sess.tts_in_q,):
        drain_aio_queue(q)

    sess.tts_stop_event = threading.Event()

    if sess.running:
        if sess.tts_task is None:
            sess.tts_task = asyncio.create_task(tts_streamer(sess))
        if sess.conversation_task is None:
            sess.conversation_task = asyncio.create_task(conversation_worker(sess))

    try:
        partial_text = await _transcribe_tts_buffer(sess)
        print("[interrupt_output] partial_text: ", partial_text)
        if partial_text != "":
            # trimmed = _trim_last_one_words(partial_text.strip())
            trimmed = partial_text.strip()
            print("[interrupt_output] trimmed: ", trimmed)
            try:
                sess.outputs[-1] = trimmed
                sess.out_q.put_nowait(jdumps({"type": "interrupt_output", "text": trimmed}))
            except Exception:
                pass
    except Exception as e:
        print("Error ", e)
    
    print("[interrupt_output] took ", time.time() - st)

async def delayed_force_turn_end(sess: Session, delay: float = 1.0, script: str = ''):
    try:
        await asyncio.sleep(delay)
        print("delayed_force_turn_end : ", sess.current_audio_state, script, delay, "\n\n")
        if sess.current_audio_state == "start":
            return

        if sess.transcript.strip():
            await sess.out_q.put(
                jdumps({"type": "transcript", "text": script, "is_final": True})
            )
            sess.answer_q.put_nowait(script)
            sess.transcript = ""

        sess.current_audio_state = "none"
        sess.audios = np.empty(0, dtype=np.float32)
        sess.end_scripting_time = time.time() % 1000
    finally:
        sess.pending_turn_task = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
