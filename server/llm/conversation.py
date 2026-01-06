import time
from app.session import Session
from utils.utils import dprint, lprint
import re 
import asyncio
from utils.constants import LLM_MODEL
if LLM_MODEL == "local":
    from llm.ollama import chat_reply, chat_greeting
else:
    from llm.openai import chat_reply, chat_greeting
import orjson as json

def jdumps(o): return json.dumps(o).decode()

SILENCE_PATTERN = re.compile(r"<\s*silence\s+(\d+(?:\.\d+)?)\s*>", re.IGNORECASE)

def split_by_silence_markers(text: str):
    """
    '<silence N>' 기준으로 텍스트/무음 명령을 순서대로 반환.
    반환 예) ["Hello.", ("__silence__", 3.0), "How are you?"]
    """
    print(f"[split_by_silence_markers] 📝 Input text: {repr(text)}")
    parts = []
    pos = 0
    for m in SILENCE_PATTERN.finditer(text):
        if m.start() > pos:
            seg = text[pos:m.start()].strip()
            if seg:
                parts.append(seg + "<cont>")
                print(f"[split_by_silence_markers]   - Text segment: {repr(seg)}")
        dur = float(m.group(1))
        parts.append(("__silence__", dur))
        print(f"[split_by_silence_markers]   - Silence: {dur}s")
        pos = m.end()
    # 꼬리 텍스트
    tail_seg = text[pos:].strip()
    if tail_seg:
        parts.append(tail_seg)
        print(f"[split_by_silence_markers]   - Tail segment: {repr(tail_seg)}")
    print(f"[split_by_silence_markers] ✅ Total parts: {len(parts)}")
    return parts

def reset_conversation(sess: Session):
    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)

async def conversation_worker(sess: Session):
    while sess.running:
        text = await sess.answer_q.get()
        await answer_one(sess, text)

async def answer_one(sess: Session, transcript: str):
    st = time.time()
    print(f"[Answer] 📥 Received transcript: '{transcript[:100]}...'")
    sess.current_transcript += " " + transcript
    print(f"[Answer] 🤖 Calling LLM...")
    answer_text = await run_answer_async(sess)  # 내부에서 run_in_executor 사용
    llm_duration = time.time() - st
    print(f"[Answer] ✅ LLM responded in {llm_duration:.2f}s - {answer_text[:100]!r}...")
    sess.answer = answer_text.strip()

    sess.transcripts.append(sess.current_transcript)
    sess.current_transcript = ""
    sess.outputs.append(sess.answer)

async def answer_greeting(sess: Session):
    loop = asyncio.get_running_loop()

    def run_blocking():
        return chat_greeting(
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time
        )

    print(f"[Greeting] Generating greeting for {sess.name} in {sess.language}")
    output = await loop.run_in_executor(None, run_blocking)
    answer_text = (output.get("text", "") or "").strip()
    sess.transcripts.append('[Call is started. User says nothing yet]')
    sess.outputs.append(answer_text)

    if answer_text:
        print(f"[Greeting] ✅ Generated greeting text: '{answer_text}'")
        print(f"[Greeting] 📤 Sending to TTS queue (tts_in_q)...")
        loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, answer_text)
        print(f"[Greeting] 📤 Sent to TTS queue successfully")
        loop.call_soon_threadsafe(
            sess.out_q.put_nowait,
            jdumps({
                "type": "speaking",
                "script": sess.current_transcript,
                "text": answer_text,
                "is_final": True
            })
        )
        print(f"[Greeting] Sent to TTS queue")
    else:
        print(f"[Greeting] WARNING: No greeting text generated")

async def run_answer_async(sess: Session) -> str:
    loop = asyncio.get_running_loop()
    sentence = ''
    sent_chars = 0  # answer_text 중 이미 보낸 문자 수

    def safe_push_tts(obj):
        loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, obj)

    def safe_push_out(msg: dict):
        loop.call_soon_threadsafe(sess.out_q.put_nowait, jdumps(msg))

    def push_pieces(pieces):
        # pieces: ["text", ("__silence__", 3.0), ...]
        push_time = time.time()
        print(f"[LLM->TTS] ⏰ [{push_time:.3f}] 📤 Pushing {len(pieces)} pieces to TTS queue")
        for i, p in enumerate(pieces):
            if isinstance(p, tuple):
                print(f"[LLM->TTS]   Piece {i}: silence {p[1]}s")
            else:
                print(f"[LLM->TTS]   Piece {i}: text '{p[:80]}...' ({len(p)} chars)")
            safe_push_tts(p)
        print(f"[LLM->TTS] ⏰ [{time.time():.3f}] ✅ All {len(pieces)} pieces sent to TTS queue (+{(time.time()-push_time)*1000:.1f}ms)")

    def flush_buffer_if_has_silence():
        nonlocal sentence, sent_chars
        if SILENCE_PATTERN.search(sentence):
            print(f"[LLM] Flushing buffer with silence markers: '{sentence[:100]}...'")
            pieces = split_by_silence_markers(sentence)
            push_pieces(pieces)
            sent_chars += len(sentence)
            sentence = ""

    def on_token(tok: str):
        nonlocal sentence, sent_chars
        # 토큰 누적
        sentence += tok
        safe_push_out({"type": "speaking", "text": tok, "is_final": False})
        # 버퍼 기준으로 <silence N> 완성 여부 확인 후 즉시 플러시
        flush_buffer_if_has_silence()
        return

    def run_blocking():
        return chat_reply(
            prev_scripts=sess.transcripts[-10:],
            prev_answers=sess.outputs[-10:],
            input_sentence=sess.current_transcript,
            language=sess.language,
            onToken=on_token,           # executor 스레드에서 호출될 가능성 높음
            name=sess.name,
            current_time=sess.current_time
        )

    # st = time.time()
    output = await loop.run_in_executor(None, run_blocking)
    answer_text = output.get("text", "") or ""
    print(f"[LLM] Complete answer: '{answer_text[:200]}...' ({len(answer_text)} chars)")
    # print(f"[{time.time() - st}s] - {answer_text}")
    
    tail = answer_text[sent_chars:]
    if tail:
        print(f"[LLM] Sending tail: '{tail[:100]}...' ({len(tail)} chars)")
        pieces = split_by_silence_markers(tail)
        push_pieces(pieces)
    
    await asyncio.sleep(0)
    cts = sess.current_transcript
    
    def _g():
        sess.out_q.put_nowait(jdumps({"type": "speaking", "script": cts, "text": answer_text, "is_final": True}))
    loop.call_soon_threadsafe(_g)
    
    return answer_text

