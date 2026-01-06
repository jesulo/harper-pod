import React, { useCallback, useEffect, useRef, useState } from "react";
import "../globals.css";
import "../styles/radix.css";
import {
  arrayBufferToPCM16LE24kBase64,
  floatTo16BitPCM,
  int16ToBase64,
  resampleLinearMono,
} from "@/components/VoiceConditionModal";
import { LANGUAGE_CODE, StreamingAudioPlayer, TARGET_SR } from "@/utils/audio";
import { Mic, MicOff, User2Icon } from "lucide-react";
import Image from "next/image";
import { MicPulseRings } from "@/components/audio/MicPulseRing";
import { playAudioOnce } from "@/components/audio/play";
import NameText from "@/components/input/NameText";
import ConversationHistory from "@/components/ConversationHistory";
import { OpusWebCodecsPlayer } from "@/components/OpusWebCodecsPlayer";
import LockScreenCall from "@/components/lockscreen";

export const AGENT_NAME = "Harper";

export default function Jennifer() {
  const [recording, setRecording] = useState(false);
  const [connected, setConnected] = useState(false);
  const [languageCode, setLanguageCode] = useState<LANGUAGE_CODE>(
    LANGUAGE_CODE.English
  );
  const [socketUrl, setSocketUrl] = useState("ws://192.168.100.198:5000/ws");
  const [isSessionLoading, setIsSessionLoading] = useState(false);
  const opusPlayerRef = useRef<OpusWebCodecsPlayer | null>(null);

  const [scripting, setScripting] = useState("");
  const [userName, setUserName] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const [responses, setResponses] = useState<
    {
      type: "user" | "Jennifer";
      text: string;
    }[]
  >([]);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<StreamingAudioPlayer | null>(null);
  const isAudioStoppedRef = useRef(false);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const procNodeRef = useRef<ScriptProcessorNode | null>(null);
  const rafLockRef = useRef(false);

  const handleUpload = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!file) return;
    setLoading(true);
    const arrayBuffer = await file.arrayBuffer();
    const b64 = await arrayBufferToPCM16LE24kBase64(arrayBuffer);

    const msg = {
      type: "scriptsession.clonevoice",
      voice: b64,
    };
    wsRef.current.send(JSON.stringify(msg));
    setLoading(false);
  };

  useEffect(() => {
    const localLanguageCode = localStorage.getItem("languageCode");
    if (localLanguageCode) {
      setLanguageCode(localLanguageCode as LANGUAGE_CODE);
    }

    const localSocketUrl = localStorage.getItem("socketUrl");
    if (localSocketUrl) {
      setSocketUrl(localSocketUrl);
    }

    const localuserName = localStorage.getItem("userName");
    if (localuserName) {
      setUserName(localuserName);
    }
  }, []);

  useEffect(() => {
    playerRef.current = new StreamingAudioPlayer();
    return () => {
      playerRef.current?.dispose();
      playerRef.current = null;
    };
  }, []);

  const lastTtsAtRef = useRef(0);

  useEffect(() => {
    let rafId = 0;
    const HOLD_MS = 1000;

    const tick = () => {
      const now = performance.now();
      const active = now - lastTtsAtRef.current < HOLD_MS;
      setIsSpeaking((prev) => (prev !== active ? active : prev));
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const openSocket = useCallback(async () => {
    const url = socketUrl;
    console.log(`[FRONTEND] Attempting to connect to: ${url}`);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`[FRONTEND] WebSocket connected successfully`);
      ws.binaryType = "arraybuffer";

      const now = new Date();
      const formatted = now.toLocaleString("en-US", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      });

      const openMsg = {
        type: "scriptsession.start",
        language: languageCode,
        use_filler: false,
        time: formatted,
        name: userName,
      };
      console.log(`[FRONTEND] Sending start message:`, openMsg);
      ws.send(JSON.stringify(openMsg));
      setIsSessionLoading(true);
    };

    ws.onerror = (err) => {
      console.error(`[FRONTEND] WebSocket error:`, err);
    };

    ws.onclose = (event) => {
      console.log(`[FRONTEND] WebSocket closed:`, event.code, event.reason);
    };

    ws.onmessage = async (ev) => {
      // binary부터 처리
      if (ev.data instanceof ArrayBuffer) {
        if (!ev.data || (ev.data as ArrayBuffer).byteLength === 0) {
          console.warn("[OpusDecoder] skip empty chunk");
          return;
        }
        
        // If audio was stopped (user interrupted), ignore incoming frames
        if (isAudioStoppedRef.current) {
          console.log("[OpusDecoder] ⚠️ Ignoring frame - audio was stopped");
          return;
        }
        
        const buf: ArrayBuffer = ev.data;
        const view = new DataView(buf);
        // MAGIC check: 0xA1 0x51
        if (view.getUint8(0) === 0xa1 && view.getUint8(1) === 0x51) {
          const flags = view.getUint8(2);
          const isFinal = (flags & 1) !== 0;
          const seq = view.getUint32(4, true);
          const pktLen = view.getUint32(16, true);
          const payload = new Uint8Array(buf, 20, pktLen);
          
          // console.log(`[OpusDecoder] 📦 Binary frame received - seq: ${seq}, pktLen: ${pktLen}, payload.length: ${payload.length}, isFinal: ${isFinal}, bufferSize: ${buf.byteLength}`);

          lastTtsAtRef.current = performance.now();

          if (!opusPlayerRef.current) {
            // console.log("[OpusDecoder] 🎵 Creating new OpusWebCodecsPlayer");
            opusPlayerRef.current = new OpusWebCodecsPlayer();
            opusPlayerRef.current.configure();
          }

          if (!payload || payload.length === 0) {
            // console.log(`[OpusDecoder] ⚠️ Empty payload - isFinal: ${isFinal}, only flushing (keeping player alive for next audio)`);
            opusPlayerRef.current.flush().catch(() => {});
            // Don't destroy the player - keep it for the next audio stream (e.g., nudge)
            // opusPlayerRef.current = null;
            isAudioStoppedRef.current = false; // Reset flag
            return;
          }
          // console.log(`[OpusDecoder] ✅ Decoding frame ${seq} with ${payload.length} bytes`);
          opusPlayerRef.current.decodeFrame(payload, seq);

          if (isFinal) {
            // console.log(`[OpusDecoder] 🏁 Final frame received (seq: ${seq}), flushing player`);
            opusPlayerRef.current.flush().catch(() => {});
            isAudioStoppedRef.current = false; // Reset flag when done
          }
          return;
        }

        return;
      }

      const msg = JSON.parse(ev.data);
      const msgType = msg.type as string;

      if (msgType === "scriptsession.started") {
        // small wait to ensure session started; or rely on user to press Mute to start — here we attempt immediately
        await startMic();
        setRecording(true);

        setConnected(true);
        setIsSessionLoading(false);
        await playAudioOnce("/audios/alarm.mp3", { volume: 0.8 });

        ws.send(JSON.stringify({ type: "scriptsession.greeting" }));
      }

      console.log("msgType", msgType, !opusPlayerRef.current);
      if (msgType === "tts_audio_meta" && msg.format === "opus") {
        console.log("[FRONTEND] 🎵 tts_audio_meta received - creating fresh player");
        // New audio stream starting - reset the stop flag
        isAudioStoppedRef.current = false;
        // Always close and recreate the player for a clean state
        if (opusPlayerRef.current) {
          try {
            opusPlayerRef.current.close();
          } catch (e) {
            console.warn("[FRONTEND] Error closing old player:", e);
          }
        }
        opusPlayerRef.current = new OpusWebCodecsPlayer();
        opusPlayerRef.current.configure();
        return;
      }

      if (msgType === "delta") {
        setScripting(msg.text ?? "");
      } else if (msgType === "transcript") {
        setScripting("");
        setResponses((ts) => [
          ...ts,
          {
            type: "user",
            text: msg.text ?? "",
          },
        ]);
      } else if (msgType === "speaking") {
        if (!msg.is_final) return;
        const clean = String(msg.text ?? "").trim();
        if (!clean) return;

        // Reset audio stop flag when new speaking message arrives (for proactive messages)
        console.log("[FRONTEND] 📢 Received speaking message, ensuring audio is ready");
        isAudioStoppedRef.current = false;

        setResponses((ts) => [
          ...ts,
          {
            type: AGENT_NAME,
            text: clean,
          },
        ]);
      } else if (msgType === "tts_stop") {
        // Backend requested to stop audio playback (user interrupted)
        console.log("[FRONTEND] 🛑 Received tts_stop, clearing audio playback");
        isAudioStoppedRef.current = true; // Set flag to ignore incoming frames
        if (opusPlayerRef.current) {
          opusPlayerRef.current.close();
          opusPlayerRef.current = null;
        }
      } else if (msgType === "session.close") {
        setConnected(false);
      } else if (msgType === "interrupt_output") {
        setResponses((ts) => [
          ...ts.slice(0, -1),
          {
            type: AGENT_NAME,
            text: msg.text,
          },
        ]);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };

    ws.onerror = (err) => {
      console.error("Socket error:", err);
      alert(
        "Server error! It looks like the temporary GPU server may be down. Please let us know."
      );
    };
  }, [languageCode, socketUrl]);

  const sendAudioCommit = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }, []);

  const stopMic = useCallback(() => {
    sendAudioCommit();
    procNodeRef.current?.disconnect();
    audioCtxRef.current?.close();
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    procNodeRef.current = null;
    audioCtxRef.current = null;
    mediaStreamRef.current = null;
  }, [sendAudioCommit]);

  const startMic = useCallback(async () => {
    console.log(`[FRONTEND] Starting microphone...`);
    const ws = wsRef.current;
    console.log(`[FRONTEND] WebSocket state when starting mic: ${ws?.readyState || 'null'}`);
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        sampleSize: 16,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      },
      video: false,
    });
    console.log(`[FRONTEND] Microphone stream obtained`);
    const audioCtx = new (window.AudioContext ||
      (window as any).webkitAudioContext)({
      sampleRate: 48000,
      latencyHint: "interactive",
    });
    const src = audioCtx.createMediaStreamSource(stream);

    const hpf = audioCtx.createBiquadFilter();
    hpf.type = "highpass";
    hpf.frequency.value = 130;

    const comp = audioCtx.createDynamicsCompressor();
    comp.threshold.value = -24;
    comp.knee.value = 20;
    comp.ratio.value = 3;
    comp.attack.value = 0.003;
    comp.release.value = 0.25;

    const proc = audioCtx.createScriptProcessor(4096, 1, 1);

    const inGain = audioCtx.createGain();
    inGain.gain.value = 0.6; // 0.4~0.8 사이에서 테스트

    src
      .connect(inGain)
      .connect(hpf)
      .connect(comp)
      .connect(proc)
      .connect(audioCtx.destination);

    proc.onaudioprocess = (ev) => {
      const inRate = audioCtx.sampleRate;
      const input = ev.inputBuffer.getChannelData(0);

      // scale down a bit
      const scaled = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) scaled[i] = input[i] * 0.95;

      const mono24k = resampleLinearMono(scaled, inRate, TARGET_SR);
      const pcm16 = floatTo16BitPCM(mono24k);
      const audioB64 = int16ToBase64(pcm16);

      if (ws?.readyState === WebSocket.OPEN) {
        // console.log(`[FRONTEND] Sending audio buffer, length: ${audioB64.length}, readyState: ${ws.readyState}`);
        ws.send(
          JSON.stringify({
            type: "input_audio_buffer.append",
            audio: audioB64,
            t0: Date.now(),
          })
        );
        // console.log(`[FRONTEND] Audio buffer sent successfully`);
      } else {
        // console.log(`[FRONTEND] WebSocket not ready, readyState: ${ws?.readyState || 'null'}`);
      }

      // voice-end detection for latency metric
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);

      // 간단한 게이트/정규화 (필요시 조정). 애니메이션을 위해 추가 animation
      const level = Math.min(1, Math.max(0, (rms - 0.005) * (1 / 0.03))); // ~0~1
      // EMA + rAF로 너무 자주 setState 안 하게
      if (!rafLockRef.current) {
        rafLockRef.current = true;
        requestAnimationFrame(() => {
          setMicLevel((prev) => prev * 0.8 + level * 0.2);
          rafLockRef.current = false;
        });
      }
    };

    mediaStreamRef.current = stream;
    audioCtxRef.current = audioCtx;
    procNodeRef.current = proc;
  }, []);

  // ============ High-level controls (renamed to Start / Mute / End) ============
  const startSession = useCallback(async () => {
    if (connected || isSessionLoading) return;
    await openSocket(); // connect first
  }, [connected, isSessionLoading, openSocket]);

  const endSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "session.close" }));
      } catch {}
      ws.close();
    }
    wsRef.current = null;
    setIsSessionLoading(false);
    setConnected(false);

    // turn mic off, too
    if (recording) {
      stopMic();
      setRecording(false);
    }
  }, [recording, stopMic]);

  const onStart = useCallback(async () => {
    if (!connected) await startSession();
    setConnected(true);
    setIsSessionLoading(false);
    setIsSpeaking(true);
  }, [connected, startSession]);

  const onToggleMute = useCallback(async () => {
    if (recording) {
      stopMic();
      setRecording(false);
    } else {
      await startMic();
      setRecording(true);
    }
  }, [recording, startMic, stopMic]);

  const sendUserName = useCallback(() => {
    localStorage.setItem("userName", userName);

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "scriptsession.setname",
          name: userName,
        })
      );
      alert("Set!");
    } else {
      alert("Failed to send prompt. 연결부터 해주세요");
    }
  }, [userName]);

  const sendSocketUrl = useCallback(() => {
    localStorage.setItem("socketUrl", socketUrl);
    console.log("socketUrl", socketUrl);
    alert("Set!");
  }, [socketUrl]);

  const [callSec, setCallSec] = useState(0);

  const minutes = String(Math.floor(callSec / 60)).padStart(2, "0");
  const seconds = String(callSec % 60).padStart(2, "0");

  useEffect(() => {
    let t: any;
    if (connected) {
      t = setInterval(() => setCallSec((s) => s + 1), 1000);
    } else {
      setCallSec(0);
    }
    return () => t && clearInterval(t);
  }, [connected]);

  return (
    <div className="min-h-screen w-full bg-[#F9FAF7] text-[#2D2F26]">
      <header className="flex items-start justify-between px-4 py-4">
        <div className="flex items-center gap-2">
          <Image src="/images/logo.ico" alt="harper" width={32} height={32} />
        </div>
        <div className="text-center">
          <div className="px-6 pt-8 pb-4 flex flex-col items-center select-none">
            <div className="text-black text-2xl tracking-wide">
              {AGENT_NAME} {minutes}:{seconds}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 bg-[#F7F9F2] rounded-xl p-2 border border-[#E9EDD9]">
              <label className="text-xs text-[#7E8766]">Input Language</label>
              <select
                className="bg-transparent outline-none ml-auto text-sm"
                value={languageCode}
                onChange={(e) => {
                  localStorage.setItem(
                    "languageCode",
                    e.target.value as LANGUAGE_CODE
                  );
                  setLanguageCode(e.target.value as LANGUAGE_CODE);
                }}
              >
                {Object.entries(LANGUAGE_CODE).map(([k, v]) => (
                  <option
                    className="bg-neutral-900 text-white"
                    key={k}
                    value={v}
                  >
                    {k}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
        <div className="w-6 h-6 rounded-full border border-[#6C7F3D] flex items-center justify-center text-xs">
          <User2Icon color="#6C7F3D" />
        </div>
      </header>

      <LockScreenCall
        connected={connected}
        isSessionLoading={isSessionLoading}
        micLevel={micLevel}
        isSpeaking={isSpeaking}
        recording={recording}
        onAnswer={onStart}
        onEnd={endSession}
        onMute={onToggleMute}
      />

      {scripting && (
        <div className="px-4 py-4 border-b border-[#EEF2E0]">
          <div className="text-[13px] text-[#7E8766] mb-1">Speaking</div>
          <div className="text-[15px] whitespace-pre-wrap break-words">
            {scripting}
          </div>
        </div>
      )}

      {/* Rolling history */}
      <br />
      <br />
      <br />
      <br />
      <br />
      <ConversationHistory responses={responses} scripting={scripting} />
      <div className="w-full max-w-xl justify-center items-center mx-auto mt-10">
        <NameText
          userName={userName}
          setUserName={(v) => setUserName(v)}
          sendUserName={sendUserName}
          text="Set My Name"
        />
        <NameText
          userName={socketUrl}
          setUserName={(v) => setSocketUrl(v)}
          sendUserName={sendSocketUrl}
          text="Set Web Socket URL"
        />
      </div>

      <div className="w-full max-w-xl justify-center items-center mx-auto mt-10 rounded-2xl bg-white shadow-md border border-zinc-100 p-6">
        <label className="mt-4 block cursor-pointer border-2 border-dashed border-zinc-300 hover:border-zinc-400 transition rounded-xl p-6 text-center">
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div className="text-zinc-600">
            {file ? (
              <>
                <div className="text-sm font-medium">{file.name}</div>
                <div className="text-xs text-zinc-500">
                  {Math.round((file.size / 1024 / 1024) * 100) / 100} MB
                </div>
              </>
            ) : (
              <>
                <div className="text-sm">Click to choose an audio file</div>
                <div className="text-xs text-zinc-500">
                  WAV/MP3/M4A/FLAC/OGG…
                </div>
              </>
            )}
          </div>
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="mt-4 w-full rounded-md bg-[#6d28d9] hover:opacity-90 disabled:opacity-50 text-white py-2.5 font-medium text-center"
        >
          {loading ? "Uploading…" : "Upload"}
        </button>
      </div>
      <br />
      <br />
      <br />
    </div>
  );
}
