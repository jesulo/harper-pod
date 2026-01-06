// utils/audio/OpusWebCodecsPlayer.ts
export class OpusWebCodecsPlayer {
  private audioCtx: AudioContext;
  private decoder: AudioDecoder | null = null;
  private playHead = 0;
  private primed = false;
  private readonly SR = 24000;
  private readonly CH = 1;
  private readonly SAFETY_BUFFER_SEC = 0.12;
  private readonly PREBUFFER_SEC = 0.25;
  private readonly FRAME_SEC = 0.02;
  private lastSeq = -1;
  private reorderWindow = new Map<number, Uint8Array>();
  private expecting = 0;
  private readonly REORDER_MAX = 8;
  private activeSources: Set<AudioBufferSourceNode> = new Set();

  constructor(audioCtx?: AudioContext) {
    this.audioCtx =
      audioCtx ??
      new (window.AudioContext || (window as any).webkitAudioContext)();
    if (!("AudioDecoder" in window)) throw new Error("WebCodecs not supported");
  }

  configure() {
    console.log("[OpusWebCodecsPlayer] 🔧 Configuring decoder...");
    try {
      this.decoder?.close();
    } catch {}
    this.decoder = new (window as any).AudioDecoder({
      output: (audioData: AudioData) => this.onDecoded(audioData),
      error: (e: any) => console.error("[OpusDecoder] error", e),
    });
    const opusHeader = this.makeOpusHead(this.CH, 0, this.SR, 0, 0);
    this.decoder?.configure({
      codec: "opus",
      sampleRate: this.SR,
      numberOfChannels: this.CH,
      description: opusHeader.buffer,
    });
    if (this.audioCtx.state === "suspended") {
      console.log("[OpusWebCodecsPlayer] ⏸️ AudioContext suspended, resuming...");
      this.audioCtx.resume().catch(() => {});
    }
    console.log(`[OpusWebCodecsPlayer] ✅ Configured - AudioContext state: ${this.audioCtx.state}, currentTime: ${this.audioCtx.currentTime.toFixed(3)}`);
    this.primed = false;
    this.playHead = Math.max(this.audioCtx.currentTime + this.PREBUFFER_SEC, 0);
    this.lastSeq = -1;
    this.expecting = -1; // Accept any first frame
    this.reorderWindow.clear();
    // console.log("[OpusWebCodecsPlayer] 🔄 Reset sequencing - expecting: -1, reorderWindow cleared");
  }

  private onDecoded(audioData: AudioData) {
    // console.log(`[OpusWebCodecsPlayer] 🎵 onDecoded called - frames: ${audioData.numberOfFrames}, channels: ${audioData.numberOfChannels}, sampleRate: ${audioData.sampleRate}`);
    const ab = this.audioCtx.createBuffer(
      audioData.numberOfChannels,
      audioData.numberOfFrames,
      audioData.sampleRate
    );
    for (let ch = 0; ch < audioData.numberOfChannels; ch++) {
      const tmp = new Float32Array(audioData.numberOfFrames);
      audioData.copyTo(tmp, { planeIndex: ch });
      ab.getChannelData(ch).set(tmp);
    }
    audioData.close();
    if (!this.primed) {
      this.playHead = Math.max(
        this.audioCtx.currentTime + this.PREBUFFER_SEC,
        this.playHead
      );
      this.primed = true;
      // console.log(`[OpusWebCodecsPlayer] ✨ Primed - playHead: ${this.playHead.toFixed(3)}s`);
    }
    const src = this.audioCtx.createBufferSource();
    src.buffer = ab;
    src.connect(this.audioCtx.destination);
    const when = Math.max(
      this.playHead,
      this.audioCtx.currentTime + this.SAFETY_BUFFER_SEC
    );
    
    // Track active sources
    this.activeSources.add(src);
    
    // console.log(`[OpusWebCodecsPlayer] 🔊 Starting playback - when: ${when.toFixed(3)}s, duration: ${ab.duration.toFixed(3)}s, currentTime: ${this.audioCtx.currentTime.toFixed(3)}s`);
    src.start(when);
    this.playHead = when + ab.duration;
    src.onended = () => {
      src.disconnect();
      this.activeSources.delete(src);
    };
  }

  decodeFrame(payload: Uint8Array, seq: number, timestampUSec?: number) {
    if (!this.decoder) return;

    // First frame after configure? Accept it and start sequencing
    if (this.expecting === -1) {
      console.log(`[OpusWebCodecsPlayer] 🎯 First frame - seq: ${seq}, starting sequencing from here`);
      this.expecting = seq;
    }

    if (seq < this.expecting) {
      // console.log(`[OpusWebCodecsPlayer] ⏮️ Skipping old frame - seq: ${seq}, expecting: ${this.expecting}`);
      return;
    }

    if (seq === this.expecting) {
      this.doDecode(payload, seq, timestampUSec);
      this.expecting++;
      this.flushReorderWindowInOrder();
      return;
    }

    // Future frame - add to reorder window
    // console.log(`[OpusWebCodecsPlayer] 🔮 Future frame - seq: ${seq}, expecting: ${this.expecting}, adding to reorder window`);
    this.reorderWindow.set(seq, payload);

    if (this.reorderWindow.size > this.REORDER_MAX) {
      const keys = Array.from(this.reorderWindow.keys()).sort((a, b) => a - b);
      while (this.reorderWindow.size > this.REORDER_MAX) {
        this.reorderWindow.delete(keys.shift()!);
      }
    }
  }

  private flushReorderWindowInOrder() {
    while (this.reorderWindow.has(this.expecting)) {
      const p = this.reorderWindow.get(this.expecting)!;
      this.reorderWindow.delete(this.expecting);
      this.doDecode(p, this.expecting);
      this.expecting++;
    }
  }

  private doDecode(payload: Uint8Array, seq: number, timestampUSec?: number) {
    if (!this.decoder || this.decoder.state !== "configured") {
      console.warn(`[OpusWebCodecsPlayer] ⚠️ Decoder not ready - state: ${this.decoder?.state}, seq: ${seq}`);
      return;
    }
    // console.log(`[OpusWebCodecsPlayer] 📥 Decoding seq ${seq}, payload size: ${payload.length}`);
    const ts = Math.trunc(timestampUSec ?? seq * this.FRAME_SEC * 1_000_000);
    const chunk = new (window as any).EncodedAudioChunk({
      type: "key",
      timestamp: ts,
      data: payload,
    });
    this.decoder!.decode(chunk);
  }

  async flush() {
    await this.decoder
      ?.flush()
      .catch((e) => console.error("[OpusDecoder] flush error", e));
    this.primed = false;
    this.playHead = Math.max(this.audioCtx.currentTime + this.PREBUFFER_SEC, 0);
    this.lastSeq = -1;
    this.reorderWindow.clear();
  }

  close() {
    console.log("[OpusWebCodecsPlayer] 🛑 Stopping playback - active sources:", this.activeSources.size);
    
    // Stop all active audio sources immediately
    this.activeSources.forEach(src => {
      try {
        src.stop();
        src.disconnect();
      } catch (e) {
        // Source might have already ended
      }
    });
    this.activeSources.clear();
    
    // Close the decoder
    try {
      this.decoder?.close();
    } catch {}
    this.decoder = null;
    this.primed = false;
    this.reorderWindow.clear();
    
    console.log("[OpusWebCodecsPlayer] ✅ All audio stopped");
  }

  get context() {
    return this.audioCtx;
  }

  private makeOpusHead(
    channels = 1,
    preSkip = 0,
    inputSampleRate = this.SR,
    gain = 0,
    channelMapping = 0
  ): Uint8Array {
    const magic = new TextEncoder().encode("OpusHead");
    const b = new Uint8Array(19);
    b.set(magic, 0);
    b[8] = 1;
    b[9] = channels;
    b[10] = preSkip & 0xff;
    b[11] = (preSkip >> 8) & 0xff;
    b[12] = inputSampleRate & 0xff;
    b[13] = (inputSampleRate >> 8) & 0xff;
    b[14] = (inputSampleRate >> 16) & 0xff;
    b[15] = (inputSampleRate >> 24) & 0xff;
    b[16] = gain & 0xff;
    b[17] = (gain >> 8) & 0xff;
    b[18] = channelMapping;
    return b;
  }
}
