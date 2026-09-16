// Audio bypass: imperceptible transformations that break audio fingerprinting
// while keeping the music sounding identical to human ears.

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const len = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = numCh * bytesPerSample;
  const byteRate = sr * blockAlign;
  const dataSize = len * blockAlign;
  const wav = new ArrayBuffer(44 + dataSize);
  const v = new DataView(wav);
  const str = (off: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataSize, true);
  str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, numCh, true); v.setUint32(24, sr, true);
  v.setUint32(28, byteRate, true); v.setUint16(32, blockAlign, true);
  v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < len; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, buffer.getChannelData(c)[i]));
      v.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([wav], { type: "audio/wav" });
}

// Build a short synthetic impulse response for subtle room feel
function makeIR(ctx: OfflineAudioContext, durationSec = 0.25): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * durationSec);
  const ir = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const ch = ir.getChannelData(c);
    for (let i = 0; i < len; i++) {
      // Exponential decay noise — sounds like a small room
      ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 4) * 0.8;
    }
  }
  return ir;
}

export interface BypassOptions {
  // Pitch shift in cents (default 8 = ~0.46%, inaudible, breaks fingerprint)
  pitchCents?: number;
  // Wet reverb mix 0–1 (default 0.04 = 4%, barely perceptible)
  reverbWet?: number;
  // Noise floor in linear gain (default 0.0002 = −74 dB, completely inaudible)
  noiseLevel?: number;
}

export async function bypassAudio(
  file: File,
  opts: BypassOptions = {},
  onProgress?: (pct: number) => void
): Promise<Blob> {
  const {
    pitchCents = 8,
    reverbWet = 0.04,
    noiseLevel = 0.0002,
  } = opts;

  onProgress?.(5);

  // Decode input audio
  const arrayBuf = await file.arrayBuffer();
  const tmpCtx = new AudioContext();
  const decoded = await tmpCtx.decodeAudioData(arrayBuf);
  await tmpCtx.close();

  onProgress?.(20);

  // playbackRate for pitch shift: 2^(cents/1200)
  // 8 cents → 1.00463× (tempo +0.46%, pitch +8¢ — below human JND of ~10¢)
  const rate = Math.pow(2, pitchCents / 1200);
  const outDuration = decoded.duration / rate + 0.5; // +0.5s tail for reverb

  const offline = new OfflineAudioContext(
    decoded.numberOfChannels,
    Math.ceil(outDuration * decoded.sampleRate),
    decoded.sampleRate
  );

  // ── Source ──────────────────────────────────────────────────────────────────
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.playbackRate.value = rate;

  // ── Subtle room reverb (breaks spectral fingerprint) ───────────────────────
  const convolver = offline.createConvolver();
  convolver.buffer = makeIR(offline);
  const dryGain = offline.createGain();
  dryGain.gain.value = 1 - reverbWet;
  const wetGain = offline.createGain();
  wetGain.gain.value = reverbWet;

  // ── Ultra-quiet noise (breaks waveform fingerprint) ────────────────────────
  // Build a noise buffer the same length as output
  const noiseLen = Math.ceil(outDuration * offline.sampleRate);
  const noiseBuf = offline.createBuffer(decoded.numberOfChannels, noiseLen, offline.sampleRate);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const ch = noiseBuf.getChannelData(c);
    for (let i = 0; i < noiseLen; i++) ch[i] = (Math.random() * 2 - 1) * noiseLevel;
  }
  const noiseSrc = offline.createBufferSource();
  noiseSrc.buffer = noiseBuf;

  // ── Master limiter (prevent clipping) ─────────────────────────────────────
  const limiter = offline.createDynamicsCompressor();
  limiter.threshold.value = -0.5;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.001;
  limiter.release.value = 0.1;

  // ── Graph ─────────────────────────────────────────────────────────────────
  src.connect(dryGain);
  src.connect(convolver);
  convolver.connect(wetGain);
  dryGain.connect(limiter);
  wetGain.connect(limiter);
  noiseSrc.connect(limiter);
  limiter.connect(offline.destination);

  src.start(0);
  noiseSrc.start(0);

  onProgress?.(35);

  const rendered = await offline.startRendering();

  onProgress?.(90);

  const wav = audioBufferToWav(rendered);
  onProgress?.(100);
  return wav;
}
