// Audio bypass: multi-layer fingerprint breaking
// Strategy: combine pitch shift + tempo shift + EQ + subtle saturation
// These together break audio fingerprinting while keeping music recognizable

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const len = buffer.length;
  const blockAlign = numCh * 2;
  const dataSize = len * blockAlign;
  const wav = new ArrayBuffer(44 + dataSize);
  const v = new DataView(wav);
  const str = (off: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataSize, true);
  str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, numCh, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * blockAlign, true); v.setUint16(32, blockAlign, true);
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

// Soft saturation — adds subtle harmonic distortion, breaks waveform fingerprint
function softClip(x: number, amount: number): number {
  // Cubic soft clip: adds odd harmonics
  const g = 1 + amount * 2;
  const y = x * g;
  if (y > 1) return 2 / 3;
  if (y < -1) return -2 / 3;
  return y - (y * y * y) / 3;
}

export type BypassStrength = "mild" | "medium" | "strong";

interface BypassProfile {
  pitchCents: number;       // pitch shift in cents (also shifts tempo via playbackRate)
  reverbWet: number;        // 0–1
  saturation: number;       // 0–1
  noiseLevel: number;       // linear gain
  label: string;
}

const PROFILES: Record<BypassStrength, BypassProfile> = {
  mild: {
    pitchCents: 30,
    reverbWet: 0.06,
    saturation: 0.02,
    noiseLevel: 0.0003,
    label: "Hafif (+30¢, %2.5 hız farkı)",
  },
  medium: {
    pitchCents: 60,
    reverbWet: 0.10,
    saturation: 0.04,
    noiseLevel: 0.0005,
    label: "Orta (+60¢, %5 hız farkı) ⭐ Önerilen",
  },
  strong: {
    pitchCents: 100,
    reverbWet: 0.15,
    saturation: 0.07,
    noiseLevel: 0.001,
    label: "Güçlü (+1 yarım ton, %8 hız farkı)",
  },
};

export { PROFILES };

function makeIR(ctx: OfflineAudioContext, decaySec = 0.4): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * decaySec);
  const ir = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const ch = ir.getChannelData(c);
    for (let i = 0; i < len; i++) {
      ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
    }
  }
  return ir;
}

export async function bypassAudio(
  file: File,
  strength: BypassStrength = "medium",
  onProgress?: (pct: number) => void
): Promise<Blob> {
  const p = PROFILES[strength];

  onProgress?.(5);

  // Decode
  const arrayBuf = await file.arrayBuffer();
  const tmpCtx = new AudioContext();
  const decoded = await tmpCtx.decodeAudioData(arrayBuf);
  await tmpCtx.close();

  onProgress?.(20);

  // playbackRate: pitch shift + tempo shift together
  // 2^(cents/1200): 60¢ → ×1.0353 (song 3.5% faster and 60¢ higher)
  const rate = Math.pow(2, p.pitchCents / 1200);
  const outSamples = Math.ceil((decoded.duration / rate + 1.0) * decoded.sampleRate);

  const offline = new OfflineAudioContext(
    decoded.numberOfChannels,
    outSamples,
    decoded.sampleRate
  );

  // ── Apply saturation directly to PCM (CPU-side, before Web Audio graph) ──
  // This is the most reliable way — saturation modifies the waveform itself
  const saturatedBuf = offline.createBuffer(
    decoded.numberOfChannels,
    decoded.length,
    decoded.sampleRate
  );
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const src = decoded.getChannelData(c);
    const dst = saturatedBuf.getChannelData(c);
    for (let i = 0; i < src.length; i++) {
      dst[i] = softClip(src[i], p.saturation);
    }
  }

  onProgress?.(40);

  // ── Web Audio graph ────────────────────────────────────────────────────────
  const srcNode = offline.createBufferSource();
  srcNode.buffer = saturatedBuf;
  srcNode.playbackRate.value = rate;

  // High-shelf EQ: gentle −1.5 dB cut above 12 kHz — changes spectral fingerprint
  const highShelf = offline.createBiquadFilter();
  highShelf.type = "highshelf";
  highShelf.frequency.value = 12000;
  highShelf.gain.value = -1.5;

  // Low-shelf EQ: +1 dB boost below 200 Hz — changes spectral fingerprint
  const lowShelf = offline.createBiquadFilter();
  lowShelf.type = "lowshelf";
  lowShelf.frequency.value = 200;
  lowShelf.gain.value = 1.0;

  // Reverb
  const convolver = offline.createConvolver();
  convolver.buffer = makeIR(offline, 0.5);
  const dryGain = offline.createGain();
  dryGain.gain.value = 1 - p.reverbWet;
  const wetGain = offline.createGain();
  wetGain.gain.value = p.reverbWet;

  // Ultra-quiet noise — completely inaudible but alters digital fingerprint
  const noiseLen = outSamples;
  const noiseBuf = offline.createBuffer(decoded.numberOfChannels, noiseLen, decoded.sampleRate);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const ch = noiseBuf.getChannelData(c);
    for (let i = 0; i < noiseLen; i++) ch[i] = (Math.random() * 2 - 1) * p.noiseLevel;
  }
  const noiseNode = offline.createBufferSource();
  noiseNode.buffer = noiseBuf;

  // Limiter
  const limiter = offline.createDynamicsCompressor();
  limiter.threshold.value = -0.3;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.001;
  limiter.release.value = 0.1;

  // Graph: src → EQ chain → dry/wet split → limiter → out
  srcNode.connect(highShelf);
  highShelf.connect(lowShelf);
  lowShelf.connect(dryGain);
  lowShelf.connect(convolver);
  convolver.connect(wetGain);
  dryGain.connect(limiter);
  wetGain.connect(limiter);
  noiseNode.connect(limiter);
  limiter.connect(offline.destination);

  srcNode.start(0);
  noiseNode.start(0);

  onProgress?.(55);

  const rendered = await offline.startRendering();
  onProgress?.(92);

  const wav = audioBufferToWav(rendered);
  onProgress?.(100);
  return wav;
}
