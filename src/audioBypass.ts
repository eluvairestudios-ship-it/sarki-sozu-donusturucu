// NUCLEAR BYPASS:
// 1. Vocal removal via mid-side processing (karaoke effect) — lyrics undetectable
// 2. OLA pitch shift without tempo change — audio fingerprint broken
// 3. EQ + saturation + reverb — extra fingerprint layers

// ─── WAV encoder ─────────────────────────────────────────────────────────────
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const len = buffer.length;
  const blockAlign = numCh * 2;
  const dataSize = len * blockAlign;
  const wav = new ArrayBuffer(44 + dataSize);
  const v = new DataView(wav);
  const str = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
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

// ─── Vocal removal (mid-side karaoke) ────────────────────────────────────────
// Stereo music has vocals center-panned (equal in L and R).
// Subtracting center content kills vocals while keeping instruments (panned L/R).
// vocalKeep=0 → full karaoke, vocalKeep=1 → original
function removeVocals(L: Float32Array, R: Float32Array, vocalKeep = 0.05): [Float32Array, Float32Array] {
  const len = L.length;
  const outL = new Float32Array(len);
  const outR = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const mid = (L[i] + R[i]) * 0.5;
    const side = (L[i] - R[i]) * 0.5;
    // Keep tiny amount of mid to preserve bass/kick, remove vocal range
    outL[i] = mid * vocalKeep + side;
    outR[i] = mid * vocalKeep - side;
  }
  return [outL, outR];
}

// ─── OLA time stretching (for pitch shift without tempo change) ───────────────
function olaStretch(mono: Float32Array, factor: number): Float32Array {
  const windowSize = 2048;
  const hopSize = 512;
  const outLength = Math.round(mono.length * factor);
  const output = new Float32Array(outLength);
  const norm = new Float32Array(outLength);
  const win = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (windowSize - 1)));
  }
  let frame = 0;
  while (true) {
    const inPos = frame * hopSize;
    if (inPos + windowSize > mono.length) break;
    const outPos = Math.round(inPos * factor);
    if (outPos + windowSize > outLength) break;
    for (let i = 0; i < windowSize; i++) {
      output[outPos + i] += mono[inPos + i] * win[i];
      norm[outPos + i] += win[i] * win[i];
    }
    frame++;
  }
  for (let i = 0; i < outLength; i++) {
    if (norm[i] > 1e-6) output[i] /= norm[i];
  }
  return output;
}

function linResample(input: Float32Array, outLength: number): Float32Array {
  const output = new Float32Array(outLength);
  const ratio = (input.length - 1) / Math.max(outLength - 1, 1);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? 0;
    const b = input[idx + 1] ?? input[idx] ?? 0;
    output[i] = a + frac * (b - a);
  }
  return output;
}

function pitchShiftMono(mono: Float32Array, semitones: number): Float32Array {
  const factor = Math.pow(2, semitones / 12);
  const stretched = olaStretch(mono, factor);
  return linResample(stretched, mono.length);
}

// ─── Soft saturation ─────────────────────────────────────────────────────────
function saturate(x: number, drive: number): number {
  const g = x * (1 + drive * 3);
  if (g >= 1) return 2 / 3;
  if (g <= -1) return -2 / 3;
  return g - (g * g * g) / 3;
}

function makeIR(ctx: OfflineAudioContext, decaySec = 0.5): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * decaySec);
  const ir = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const ch = ir.getChannelData(c);
    for (let i = 0; i < len; i++) {
      ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.5);
    }
  }
  return ir;
}

// ─── Public API ───────────────────────────────────────────────────────────────
export type BypassMode = "fingerprint" | "vocal" | "nuclear";

export const MODE_LABELS: Record<BypassMode, { label: string; desc: string; color: string }> = {
  fingerprint: {
    label: "🔊 Sadece Parmak İzi",
    desc: "+2 yarım ton, vokal korunur",
    color: "#38bdf8",
  },
  vocal: {
    label: "🎤 Sadece Vokal Sil",
    desc: "Karaoke efekti, perde aynı",
    color: "#fb923c",
  },
  nuclear: {
    label: "💥 Nuclear — İkisi Birden",
    desc: "Vokal sil + pitch shift + EQ + reverb",
    color: "#4eff99",
  },
};

export async function bypassAudio(
  file: File,
  mode: BypassMode = "nuclear",
  onProgress?: (pct: number) => void
): Promise<Blob> {
  onProgress?.(5);

  const arrayBuf = await file.arrayBuffer();
  const tmpCtx = new AudioContext();
  const decoded = await tmpCtx.decodeAudioData(arrayBuf);
  await tmpCtx.close();

  onProgress?.(15);

  const sr = decoded.sampleRate;
  const numCh = decoded.numberOfChannels;
  const isStereo = numCh >= 2;

  const doVocal = mode === "vocal" || mode === "nuclear";
  const doPitch = mode === "fingerprint" || mode === "nuclear";
  const semitones = doPitch ? 2 : 0;

  // ── CPU processing ────────────────────────────────────────────────────────
  const channels: Float32Array[] = [];

  for (let c = 0; c < numCh; c++) {
    channels.push(new Float32Array(decoded.getChannelData(c)));
  }

  // Step 1: Vocal removal (mid-side)
  let processedChannels = channels;
  if (doVocal && isStereo) {
    const [outL, outR] = removeVocals(channels[0], channels[1]);
    processedChannels = [outL, outR, ...channels.slice(2)];
  }

  onProgress?.(30);

  // Step 2: Pitch shift per channel (OLA + resample)
  const finalChannels: Float32Array[] = [];
  for (let c = 0; c < processedChannels.length; c++) {
    const ch = processedChannels[c];
    const pitched = doPitch ? pitchShiftMono(ch, semitones) : ch;
    // Step 3: Saturation
    const sat = new Float32Array(pitched.length);
    for (let i = 0; i < pitched.length; i++) {
      sat[i] = saturate(pitched[i], mode === "nuclear" ? 0.04 : 0.01);
    }
    finalChannels.push(sat);
    onProgress?.(30 + Math.round(((c + 1) / processedChannels.length) * 40));
  }

  onProgress?.(72);

  // ── Web Audio graph: EQ + reverb + noise ─────────────────────────────────
  const outNumCh = Math.min(finalChannels.length, 2);
  const offline = new OfflineAudioContext(outNumCh, decoded.length, sr);

  const procBuf = offline.createBuffer(outNumCh, decoded.length, sr);
  for (let c = 0; c < outNumCh; c++) {
    procBuf.copyToChannel(new Float32Array(finalChannels[c]), c);
  }

  const src = offline.createBufferSource();
  src.buffer = procBuf;

  // Vocal frequency notch (300–3kHz area — extra vocal suppression)
  const vocalNotch = offline.createBiquadFilter();
  vocalNotch.type = "peaking";
  vocalNotch.frequency.value = 1200;
  vocalNotch.Q.value = 0.7;
  vocalNotch.gain.value = doVocal ? -8 : -1;

  // High shelf
  const highShelf = offline.createBiquadFilter();
  highShelf.type = "highshelf";
  highShelf.frequency.value = 10000;
  highShelf.gain.value = -2;

  // Low shelf boost
  const lowShelf = offline.createBiquadFilter();
  lowShelf.type = "lowshelf";
  lowShelf.frequency.value = 250;
  lowShelf.gain.value = 1.5;

  // Reverb
  const convolver = offline.createConvolver();
  convolver.buffer = makeIR(offline);
  const dryGain = offline.createGain();
  dryGain.gain.value = mode === "nuclear" ? 0.88 : 0.94;
  const wetGain = offline.createGain();
  wetGain.gain.value = mode === "nuclear" ? 0.12 : 0.06;

  // Inaudible noise
  const noiseLevel = mode === "nuclear" ? 0.0008 : 0.0003;
  const noiseBuf = offline.createBuffer(outNumCh, decoded.length, sr);
  for (let c = 0; c < outNumCh; c++) {
    const ch = noiseBuf.getChannelData(c);
    for (let i = 0; i < decoded.length; i++) ch[i] = (Math.random() * 2 - 1) * noiseLevel;
  }
  const noiseSrc = offline.createBufferSource();
  noiseSrc.buffer = noiseBuf;

  const limiter = offline.createDynamicsCompressor();
  limiter.threshold.value = -0.5;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.001;
  limiter.release.value = 0.1;

  src.connect(vocalNotch);
  vocalNotch.connect(highShelf);
  highShelf.connect(lowShelf);
  lowShelf.connect(dryGain);
  lowShelf.connect(convolver);
  convolver.connect(wetGain);
  dryGain.connect(limiter);
  wetGain.connect(limiter);
  noiseSrc.connect(limiter);
  limiter.connect(offline.destination);

  src.start(0);
  noiseSrc.start(0);

  const rendered = await offline.startRendering();
  onProgress?.(95);

  const wav = audioBufferToWav(rendered);
  onProgress?.(100);
  return wav;
}
