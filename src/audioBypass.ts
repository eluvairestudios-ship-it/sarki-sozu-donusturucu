// Audio bypass: pitch shift WITHOUT tempo change + multi-layer fingerprint breaking
// OLA (Overlap-Add) time stretching + linear resampling = true pitch shift

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

// ─── OLA time stretching ──────────────────────────────────────────────────────
// Stretches audio by `factor` without changing pitch.
// e.g. factor=1.122 → audio 12.2% longer, same pitch
function olaStretch(mono: Float32Array, factor: number): Float32Array {
  const windowSize = 2048;
  const hopSize = 512;
  const outLength = Math.round(mono.length * factor);
  const output = new Float32Array(outLength);
  const norm = new Float32Array(outLength);

  // Hann window
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

// ─── Linear interpolation resampler ──────────────────────────────────────────
function linResample(input: Float32Array, outLength: number): Float32Array {
  const output = new Float32Array(outLength);
  const ratio = (input.length - 1) / (outLength - 1);
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

// ─── Pitch shift (no tempo change) ───────────────────────────────────────────
// semitones > 0: pitch up, semitones < 0: pitch down
// Tempo stays the same — only the key/pitch of the song changes.
function pitchShiftMono(mono: Float32Array, semitones: number): Float32Array {
  const factor = Math.pow(2, semitones / 12);
  // Step 1: OLA stretch by factor → same pitch, longer duration
  const stretched = olaStretch(mono, factor);
  // Step 2: resample back to original length → raises pitch, original tempo
  return linResample(stretched, mono.length);
}

// ─── Soft saturation ─────────────────────────────────────────────────────────
function saturate(x: number, drive: number): number {
  const g = x * (1 + drive * 3);
  if (g >= 1) return 2 / 3;
  if (g <= -1) return -2 / 3;
  return g - (g * g * g) / 3;
}

// ─── Impulse response for room reverb ────────────────────────────────────────
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
export type BypassStrength = "mild" | "medium" | "strong";

interface Profile {
  semitones: number;
  saturation: number;
  reverbWet: number;
  noiseLevel: number;
  label: string;
  tempoNote: string;
}

export const PROFILES: Record<BypassStrength, Profile> = {
  mild: {
    semitones: 1,
    saturation: 0.02,
    reverbWet: 0.06,
    noiseLevel: 0.0003,
    label: "Hafif — +1 yarım ton, tempo aynı",
    tempoNote: "Pek fark edilmez",
  },
  medium: {
    semitones: 2,
    saturation: 0.04,
    reverbWet: 0.10,
    noiseLevel: 0.0005,
    label: "Orta ⭐ — +2 yarım ton, tempo aynı",
    tempoNote: "Dikkatli dinleyenler fark eder",
  },
  strong: {
    semitones: 3,
    saturation: 0.07,
    reverbWet: 0.14,
    noiseLevel: 0.001,
    label: "Güçlü — +3 yarım ton, tempo aynı",
    tempoNote: "Perde farkı belirgin ama müzik kaliteli",
  },
};

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

  onProgress?.(15);

  const sr = decoded.sampleRate;
  const numCh = decoded.numberOfChannels;

  // ── CPU-side: pitch shift + saturation per channel ────────────────────────
  const processedBuf = new AudioBuffer({ numberOfChannels: numCh, length: decoded.length, sampleRate: sr });

  for (let c = 0; c < numCh; c++) {
    const raw = decoded.getChannelData(c);

    // 1. Pitch shift (OLA + resample) — tempo unchanged
    const pitched = pitchShiftMono(raw, p.semitones);

    // 2. Soft saturation — changes waveform shape, breaks PCM fingerprint
    const sat = new Float32Array(pitched.length);
    for (let i = 0; i < pitched.length; i++) {
      sat[i] = saturate(pitched[i], p.saturation);
    }

    processedBuf.copyToChannel(sat, c);
    onProgress?.(15 + Math.round((c + 1) / numCh * 55)); // 15→70%
  }

  onProgress?.(72);

  // ── Web Audio graph: EQ + reverb + noise ─────────────────────────────────
  const offline = new OfflineAudioContext(numCh, decoded.length, sr);

  const src = offline.createBufferSource();
  src.buffer = processedBuf;

  // High-shelf cut: −2 dB above 10 kHz → changes spectral fingerprint
  const highShelf = offline.createBiquadFilter();
  highShelf.type = "highshelf";
  highShelf.frequency.value = 10000;
  highShelf.gain.value = -2.0;

  // Low-shelf boost: +1.5 dB below 250 Hz
  const lowShelf = offline.createBiquadFilter();
  lowShelf.type = "lowshelf";
  lowShelf.frequency.value = 250;
  lowShelf.gain.value = 1.5;

  // Notch at 1 kHz (subtle — changes mid-range fingerprint)
  const notch = offline.createBiquadFilter();
  notch.type = "notch";
  notch.frequency.value = 1000;
  notch.Q.value = 0.5;
  notch.gain.value = -1.0;

  // Reverb (changes room acoustic fingerprint)
  const convolver = offline.createConvolver();
  convolver.buffer = makeIR(offline, 0.5);
  const dryGain = offline.createGain();
  dryGain.gain.value = 1 - p.reverbWet;
  const wetGain = offline.createGain();
  wetGain.gain.value = p.reverbWet;

  // Inaudible noise (changes raw waveform fingerprint)
  const noiseLen = decoded.length;
  const noiseBuf = offline.createBuffer(numCh, noiseLen, sr);
  for (let c = 0; c < numCh; c++) {
    const ch = noiseBuf.getChannelData(c);
    for (let i = 0; i < noiseLen; i++) ch[i] = (Math.random() * 2 - 1) * p.noiseLevel;
  }
  const noiseSrc = offline.createBufferSource();
  noiseSrc.buffer = noiseBuf;

  // Limiter to prevent clipping
  const limiter = offline.createDynamicsCompressor();
  limiter.threshold.value = -0.5;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.001;
  limiter.release.value = 0.1;

  // Graph
  src.connect(highShelf);
  highShelf.connect(lowShelf);
  lowShelf.connect(notch);
  notch.connect(dryGain);
  notch.connect(convolver);
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
