import { Midi } from "@tonejs/midi";

// ─── Hann window (reduces spectral leakage) ───────────────────────────────────
function hannWindow(buf: Float32Array): Float32Array {
  const n = buf.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = buf[i] * 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return out;
}

// ─── YIN pitch detection (much more accurate than autocorrelation) ────────────
function yin(buf: Float32Array, sr: number, threshold = 0.12): number {
  const windowed = hannWindow(buf);
  const n = windowed.length;
  const half = Math.floor(n / 2);

  // RMS gate
  let rms = 0;
  for (let i = 0; i < n; i++) rms += windowed[i] * windowed[i];
  if (Math.sqrt(rms / n) < 0.008) return -1;

  // Step 1: difference function
  const d = new Float32Array(half);
  for (let tau = 1; tau < half; tau++) {
    for (let j = 0; j < half; j++) {
      const delta = windowed[j] - windowed[j + tau];
      d[tau] += delta * delta;
    }
  }

  // Step 2: cumulative mean normalized difference (CMNDF)
  const cmndf = new Float32Array(half);
  cmndf[0] = 1;
  let sum = 0;
  for (let tau = 1; tau < half; tau++) {
    sum += d[tau];
    cmndf[tau] = sum > 0 ? (d[tau] * tau) / sum : 1;
  }

  // Step 3: find first dip below threshold
  // Search range: 60 Hz – 2000 Hz
  const tauMin = Math.floor(sr / 2000);
  const tauMax = Math.floor(sr / 60);
  let bestTau = -1;
  for (let tau = Math.max(2, tauMin); tau < Math.min(half, tauMax); tau++) {
    if (cmndf[tau] < threshold) {
      // Walk to local minimum
      while (tau + 1 < half && cmndf[tau + 1] < cmndf[tau]) tau++;
      bestTau = tau;
      break;
    }
  }
  if (bestTau === -1) return -1;

  // Step 4: parabolic interpolation for sub-sample accuracy
  const prev = cmndf[bestTau - 1] ?? cmndf[bestTau];
  const curr = cmndf[bestTau];
  const next = cmndf[bestTau + 1] ?? cmndf[bestTau];
  const a = (prev + next - 2 * curr) / 2;
  const b = (next - prev) / 2;
  const refined = a !== 0 ? bestTau - b / (2 * a) : bestTau;

  const freq = sr / refined;
  return freq >= 60 && freq <= 2000 ? freq : -1;
}

// ─── Median filter for pitch smoothing ───────────────────────────────────────
function medianFilter(arr: number[], k = 5): number[] {
  const half = Math.floor(k / 2);
  return arr.map((_, i) => {
    const window = arr.slice(Math.max(0, i - half), i + half + 1).filter(v => v > 0);
    if (!window.length) return arr[i];
    window.sort((a, b) => a - b);
    return window[Math.floor(window.length / 2)];
  });
}

function freqToMidi(freq: number): number {
  return Math.round(12 * Math.log2(freq / 440) + 69);
}

// ─── Audio buffer → MIDI ──────────────────────────────────────────────────────
export async function audioToMidi(
  file: File,
  bpmShift = 1.00,   // default: no tempo change
  transpose = 0      // default: no transpose — preserve original pitch
): Promise<Blob> {
  const arrayBuf = await file.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arrayBuf);
  await ctx.close();

  // Mix to mono
  const numCh = decoded.numberOfChannels;
  const len = decoded.length;
  const mono = new Float32Array(len);
  for (let c = 0; c < numCh; c++) {
    const ch = decoded.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += ch[i] / numCh;
  }

  const sr = decoded.sampleRate;
  const frameSize = 4096; // larger frame = better frequency resolution
  const hopSize   = 512;  // ~12ms hop for fine temporal resolution

  // Detect pitch for every hop
  const rawFreqs: number[] = [];
  const times: number[] = [];

  for (let offset = 0; offset + frameSize < len; offset += hopSize) {
    const frame = mono.slice(offset, offset + frameSize);
    const freq = yin(frame, sr);
    rawFreqs.push(freq);
    times.push(offset / sr);
  }

  // Smooth with median filter to remove spurious detections
  const smoothed = medianFilter(rawFreqs, 7);

  // Convert to MIDI note events
  type RawNote = { time: number; note: number; duration: number };
  const notes: RawNote[] = [];
  const hopDur = hopSize / sr;

  let i = 0;
  while (i < smoothed.length) {
    const freq = smoothed[i];
    if (freq <= 0) { i++; continue; }

    const note = Math.max(21, Math.min(108, freqToMidi(freq) + transpose));
    const startTime = times[i];

    // Extend while same note (within ±1 semitone tolerance)
    let j = i + 1;
    while (j < smoothed.length) {
      const f2 = smoothed[j];
      if (f2 <= 0) break;
      const n2 = freqToMidi(f2) + transpose;
      if (Math.abs(n2 - note) > 1) break;
      j++;
    }

    const duration = (j - i) * hopDur;
    // Only keep notes longer than 60ms
    if (duration >= 0.06) {
      notes.push({ time: startTime, note, duration: duration * 0.93 });
    }
    i = j;
  }

  // Build MIDI
  const midi = new Midi();
  midi.header.setTempo(Math.round(120 * bpmShift));
  const track = midi.addTrack();
  track.name = "Melody";

  for (const n of notes) {
    track.addNote({
      midi: n.note,
      time: n.time,
      duration: n.duration,
      // Slight velocity variation for naturalness
      velocity: 0.72 + Math.random() * 0.12,
    });
  }

  return new Blob([midi.toArray()], { type: "audio/midi" });
}
