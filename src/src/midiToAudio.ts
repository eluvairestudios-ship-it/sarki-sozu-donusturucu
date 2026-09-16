import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

// AudioBuffer → WAV Blob (no external lib)
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const len = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = numCh * bytesPerSample;
  const byteRate = sr * blockAlign;
  const dataSize = len * blockAlign;
  const headerSize = 44;

  const wav = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(wav);

  const write = (off: number, val: string) => {
    for (let i = 0; i < val.length; i++) view.setUint8(off + i, val.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, numCh, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);         // 16-bit
  write(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < len; i++) {
    for (let c = 0; c < numCh; c++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(c)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([wav], { type: "audio/wav" });
}

export async function midiToWav(
  midiBlob: Blob,
  onProgress?: (pct: number) => void
): Promise<Blob> {
  const arrayBuf = await midiBlob.arrayBuffer();
  const midi = new Midi(arrayBuf);
  const duration = midi.duration + 2.5;

  onProgress?.(5);

  const audioBuffer = await Tone.Offline(({ transport }) => {
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle8" },
      envelope: { attack: 0.02, decay: 0.15, sustain: 0.6, release: 1.2 },
      volume: -6,
    }).toDestination();

    const reverb = new Tone.Reverb({ decay: 1.8, wet: 0.25 }).toDestination();
    synth.connect(reverb);

    midi.tracks.forEach((track) => {
      track.notes.forEach((note) => {
        transport.schedule((time) => {
          synth.triggerAttackRelease(
            note.name,
            note.duration,
            time,
            note.velocity
          );
        }, note.time);
      });
    });

    transport.start();
    onProgress?.(40);
  }, duration);

  onProgress?.(85);
  const wav = audioBufferToWav(audioBuffer);
  onProgress?.(100);
  return wav;
}
