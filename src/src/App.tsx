import { useState, useRef, useCallback } from "react";
import { audioToMidi } from "./midi";
import { midiToWav } from "./midiToAudio";

const ENV_KEY = import.meta.env.VITE_OPENAI_API_KEY as string | undefined;

// ─── Whisper transcription ────────────────────────────────────────────────────
type Provider = "openai" | "groq";
function detectProvider(key: string): Provider {
  return key.startsWith("gsk_") ? "groq" : "openai";
}

interface WhisperWord { word: string; start: number; end: number; }
interface WhisperSegment { text: string; start: number; end: number; words?: WhisperWord[]; }
interface WhisperVerbose { segments: WhisperSegment[]; }

function assertAsciiKey(key: string) {
  for (let i = 0; i < key.length; i++) {
    if (key.charCodeAt(i) > 127) {
      throw new Error(`API anahtarında geçersiz karakter var (konum ${i}: '${key[i]}'). Anahtarınızı kontrol edin — Türkçe karakter içermemeli.`);
    }
  }
}

async function transcribeAudioTimed(file: File, key: string): Promise<WhisperVerbose> {
  assertAsciiKey(key);
  const p = detectProvider(key);
  // Groq supports word timestamps too
  const url = p === "groq"
    ? "https://api.groq.com/openai/v1/audio/transcriptions"
    : "https://api.openai.com/v1/audio/transcriptions";
  const model = p === "groq" ? "whisper-large-v3" : "whisper-1";
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("language", "tr");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "word");
  form.append("timestamp_granularities[]", "segment");
  const res = await fetch(url, { method: "POST", headers: { Authorization: `Bearer ${key}` }, body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── Timing-aware lyric formatter ─────────────────────────────────────────────
// Returns Suno-annotated lyrics based on actual word durations and gaps
function buildTimedLyrics(data: WhisperVerbose): string {
  // Flatten all words with timing
  const words: WhisperWord[] = [];
  for (const seg of data.segments) {
    if (seg.words?.length) {
      words.push(...seg.words);
    } else {
      // Fallback: distribute segment evenly
      const segWords = seg.text.trim().split(/\s+/);
      const dur = (seg.end - seg.start) / segWords.length;
      segWords.forEach((w, i) => {
        words.push({ word: w, start: seg.start + i * dur, end: seg.start + (i + 1) * dur });
      });
    }
  }

  if (!words.length) return data.segments.map(s => s.text).join("\n");

  // Calculate average word duration for relative comparison
  const durations = words.map(w => w.end - w.start);
  const avgDur = durations.reduce((a, b) => a + b, 0) / durations.length;

  const lines: string[] = [];
  let currentLine: string[] = [];

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const dur = w.end - w.start;
    const gapAfter = i < words.length - 1 ? words[i + 1].start - w.end : 0;

    // Annotate this word
    let token = w.word.trim();

    // Long held note (duration > 1.8× average) → ~~
    if (dur > avgDur * 1.8) {
      token = addExtension(token, "~~");
    }
    // Medium held note (duration > 1.3× average) → ~
    else if (dur > avgDur * 1.3) {
      token = addExtension(token, "~");
    }

    // Rising pitch: mark with uppercase if word is short and high-energy
    // (heuristic: very short gap before = rushed/rising)
    const gapBefore = i > 0 ? w.start - words[i - 1].end : 1;
    if (gapBefore < 0.05 && dur < avgDur * 0.7) {
      token = token.toUpperCase();
    }

    currentLine.push(token);

    // Line break decisions:
    const longPause = gapAfter > 0.8;   // >0.8s pause → new line
    const medPause  = gapAfter > 0.4;   // >0.4s pause → new line if line has 3+ words
    const longLine  = currentLine.length >= 6; // max 6 words per line

    if (longPause || (medPause && currentLine.length >= 3) || longLine) {
      // Add pause marker if significant gap
      if (longPause && gapAfter > 1.5) currentLine.push("...");
      else if (longPause) currentLine.push(",");
      lines.push(currentLine.join(" "));
      currentLine = [];
    }
  }
  if (currentLine.length) lines.push(currentLine.join(" "));

  return lines.join("\n");
}

function addExtension(word: string, ext: string): string {
  // Add extension marker after the last vowel
  const vowelRe = /([aeıioöuüAEIİOÖUÜ])(?=[^aeıioöuüAEIİOÖUÜ]*$)/;
  return vowelRe.test(word) ? word.replace(vowelRe, `$1${ext}`) : word + ext;
}

// ─── GPT: Doğru sözleri Whisper zamanlamasıyla hizala ────────────────────────
async function alignLyricsWithTiming(
  correctLyrics: string,
  timedLyrics: string,
  key: string
): Promise<string> {
  assertAsciiKey(key);
  const p = detectProvider(key);
  const url = p === "groq"
    ? "https://api.groq.com/openai/v1/chat/completions"
    : "https://api.openai.com/v1/chat/completions";
  const model = p === "groq" ? "llama-3.3-70b-versatile" : "gpt-4o-mini";

  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content:
            "Sen iki şarkı sözü metnini karşılaştıran ve birleştiren uzmansın.\n\n" +
            "Sana iki metin verilecek:\n" +
            "1. DOĞRU SÖZLER: Kullanıcının internetten kopyaladığı, %100 doğru kelimeler\n" +
            "2. ZAMANLAMALI SÖZLER: Whisper'ın sesden çıkardığı, ~ ve ~~ işaretleri olan, satır kesmeli versiyon\n\n" +
            "Görevin:\n" +
            "- DOĞRU SÖZLER'deki kelimeleri kullan (bunlar kesinlikle doğru)\n" +
            "- ZAMANLAMALI SÖZLER'deki ~ ~~ , ... işaretlerini ve satır kesmelerini al\n" +
            "- [Verse], [Chorus] gibi etiketleri koru veya ekle\n" +
            "- İki metni hizala: doğru kelime + doğru zamanlama işareti\n" +
            "- Kelimeleri ASLA değiştirme, ekleyip çıkarma\n\n" +
            "Sadece sonuç metnini döndür, açıklama yazma.",
        },
        {
          role: "user",
          content: `DOĞRU SÖZLER:\n${correctLyrics}\n\nZAMANLAMALI SÖZLER:\n${timedLyrics}`,
        },
      ],
      temperature: 0,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const json = await res.json();
  return json.choices[0].message.content.trim();
}

// ─── GPT lyrics formatter ─────────────────────────────────────────────────────
async function formatLyricsWithGPT(timedLyrics: string, key: string): Promise<string> {
  assertAsciiKey(key);
  const p = detectProvider(key);
  const url = p === "groq"
    ? "https://api.groq.com/openai/v1/chat/completions"
    : "https://api.openai.com/v1/chat/completions";
  const model = p === "groq" ? "llama-3.3-70b-versatile" : "gpt-4o-mini";

  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content:
            "Sen bir şarkı sözü DÜZEN editörüsün. Görevin SADECE satır kesmek ve bölüm etiketi eklemek.\n\n" +
            "KESİN YASAKLAR — bunları ASLA yapma:\n" +
            "- Hiçbir kelimeyi değiştirme, düzeltme, tahmin etme\n" +
            "- Hiçbir kelime ekleme veya silme\n" +
            "- Türkçe karakterleri değiştirme (ş, ç, ğ, ü, ö, ı harfleri AYNEN kalır)\n" +
            "- ~ ve ~~ işaretlerini kaldırma veya değiştirme\n" +
            "- , ve ... işaretlerini kaldırma\n" +
            "- Büyük/küçük harf değiştirme\n\n" +
            "SADECE yapabileceklerin:\n" +
            "1. Satırları müzik nefesine göre böl (her satır 3-6 kelime)\n" +
            "2. [Verse 1], [Chorus], [Verse 2], [Bridge], [Outro] etiketleri ekle\n" +
            "3. Tekrar eden satır gruplarını [Chorus] yap\n" +
            "4. Bölümler arasına boş satır koy\n\n" +
            "Çıktıda SADECE şarkı sözleri olsun, açıklama yazma.",
        },
        { role: "user", content: timedLyrics },
      ],
      temperature: 0,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const json = await res.json();
  return json.choices[0].message.content.trim();
}

// ─── Suno annotation ──────────────────────────────────────────────────────────
const VOWELS = "aeıioöuüAEIİOÖUÜ";
const isVowel = (c: string) => VOWELS.includes(c);
const ZWS = "​";

function syllabify(word: string): string[] {
  const syls: string[] = [];
  let cur = "";
  for (let i = 0; i < word.length; i++) {
    cur += word[i];
    if (isVowel(word[i])) {
      const next = word[i + 1] ?? "";
      const afterNext = word[i + 2] ?? "";
      if (!isVowel(next) && !isVowel(afterNext) && next !== "") { cur += next; i++; }
      syls.push(cur);
      cur = "";
    }
  }
  if (cur) syls.length ? (syls[syls.length - 1] += cur) : syls.push(cur);
  return syls.length ? syls : [word];
}

function annotateWord(word: string, isLineEnd: boolean): string {
  if (/^\[.*\]$/.test(word)) return word;
  const syls = syllabify(word);
  const annotated = syls.map((syl, idx) => {
    const isLast = idx === syls.length - 1;
    if (isLast && isLineEnd) return syl.replace(/([aeıioöuüAEIİOÖUÜ])(?=[^aeıioöuüAEIİOÖUÜ]*$)/, "$1~~");
    if (isLast && syls.length >= 3) return syl.replace(/([aeıioöuüAEIİOÖUÜ])(?=[^aeıioöuüAEIİOÖUÜ]*$)/, "$1~");
    return syl;
  });
  const joined = annotated.join("-");
  if (joined.length <= 2) return joined;
  const mid = Math.ceil(joined.length / 2);
  return joined.slice(0, mid) + ZWS + joined.slice(mid);
}

function annotateLine(line: string): string {
  const trimmed = line.trim();
  if (!trimmed || /^\[.*\]$/.test(trimmed)) return line;
  const tokens = line.split(/(\s+)/);
  const realWords = tokens.filter((t) => t.trim().length > 0);
  return tokens.map((tok) => {
    if (!tok.trim()) return tok;
    return annotateWord(tok, tok === realWords[realWords.length - 1]);
  }).join("");
}

type Line = { original: string; converted: string };

// ─── Download helper ──────────────────────────────────────────────────────────
function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

// ─── App ──────────────────────────────────────────────────────────────────────
type Stage = "idle" | "transcribing" | "formatting" | "done" | "error";
type MidiStage = "idle" | "processing" | "done" | "error";

export default function App() {
  const [apiKey, setApiKey] = useState(ENV_KEY ?? "");
  const [showKey, setShowKey] = useState(false);

  // Lyrics flow
  const [stage, setStage] = useState<Stage>("idle");
  const [stageMsg, setStageMsg] = useState("");
  const [audioName, setAudioName] = useState("");
  const [rawLyrics, setRawLyrics] = useState("");
  const [correctLyrics, setCorrectLyrics] = useState("");
  const [timedLyricsRaw, setTimedLyricsRaw] = useState("");
  const [formattedLyrics, setFormattedLyrics] = useState("");
  const [lines, setLines] = useState<Line[]>([]);
  const [pasteZone, setPasteZone] = useState("");
  const [lyricsError, setLyricsError] = useState("");
  const [alignStage, setAlignStage] = useState<"idle"|"aligning"|"done"|"error">("idle");

  // MIDI flow
  const [midiStage, setMidiStage] = useState<MidiStage>("idle");
  const [midiError, setMidiError] = useState("");
  const [midiAudioName, setMidiAudioName] = useState("");
  const [transpose, setTranspose] = useState(0);
  const [tempoShift, setTempoShift] = useState(0);
  const [midiDragOver, setMidiDragOver] = useState(false);
  const [midiBlob, setMidiBlob] = useState<Blob | null>(null);
  const [wavProgress, setWavProgress] = useState(0);
  const [wavStage, setWavStage] = useState<"idle"|"processing"|"done"|"error">("idle");

  // ElevenLabs cover
  const [elKey, setElKey] = useState("");
  const [elVoices, setElVoices] = useState<{voice_id:string;name:string}[]>([]);
  const [elVoiceId, setElVoiceId] = useState("");
  const [elText, setElText] = useState("");
  const [elStage, setElStage] = useState<"idle"|"loading"|"tts"|"done"|"error">("idle");
  const [elError, setElError] = useState("");

  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const midiFileRef = useRef<HTMLInputElement>(null);

  const keyStatus = apiKey.startsWith("gsk_")
    ? { label: "Groq — Ücretsiz ✓", color: "#4eff99" }
    : apiKey.startsWith("sk-")
    ? { label: "OpenAI ✓", color: "#38bdf8" }
    : { label: "Anahtar gerekli", color: "#fb923c" };

  // ── Lyrics pipeline ──
  const handleLyricsFile = useCallback(async (file: File) => {
    if (!apiKey.trim()) { setLyricsError("API anahtarı gir."); return; }
    setAudioName(file.name);
    setStage("transcribing");
    setStageMsg("Whisper ile transkripsiyon yapılıyor…");
    setLyricsError("");
    setRawLyrics(""); setFormattedLyrics(""); setLines([]); setPasteZone("");
    try {
      const timed = await transcribeAudioTimed(file, apiKey.trim());
      const rawText = timed.segments.map(s => s.text).join(" ").trim();
      setRawLyrics(rawText);

      setStage("formatting");
      setStageMsg("Zamanlama analiz ediliyor — duraklar, uzatmalar, yükselmeler işaretleniyor…");
      const timedLyrics = buildTimedLyrics(timed);
      setTimedLyricsRaw(timedLyrics);

      setStageMsg("GPT ile bölümler ve yapı düzenleniyor…");
      const fmt = await formatLyricsWithGPT(timedLyrics, apiKey.trim());
      setFormattedLyrics(fmt);
      const ls: Line[] = fmt.split("\n").map((line) => ({
        original: line,
        converted: annotateLine(line),
      }));
      setLines(ls);
      setPasteZone(ls.map((l) => l.converted).join("\n"));
      setStage("done");
      setStageMsg("");
    } catch (e: unknown) {
      setLyricsError(e instanceof Error ? e.message : "Hata");
      setStage("error");
    }
  }, [apiKey]);

  const handleAlign = async () => {
    if (!correctLyrics.trim() || !apiKey.trim()) return;
    setAlignStage("aligning");
    setLyricsError("");
    try {
      const timed = timedLyricsRaw || rawLyrics;
      const aligned = await alignLyricsWithTiming(correctLyrics, timed, apiKey.trim());
      setFormattedLyrics(aligned);
      const ls: Line[] = aligned.split("\n").map((line) => ({
        original: line, converted: annotateLine(line),
      }));
      setLines(ls);
      setPasteZone(ls.map((l) => l.converted).join("\n"));
      setAlignStage("done");
      setStage("done");
    } catch (e: unknown) {
      setLyricsError(e instanceof Error ? e.message : "Hata");
      setAlignStage("error");
    }
  };

  const handleManualFormat = async () => {
    if (!rawLyrics.trim() || !apiKey.trim()) return;
    setStage("formatting");
    setStageMsg("GPT ile düzenleniyor…");
    setLyricsError("");
    try {
      const fmt = await formatLyricsWithGPT(rawLyrics, apiKey.trim());
      setFormattedLyrics(fmt);
      const ls: Line[] = fmt.split("\n").map((line) => ({
        original: line, converted: annotateLine(line),
      }));
      setLines(ls);
      setPasteZone(ls.map((l) => l.converted).join("\n"));
      setStage("done"); setStageMsg("");
    } catch (e: unknown) {
      setLyricsError(e instanceof Error ? e.message : "Hata");
      setStage("error");
    }
  };

  const handleManualAnnotate = () => {
    if (!formattedLyrics.trim()) return;
    const ls: Line[] = formattedLyrics.split("\n").map((line) => ({
      original: line, converted: annotateLine(line),
    }));
    setLines(ls);
    setPasteZone(ls.map((l) => l.converted).join("\n"));
    setStage("done");
  };

  // ── MIDI pipeline ──
  const handleMidiFile = useCallback(async (file: File) => {
    setMidiAudioName(file.name);
    setMidiStage("processing");
    setMidiError("");
    setMidiBlob(null);
    setWavStage("idle");
    try {
      const blob = await audioToMidi(file, 1 + tempoShift / 100, transpose);
      setMidiBlob(blob);
      const baseName = file.name.replace(/\.[^.]+$/, "");
      downloadBlob(blob, `${baseName}_suno.mid`);
      setMidiStage("done");
    } catch (e: unknown) {
      setMidiError(e instanceof Error ? e.message : "Hata");
      setMidiStage("error");
    }
  }, [transpose, tempoShift]);

  const handleMidiToWav = useCallback(async () => {
    if (!midiBlob) return;
    setWavStage("processing");
    setWavProgress(0);
    try {
      const wav = await midiToWav(midiBlob, setWavProgress);
      downloadBlob(wav, `${midiAudioName.replace(/\.[^.]+$/,"")}_suno.wav`);
      setWavStage("done");
    } catch (e: unknown) {
      setWavStage("error");
    }
  }, [midiBlob, midiAudioName]);

  // ── ElevenLabs ──
  // Hardcoded popular ElevenLabs voices — no voices_read permission needed
  const PRESET_VOICES = [
    { voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel — Kadın, Sakin" },
    { voice_id: "EXAVITQu4vr4xnSDxMaL", name: "Bella — Kadın, Enerjik" },
    { voice_id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli — Kadın, Genç" },
    { voice_id: "AZnzlk1XvdvUeBnXmlld", name: "Domi — Kadın, Güçlü" },
    { voice_id: "pNInz6obpgDQGcFmaJgB", name: "Adam — Erkek, Derin" },
    { voice_id: "ErXwobaYiN019PkySvjV", name: "Antoni — Erkek, Sakin" },
    { voice_id: "VR6AewLTigWG4xSOukaG", name: "Arnold — Erkek, Güçlü" },
    { voice_id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh — Erkek, Genç" },
    { voice_id: "yoZ06aMxZJJ28mfd3POQ", name: "Sam — Erkek, Enerjik" },
    { voice_id: "N2lVS1w4EtoT3dr4eOWO", name: "Callum — Erkek, Dramatik" },
  ];

  const handleLoadVoices = () => {
    if (!elKey.trim()) return;
    setElVoices(PRESET_VOICES);
    setElVoiceId(PRESET_VOICES[0].voice_id);
    setElStage("idle");
    setElError("");
  };

  const handleTTS = async () => {
    if (!elKey.trim() || !elVoiceId || !elText.trim()) return;
    setElStage("tts"); setElError("");
    try {
      const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${elVoiceId}`, {
        method: "POST",
        headers: { "xi-api-key": elKey.trim(), "Content-Type": "application/json" },
        body: JSON.stringify({
          text: elText,
          model_id: "eleven_multilingual_v2",
          voice_settings: { stability: 0.4, similarity_boost: 0.75, style: 0.3, use_speaker_boost: true },
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      downloadBlob(new Blob([blob], { type: "audio/mpeg" }), "cover_vocals.mp3");
      setElStage("done");
    } catch (e: unknown) {
      setElError(e instanceof Error ? e.message : "Hata");
      setElStage("error");
    }
  };

  const isLoading = stage === "transcribing" || stage === "formatting";

  return (
    <>
      <div className="bg-anim" />
      <div style={{ position: "relative", zIndex: 1, minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", padding: "32px 16px 48px" }}>

        {/* HEADER */}
        <header style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ fontSize: 42, marginBottom: 8, filter: "drop-shadow(0 0 18px rgba(160,80,255,0.6))" }}>🎙️</div>
          <h1 style={{ fontFamily: "'Outfit',sans-serif", fontWeight: 900, fontSize: "clamp(17px,4vw,30px)", letterSpacing: "0.14em", color: "#fff", textShadow: "0 0 28px rgba(160,80,255,0.45)", marginBottom: 6 }}>
            ŞARKI SÖZÜ DÖNÜŞTÜRÜCÜ
          </h1>
          <p style={{ fontFamily: "'Outfit',sans-serif", fontSize: 13, color: "rgba(200,170,255,0.65)", letterSpacing: "0.05em" }}>
            MP3 → Türkçe Sözler · MIDI · Suno Copyright Bypass
          </p>
        </header>

        <div style={{ width: "100%", maxWidth: 1080, display: "flex", flexDirection: "column", gap: 16 }}>

          {/* API KEY */}
          <Panel>
            <PanelHeader left="🔑 API Anahtarı" right={<Mono color={keyStatus.color}>{keyStatus.label}</Mono>} />
            <div style={{ padding: "12px 16px", display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value.replace(/[^\x00-\x7F]/g, ""))}
                placeholder="gsk_... (Groq ücretsiz) veya sk-... (OpenAI)"
                style={{ flex: 1, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 7, padding: "8px 12px", fontFamily: "'JetBrains Mono',monospace", fontSize: 12.5, color: "rgba(235,220,255,0.9)", outline: "none" }}
              />
              <SmallBtn onClick={() => setShowKey(!showKey)}>{showKey ? "Gizle" : "Göster"}</SmallBtn>
            </div>
            <div style={{ padding: "0 16px 12px", fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: "rgba(200,170,255,0.4)" }}>
              Groq ücretsiz → console.groq.com &nbsp;|&nbsp; OpenAI ücretli → platform.openai.com
            </div>
          </Panel>

          {/* ═══ SECTION A: LYRICS ═══ */}
          <SectionLabel>① ŞARKI SÖZLERİ — Transkripsiyon & Düzenleme</SectionLabel>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(290px,1fr))", gap: 16 }}>

            {/* Upload */}
            <Panel>
              <PanelHeader left="🎵 MP3 Yükle → Sözleri Çıkar" right={audioName ? <Mono color="#4eff99">📎 {audioName}</Mono> : null} />
              <div style={{ padding: 16 }}>
                <DropZone
                  dragOver={dragOver}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleLyricsFile(f); }}
                  onClick={() => fileRef.current?.click()}
                  icon="🎙️"
                  text="MP3 sürükle veya tıkla"
                  hint="Whisper transkripsiyon + GPT düzenleme"
                />
                <input ref={fileRef} type="file" accept="audio/*" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleLyricsFile(f); }} />
                {isLoading && <StatusBar msg={stageMsg} />}
                {lyricsError && <ErrorBar msg={lyricsError} />}
              </div>
            </Panel>

            {/* Raw */}
            <Panel>
              <PanelHeader left="📝 Ham Transkripsiyon — Düzenle" right={rawLyrics ? <Mono color="#fb923c">⚠ Yanlış kelimeleri buradan düzelt!</Mono> : null} />
              <textarea
                value={rawLyrics}
                onChange={(e) => setRawLyrics(e.target.value)}
                spellCheck={false}
                placeholder={"⚠ ÖNEMLİ: Whisper bazen kelimeleri yanlış duyar!\nÖrn: 'yazgım' → 'yaşgım' gibi hatalar olabilir.\n\nGPT'ye göndermeden ÖNCE tüm kelimeleri kontrol et!\nYanlış kelimeleri buradan düzelt, sonra 'GPT ile Düzenle' bas."}
                style={textareaStyle(240)}
              />
              <PanelFooter>
                <PrimaryBtn onClick={handleManualFormat} disabled={!rawLyrics.trim() || !apiKey.trim()}>
                  🤖 GPT ile Düzenle
                </PrimaryBtn>
              </PanelFooter>
            </Panel>
          </div>

          {/* Correct Lyrics alignment panel — shown after transcription */}
          {(stage === "done" || timedLyricsRaw) && (
            <Panel>
              <PanelHeader
                left="🔍 Doğru Sözleri Yapıştır → Zamanlama ile Hizala"
                right={<Mono color="rgba(200,170,255,0.5)">İnternetten kopyaladığın doğru sözler + Whisper ritim zamanlaması</Mono>}
              />
              <div style={{ padding: "12px 16px 4px", fontFamily: "'Outfit',sans-serif", fontSize: 12, color: "rgba(200,170,255,0.55)", lineHeight: 1.7 }}>
                💡 Başka bir siteden kopyaladığın <strong style={{ color: "rgba(200,170,255,0.85)" }}>doğru şarkı sözlerini</strong> aşağıya yapıştır.
                Program Whisper'ın bulduğu <strong style={{ color: "#4eff99" }}>ritim zamanlamasını</strong> bu sözlere uygular — hem kelimeler doğru hem tempo korunur.
              </div>
              <div style={{ padding: "8px 16px" }}>
                <textarea
                  value={correctLyrics}
                  onChange={(e) => setCorrectLyrics(e.target.value)}
                  spellCheck={false}
                  placeholder={"Buraya başka siteden kopyaladığın DOĞRU şarkı sözlerini yapıştır.\nÖrn:\nBizim olsun bu gece\nSenin arzun benim yazgım\n...\n\nProgram Whisper zamanlaması ile karşılaştırıp hizalar."}
                  style={textareaStyle(180)}
                />
              </div>
              <PanelFooter>
                <PrimaryBtn
                  onClick={handleAlign}
                  disabled={!correctLyrics.trim() || !apiKey.trim() || alignStage === "aligning"}
                >
                  {alignStage === "aligning" ? "⏳ Hizalanıyor…" : alignStage === "done" ? "✓ Hizalandı" : "🔀 Hizala — Doğru Kelime + Doğru Ritim"}
                </PrimaryBtn>
                {alignStage === "error" && <ErrorBar msg={lyricsError} />}
              </PanelFooter>
            </Panel>
          )}

          {/* Formatted + Annotated */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(290px,1fr))", gap: 16 }}>

            <Panel>
              <PanelHeader left="✂️ Düzenlenmiş Sözler — [Verse]/[Chorus]" right={formattedLyrics ? <Mono color="rgba(200,170,255,0.4)">{formattedLyrics.split("\n").length} satır</Mono> : null} />
              <textarea
                value={formattedLyrics}
                onChange={(e) => { setFormattedLyrics(e.target.value); }}
                spellCheck={false}
                placeholder={"GPT satırları ve bölümleri burada düzenler.\n[Verse 1]\n...\n[Chorus]\n..."}
                style={textareaStyle(240)}
              />
              <PanelFooter>
                <PrimaryBtn onClick={handleManualAnnotate} disabled={!formattedLyrics.trim()}>
                  ⚡ Suno Notasyonu Ekle
                </PrimaryBtn>
              </PanelFooter>
            </Panel>

            <Panel>
              <PanelHeader
                left="✨ Suno-Uyumlu Çıktı"
                right={lines.length ? <Mono color="#4eff99">he-ce · ~ · ZWS bypass</Mono> : null}
              />
              <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px", minHeight: 240, maxHeight: 300 }}>
                {!lines.length ? (
                  <Placeholder text="Çıktı burada belirecek" hint="he-ce-le-me · ~ uzatma · ZWS copyright bypass" />
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    {lines.map((line, i) => {
                      const t = line.converted.trim();
                      if (!t) return <div key={i} style={{ height: 10 }} />;
                      if (/^\[.*\]$/.test(t)) return <div key={i} style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "#a78bfa", padding: "5px 0 2px" }}>{t}</div>;
                      return (
                        <div key={i} style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, lineHeight: 1.85, color: "rgba(235,220,255,0.85)", display: "flex", gap: 6, alignItems: "baseline" }}>
                          <span style={{ width: 4, height: 4, borderRadius: "50%", background: "#4eff99", flexShrink: 0, display: "inline-block", marginBottom: 2 }} />
                          <span>{renderAnnotated(line.converted)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              <PanelFooter>
                <GhostBtn disabled={!lines.length} onClick={async () => { await navigator.clipboard.writeText(pasteZone); }}>
                  ⇨ Kopyala
                </GhostBtn>
              </PanelFooter>
            </Panel>
          </div>

          {/* Paste zone */}
          <Panel>
            <PanelHeader left="📋 Suno'ya Yapıştırma Alanı" right={pasteZone ? <Mono color="#4eff99">Hazır</Mono> : null} />
            <textarea
              value={pasteZone}
              onChange={(e) => setPasteZone(e.target.value)}
              spellCheck={false}
              placeholder="Suno-uyumlu çıktı burada belirir. Buradan kopyalayıp Suno lyrics kutusuna yapıştır."
              style={textareaStyle(140)}
            />
          </Panel>

          {/* ═══ SECTION B: MIDI ═══ */}
          <SectionLabel>② MIDI — Müziği Bire Bir Koru, Suno Copyright'ı Atla</SectionLabel>

          <Panel>
            <PanelHeader
              left="🎹 MP3 → MIDI Dönüştür"
              right={midiAudioName ? <Mono color="#4eff99">📎 {midiAudioName}</Mono> : null}
            />
            <div style={{ padding: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 16 }}>

              {/* Drop zone */}
              <div>
                <DropZone
                  dragOver={midiDragOver}
                  onDragOver={(e) => { e.preventDefault(); setMidiDragOver(true); }}
                  onDragLeave={() => setMidiDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setMidiDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleMidiFile(f); }}
                  onClick={() => midiFileRef.current?.click()}
                  icon="🎹"
                  text="MP3 sürükle veya tıkla"
                  hint="MIDI indir → Suno'ya yükle"
                />
                <input ref={midiFileRef} type="file" accept="audio/*" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleMidiFile(f); }} />
                {midiStage === "processing" && <StatusBar msg="Pitch analizi yapılıyor, MIDI oluşturuluyor…" />}
                {midiStage === "done" && (
                  <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                    <SuccessBar msg="MIDI hazır! Aşağıdan WAV olarak da indir." />
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      <PrimaryBtn onClick={() => midiBlob && downloadBlob(midiBlob, `${midiAudioName.replace(/\.[^.]+$/,"")}_suno.mid`)}>
                        ⬇ MIDI İndir
                      </PrimaryBtn>
                      <PrimaryBtn onClick={handleMidiToWav} disabled={wavStage === "processing"}>
                        {wavStage === "processing" ? `⏳ WAV %${wavProgress}` : wavStage === "done" ? "✓ WAV İndirildi" : "⬇ WAV İndir (Suno için)"}
                      </PrimaryBtn>
                    </div>
                  </div>
                )}
                {midiError && <ErrorBar msg={midiError} />}
              </div>

              {/* Controls */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ fontFamily: "'Outfit',sans-serif", fontSize: 12, color: "rgba(200,170,255,0.7)", display: "block", marginBottom: 8 }}>
                    Transpozisyon: <span style={{ color: "#a78bfa", fontWeight: 700 }}>+{transpose} yarım ton</span>
                    <span style={{ color: "rgba(200,170,255,0.4)", marginLeft: 8 }}>— aynı duygu, farklı parmak izi</span>
                  </label>
                  <input type="range" min={0} max={6} value={transpose} onChange={(e) => setTranspose(+e.target.value)}
                    style={{ width: "100%", accentColor: "#a78bfa" }} />
                </div>
                <div>
                  <label style={{ fontFamily: "'Outfit',sans-serif", fontSize: 12, color: "rgba(200,170,255,0.7)", display: "block", marginBottom: 8 }}>
                    Tempo kayması: <span style={{ color: "#38bdf8", fontWeight: 700 }}>+{tempoShift}%</span>
                    <span style={{ color: "rgba(200,170,255,0.4)", marginLeft: 8 }}>— neredeyse duyulmaz</span>
                  </label>
                  <input type="range" min={0} max={10} value={tempoShift} onChange={(e) => setTempoShift(+e.target.value)}
                    style={{ width: "100%", accentColor: "#38bdf8" }} />
                </div>
                <div style={{ padding: "12px 14px", background: "rgba(78,255,153,0.05)", border: "1px solid rgba(78,255,153,0.15)", borderRadius: 8 }}>
                  {[
                    ["Pitch tespiti", "Otomatik frekans analizi"],
                    ["Transpozisyon", `+${transpose} yarım ton`],
                    ["Tempo", `+${tempoShift}% hızlandırma`],
                    ["Velocity", "Rastgele insanlaştırma"],
                  ].map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "rgba(200,170,255,0.5)" }}>{k}</span>
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "#4eff99" }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Panel>

          {/* ═══ SECTION C: COVER VOICE ═══ */}
          <SectionLabel>③ COVER SES — ElevenLabs ile Sözleri Farklı Sesle Söyle</SectionLabel>

          <Panel>
            <PanelHeader left="🎤 Cover Vokal Üret" right={<Mono color="rgba(200,170,255,0.4)">elevenlabs.io · ücretsiz 10k karakter/ay</Mono>} />
            <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 14 }}>

              {/* EL API Key */}
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="password"
                  value={elKey}
                  onChange={(e) => setElKey(e.target.value)}
                  placeholder="ElevenLabs API anahtarı — elevenlabs.io/app/settings/api-keys"
                  style={{ flex: 1, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 7, padding: "8px 12px", fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "rgba(235,220,255,0.9)", outline: "none" }}
                />
                <PrimaryBtn onClick={handleLoadVoices} disabled={!elKey.trim() || elStage === "loading"}>
                  {elStage === "loading" ? "Yükleniyor…" : "Sesleri Yükle"}
                </PrimaryBtn>
              </div>

              {/* Voice picker */}
              {elVoices.length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 10 }}>
                  <div>
                    <label style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11.5, color: "rgba(200,170,255,0.6)", display: "block", marginBottom: 6 }}>Ses Seç ({elVoices.length} ses)</label>
                    <select
                      value={elVoiceId}
                      onChange={(e) => setElVoiceId(e.target.value)}
                      style={{ width: "100%", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 7, padding: "8px 10px", color: "rgba(235,220,255,0.9)", fontFamily: "'Outfit',sans-serif", fontSize: 13, outline: "none" }}
                    >
                      {elVoices.map((v) => (
                        <option key={v.voice_id} value={v.voice_id} style={{ background: "#1a0a2e" }}>{v.name}</option>
                      ))}
                    </select>
                  </div>
                  <div style={{ padding: "10px 14px", background: "rgba(78,255,153,0.04)", border: "1px solid rgba(78,255,153,0.12)", borderRadius: 8, fontSize: 11.5, fontFamily: "'Outfit',sans-serif", color: "rgba(200,170,255,0.55)", lineHeight: 1.7 }}>
                    💡 <strong style={{ color: "rgba(200,170,255,0.8)" }}>İpucu:</strong> "Rachel", "Bella", "Antoni" gibi sesleri dene.
                    Türkçe için <strong style={{ color: "rgba(200,170,255,0.8)" }}>eleven_multilingual_v2</strong> modeli kullanılıyor.
                  </div>
                </div>
              )}

              {/* Lyrics text */}
              <div>
                <label style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11.5, color: "rgba(200,170,255,0.6)", display: "block", marginBottom: 6 }}>
                  Söylenecek Sözler
                  {pasteZone && <button onClick={() => setElText(formattedLyrics)} style={{ marginLeft: 10, background: "rgba(167,139,250,0.12)", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 4, padding: "1px 8px", color: "#a78bfa", fontSize: 10.5, cursor: "pointer", fontFamily: "'JetBrains Mono',monospace" }}>← Düzenlenmiş sözleri getir</button>}
                </label>
                <textarea
                  value={elText}
                  onChange={(e) => setElText(e.target.value)}
                  spellCheck={false}
                  placeholder="Buraya şarkı sözlerini yapıştır — seçilen ses onları okuyacak…"
                  style={{ ...textareaStyle(140), border: "1px solid rgba(255,255,255,0.08)", borderRadius: 7 }}
                />
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <PrimaryBtn onClick={handleTTS} disabled={!elVoiceId || !elText.trim() || elStage === "tts"}>
                  {elStage === "tts" ? "⏳ Ses Üretiliyor…" : "🎤 Cover Ses Üret & İndir"}
                </PrimaryBtn>
                {elStage === "done" && <span style={{ fontFamily: "'Outfit',sans-serif", fontSize: 12.5, color: "#4eff99" }}>✓ cover_vocals.mp3 indirildi!</span>}
                {elError && <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "#f87171" }}>⚠ {elError}</span>}
              </div>

              <div style={{ padding: "12px 14px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8 }}>
                <p style={{ fontFamily: "'Outfit',sans-serif", fontSize: 12, color: "rgba(200,170,255,0.5)", lineHeight: 1.7, margin: 0 }}>
                  <strong style={{ color: "rgba(200,170,255,0.75)" }}>Suno'da cover nasıl yapılır:</strong><br/>
                  1. Vokal MP3'ü indir → Audacity/GarageBand ile müziğin üstüne ekle<br/>
                  2. Ya da Suno'da "Upload Audio" → WAV müziği yükle → lyrics kutusuna sözleri yapıştır → Suno kendi sesini kullanır<br/>
                  3. En iyi sonuç: WAV müzik + ElevenLabs vokal → DAW'da birleştir
                </p>
              </div>

            </div>
          </Panel>

        </div>

        <footer style={{ marginTop: 28, fontFamily: "'Outfit',sans-serif", fontSize: 11, color: "rgba(200,170,255,0.25)", letterSpacing: "0.06em", textAlign: "center" }}>
          🎙️ ŞARKI SÖZÜ DÖNÜŞTÜRÜCÜ &nbsp;·&nbsp; HuBBez Production &nbsp;·&nbsp; 2026
        </footer>
      </div>
    </>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function renderAnnotated(text: string) {
  return text.split(/(~~|~|-)/g).map((p, i) => {
    if (p === "~~") return <span key={i} style={{ color: "#38bdf8", fontWeight: 700 }}>~~</span>;
    if (p === "~")  return <span key={i} style={{ color: "#4eff99", fontWeight: 700 }}>~</span>;
    if (p === "-")  return <span key={i} style={{ color: "#a78bfa" }}>-</span>;
    return <span key={i}>{p}</span>;
  });
}

const textareaStyle = (minH: number): React.CSSProperties => ({
  flex: 1, width: "100%", background: "transparent", border: "none", outline: "none",
  resize: "none", padding: "14px 16px", fontFamily: "'JetBrains Mono',monospace",
  fontSize: 12.5, lineHeight: 1.8, color: "rgba(235,220,255,0.88)", minHeight: minH,
});

// ─── UI atoms ─────────────────────────────────────────────────────────────────
function Panel({ children }: { children: React.ReactNode }) {
  return <div style={{ display: "flex", flexDirection: "column", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.09)", borderRadius: 12, backdropFilter: "blur(16px)", overflow: "hidden", boxShadow: "0 6px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)" }}>{children}</div>;
}

function PanelHeader({ left, right }: { left: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderBottom: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.03)" }}>
      <span style={{ fontFamily: "'Outfit',sans-serif", fontWeight: 700, fontSize: 13, color: "rgba(235,220,255,0.88)" }}>{left}</span>
      {right}
    </div>
  );
}

function PanelFooter({ children }: { children: React.ReactNode }) {
  return <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, padding: "10px 12px", borderTop: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.03)" }}>{children}</div>;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div style={{ fontFamily: "'Outfit',sans-serif", fontWeight: 800, fontSize: 11.5, letterSpacing: "0.14em", color: "rgba(200,170,255,0.45)", textTransform: "uppercase", paddingLeft: 2 }}>{children}</div>;
}

function Mono({ children, color }: { children: React.ReactNode; color: string }) {
  return <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color }}>{children}</span>;
}

function Placeholder({ text, hint }: { text: string; hint: string }) {
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, opacity: 0.3, padding: "30px 0" }}>
      <span style={{ fontSize: 30 }}>🎵</span>
      <p style={{ fontSize: 13, color: "rgba(200,170,255,0.8)", textAlign: "center" }}>{text}</p>
      <p style={{ fontSize: 11, color: "rgba(200,170,255,0.5)", textAlign: "center" }}>{hint}</p>
    </div>
  );
}

function StatusBar({ msg }: { msg: string }) {
  return (
    <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", background: "rgba(167,139,250,0.08)", border: "1px solid rgba(167,139,250,0.2)", borderRadius: 7 }}>
      <div style={{ width: 14, height: 14, border: "2px solid rgba(167,139,250,0.3)", borderTop: "2px solid #a78bfa", borderRadius: "50%", flexShrink: 0, animation: "spin 0.8s linear infinite" }} />
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      <span style={{ fontFamily: "'Outfit',sans-serif", fontSize: 12.5, color: "rgba(200,170,255,0.85)" }}>{msg}</span>
    </div>
  );
}

function SuccessBar({ msg }: { msg: string }) {
  return <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(78,255,153,0.07)", border: "1px solid rgba(78,255,153,0.2)", borderRadius: 7, fontFamily: "'Outfit',sans-serif", fontSize: 12.5, color: "#4eff99" }}>✓ {msg}</div>;
}

function ErrorBar({ msg }: { msg: string }) {
  return <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 7, fontFamily: "'JetBrains Mono',monospace", fontSize: 11.5, color: "#f87171" }}>⚠ {msg}</div>;
}

function DropZone({ dragOver, onDragOver, onDragLeave, onDrop, onClick, icon, text, hint }: {
  dragOver: boolean; onDragOver: React.DragEventHandler; onDragLeave: React.DragEventHandler;
  onDrop: React.DragEventHandler; onClick: () => void; icon: string; text: string; hint: string;
}) {
  return (
    <div onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop} onClick={onClick}
      style={{ border: `2px dashed ${dragOver ? "#a78bfa" : "rgba(255,255,255,0.14)"}`, borderRadius: 9, padding: "28px 16px", textAlign: "center", cursor: "pointer", transition: "all 0.2s", background: dragOver ? "rgba(167,139,250,0.06)" : "rgba(255,255,255,0.02)" }}>
      <div style={{ fontSize: 30, marginBottom: 10 }}>{icon}</div>
      <p style={{ fontFamily: "'Outfit',sans-serif", fontSize: 13.5, color: "rgba(235,220,255,0.78)", marginBottom: 5 }}>{text}</p>
      <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: "rgba(200,170,255,0.38)" }}>{hint}</p>
    </div>
  );
}

function GhostBtn({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled}
      onMouseEnter={(e) => { if (!disabled) { e.currentTarget.style.borderColor = "rgba(255,255,255,0.3)"; e.currentTarget.style.color = "#fff"; } }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.14)"; e.currentTarget.style.color = disabled ? "rgba(200,170,255,0.22)" : "rgba(200,170,255,0.72)"; }}
      style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 7, color: disabled ? "rgba(200,170,255,0.22)" : "rgba(200,170,255,0.72)", fontFamily: "'Outfit',sans-serif", fontSize: 12.5, fontWeight: 600, padding: "7px 16px", cursor: disabled ? "not-allowed" : "pointer", transition: "all 0.15s" }}>
      {children}
    </button>
  );
}

function PrimaryBtn({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled}
      onMouseEnter={(e) => { if (!disabled) { e.currentTarget.style.boxShadow = "0 4px 24px rgba(139,92,246,0.6)"; e.currentTarget.style.transform = "translateY(-1px)"; } }}
      onMouseLeave={(e) => { e.currentTarget.style.boxShadow = "0 3px 16px rgba(139,92,246,0.35)"; e.currentTarget.style.transform = "translateY(0)"; }}
      style={{ background: disabled ? "rgba(120,60,220,0.18)" : "linear-gradient(135deg,#7c3aed,#a855f7)", border: "none", borderRadius: 7, color: disabled ? "rgba(200,170,255,0.28)" : "#fff", fontFamily: "'Outfit',sans-serif", fontSize: 12.5, fontWeight: 700, padding: "7px 18px", cursor: disabled ? "not-allowed" : "pointer", transition: "all 0.15s", boxShadow: disabled ? "none" : "0 3px 16px rgba(139,92,246,0.35)" }}>
      {children}
    </button>
  );
}

function SmallBtn({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button onClick={onClick}
      style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 7, padding: "8px 12px", color: "rgba(200,170,255,0.65)", cursor: "pointer", fontFamily: "'JetBrains Mono',monospace", fontSize: 11.5 }}>
      {children}
    </button>
  );
}
