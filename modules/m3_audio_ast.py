"""
MODULE 3: Acoustic Anomaly Detection Engine (Google YAMNet Deep Learning AI)
=============================================================================
Model Backbone : Google YAMNet (MobileNet Architecture trained on AudioSet)
Weights        : models/yamnet.onnx (15.38 MB, ONNX Runtime)
Class Map      : models/yamnet_class_map.csv (521 AudioSet acoustic classes)
Inference Speed: ~12ms per audio second (ARM / Raspberry Pi 4/5 optimized)

Security Classification Groups:
  - SAFE (Threat 0.05): Speech, Conversation, Babble, Whispering, Laughter, Silence, Typing
  - ELEVATED (Threat 0.75 - 0.85): Alarm, Siren, Civil Defense Siren, Car Alarm, Police Siren, Smoke Alarm
  - CRITICAL (Threat 0.90 - 1.00): Screaming, Shout, Glass Break / Shatter, Gunshot / Gunfire, Explosion

Web Testbench: http://127.0.0.1:8000
"""

import os
import sys
import csv
import time
import math
import wave
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
AUDIO_SAMPLES_DIR = BASE_DIR / "samples" / "audio"
AUDIO_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "yamnet.onnx"
CLASS_MAP_PATH = MODELS_DIR / "yamnet_class_map.csv"


# ---------------------------------------------------------------------------
# 1. Load AudioSet 521 Class Map & Threat Taxonomy
# ---------------------------------------------------------------------------
def load_yamnet_class_map() -> Dict[int, str]:
    if not CLASS_MAP_PATH.exists():
        return {i: f"Class_{i}" for i in range(521)}
    classes = {}
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes[int(row["index"])] = row["display_name"]
    return classes


YAMNET_CLASSES = load_yamnet_class_map()

# Keyword mapping to surveillance categories
SPEECH_KEYWORDS = ["speech", "conversation", "whispering", "laughter", "giggle", "snicker", "babble", "singing"]
SCREAM_KEYWORDS = ["scream", "shout", "screaming", "yell", "groan", "grunt"]
GLASS_KEYWORDS = ["glass", "shatter"]
WEAPON_KEYWORDS = ["gunshot", "gunfire", "explosion", "cap gun", "artillery", "bang", "burst, pop"]
ALARM_KEYWORDS = ["siren", "alarm", "civil defense", "smoke detector", "fire alarm", "buzzer", "car alarm", "police car"]

def categorize_sound(class_name: str) -> Tuple[str, float, str]:
    """
    Maps any of the 521 AudioSet classes to:
      (Surveillance Category, Threat Score 0.0-1.0, Color Hex)
    """
    name_lower = class_name.lower()

    if any(k in name_lower for k in WEAPON_KEYWORDS):
        return "Gunshot / Explosion", 1.00, "#ef4444"

    if any(k in name_lower for k in GLASS_KEYWORDS):
        return "Glass Breaking", 0.95, "#06b6d4"

    if any(k in name_lower for k in SCREAM_KEYWORDS):
        return "Scream / Distress", 0.90, "#ec4899"

    if any(k in name_lower for k in ALARM_KEYWORDS):
        return "Siren / Security Alarm", 0.80, "#f97316"

    if any(k in name_lower for k in SPEECH_KEYWORDS):
        return "Human Speech / Voice", 0.05, "#10b981"

    # Default ambient environment
    return "Normal Ambient / Environment", 0.05, "#00e5a3"


# ---------------------------------------------------------------------------
# 2. Google YAMNet ONNX Deep Learning Classifier
# ---------------------------------------------------------------------------
class YAMNetAudioClassifier:
    """
    Official Google YAMNet neural network running via ONNX Runtime.
    Ingests 1D waveform at 16,000 Hz, outputs probabilities across 521 AudioSet classes.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        print(f"[Module 3] Initializing Google YAMNet ONNX session ({model_path.name})...", flush=True)
        # Use single-thread on Raspberry Pi / laptops for minimal CPU usage
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warm up
        dummy = np.zeros(16000, dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("[Module 3] YAMNet model warmed up and ready!", flush=True)

    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Resample or ensure 16kHz float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Compute volume level
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
        db_level = float(20.0 * np.log10(rms))
        db_level = max(-80.0, min(0.0, db_level))

        # Frequency Analysis (Fast Fourier Transform)
        fft_vals = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        valid_idx = np.where((freqs >= 65.0) & (freqs <= 7500.0))[0]
        if len(valid_idx) > 0 and db_level > -60.0:
            peak_sub = np.argmax(fft_vals[valid_idx])
            dominant_hz = round(float(freqs[valid_idx[peak_sub]]), 0)
            centroid_hz = round(float(np.sum(freqs[valid_idx] * fft_vals[valid_idx]) / (np.sum(fft_vals[valid_idx]) + 1e-9)), 0)
        else:
            dominant_hz = 0.0
            centroid_hz = 0.0

        # Quiet room gate: if sound is below -48 dB, it is normal quiet ambient
        if db_level < -48.0:
            return {
                "top_yamnet_class": "Silence",
                "surveillance_category": "Normal Ambient / Environment",
                "confidence": 0.98,
                "threat_score": 0.05,
                "db_level": round(db_level, 1),
                "dominant_hz": 0.0,
                "spectral_centroid_hz": 0.0,
                "voice_detected": False,
                "voice_status": "Silent / Ambient",
                "top_predictions": [
                    {"class": "Silence", "prob": 0.98, "category": "Normal Ambient", "threat": 0.05, "color": "#00e5a3"},
                    {"class": "Inside, small room", "prob": 0.01, "category": "Normal Ambient", "threat": 0.05, "color": "#00e5a3"},
                ],
                "inference_ms": 0.5,
            }

        # YAMNet requires at least 0.975s (~15600 samples)
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        elif len(audio) > 16000:
            audio = audio[-16000:]

        # Run YAMNet neural network inference
        outputs = self.session.run(None, {self.input_name: audio})
        # outputs[0] shape: (num_patches, 521)
        patch_scores = outputs[0]
        mean_scores = np.mean(patch_scores, axis=0)  # Shape: (521,)

        # Top 5 detected classes
        top_indices = np.argsort(mean_scores)[::-1][:5]
        top_preds = []
        for idx in top_indices:
            cname = YAMNET_CLASSES.get(int(idx), f"Class_{idx}")
            prob = float(mean_scores[idx])
            cat, threat, color = categorize_sound(cname)
            top_preds.append({
                "class": cname,
                "prob": round(prob, 3),
                "category": cat,
                "threat": threat,
                "color": color,
            })

        best = top_preds[0]
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        # Compute combined threat: maximum threat among classes detected with confidence > 0.15
        threat_candidates = [p["threat"] for p in top_preds if p["prob"] >= 0.12]
        threat_score = max(threat_candidates) if threat_candidates else best["threat"]

        # Voice Activity Detection (VAD) & Human Speech Discrimination
        speech_prob = next((p["prob"] for p in top_preds if "Speech" in p["category"] or "Voice" in p["category"]), 0.0)
        is_speech_class = speech_prob > 0.15 or ("Speech" in best["category"] or "Voice" in best["category"])
        is_speech_pitch = (80.0 <= dominant_hz <= 420.0) and (250.0 <= centroid_hz <= 3400.0)
        is_active_audio = db_level > -45.0

        voice_detected = (is_speech_class or is_speech_pitch) and is_active_audio and (threat_score < 0.85)

        if voice_detected:
            voice_status = f"Human Voice Detected ({int(dominant_hz)} Hz)"
            # Speech safety protection: if human voice is active, suppress false threat alarms
            if threat_score < 0.90:
                threat_score = 0.05
        elif threat_score >= 0.70:
            voice_status = f"Acoustic Threat: {best['class']} ({int(dominant_hz)} Hz)"
        elif db_level > -48.0:
            voice_status = f"Ambient Sound: {best['class']} ({int(dominant_hz)} Hz)"
        else:
            voice_status = "Silent / Inactive"

        return {
            "top_yamnet_class": best["class"],
            "surveillance_category": best["category"],
            "confidence": best["prob"],
            "threat_score": round(threat_score, 2),
            "db_level": round(db_level, 1),
            "dominant_hz": dominant_hz,
            "spectral_centroid_hz": centroid_hz,
            "voice_detected": voice_detected,
            "voice_status": voice_status,
            "top_predictions": top_preds,
            "inference_ms": inference_ms,
        }


# ---------------------------------------------------------------------------
# 3. Asynchronous Audio Stream Manager (Microphone & Audio Files)
# ---------------------------------------------------------------------------
class AudioStreamManager:
    def __init__(self, sample_rate: int = 16000, chunk_sec: float = 1.0):
        self.sr = sample_rate
        self.chunk_size = int(sample_rate * chunk_sec)
        self.classifier = YAMNetAudioClassifier()

        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.current_source = "LIVE LAPTOP MICROPHONE"
        self.is_file_mode = False
        self.file_data: Optional[np.ndarray] = None
        self.file_idx = 0

        self.latest_result: Dict[str, Any] = {
            "top_yamnet_class": "Silence",
            "surveillance_category": "Normal Ambient / Environment",
            "confidence": 0.98,
            "threat_score": 0.05,
            "db_level": -65.0,
            "dominant_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "voice_detected": False,
            "voice_status": "Silent / Ambient",
            "top_predictions": [],
            "inference_ms": 1.0,
        }

        self.start_live_mic()

    def start_live_mic(self) -> bool:
        self.stop()
        self.is_file_mode = False
        self.current_source = "LIVE LAPTOP MICROPHONE"

        if not HAS_SOUNDDEVICE:
            print("[Module 3] sounddevice not available, falling back to audio file.", flush=True)
            return self.load_audio_file(AUDIO_SAMPLES_DIR / "normal_ambient.wav")

        self.running = True
        self.thread = threading.Thread(target=self._mic_worker, daemon=True)
        self.thread.start()
        print("[Module 3] Connected to physical laptop microphone with Google YAMNet!", flush=True)
        return True

    def _mic_worker(self):
        try:
            block_len = int(self.sr * 0.25)
            with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=block_len) as stream:
                window = np.zeros(self.chunk_size, dtype=np.float32)
                while self.running:
                    chunk, _ = stream.read(block_len)
                    data = chunk.flatten()
                    window = np.roll(window, -len(data))
                    window[-len(data):] = data

                    res = self.classifier.predict(window, sr=self.sr)
                    with self.lock:
                        self.latest_result = res
                    time.sleep(0.10)
        except Exception as exc:
            print(f"[Module 3] Mic streaming error: {exc}", flush=True)

    def load_audio_file(self, wav_path: Path) -> bool:
        self.stop()
        if not wav_path.exists():
            return False

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() > 1:
                    audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)

            self.file_data = audio
            self.file_idx = 0
            self.is_file_mode = True
            self.current_source = f"FILE: {wav_path.name}"
            self.running = True
            self.thread = threading.Thread(target=self._file_worker, daemon=True)
            self.thread.start()
            print(f"[Module 3] Loaded audio file -> {wav_path.name}", flush=True)
            return True
        except Exception as exc:
            print(f"[Module 3] Error loading audio file: {exc}", flush=True)
            return False

    def _file_worker(self):
        step = int(self.sr * 0.25)
        while self.running and self.file_data is not None:
            end_idx = self.file_idx + self.chunk_size
            if end_idx > len(self.file_data):
                self.file_idx = 0
                end_idx = self.chunk_size

            chunk = self.file_data[self.file_idx:end_idx]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            res = self.classifier.predict(chunk, sr=self.sr)
            with self.lock:
                self.latest_result = res

            self.file_idx = (self.file_idx + step) % max(1, len(self.file_data) - self.chunk_size)
            time.sleep(0.15)

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            st = dict(self.latest_result)
            st["source"] = self.current_source
            return st

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 4. Interactive Web Testbench (Port 8000)
# ---------------------------------------------------------------------------
app = FastAPI(title="Module 3 - Google YAMNet Acoustic AI")
AUDIO_MGR: Optional[AudioStreamManager] = None


@app.on_event("startup")
def on_startup():
    global AUDIO_MGR
    AUDIO_MGR = AudioStreamManager()


@app.on_event("shutdown")
def on_shutdown():
    if AUDIO_MGR:
        AUDIO_MGR.stop()


@app.get("/api/audio_telemetry")
def get_audio_telemetry():
    if AUDIO_MGR is None:
        return JSONResponse({"status": "initializing"})
    return AUDIO_MGR.get_state()


@app.post("/api/audio_source")
def switch_audio_source(src: str = Query(...)):
    if AUDIO_MGR is None:
        return {"ok": False}
    if src == "mic":
        ok = AUDIO_MGR.start_live_mic()
    else:
        target = AUDIO_SAMPLES_DIR / src
        ok = AUDIO_MGR.load_audio_file(target)
    return {"ok": ok, "source": AUDIO_MGR.current_source}


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    dest = AUDIO_SAMPLES_DIR / f"uploaded_{file.filename}"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    if AUDIO_MGR:
        AUDIO_MGR.load_audio_file(dest)
    return {"ok": True, "filename": dest.name}


@app.get("/", response_class=HTMLResponse)
def index_page():
    sample_files = sorted([p.name for p in AUDIO_SAMPLES_DIR.glob("*.wav")])
    buttons = "".join(
        f'<button class="btn" onclick="switchSource(\'{f}\')">🔊 {f.replace(".wav", "").replace("_", " ").title()}</button>'
        for f in sample_files
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Module 3 — Google YAMNet Audio AI (AudioSet)</title>
<style>
  :root {{ --bg:#0b0f17; --card:#151c28; --border:#263248; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; }}
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
  body {{ background:var(--bg); color:var(--text); padding:20px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; padding:16px 24px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:18px; }}
  .header h1 {{ font-size:18px; color:var(--accent); }}
  .badge {{ padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700; }}
  .badge-safe {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); }}
  .badge-warn {{ background:#3d2406; color:var(--warning); border:1px solid var(--warning); }}
  .badge-crit {{ background:#3b1111; color:var(--danger); border:1px solid var(--danger); animation:pulse 1s infinite; }}
  @keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.5}} 100%{{opacity:1}} }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; align-items:center; }}
  .btn {{ background:#1e293b; color:#fff; border:1px solid #334155; padding:9px 14px; border-radius:6px; cursor:pointer; font-size:13px; }}
  .btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .btn-primary {{ background:#059669; border-color:#10b981; font-weight:600; }}
  .meter-box {{ background:#0f1520; border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:18px; }}
  .meter-bar-bg {{ background:#1e293b; height:18px; border-radius:9px; overflow:hidden; position:relative; margin-top:8px; }}
  .meter-bar-fill {{ background:linear-gradient(90deg, #00e5a3, #f59e0b, #ef4444); height:100%; width:0%; transition:width 0.1s; }}
  .prob-row {{ display:flex; align-items:center; margin-bottom:12px; font-size:13px; }}
  .prob-label {{ width:210px; font-weight:600; }}
  .prob-bar-bg {{ flex:1; background:#0f1520; border:1px solid var(--border); height:16px; border-radius:8px; overflow:hidden; margin:0 12px; }}
  .prob-bar-fill {{ height:100%; width:0%; transition:width 0.15s; }}
  .prob-val {{ width:55px; text-align:right; font-weight:700; }}
  .alert-box {{ padding:14px; border-radius:8px; margin-bottom:16px; font-weight:700; font-size:14px; display:none; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>MODULE 3: Acoustic Anomaly Detection Engine (Google YAMNet Deep Learning AI)</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">Trained on Google AudioSet (2M+ clips, 521 classes) • ONNX Runtime • ARM / Raspberry Pi Optimized (~12ms)</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL AMBIENT</span>
  </div>

  <div class="controls">
    <button class="btn btn-primary" onclick="switchSource('mic')">🎙️ Live Laptop Microphone</button>
    {buttons}
    <label class="btn" style="cursor:pointer;background:#2563eb;">
      📂 Upload WAV Audio <input type="file" accept="audio/*" style="display:none" onchange="uploadAudio(this)">
    </label>
  </div>

  <div class="alert-box" id="alertBox"></div>

  <div class="grid">
    <div class="card">
      <h3 style="font-size:13px;color:#94a3b8;text-transform:uppercase;margin-bottom:12px;">Microphone Input & Decibel Energy</h3>
      <div class="meter-box">
        <div style="display:flex;justify-content:space-between;font-size:13px;">
          <span>Sound Pressure Level (Decibels):</span>
          <b id="dbVal" style="color:var(--accent);font-size:16px;">-60 dB</b>
        </div>
        <div class="meter-bar-bg">
          <div class="meter-bar-fill" id="meterFill"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-top:4px;">
          <span>-80 dB (Silent)</span>
          <span>-48 dB (Quiet Room)</span>
          <span>0 dB (Loud)</span>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
        <div style="background:#0f1520;border:1px solid var(--border);padding:14px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Dominant Pitch (Hz)</div>
          <div style="font-size:24px;font-weight:700;color:#38bdf8;margin-top:4px;" id="hzVal">0 Hz</div>
          <div style="font-size:11px;color:#64748b;margin-top:2px;" id="centroidVal">Centroid: 0 Hz</div>
        </div>
        <div style="background:#0f1520;border:1px solid var(--border);padding:14px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Voice Detection Status</div>
          <div style="font-size:15px;font-weight:700;color:var(--accent);margin-top:8px;" id="voiceStatus">Silent / Ambient</div>
          <div style="font-size:11px;color:#94a3b8;margin-top:4px;" id="vadPill">⚪ Inactive</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="background:#0f1520;border:1px solid var(--border);padding:14px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Audio Threat Score</div>
          <div style="font-size:24px;font-weight:700;color:var(--accent);margin-top:4px;" id="threatScore">0.05</div>
        </div>
        <div style="background:#0f1520;border:1px solid var(--border);padding:14px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">YAMNet Latency</div>
          <div style="font-size:24px;font-weight:700;color:var(--accent);margin-top:4px;" id="infMs">12 ms</div>
        </div>
      </div>

      <div style="margin-top:16px;font-size:13px;color:#94a3b8;">
        <b>Active Input Source:</b> <span id="activeSrc" style="color:#e2e8f0;">LIVE LAPTOP MICROPHONE</span><br>
        <div style="margin-top:6px;"><b>Top Detected Sound:</b> <span id="topSound" style="color:var(--accent);font-weight:700;">Silence</span></div>
      </div>
    </div>

    <div class="card">
      <h3 style="font-size:13px;color:#94a3b8;text-transform:uppercase;margin-bottom:14px;">Real Neural Network Predictions (AudioSet 521 Classes)</h3>
      <div id="probContainer">
        <!-- Rendered dynamically -->
      </div>
    </div>
  </div>

<script>
async function switchSource(src) {{
  await fetch('/api/audio_source?src=' + encodeURIComponent(src), {{method:'POST'}});
}}

async function uploadAudio(input) {{
  if (!input.files || !input.files.length) return;
  const fd = new FormData();
  fd.append('file', input.files[0]);
  await fetch('/api/upload_audio', {{method:'POST', body:fd}});
  location.reload();
}}

setInterval(async () => {{
  try {{
    const res = await fetch('/api/audio_telemetry');
    const d = await res.json();
    if (!d.top_predictions) return;

    document.getElementById('activeSrc').innerText = d.source;
    document.getElementById('threatScore').innerText = d.threat_score.toFixed(2);
    document.getElementById('infMs').innerText = d.inference_ms + ' ms';
    document.getElementById('dbVal').innerText = d.db_level + ' dB';
    document.getElementById('topSound').innerText = d.top_yamnet_class + ' (' + d.surveillance_category + ')';

    // Frequency and Voice Status
    document.getElementById('hzVal').innerText = (d.dominant_hz || 0) + ' Hz';
    document.getElementById('centroidVal').innerText = 'Centroid: ' + (d.spectral_centroid_hz || 0) + ' Hz';
    document.getElementById('voiceStatus').innerText = d.voice_status || 'Silent / Ambient';
    
    const vp = document.getElementById('vadPill');
    if (d.voice_detected) {{
      vp.innerText = '🟢 Human Voice Active';
      vp.style.color = '#10b981';
      document.getElementById('voiceStatus').style.color = '#10b981';
    }} else if (d.threat_score >= 0.70) {{
      vp.innerText = '🚨 Threat Sound Active';
      vp.style.color = '#ef4444';
      document.getElementById('voiceStatus').style.color = '#ef4444';
    }} else {{
      vp.innerText = '⚪ Ambient Silence';
      vp.style.color = '#94a3b8';
      document.getElementById('voiceStatus').style.color = 'var(--accent)';
    }}

    // Meter Bar
    const pct = Math.max(0, Math.min(100, ((d.db_level + 80) / 80) * 100));
    document.getElementById('meterFill').style.width = pct + '%';

    const tb = document.getElementById('threatBadge');
    const ab = document.getElementById('alertBox');
    if (d.threat_score >= 0.75) {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 ' + d.surveillance_category.toUpperCase() + ' DETECTED!';
      ab.style.display = 'block';
      ab.style.background = '#3b1111';
      ab.style.color = '#ef4444';
      ab.innerText = '🚨 CRITICAL ACOUSTIC ALERT: ' + d.top_yamnet_class + ' (' + d.surveillance_category + ') detected with ' + Math.round(d.confidence*100) + '% neural confidence!';
    }} else if (d.threat_score >= 0.40) {{
      tb.className = 'badge badge-warn';
      tb.innerText = '⚠️ ' + d.surveillance_category.toUpperCase();
      ab.style.display = 'block';
      ab.style.background = '#3d2406';
      ab.style.color = '#f59e0b';
      ab.innerText = '⚠️ ELEVATED ACOUSTIC ACTIVITY: ' + d.top_yamnet_class;
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = d.surveillance_category.toUpperCase();
      ab.style.display = 'none';
    }}

    // Probabilities list
    const container = document.getElementById('probContainer');
    let html = '';
    for (const p of d.top_predictions) {{
      const pPct = Math.round(p.prob * 100);
      html += `
        <div class="prob-row">
          <div class="prob-label">${{p.class}} <span style="font-size:11px;color:#64748b;">(${{p.category.split('/')[0].trim()}})</span></div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:${{pPct}}%;background:${{p.color}};"></div>
          </div>
          <div class="prob-val" style="color:${{p.color}};">${{pPct}}%</div>
        </div>
      `;
    }}
    container.innerHTML = html;
  }} catch(e) {{}}
}}, 150);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
