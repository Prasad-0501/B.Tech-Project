"""
SMARTGUARD: UNIFIED MULTI-MODAL EDGE AI SURVEILLANCE MISSION CONTROL
=====================================================================
Integrates:
  - Module 1: YOLOv8 Object Detection & ByteTrack Kalman Tracking
  - Module 2: Spatial Heatmap Projection & 2D ConvLSTM Spatiotemporal Behavior
  - Module 3: Google YAMNet Audio AI (AudioSet 521 Classes)
  - Module 4: Multi-Modal Fusion, Cross-Modal Synergy & SQLite Evidence Logger

Mission Control: http://127.0.0.1:8000
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import our verified modules
from modules.m1_yolo_tracker import ZeroLagVideoStream, YOLOTrackerEngine, CLASS_COLORS, safe_sample_path, list_samples
from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer, draw_module2_hud
from modules.m3_audio_ast import AudioStreamManager, AUDIO_SAMPLES_DIR
from modules.m4_fusion_engine import MultiModalFusionEngine, SNAPSHOTS_DIR

app = FastAPI(title="SmartGuard Multi-Modal Edge AI Surveillance")

# Mount snapshots directory to serve captured evidence images in UI
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

# Global subsystem state
STREAM: Optional[ZeroLagVideoStream] = None
TRACKER: Optional[YOLOTrackerEngine] = None
BEHAVIOR: YOLOConvLSTMBehaviorAnalyzer = YOLOConvLSTMBehaviorAnalyzer()
AUDIO_MGR: Optional[AudioStreamManager] = None
FUSION: MultiModalFusionEngine = MultiModalFusionEngine()

LATEST_FUSION_RESULT: Dict[str, Any] = {
    "threat_level": "NORMAL",
    "fused_score": 0.05,
    "visual_score": 0.05,
    "audio_score": 0.05,
    "behavior_score": 0.05,
    "synergy_boost": 0.0,
    "behavior_desc": "Normal Movement",
    "top_sound": "Silence",
    "audio_cat": "Normal Ambient",
}


@app.on_event("startup")
def on_startup():
    global STREAM, TRACKER, AUDIO_MGR
    print("=" * 70, flush=True)
    print("  [+] SMARTGUARD MULTI-MODAL EDGE AI SURVEILLANCE RUNNING", flush=True)
    print("  [+] Vision Engine   : YOLOv8n + ByteTrack (MS COCO)", flush=True)
    print("  [+] Behavior Engine : 2D ConvLSTM Spatiotemporal Heatmaps", flush=True)
    print("  [+] Acoustic AI     : Google YAMNet (AudioSet 521 Classes)", flush=True)
    print("  [+] Mission Control : http://127.0.0.1:8000", flush=True)
    print("=" * 70, flush=True)

    STREAM = ZeroLagVideoStream(source=0, width=640, height=480)
    TRACKER = YOLOTrackerEngine(conf_threshold=0.45, imgsz=416)
    TRACKER.start_async_worker(STREAM)
    AUDIO_MGR = AudioStreamManager()


@app.on_event("shutdown")
def on_shutdown():
    if TRACKER:
        TRACKER.stop_async_worker()
    if STREAM:
        STREAM.stop()
    if AUDIO_MGR:
        AUDIO_MGR.stop()


# ---------------------------------------------------------------------------
# MJPEG Video Feed with Full Visual Telemetry
# ---------------------------------------------------------------------------
def mjpeg_generator():
    global LATEST_FUSION_RESULT
    last_fid = -1
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 82]

    while True:
        if STREAM is None or TRACKER is None or AUDIO_MGR is None:
            time.sleep(0.02)
            continue

        pkt = STREAM.read()
        if pkt is None or pkt.frame_id == last_fid:
            time.sleep(0.005)
            continue
        last_fid = pkt.frame_id
        frame = pkt.image
        sfps = pkt.fps

        # 1. Vision Detections
        state = TRACKER.get_latest_state()
        raw_dets = state.detections if hasattr(state, "detections") else state[0]
        inf_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]

        # 2. Spatiotemporal Behavior (ConvLSTM)
        fw = frame.shape[1]
        fh = frame.shape[0]
        b_data = BEHAVIOR.update(raw_dets, frame_w=fw, frame_h=fh)

        # 3. Audio AI (Google YAMNet)
        a_data = AUDIO_MGR.get_state()

        # 4. Multi-Modal Fusion Engine
        fusion_res = FUSION.fuse(
            visual_data={"detections": raw_dets},
            behavior_data=b_data,
            audio_data=a_data,
        )
        fusion_res["detections"] = raw_dets
        LATEST_FUSION_RESULT = fusion_res

        # 5. Draw Visual HUD
        vis = draw_module2_hud(
            frame=frame,
            behavior_results=b_data,
            zones=BEHAVIOR.zones,
            source_label=STREAM.source_label,
            stream_fps=sfps,
            yolo_ms=inf_ms,
        )

        # 6. Automatic Snapshot Capture on Incident
        if fusion_res.get("should_record", False):
            snap_name = f"incident_{int(time.time())}.jpg"
            snap_path = SNAPSHOTS_DIR / snap_name
            cv2.imwrite(str(snap_path), vis)
            FUSION.log_incident_to_db(fusion_res, snapshot_path=f"/snapshots/{snap_name}")

        ret, jpg = cv2.imencode(".jpg", vis, encode_param)
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Telemetry & Control API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/telemetry")
def get_telemetry():
    if STREAM is None or TRACKER is None or AUDIO_MGR is None:
        return JSONResponse({"status": "initializing"})

    pkt = STREAM.read()
    sfps = pkt.fps if pkt is not None else 0.0
    state = TRACKER.get_latest_state()
    inf_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]
    raw_dets = state.detections if hasattr(state, "detections") else state[0]
    a_data = AUDIO_MGR.get_state()

    return {
        "source": STREAM.source_label,
        "audio_source": a_data.get("source", "Mic"),
        "stream_fps": sfps,
        "yolo_ms": inf_ms,
        "audio_db": a_data.get("db_level", -60.0),
        "top_sound": a_data.get("top_yamnet_class", "Silence"),
        "audio_category": a_data.get("surveillance_category", "Normal Ambient"),
        "audio_confidence": a_data.get("confidence", 0.98),
        "fused_threat": LATEST_FUSION_RESULT.get("threat_level", "NORMAL"),
        "fused_score": LATEST_FUSION_RESULT.get("fused_score", 0.05),
        "visual_score": LATEST_FUSION_RESULT.get("visual_score", 0.05),
        "audio_score": LATEST_FUSION_RESULT.get("audio_score", 0.05),
        "behavior_score": LATEST_FUSION_RESULT.get("behavior_score", 0.05),
        "synergy_boost": LATEST_FUSION_RESULT.get("synergy_boost", 0.0),
        "behavior_desc": LATEST_FUSION_RESULT.get("behavior_desc", "Normal Movement"),
        "detected_count": len(raw_dets),
        "detections": raw_dets,
    }


@app.get("/api/incidents")
def get_incidents():
    return FUSION.get_recent_incidents(limit=10)


@app.post("/api/switch_video")
def switch_video(src: str = Query(...)):
    if STREAM is None or TRACKER is None:
        return {"ok": False}
    TRACKER.reset_tracking()
    if src == "camera":
        ok = STREAM.open_source(0)
    else:
        target = BASE_DIR / "samples" / src
        ok = STREAM.open_source(str(target))
    return {"ok": ok, "source": STREAM.source_label}


@app.post("/api/switch_audio")
def switch_audio(src: str = Query(...)):
    if AUDIO_MGR is None:
        return {"ok": False}
    if src == "mic":
        ok = AUDIO_MGR.start_live_mic()
    else:
        target = AUDIO_SAMPLES_DIR / src
        ok = AUDIO_MGR.load_audio_file(target)
    return {"ok": ok, "source": AUDIO_MGR.current_source}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    samples_dir = BASE_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    dest = samples_dir / f"uploaded_{file.filename}"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    if STREAM and TRACKER:
        TRACKER.reset_tracking()
        STREAM.open_source(str(dest))
    return {"ok": True, "filename": dest.name}


# ---------------------------------------------------------------------------
# Unified Mission Control Web UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    samples_dir = BASE_DIR / "samples"
    videos = sorted([p.name for p in samples_dir.glob("*.mp4")]) if samples_dir.exists() else []
    vid_buttons = "".join(
        f'<button class="btn" onclick="switchVideo(\'{v}\')">🎬 {v[:22]}</button>'
        for v in videos
    )

    audio_files = sorted([p.name for p in AUDIO_SAMPLES_DIR.glob("*.wav")])
    audio_buttons = "".join(
        f'<button class="btn btn-audio" onclick="switchAudio(\'{a}\')">🔊 {a.replace(".wav","").replace("_"," ").title()[:18]}</button>'
        for a in audio_files
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SmartGuard — Multi-Modal Edge AI Surveillance</title>
<style>
  :root {{ --bg:#090d16; --card:#131a29; --card-header:#182235; --border:#223049; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; }}
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
  body {{ background:var(--bg); color:var(--text); padding:16px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; padding:14px 22px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; }}
  .header h1 {{ font-size:18px; color:var(--accent); display:flex; align-items:center; gap:8px; }}
  .badge {{ padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700; letter-spacing:0.5px; }}
  .badge-safe {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); }}
  .badge-warn {{ background:#3d2406; color:var(--warning); border:1px solid var(--warning); }}
  .badge-crit {{ background:#3b1111; color:var(--danger); border:1px solid var(--danger); animation:pulse 1s infinite; }}
  @keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.5}} 100%{{opacity:1}} }}
  .layout {{ display:grid; grid-template-columns: 2fr 1fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }}
  .card-title {{ font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:700; margin-bottom:12px; display:flex; justify-content:space-between; }}
  .video-box {{ width:100%; border-radius:8px; overflow:hidden; background:#000; border:1px solid var(--border); position:relative; }}
  .video-box img {{ width:100%; display:block; }}
  .controls-bar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }}
  .btn {{ background:#1e293b; color:#fff; border:1px solid #334155; padding:8px 12px; border-radius:6px; cursor:pointer; font-size:12px; transition:0.15s; }}
  .btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .btn-primary {{ background:#059669; border-color:#10b981; font-weight:600; }}
  .btn-upload {{ background:#2563eb; border-color:#3b82f6; font-weight:600; }}
  .btn-audio {{ background:#312e81; border-color:#4338ca; }}
  .meter-container {{ background:#0d131f; border:1px solid var(--border); border-radius:8px; padding:14px; margin-bottom:12px; }}
  .meter-bar {{ background:#1e293b; height:16px; border-radius:8px; overflow:hidden; margin-top:6px; }}
  .meter-fill {{ height:100%; width:0%; transition:width 0.2s; }}
  .fusion-dial {{ text-align:center; padding:14px; background:#0d131f; border:1px solid var(--border); border-radius:8px; margin-bottom:12px; }}
  .fusion-score {{ font-size:38px; font-weight:800; color:var(--accent); }}
  .grid-metrics {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; margin-bottom:12px; }}
  .metric-pill {{ background:#0d131f; border:1px solid var(--border); padding:10px; border-radius:6px; text-align:center; }}
  .metric-label {{ font-size:10px; color:#94a3b8; text-transform:uppercase; }}
  .metric-val {{ font-size:16px; font-weight:700; color:var(--accent); margin-top:3px; }}
  .incident-row {{ display:flex; gap:10px; padding:10px; border-bottom:1px solid var(--border); font-size:12px; align-items:center; }}
  .incident-snap {{ width:60px; height:42px; border-radius:4px; object-fit:cover; border:1px solid var(--border); background:#000; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🛡️ SMARTGUARD: Multi-Modal Edge AI Surveillance</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">YOLOv8 + ByteTrack • 2D ConvLSTM Spatiotemporal • Google YAMNet AudioSet • Cross-Modal Fusion</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL MONITORING</span>
  </div>

  <div class="layout">
    <!-- Left Column: Vision & Video Feed -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>Live Visual Stream (YOLOv8 + ConvLSTM Heatmap)</span>
          <span id="srcLabel" style="color:var(--accent);">Camera 0</span>
        </div>
        <div class="video-box">
          <img src="/video_feed" alt="Surveillance Video Feed" />
        </div>
        <div class="controls-bar">
          <button class="btn btn-primary" onclick="switchVideo('camera')">📷 Live Camera</button>
          {vid_buttons}
          <label class="btn btn-upload" style="cursor:pointer;">
            📂 Upload MP4 <input type="file" accept="video/mp4,video/*" style="display:none" onchange="uploadVid(this)">
          </label>
        </div>
      </div>

      <!-- Audio Anomaly Controls -->
      <div class="card">
        <div class="card-title">Acoustic Anomaly Channel (Google YAMNet AudioSet)</div>
        <div class="controls-bar" style="margin-top:0;">
          <button class="btn btn-primary" onclick="switchAudio('mic')">🎙️ Live Laptop Mic</button>
          {audio_buttons}
        </div>
        <div class="meter-container" style="margin-top:12px;margin-bottom:0;">
          <div style="display:flex;justify-content:space-between;font-size:12px;">
            <span>Microphone Sound Level: <b id="dbVal">-60 dB</b></span>
            <span>Acoustic AI: <b id="topSound" style="color:var(--accent);">Silence</b></span>
          </div>
          <div class="meter-bar">
            <div class="meter-fill" id="meterFill" style="background:linear-gradient(90deg, #00e5a3, #f59e0b, #ef4444);"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Decision Fusion & Incidents -->
    <div>
      <!-- Fusion Dial -->
      <div class="card">
        <div class="card-title">Multi-Modal Threat Fusion Engine</div>
        <div class="fusion-dial">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Fused Decision Score (S<sub>fused</sub>)</div>
          <div class="fusion-score" id="fusedScore">0.05</div>
          <div id="threatStatusText" style="font-size:13px;font-weight:700;color:var(--accent);">NORMAL MONITORING</div>
        </div>
        <div class="grid-metrics">
          <div class="metric-pill"><div class="metric-label">Visual (YOLO)</div><div class="metric-val" id="mVis">0.05</div></div>
          <div class="metric-pill"><div class="metric-label">Acoustic (YAMNet)</div><div class="metric-val" id="mAud">0.05</div></div>
          <div class="metric-pill"><div class="metric-label">Behavior (LSTM)</div><div class="metric-val" id="mBeh">0.05</div></div>
        </div>
        <div style="font-size:11px;color:#94a3b8;background:#0d131f;padding:8px;border-radius:6px;border:1px solid var(--border);">
          <b>Cross-Modal Synergy:</b> <span id="mSyn" style="color:var(--accent);">+0.00</span> | 
          <b>Behavior:</b> <span id="mDesc" style="color:#e2e8f0;">Normal Movement</span>
        </div>
      </div>

      <!-- Real-time Incident Gallery -->
      <div class="card" style="margin-bottom:0;">
        <div class="card-title">Live Security Incident Log (SQLite + Evidence)</div>
        <div id="incidentList" style="max-height:360px;overflow-y:auto;">
          <div style="color:#64748b;font-size:12px;padding:12px;text-align:center;">No incidents recorded yet. System secure.</div>
        </div>
      </div>
    </div>
  </div>

<script>
async function switchVideo(src) {{
  await fetch('/api/switch_video?src=' + encodeURIComponent(src), {{method:'POST'}});
}}
async function switchAudio(src) {{
  await fetch('/api/switch_audio?src=' + encodeURIComponent(src), {{method:'POST'}});
}}
async function uploadVid(input) {{
  if (!input.files || !input.files.length) return;
  const fd = new FormData();
  fd.append('file', input.files[0]);
  await fetch('/api/upload_video', {{method:'POST', body:fd}});
  setTimeout(() => location.reload(), 600);
}}

setInterval(async () => {{
  try {{
    // 1. Fetch live telemetry
    const r = await fetch('/api/telemetry');
    const d = await r.json();
    if (!d.source) return;

    document.getElementById('srcLabel').innerText = d.source;
    document.getElementById('dbVal').innerText = d.audio_db + ' dB';
    document.getElementById('topSound').innerText = d.top_sound + ' (' + d.audio_category + ')';

    // Mic meter
    const pct = Math.max(0, Math.min(100, ((d.audio_db + 80) / 80) * 100));
    document.getElementById('meterFill').style.width = pct + '%';

    // Fusion metrics
    document.getElementById('fusedScore').innerText = d.fused_score.toFixed(2);
    document.getElementById('mVis').innerText = d.visual_score.toFixed(2);
    document.getElementById('mAud').innerText = d.audio_score.toFixed(2);
    document.getElementById('mBeh').innerText = d.behavior_score.toFixed(2);
    document.getElementById('mSyn').innerText = '+' + d.synergy_boost.toFixed(2);
    document.getElementById('mDesc').innerText = d.behavior_desc;

    const tb = document.getElementById('threatBadge');
    const fScore = document.getElementById('fusedScore');
    const tText = document.getElementById('threatStatusText');

    if (d.fused_threat === 'CRITICAL') {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 CRITICAL SECURITY BREACH';
      fScore.style.color = '#ef4444';
      tText.style.color = '#ef4444';
      tText.innerText = '🚨 CRITICAL BREACH - DISPATCH';
    }} else if (d.fused_threat === 'WARNING') {{
      tb.className = 'badge badge-warn';
      tb.innerText = '⚠️ ELEVATED SUSPICION';
      fScore.style.color = '#f59e0b';
      tText.style.color = '#f59e0b';
      tText.innerText = '⚠️ ELEVATED SUSPICION';
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = 'NORMAL MONITORING';
      fScore.style.color = '#00e5a3';
      tText.style.color = '#00e5a3';
      tText.innerText = 'NORMAL MONITORING';
    }}
  }} catch(e) {{}}
}}, 250);

// Poll incidents every 2 seconds
setInterval(async () => {{
  try {{
    const res = await fetch('/api/incidents');
    const list = await res.json();
    const box = document.getElementById('incidentList');
    if (!list || list.length === 0) {{
      box.innerHTML = '<div style="color:#64748b;font-size:12px;padding:12px;text-align:center;">No incidents recorded yet. System secure.</div>';
      return;
    }}
    box.innerHTML = list.map(inc => `
      <div class="incident-row">
        ${{inc.snapshot_path ? `<a href="${{inc.snapshot_path}}" target="_blank"><img class="incident-snap" src="${{inc.snapshot_path}}" alt="Snapshot"></a>` : ''}}
        <div style="flex:1;">
          <div style="display:flex;justify-content:space-between;font-weight:700;">
            <span style="color:${{inc.threat_level==='CRITICAL'?'#ef4444':'#f59e0b'}};">${{inc.threat_level}} (${{inc.fused_score.toFixed(2)}})</span>
            <span style="color:#64748b;font-size:11px;">${{inc.timestamp.split(' ')[1]}}</span>
          </div>
          <div style="font-size:11px;color:#cbd5e1;margin-top:2px;">${{inc.llm_summary}}</div>
        </div>
      </div>
    `).join('');
  }} catch(e) {{}}
}}, 2000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
