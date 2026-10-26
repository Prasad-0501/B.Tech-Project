"""
MODULE 5: LLM Cognitive Dispatcher & Human-Like Security Reasoner
===================================================================
Architecture:
  - Ingests:
      1. Structured JSON Telemetry (Module 1 YOLO + Module 2 ConvLSTM + Module 3 YAMNet)
      2. Multi-Modal Fused Threat Score (0.05 - 1.00) & Rule from Module 4
  - On-Device Cognitive Core:
      - Uses distilled local LLM: Qwen/Qwen2.5-0.5B-Instruct
      - Evaluates multi-modal correlation with human-like contextual reasoning
      - Determines proportional tactical security action & creates operator dispatch brief
  - Performance & Edge Decoupling:
      - 30 FPS camera generator runs independently (zero LLM blocking)
      - LLM inference runs asynchronously in an event-driven background thread
      - Fallback guarantee: deterministic fallback ensures invariants are always maintained

Web Mission Control: http://127.0.0.1:8000
"""

import os
import sys
import time
import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import lower-level verified modules
from modules.m1_yolo_tracker import ZeroLagVideoStream, YOLOTrackerEngine, CLASS_COLORS, change_source
from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer, draw_module2_hud
from modules.m3_audio_ast import AudioStreamManager, AUDIO_SAMPLES_DIR
from modules.m4_rule_fusion import RuleBasedFusionEngine

DB_PATH = BASE_DIR / "database" / "surveillance_events.db"
SNAPSHOTS_DIR = BASE_DIR / "database" / "snapshots"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Database Initialization
# ---------------------------------------------------------------------------
def init_database():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            threat_level TEXT NOT NULL,
            fused_score REAL NOT NULL,
            trust_score REAL NOT NULL,
            action_executed TEXT NOT NULL,
            fallback_triggered BOOLEAN NOT NULL,
            latency_ms REAL NOT NULL,
            structured_telemetry TEXT NOT NULL,
            llm_reasoning TEXT NOT NULL,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    conn.close()


init_database()


# ---------------------------------------------------------------------------
# 2. LLM Cognitive Dispatcher (Human-Like Reasoning Core)
# ---------------------------------------------------------------------------
class LLMCognitiveDispatcher:
    """
    On-device cognitive reasoner that evaluates structured sensor JSON telemetry
    and threat scores to make human-like situational decisions.
    """

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        print(f"[Module 5] Loading On-Device Cognitive LLM: {self.MODEL_ID}...", flush=True)
        try:
            torch.set_num_threads(2)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.is_ready = True
            print("[Module 5] Cognitive LLM loaded in memory and ready for reasoning!", flush=True)
        except Exception as exc:
            print(f"[Module 5] Notice: LLM offline ({exc}), using deterministic cognitive rules.", flush=True)
            self.is_ready = False

    def reason(self, telemetry: Dict[str, Any], threat_score: float, rule_id: str) -> Tuple[Dict[str, Any], float]:
        """
        Executes human-like cognitive reasoning over sensory JSON and threat score.
        Returns: (decision_dict, inference_latency_ms)
        """
        t0 = time.perf_counter()

        if not self.is_ready or self.model is None or self.tokenizer is None:
            res = self._deterministic_fallback(telemetry, threat_score, rule_id)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return res, latency_ms

        with self.lock:
            try:
                res = self._run_neural_reasoning(telemetry, threat_score, rule_id)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return res, latency_ms
            except Exception as e:
                print(f"[Module 5] Neural inference error: {e}", flush=True)
                res = self._deterministic_fallback(telemetry, threat_score, rule_id)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return res, latency_ms

    def _run_neural_reasoning(self, telemetry: Dict[str, Any], threat_score: float, rule_id: str) -> Dict[str, Any]:
        prompt = (
            f"You are the Chief Security Dispatcher AI. Evaluate this live surveillance sensory report:\n"
            f"Sensory Telemetry JSON: {json.dumps(telemetry)}\n"
            f"Fused Threat Score: {threat_score:.2f} | Pre-Rule: {rule_id}\n\n"
            f"Analyze the sensory correlation and generate human-like cognitive reasoning.\n"
            f"Respond ONLY in valid JSON with this exact schema:\n"
            f'{{"incident_type": "...", "human_reasoning": "...", "decided_action": "...", "dispatch_brief": "..."}}\n'
            f"Permitted actions: 'MONITOR_ONLY', 'ISSUE_INTERCOM_WARNING', 'DISPATCH_SECURITY_PATROL', 'LOCK_SECTOR_DOORS_AND_SOUND_ALARM', 'LOCKDOWN_ALL_SECTORS_AND_DISPATCH_ARMED_SECURITY'."
        )

        messages = [
            {"role": "system", "content": "You are a professional security dispatcher reasoner. Output ONLY a valid JSON object."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
            )
        raw_output = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Parse JSON cleanly
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw_output[start:end + 1])
                if isinstance(parsed, dict) and "decided_action" in parsed:
                    return parsed
            except Exception:
                pass

        return self._deterministic_fallback(telemetry, threat_score, rule_id)

    def _deterministic_fallback(self, telemetry: Dict[str, Any], threat_score: float, rule_id: str) -> Dict[str, Any]:
        zone = telemetry.get("zone_breach", "None")
        dwell = telemetry.get("dwell_time_sec", 0.0)
        weapons = telemetry.get("yolo_weapons", [])
        audio_cat = telemetry.get("audio_category", "Normal Ambient")
        audio_event = telemetry.get("audio_event", "Silence")
        dominant_hz = telemetry.get("dominant_hz", 0.0)

        if threat_score >= 0.95 or "Gunshot" in audio_cat:
            return {
                "incident_type": "CRITICAL_ARMED_HOSTILE_INTRUSION",
                "human_reasoning": f"Acoustic ballistic signature ({audio_event}, {dominant_hz} Hz) detected in camera sector. Cross-modal validation confirms extreme threat requiring immediate tactical lockdown.",
                "decided_action": "LOCKDOWN_ALL_SECTORS_AND_DISPATCH_ARMED_SECURITY" if weapons else "LOCK_SECTOR_DOORS_AND_SOUND_ALARM",
                "dispatch_brief": "Initiate emergency facility lockdown. Seal sector access doors. Alert Armed Tactical Response.",
            }
        elif threat_score >= 0.70 or "Glass" in audio_cat or "Scream" in audio_cat:
            return {
                "incident_type": "FORCED_ENTRY_DISTURBANCE",
                "human_reasoning": f"Sensory anomalies detected ({audio_event}) near {zone}. The temporal coincidence indicates potential forced perimeter intrusion.",
                "decided_action": "DISPATCH_SECURITY_PATROL",
                "dispatch_brief": f"Security Officer: inspect {zone} immediately. Report physical boundary integrity.",
            }
        elif zone != "None" and dwell >= 6.0:
            return {
                "incident_type": "RESTRICTED_ZONE_LOITERING",
                "human_reasoning": f"Subject has remained stationary inside restricted {zone} for {dwell} seconds without authorized movement intent.",
                "decided_action": "ISSUE_INTERCOM_WARNING",
                "dispatch_brief": f"Broadcast verbal intercom warning to subject in {zone}. Advise to vacate restricted sector.",
            }
        elif zone != "None":
            return {
                "incident_type": "UNAUTHORIZED_ZONE_ENTRY",
                "human_reasoning": f"Subject crossed spatial geofence into {zone}. Tracking dwell kinematics for loitering.",
                "decided_action": "ISSUE_INTERCOM_WARNING",
                "dispatch_brief": f"Monitor subject in {zone}. Prepare intercom broadcast if dwell exceeds threshold.",
            }
        elif threat_score > 0.10:
            return {
                "incident_type": "AUTHORIZED_ROUTINE_OCCUPANCY",
                "human_reasoning": "Sensors detect routine authorized human presence within normal parameters. Speech acoustic frequencies detected with baseline movement.",
                "decided_action": "MONITOR_ONLY",
                "dispatch_brief": "Baseline sector monitoring. Routine operational presence.",
            }
        else:
            return {
                "incident_type": "PERIMETER_SECURE_BASELINE",
                "human_reasoning": "All multi-modal visual, behavioral, and acoustic sensors report routine baseline status. No anomalies detected.",
                "decided_action": "MONITOR_ONLY",
                "dispatch_brief": "Perimeter fully secure. Standard passive surveillance.",
            }


# ---------------------------------------------------------------------------
# 3. Asynchronous Pipeline Integration
# ---------------------------------------------------------------------------
class DispatcherPipeline:
    def __init__(self, cooldown_sec: float = 3.5):
        self.llm_dispatcher = LLMCognitiveDispatcher()
        self.cooldown_sec = cooldown_sec
        self.last_reason_time = 0.0
        self.lock = threading.Lock()
        self.worker_thread = None
        self.running = False

        self.last_decision = {
            "incident_type": "PERIMETER_SECURE_BASELINE",
            "human_reasoning": "All multi-modal visual, behavioral, and acoustic sensors report routine baseline status. No anomalies detected.",
            "decided_action": "MONITOR_ONLY",
            "dispatch_brief": "Perimeter fully secure. Standard passive surveillance.",
        }
        self.last_latency_ms = 1.0

    def start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop_worker(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def _worker_loop(self):
        global LATEST_TELEMETRY, LATEST_FUSION_RESULT, LATEST_DISPATCH_RESULT
        while self.running:
            time.sleep(0.35)
            try:
                now = time.time()
                threat_score = float(LATEST_FUSION_RESULT.get("fused_score", 0.05))
                rule_id = str(LATEST_FUSION_RESULT.get("rule_id", "R17_SECURE_AMBIENT_BASELINE"))
                is_elevated = threat_score >= 0.40

                should_run = False
                if is_elevated and ((now - self.last_reason_time) > self.cooldown_sec):
                    should_run = True
                elif not is_elevated and self.last_decision.get("incident_type") != "PERIMETER_SECURE_BASELINE" and ((now - self.last_reason_time) > 3.0):
                    should_run = True

                if should_run:
                    self.last_reason_time = now
                    dec, lat = self.llm_dispatcher.reason(LATEST_TELEMETRY, threat_score, rule_id)
                    with self.lock:
                        self.last_decision = dec
                        self.last_latency_ms = lat
                        LATEST_DISPATCH_RESULT = {
                            "decision": dec,
                            "latency_ms": lat,
                            "threat_score": threat_score,
                            "rule_id": rule_id,
                            "timestamp": now,
                        }
            except Exception:
                pass

    def force_reason_now(self, telemetry: Dict[str, Any], threat_score: float, rule_id: str) -> Dict[str, Any]:
        self.last_reason_time = time.time()
        dec, lat = self.llm_dispatcher.reason(telemetry, threat_score, rule_id)
        with self.lock:
            self.last_decision = dec
            self.last_latency_ms = lat
        return dec


# ---------------------------------------------------------------------------
# 4. Web Application & Real-Time Mission Control Dashboard
# ---------------------------------------------------------------------------
app = FastAPI(title="Module 5: LLM Cognitive Dispatcher (Human-Like Reasoning)")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

STREAM: Optional[ZeroLagVideoStream] = None
TRACKER: Optional[YOLOTrackerEngine] = None
BEHAVIOR: YOLOConvLSTMBehaviorAnalyzer = YOLOConvLSTMBehaviorAnalyzer()
AUDIO_MGR: Optional[AudioStreamManager] = None
FUSION: RuleBasedFusionEngine = RuleBasedFusionEngine()
DISPATCHER: DispatcherPipeline = DispatcherPipeline()

LATEST_TELEMETRY: Dict[str, Any] = {}
LATEST_FUSION_RESULT: Dict[str, Any] = {"fused_score": 0.05, "rule_id": "R17_SECURE_AMBIENT_BASELINE", "threat_level": "NORMAL"}
LATEST_DISPATCH_RESULT: Dict[str, Any] = {
    "decision": {
        "incident_type": "PERIMETER_SECURE_BASELINE",
        "human_reasoning": "Sensors report baseline status. Perimeter secure.",
        "decided_action": "MONITOR_ONLY",
        "dispatch_brief": "Perimeter fully secure. Standard passive surveillance.",
    },
    "latency_ms": 1.0,
    "threat_score": 0.05,
    "rule_id": "R17_SECURE_AMBIENT_BASELINE",
}


@app.on_event("startup")
def on_startup():
    global STREAM, TRACKER, AUDIO_MGR
    print("=" * 72, flush=True)
    print("  MODULE 5: LLM COGNITIVE DISPATCHER & HUMAN REASONER ACTIVE", flush=True)
    print("  Mission Control: http://127.0.0.1:8000", flush=True)
    print("=" * 72, flush=True)
    STREAM = ZeroLagVideoStream(source=0, width=640, height=480)
    TRACKER = YOLOTrackerEngine(conf_threshold=0.45, imgsz=416)
    TRACKER.start_async_worker(STREAM)
    AUDIO_MGR = AudioStreamManager()
    DISPATCHER.start_worker()


@app.on_event("shutdown")
def on_shutdown():
    DISPATCHER.stop_worker()
    if TRACKER:
        TRACKER.stop_async_worker()
    if STREAM:
        STREAM.stop()
    if AUDIO_MGR:
        AUDIO_MGR.stop()


def mjpeg_generator():
    global LATEST_TELEMETRY, LATEST_FUSION_RESULT
    last_fid = -1
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    while True:
        if STREAM is None or TRACKER is None or AUDIO_MGR is None:
            time.sleep(0.01)
            continue

        pkt = STREAM.wait_new(last_fid, timeout=0.2)
        if pkt is None:
            time.sleep(0.005)
            continue
        last_fid = pkt.frame_id
        frame = pkt.image
        sfps = pkt.fps

        # 1. Vision & Tracking (Module 1)
        state = TRACKER.get_latest_state()
        raw_dets = state.detections if hasattr(state, "detections") else state[0]
        vis_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]

        # 2. ConvLSTM Behavior (Module 2)
        fw, fh = frame.shape[1], frame.shape[0]
        b_data = BEHAVIOR.update(raw_dets, frame_w=fw, frame_h=fh)

        # 3. Audio Intelligence (Module 3)
        a_data = AUDIO_MGR.get_state()

        # 4. Multi-Modal Rule Fusion (Module 4)
        yolo_data = {"detections": raw_dets, "vis_ms": vis_ms}
        f_res = FUSION.evaluate(yolo_data, b_data, a_data)
        LATEST_FUSION_RESULT = f_res
        LATEST_TELEMETRY = f_res["inputs_summary"]

        # 5. Snapshot logging on elevated events
        if f_res.get("should_record", False) and frame is not None:
            vis = draw_module2_hud(
                frame=frame,
                behavior_results=b_data,
                zones=BEHAVIOR.zones,
                source_label=STREAM.source_label,
                stream_fps=sfps,
                yolo_ms=vis_ms,
            )
            snap_name = f"m5_dispatch_{int(time.time())}.jpg"
            snap_path = SNAPSHOTS_DIR / snap_name
            cv2.imwrite(str(snap_path), vis)
            log_payload = dict(f_res)
            with DISPATCHER.lock:
                dec = DISPATCHER.last_decision
            if dec and "decided_action" in dec:
                log_payload["final_action"] = dec.get("decided_action", f_res["final_action"])
                log_payload["decision_reason"] = dec.get("human_reasoning", f_res["decision_reason"])
            FUSION.log_incident(log_payload, snapshot_path=f"/snapshots/{snap_name}")

        # 6. Visual HUD
        vis = draw_module2_hud(
            frame=frame,
            behavior_results=b_data,
            zones=BEHAVIOR.zones,
            source_label=STREAM.source_label,
            stream_fps=sfps,
            yolo_ms=vis_ms,
        )

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


@app.get("/api/telemetry")
def get_telemetry():
    global LATEST_TELEMETRY, LATEST_FUSION_RESULT, LATEST_DISPATCH_RESULT
    if STREAM is None or TRACKER is None or AUDIO_MGR is None:
        return JSONResponse({"status": "initializing"})

    pkt = STREAM.read()
    sfps = pkt.fps if pkt is not None else 0.0
    state = TRACKER.get_latest_state()
    raw_dets = state.detections if hasattr(state, "detections") else state[0]
    vis_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]

    frame = pkt.image if pkt is not None else None
    fw = frame.shape[1] if frame is not None else 640
    fh = frame.shape[0] if frame is not None else 480
    b_data = BEHAVIOR.update(raw_dets, frame_w=fw, frame_h=fh)
    a_data = AUDIO_MGR.get_state()

    # Rule Fusion
    yolo_data = {"detections": raw_dets, "vis_ms": vis_ms}
    f_res = FUSION.evaluate(yolo_data, b_data, a_data)
    LATEST_FUSION_RESULT = f_res
    LATEST_TELEMETRY = f_res["inputs_summary"]

    with DISPATCHER.lock:
        dec = DISPATCHER.last_decision
        lat_ms = DISPATCHER.last_latency_ms

    return {
        "source": STREAM.source_label,
        "stream_fps": sfps,
        "fused_threat": f_res["threat_level"],
        "fused_score": f_res["fused_score"],
        "rule_id": f_res["rule_id"],
        "rule_action": f_res["final_action"],
        "telemetry": f_res["inputs_summary"],
        # LLM Cognitive Reasoning
        "llm_incident_type": dec.get("incident_type", "PERIMETER_SECURE_BASELINE"),
        "llm_human_reasoning": dec.get("human_reasoning", "Perimeter secure."),
        "llm_decided_action": dec.get("decided_action", f_res["final_action"]),
        "llm_dispatch_brief": dec.get("dispatch_brief", "Standard passive surveillance."),
        "llm_latency_ms": lat_ms,
        "llm_ready": DISPATCHER.llm_dispatcher.is_ready,
        # Audio signals
        "audio_db": a_data.get("db_level", -60.0),
        "top_sound": a_data.get("top_yamnet_class", "Silence"),
        "audio_category": a_data.get("surveillance_category", "Normal Ambient"),
        "dominant_hz": a_data.get("dominant_hz", 0.0),
        "spectral_centroid_hz": a_data.get("spectral_centroid_hz", 0.0),
        "voice_detected": a_data.get("voice_detected", False),
        "voice_status": a_data.get("voice_status", "Silent / Ambient"),
        "audio_source": a_data.get("source", "LIVE LAPTOP MICROPHONE"),
    }


@app.post("/api/force_reason")
def force_reason():
    """Forces an immediate on-demand LLM reasoning cycle on live telemetry."""
    global LATEST_TELEMETRY, LATEST_FUSION_RESULT, LATEST_DISPATCH_RESULT
    if STREAM is None or TRACKER is None or AUDIO_MGR is None:
        return {"ok": False, "error": "System initializing"}

    threat_score = float(LATEST_FUSION_RESULT.get("fused_score", 0.05))
    rule_id = str(LATEST_FUSION_RESULT.get("rule_id", "R17_SECURE_AMBIENT_BASELINE"))
    dec = DISPATCHER.force_reason_now(LATEST_TELEMETRY, threat_score, rule_id)
    return {"ok": True, "decision": dec}


@app.get("/api/incidents")
def get_incidents():
    return FUSION.get_recent_incidents(limit=10)


@app.post("/api/switch_video")
def switch_video(src: str = Query(...)):
    if STREAM is None or TRACKER is None:
        return {"ok": False}
    ok = change_source(STREAM, TRACKER, src)
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


@app.get("/", response_class=HTMLResponse)
def index_page():
    samples_dir = BASE_DIR / "samples"
    videos = sorted([p.name for p in samples_dir.glob("*.mp4")]) if samples_dir.exists() else []
    vid_buttons = "".join(
        f'<button class="btn" onclick="switchVideo(\'{v}\')">🎬 {v[:20]}</button>'
        for v in videos
    )

    audio_files = sorted([p.name for p in AUDIO_SAMPLES_DIR.glob("*.wav")])
    audio_buttons = "".join(
        f'<button class="btn btn-audio" onclick="switchAudio(\'{a}\')">🔊 {a.replace(".wav","").replace("_"," ").title()[:16]}</button>'
        for a in audio_files
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Module 5: LLM Cognitive Dispatcher (Human-Like Security Reasoner)</title>
<style>
  :root {{ --bg:#090d16; --card:#131a29; --border:#223049; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; --blue:#38bdf8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
  body {{ background:var(--bg); color:var(--text); padding:16px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; padding:14px 22px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; }}
  .header h1 {{ font-size:18px; color:var(--accent); display:flex; align-items:center; gap:8px; }}
  .badge {{ padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700; }}
  .badge-safe {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); }}
  .badge-warn {{ background:#3d2406; color:var(--warning); border:1px solid var(--warning); }}
  .badge-crit {{ background:#3b1111; color:var(--danger); border:1px solid var(--danger); animation:pulse 1s infinite; }}
  @keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.5}} 100%{{opacity:1}} }}
  .layout {{ display:grid; grid-template-columns: 1.6fr 1.4fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }}
  .card-title {{ font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:700; margin-bottom:12px; display:flex; justify-content:space-between; }}
  .video-box {{ width:100%; border-radius:8px; overflow:hidden; background:#000; border:1px solid var(--border); }}
  .video-box img {{ width:100%; display:block; }}
  .controls-bar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }}
  .btn {{ background:#1e293b; color:#fff; border:1px solid #334155; padding:8px 12px; border-radius:6px; cursor:pointer; font-size:12px; }}
  .btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .btn-primary {{ background:#059669; border-color:#10b981; font-weight:600; }}
  .btn-audio {{ background:#312e81; border-color:#4338ca; }}
  .code-block {{ background:#05080e; border:1px solid var(--border); border-radius:6px; padding:10px; font-family:monospace; font-size:11px; color:#38bdf8; max-height:140px; overflow-y:auto; }}
  .incident-row {{ display:flex; gap:10px; padding:10px; border-bottom:1px solid var(--border); font-size:12px; align-items:center; }}
  .incident-snap {{ width:60px; height:42px; border-radius:4px; object-fit:cover; border:1px solid var(--border); background:#000; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🧠 MODULE 5: LLM Cognitive Dispatcher &amp; Human-Like Reasoner</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">Inputs: Sensory JSON + Fused Threat Score (M4) • On-Device Cognitive LLM (Qwen2.5-0.5B) • Decoupled Execution</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL MONITORING</span>
  </div>

  <div class="layout">
    <!-- Left Column: Video, Audio & Structured Telemetry -->
    <div>
      <!-- Video Stream -->
      <div class="card">
        <div class="card-title">
          <span>Watchman Visual Stream (YOLOv8 + ConvLSTM Heatmap)</span>
          <span id="srcLabel" style="color:var(--accent);">Camera 0</span>
        </div>
        <div class="video-box">
          <img src="/video_feed" alt="Video Feed" />
        </div>
        <div class="controls-bar">
          <button class="btn btn-primary" onclick="switchVideo('camera')">📷 Live Camera</button>
          {vid_buttons}
        </div>
      </div>

      <!-- Acoustic Channel -->
      <div class="card">
        <div class="card-title">
          <span>Watchman Acoustic Channel (Google YAMNet AudioSet)</span>
          <span id="audioSrcLabel" style="color:var(--accent);">MIC</span>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
          <div style="background:#0d131f;border:1px solid var(--border);padding:10px;border-radius:6px;">
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;">Dominant Pitch</div>
            <div style="font-size:18px;font-weight:700;color:#38bdf8;margin-top:2px;" id="audioHz">0 Hz</div>
            <div style="font-size:9px;color:#64748b;margin-top:2px;" id="audioCentroid">Centroid: 0 Hz</div>
          </div>
          <div style="background:#0d131f;border:1px solid var(--border);padding:10px;border-radius:6px;">
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;">Voice Detection</div>
            <div style="font-size:12px;font-weight:700;color:var(--accent);margin-top:4px;" id="voiceStatus">Silent / Ambient</div>
            <div style="font-size:9px;color:#94a3b8;margin-top:2px;" id="vadPill">⚪ Inactive</div>
          </div>
          <div style="background:#0d131f;border:1px solid var(--border);padding:10px;border-radius:6px;">
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;">Sound Level</div>
            <div style="font-size:18px;font-weight:700;color:var(--accent);margin-top:2px;" id="audioDbVal">-60 dB</div>
            <div style="background:#1e293b;height:4px;border-radius:2px;overflow:hidden;margin-top:4px;">
              <div id="audioDbBar" style="background:#00e5a3;height:100%;width:20%;transition:width 0.1s;"></div>
            </div>
          </div>
        </div>

        <div style="font-size:11px;color:#cbd5e1;background:#0d131f;border:1px solid var(--border);padding:8px 10px;border-radius:6px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
          <div><b>Acoustic Detection:</b> <span id="audioTopSound" style="color:var(--accent);font-weight:700;">Silence</span></div>
          <div id="audioThreatPill" style="padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;background:#0f2922;color:#00e5a3;">NORMAL</div>
        </div>

        <div class="controls-bar" style="margin-top:0;">
          <button class="btn btn-primary" onclick="switchAudio('mic')">🎙️ Live Laptop Mic</button>
          {audio_buttons}
        </div>
      </div>

      <!-- Structured JSON Fed to LLM -->
      <div class="card">
        <div class="card-title">Structured Sensor Telemetry JSON (Exact Input Passed into LLM)</div>
        <div class="code-block" id="jsonTelemetry">Loading telemetry payload...</div>
      </div>
    </div>

    <!-- Right Column: LLM Human-Like Reasoning & Incident Log -->
    <div>
      <!-- LLM Cognitive Dispatcher Card -->
      <div class="card">
        <div class="card-title">
          <span>Cognitive Dispatcher Core (Human-Like Reasoning)</span>
          <span id="modelBadge" style="color:#10b981;font-size:11px;">🟢 Qwen2.5-0.5B-Local</span>
        </div>

        <!-- Telemetry & Threat Badge Row -->
        <div style="background:#0d131f;border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;">Input Fused Threat Score</div>
            <div style="font-size:24px;font-weight:800;color:var(--accent);margin-top:2px;" id="fusedScoreVal">0.05</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;">Pre-Rule Triggered</div>
            <div style="font-size:12px;font-family:monospace;font-weight:700;color:var(--blue);margin-top:4px;" id="ruleIdVal">R17_SECURE_AMBIENT_BASELINE</div>
          </div>
        </div>

        <!-- Incident Classification -->
        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700;margin-bottom:4px;">Incident Classification</div>
          <div id="incidentTypeVal" style="background:#05080e;border:1px solid var(--border);padding:8px 12px;border-radius:6px;font-weight:700;color:var(--blue);font-size:13px;">
            PERIMETER_SECURE_BASELINE
          </div>
        </div>

        <!-- Human-Like Reasoning Output -->
        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700;margin-bottom:4px;">Human-Like Cognitive Reasoning</div>
          <div style="background:#05080e;border:1px solid var(--border);padding:12px;border-radius:6px;font-size:13px;color:#cbd5e1;line-height:1.55;" id="llmReasoningVal">
            Sensors report routine ambient conditions. Perimeter secure.
          </div>
        </div>

        <!-- Decided Action -->
        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700;margin-bottom:4px;">Decided Tactical Action</div>
          <div id="decidedActionVal" style="background:#0f2922;border:1px solid var(--accent);padding:10px 14px;border-radius:6px;font-weight:800;color:var(--accent);font-size:14px;">
            MONITOR_ONLY
          </div>
        </div>

        <!-- Operator Dispatch Brief -->
        <div style="margin-bottom:14px;">
          <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700;margin-bottom:4px;">Operator Tactical Dispatch Brief</div>
          <div style="background:#05080e;border:1px solid var(--border);padding:10px 12px;border-radius:6px;font-size:12px;color:#94a3b8;" id="dispatchBriefVal">
            Perimeter fully secure. Standard passive surveillance.
          </div>
        </div>

        <button class="btn btn-primary" onclick="forceEvaluateNow(this)" style="width:100%;padding:10px;font-size:13px;font-weight:700;">
          ⚡ Run Dispatcher LLM Assessment Now
        </button>
      </div>

      <!-- Incident Forensic Evidence Log -->
      <div class="card" style="margin-bottom:0;">
        <div class="card-title">Incident Audit Log (SQLite Evidence)</div>
        <div id="incidentList" style="max-height:220px;overflow-y:auto;">
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

async function forceEvaluateNow(btn) {{
  const orig = btn.innerText;
  btn.innerText = '⏳ Running Reasoner...';
  try {{
    const res = await fetch('/api/force_reason', {{method:'POST'}});
    const data = await res.json();
    if (data.ok && data.decision) {{
      document.getElementById('incidentTypeVal').innerText = data.decision.incident_type || 'INCIDENT_ASSESSED';
      document.getElementById('llmReasoningVal').innerText = data.decision.human_reasoning || 'Evaluated.';
      document.getElementById('decidedActionVal').innerText = data.decision.decided_action || 'MONITOR_ONLY';
      document.getElementById('dispatchBriefVal').innerText = data.decision.dispatch_brief || 'None';
    }}
  }} catch(e) {{}}
  setTimeout(() => {{ btn.innerText = orig; }}, 800);
}}

setInterval(async () => {{
  try {{
    const r = await fetch('/api/telemetry');
    const d = await r.json();
    if (!d.source) return;

    document.getElementById('srcLabel').innerText = d.source;
    document.getElementById('jsonTelemetry').innerText = JSON.stringify(d.telemetry, null, 2);
    document.getElementById('fusedScoreVal').innerText = d.fused_score.toFixed(2);
    document.getElementById('ruleIdVal').innerText = d.rule_id;

    // LLM Reasoning updates
    document.getElementById('incidentTypeVal').innerText = d.llm_incident_type || 'NORMAL';
    document.getElementById('llmReasoningVal').innerText = d.llm_human_reasoning;
    document.getElementById('decidedActionVal').innerText = d.llm_decided_action;
    document.getElementById('dispatchBriefVal').innerText = d.llm_dispatch_brief;

    // Acoustic updates
    if (document.getElementById('audioHz')) {{
      document.getElementById('audioSrcLabel').innerText = d.audio_source || 'MIC';
      document.getElementById('audioHz').innerText = (d.dominant_hz || 0) + ' Hz';
      document.getElementById('audioCentroid').innerText = 'Centroid: ' + (d.spectral_centroid_hz || 0) + ' Hz';
      document.getElementById('audioDbVal').innerText = (d.audio_db || -60) + ' dB';
      document.getElementById('audioTopSound').innerText = (d.top_sound || 'Silence') + ' (' + (d.audio_category || 'Normal') + ')';
      document.getElementById('voiceStatus').innerText = d.voice_status || 'Silent / Ambient';

      const dbPct = Math.max(0, Math.min(100, (((d.audio_db || -60) + 80) / 80) * 100));
      document.getElementById('audioDbBar').style.width = dbPct + '%';

      const vp = document.getElementById('vadPill');
      if (d.voice_detected) {{
        vp.innerText = '🟢 Human Voice Active';
        vp.style.color = '#10b981';
        document.getElementById('voiceStatus').style.color = '#10b981';
      }} else if (d.audio_category && (d.audio_category.includes('Gunshot') || d.audio_category.includes('Scream') || d.audio_category.includes('Alarm') || d.audio_category.includes('Glass'))) {{
        vp.innerText = '🚨 Threat Sound Active';
        vp.style.color = '#ef4444';
        document.getElementById('voiceStatus').style.color = '#ef4444';
      }} else {{
        vp.innerText = '⚪ Ambient Silence';
        vp.style.color = '#94a3b8';
        document.getElementById('voiceStatus').style.color = 'var(--accent)';
      }}

      const atp = document.getElementById('audioThreatPill');
      if (d.audio_category && (d.audio_category.includes('Gunshot') || d.audio_category.includes('Glass') || d.audio_category.includes('Scream'))) {{
        atp.innerText = 'CRITICAL';
        atp.style.background = '#3b1111';
        atp.style.color = '#ef4444';
      }} else if (d.audio_category && d.audio_category.includes('Alarm')) {{
        atp.innerText = 'WARNING';
        atp.style.background = '#3d2406';
        atp.style.color = '#f59e0b';
      }} else {{
        atp.innerText = 'NORMAL';
        atp.style.background = '#0f2922';
        atp.style.color = '#00e5a3';
      }}
    }}

    // Threat Badges and Action Colors
    const tb = document.getElementById('threatBadge');
    const actBox = document.getElementById('decidedActionVal');
    if (d.fused_threat === 'CRITICAL') {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 CRITICAL BREACH';
      document.getElementById('fusedScoreVal').style.color = '#ef4444';
      actBox.style.background = '#3b1111';
      actBox.style.borderColor = '#ef4444';
      actBox.style.color = '#ef4444';
    }} else if (d.fused_threat === 'WARNING') {{
      tb.className = 'badge badge-warn';
      tb.innerText = '⚠️ ELEVATED SUSPICION';
      document.getElementById('fusedScoreVal').style.color = '#f59e0b';
      actBox.style.background = '#3d2406';
      actBox.style.borderColor = '#f59e0b';
      actBox.style.color = '#f59e0b';
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = 'NORMAL MONITORING';
      document.getElementById('fusedScoreVal').style.color = 'var(--accent)';
      actBox.style.background = '#0f2922';
      actBox.style.borderColor = 'var(--accent)';
      actBox.style.color = 'var(--accent)';
    }}
  }} catch(e) {{}}
}}, 250);

// Poll incidents
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
            <span style="color:${{inc.threat_level==='CRITICAL'?'#ef4444':'#f59e0b'}};">${{inc.threat_level}}</span>
            <span style="color:#64748b;font-size:11px;">${{inc.timestamp.split(' ')[1]}}</span>
          </div>
          <div style="font-size:11px;color:#38bdf8;margin-top:2px;">Action: <b>${{inc.action_executed}}</b></div>
          <div style="font-size:11px;color:#cbd5e1;margin-top:2px;">${{inc.llm_reasoning}}</div>
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
