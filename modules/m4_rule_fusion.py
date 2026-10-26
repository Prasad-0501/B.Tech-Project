"""
MODULE 4: Multi-Modal Rule-Based Decision Fusion Engine (Deterministic If-Else Ladder)
========================================================================================
Architecture:
  - Takes input strictly from:
      1. Module 1: YOLOv8 Object Detection & ByteTrack Kalman Tracking
      2. Module 2: 2D ConvLSTM Spatiotemporal Heatmaps & Restricted Zones
      3. Module 3: Google YAMNet Acoustic AI (521 AudioSet Classes)
  - Zero LLM Dependency: Completely decoupled from any LLMs for pure deterministic edge speed (< 0.1ms).
  - Explicit If-Else Ladder: Prioritized, hierarchical multi-modal security rules.
  - Generates Fused Threat Score (0.05 - 1.00), Threat Level (NORMAL, WARNING, CRITICAL), and Action.

Web Console: http://127.0.0.1:8000
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
import uvicorn
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import existing modular components (YOLO, ConvLSTM, Audio)
from modules.m1_yolo_tracker import ZeroLagVideoStream, YOLOTrackerEngine, CLASS_COLORS, change_source
from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer, draw_module2_hud
from modules.m3_audio_ast import AudioStreamManager, AUDIO_SAMPLES_DIR

DB_PATH = BASE_DIR / "database" / "surveillance_events.db"
SNAPSHOTS_DIR = BASE_DIR / "database" / "snapshots"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Database Schema
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
# 2. Rule-Based Fusion Engine (Deterministic If-Else Ladder)
# ---------------------------------------------------------------------------
class RuleBasedFusionEngine:
    """
    Multi-modal decision engine that fuses YOLO detections, ConvLSTM behavior,
    and YAMNet acoustic intelligence using an explicit, prioritized if-else ladder.
    Zero LLM dependencies; executes in < 0.1ms.
    """

    def __init__(self, cooldown_sec: float = 3.0):
        self.cooldown_sec = cooldown_sec
        self.last_incident_time = 0.0

    def evaluate(
        self,
        yolo_data: Dict[str, Any],
        behavior_data: Dict[str, Any],
        audio_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Parse Input Signals from Module 1 (YOLO)
        # -------------------------------------------------------------
        detections = yolo_data.get("detections", [])
        has_weapon = any(d.get("class_name") in ["Knife", "Scissors", "Baseball Bat", "Gun"] for d in detections)
        weapon_names = [d.get("class_name") for d in detections if d.get("class_name") in ["Knife", "Scissors", "Baseball Bat", "Gun"]]
        has_person = any(d.get("class_name") == "Person" for d in detections)
        person_count = sum(1 for d in detections if d.get("class_name") == "Person")

        # -------------------------------------------------------------
        # 2. Parse Input Signals from Module 2 (ConvLSTM & Zones)
        # -------------------------------------------------------------
        convlstm_energy = float(behavior_data.get("convlstm_energy", 0.0))
        intrusions = behavior_data.get("intrusions", [])
        loiterings = behavior_data.get("loitering_events", [])
        runnings = behavior_data.get("running_events", [])
        has_intrusion = len(intrusions) > 0
        intrusion_zone = intrusions[0].get("zone_name", "Restricted Area") if intrusions else ""
        max_dwell = max([l.get("dwell_sec", 0.0) for l in loiterings], default=0.0)
        max_speed = float(behavior_data.get("max_speed_px_s", 0.0))
        is_running = len(runnings) > 0 or max_speed > 160.0

        # -------------------------------------------------------------
        # 3. Parse Input Signals from Module 3 (AST / YAMNet Audio)
        # -------------------------------------------------------------
        audio_class = str(audio_data.get("top_yamnet_class", "Silence"))
        audio_cat = str(audio_data.get("surveillance_category", "Normal Ambient"))
        audio_threat = float(audio_data.get("threat_score", 0.05))
        audio_db = float(audio_data.get("db_level", -60.0))
        dominant_hz = float(audio_data.get("dominant_hz", 0.0))
        voice_detected = bool(audio_data.get("voice_detected", False))

        is_gunfire = (
            "gunshot" in audio_class.lower()
            or "gunfire" in audio_class.lower()
            or "explosion" in audio_class.lower()
            or audio_cat == "Gunshot / Explosion"
            or audio_threat >= 0.99
        )
        is_glass = "glass" in audio_class.lower() or audio_cat == "Glass Breaking"
        is_scream = "scream" in audio_class.lower() or "shout" in audio_class.lower() or audio_cat == "Scream / Distress"
        is_siren = "siren" in audio_class.lower() or "alarm" in audio_class.lower() or audio_cat == "Siren / Security Alarm"

        # -------------------------------------------------------------
        # 4. EXPLICIT MULTI-MODAL IF-ELSE DECISION LADDER
        # -------------------------------------------------------------
        if has_weapon and is_gunfire:
            rule_id = "R01_ARMED_GUNFIRE_BREACH"
            threat_level = "CRITICAL"
            threat_score = 1.00
            action = "LOCKDOWN_ALL_SECTORS_AND_DISPATCH_ARMED_SECURITY"
            reason = f"Armed subject confirmed ({', '.join(weapon_names)}) with active gunfire acoustic detonation."

        elif is_gunfire:
            rule_id = "R02_ACTIVE_GUNFIRE_DETONATION"
            threat_level = "CRITICAL"
            threat_score = 0.98
            action = "LOCK_SECTOR_DOORS_AND_SOUND_ALARM"
            reason = f"Gunshot / Explosive acoustic detonation confirmed by YAMNet ({audio_class}, {audio_db:.1f} dB)."

        elif has_weapon and has_intrusion:
            rule_id = "R03_ARMED_ZONE_INTRUSION"
            threat_level = "CRITICAL"
            threat_score = 0.95
            action = "LOCK_SECTOR_DOORS_AND_DISPATCH_PATROL"
            reason = f"Armed subject brandishing {', '.join(weapon_names)} inside {intrusion_zone}."

        elif has_weapon:
            rule_id = "R04_WEAPON_BRANDISHED"
            threat_level = "CRITICAL"
            threat_score = 0.92
            action = "DISPATCH_SECURITY_PATROL"
            reason = f"Dangerous weapon detected ({', '.join(weapon_names)}) in monitored camera sector."

        elif is_glass and has_intrusion:
            rule_id = "R05_FORCED_ENTRY_GLASS_BREAK"
            threat_level = "CRITICAL"
            threat_score = 0.92
            action = "DISPATCH_SECURITY_PATROL"
            reason = f"Forced entry confirmed: Glass break sound ({int(dominant_hz)} Hz) coinciding with visual intrusion into {intrusion_zone}."

        elif is_scream and (has_person or has_intrusion):
            rule_id = "R06_DISTRESS_SCREAM_IN_ZONE"
            threat_level = "CRITICAL"
            threat_score = 0.90
            action = "DISPATCH_SECURITY_PATROL"
            reason = "Human distress scream detected in immediate vicinity of detected subject."

        elif is_siren:
            rule_id = "R07_SECURITY_SIREN_ACTIVE"
            threat_level = "CRITICAL"
            threat_score = 0.85
            action = "TRIGGER_EMERGENCY_SIREN_PROTOCOL"
            reason = f"Facility siren / civil alarm acoustic resonance detected ({audio_class})."

        elif has_intrusion and (is_running or convlstm_energy > 0.45):
            rule_id = "R08_RAPID_BREACH_RUNNING"
            threat_level = "CRITICAL"
            threat_score = 0.82
            action = "DISPATCH_SECURITY_PATROL"
            reason = f"High-speed intrusion: Subject sprinting inside {intrusion_zone} ({int(max_speed)} px/s, ConvLSTM energy: {convlstm_energy:.2f})."

        elif has_intrusion and max_dwell >= 6.0:
            rule_id = "R09_RESTRICTED_ZONE_LOITERING"
            threat_level = "WARNING"
            threat_score = 0.75
            action = "ISSUE_INTERCOM_WARNING"
            reason = f"Persistent loitering violation: Person lingering in {intrusion_zone} for {max_dwell:.1f}s."

        elif is_glass:
            rule_id = "R10_PERIMETER_GLASS_BREAK"
            threat_level = "WARNING"
            threat_score = 0.70
            action = "ALERT_OPERATOR_CHECK_SECTOR"
            reason = f"Glass breaking acoustic event detected ({int(dominant_hz)} Hz) without confirmed visual zone breach."

        elif has_intrusion:
            rule_id = "R11_UNAUTHORIZED_ZONE_ENTRY"
            threat_level = "WARNING"
            threat_score = 0.60
            action = "ISSUE_INTERCOM_WARNING"
            reason = f"Unauthorized zone entry: Subject crossed boundary into {intrusion_zone}."

        elif max_dwell >= 6.0:
            rule_id = "R12_PERIMETER_LOITERING"
            threat_level = "WARNING"
            threat_score = 0.50
            action = "MONITOR_TRACK"
            reason = f"Perimeter dwell alert: Subject stationary in camera sector for {max_dwell:.1f}s."

        elif is_scream:
            rule_id = "R13_DISTANT_SCREAM_AUDIO"
            threat_level = "WARNING"
            threat_score = 0.45
            action = "MONITOR_TRACK"
            reason = "Potential scream / vocal distress acoustic frequency detected."

        elif convlstm_energy > 0.50 and max_speed > 140.0:
            rule_id = "R14_ERRATIC_KINETIC_MOTION"
            threat_level = "WARNING"
            threat_score = 0.40
            action = "MONITOR_TRACK"
            reason = f"ConvLSTM kinetic motion anomaly: high velocity shift ({int(max_speed)} px/s)."

        elif has_person and voice_detected:
            rule_id = "R15_ROUTINE_HUMAN_CONVERSATION"
            threat_level = "NORMAL"
            threat_score = 0.15
            action = "MONITOR_ONLY"
            reason = f"Normal routine presence: {person_count} person(s) with active conversation voice ({int(dominant_hz)} Hz)."

        elif has_person:
            rule_id = "R16_BASELINE_PERSON_DETECTED"
            threat_level = "NORMAL"
            threat_score = 0.10
            action = "MONITOR_ONLY"
            reason = f"Baseline occupancy: {person_count} authorized person(s) detected in standard sector."

        else:
            rule_id = "R17_SECURE_AMBIENT_BASELINE"
            threat_level = "NORMAL"
            threat_score = 0.05
            action = "MONITOR_ONLY"
            reason = "All multi-modal sensors report secure baseline status. Zero violations."

        inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        now = time.time()
        should_record = (threat_level in ["WARNING", "CRITICAL"]) and ((now - self.last_incident_time) > self.cooldown_sec)
        if should_record:
            self.last_incident_time = now

        return {
            "timestamp": now,
            "rule_id": rule_id,
            "threat_level": threat_level,
            "fused_score": threat_score,
            "final_action": action,
            "decision_reason": reason,
            "fusion_latency_ms": inference_ms,
            "should_record": should_record,
            "inputs_summary": {
                "yolo_persons": person_count,
                "yolo_weapons": weapon_names,
                "convlstm_energy": round(convlstm_energy, 2),
                "zone_breach": intrusion_zone if has_intrusion else "None",
                "dwell_time_sec": round(max_dwell, 1),
                "audio_event": audio_class,
                "audio_category": audio_cat,
                "dominant_hz": round(dominant_hz, 0),
                "audio_db": round(audio_db, 1),
            },
        }

    def log_incident(self, res: Dict[str, Any], snapshot_path: Optional[str] = None):
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    timestamp, threat_level, fused_score, trust_score, action_executed,
                    fallback_triggered, latency_ms, structured_telemetry, llm_reasoning, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(res.get("timestamp", time.time()))),
                res.get("threat_level", "NORMAL"),
                res.get("fused_score", 0.0),
                1.0,
                res.get("final_action", "MONITOR_ONLY"),
                False,
                res.get("fusion_latency_ms", 0.0),
                json.dumps(res.get("inputs_summary", {})),
                f"[{res.get('rule_id')}] {res.get('decision_reason')}",
                snapshot_path,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Rule Fusion] DB Error: {e}", flush=True)

    def get_recent_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# 3. Web Application & Real-Time Mission Control Dashboard
# ---------------------------------------------------------------------------
app = FastAPI(title="Module 4: Rule-Based Multi-Modal Decision Fusion (No LLM)")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

STREAM: Optional[ZeroLagVideoStream] = None
TRACKER: Optional[YOLOTrackerEngine] = None
BEHAVIOR: YOLOConvLSTMBehaviorAnalyzer = YOLOConvLSTMBehaviorAnalyzer()
AUDIO_MGR: Optional[AudioStreamManager] = None
FUSION: RuleBasedFusionEngine = RuleBasedFusionEngine()

LATEST_FUSION_STATE: Dict[str, Any] = {
    "rule_id": "R17_SECURE_AMBIENT_BASELINE",
    "threat_level": "NORMAL",
    "fused_score": 0.05,
    "final_action": "MONITOR_ONLY",
    "decision_reason": "All multi-modal sensors report secure baseline status. Zero violations.",
    "fusion_latency_ms": 0.05,
    "inputs_summary": {},
}


@app.on_event("startup")
def on_startup():
    global STREAM, TRACKER, AUDIO_MGR
    print("=" * 72, flush=True)
    print("  MODULE 4: MULTI-MODAL RULE-BASED FUSION (ZERO LLM) ACTIVE", flush=True)
    print("  Mission Control: http://127.0.0.1:8000", flush=True)
    print("=" * 72, flush=True)
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


def mjpeg_generator():
    global LATEST_FUSION_STATE
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

        # 4. Multi-Modal Rule Fusion (Module 4 - If-Else Ladder)
        yolo_data = {"detections": raw_dets, "vis_ms": vis_ms}
        res = FUSION.evaluate(yolo_data, b_data, a_data)
        LATEST_FUSION_STATE = res

        # 5. Snapshot logging on Warning / Critical
        if res.get("should_record", False) and frame is not None:
            vis = draw_module2_hud(
                frame=frame,
                behavior_results=b_data,
                zones=BEHAVIOR.zones,
                source_label=STREAM.source_label,
                stream_fps=sfps,
                yolo_ms=vis_ms,
            )
            snap_name = f"rule_fusion_{int(time.time())}.jpg"
            snap_path = SNAPSHOTS_DIR / snap_name
            cv2.imwrite(str(snap_path), vis)
            FUSION.log_incident(res, snapshot_path=f"/snapshots/{snap_name}")

        # 6. Render Visual HUD
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
    global LATEST_FUSION_STATE
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

    # Rule evaluation (0.05ms)
    yolo_data = {"detections": raw_dets, "vis_ms": vis_ms}
    res = FUSION.evaluate(yolo_data, b_data, a_data)
    LATEST_FUSION_STATE = res

    # Record snapshot if elevated
    if res.get("should_record", False) and frame is not None:
        vis = draw_module2_hud(
            frame=frame,
            behavior_results=b_data,
            zones=BEHAVIOR.zones,
            source_label=STREAM.source_label,
            stream_fps=sfps,
            yolo_ms=vis_ms,
        )
        snap_name = f"rule_fusion_{int(time.time())}.jpg"
        snap_path = SNAPSHOTS_DIR / snap_name
        cv2.imwrite(str(snap_path), vis)
        FUSION.log_incident(res, snapshot_path=f"/snapshots/{snap_name}")

    return {
        "source": STREAM.source_label,
        "stream_fps": sfps,
        "rule_id": res["rule_id"],
        "threat_level": res["threat_level"],
        "fused_score": res["fused_score"],
        "final_action": res["final_action"],
        "decision_reason": res["decision_reason"],
        "fusion_latency_ms": res["fusion_latency_ms"],
        "inputs_summary": res["inputs_summary"],
        "audio_db": a_data.get("db_level", -60.0),
        "top_sound": a_data.get("top_yamnet_class", "Silence"),
        "audio_category": a_data.get("surveillance_category", "Normal Ambient"),
        "dominant_hz": a_data.get("dominant_hz", 0.0),
        "spectral_centroid_hz": a_data.get("spectral_centroid_hz", 0.0),
        "voice_detected": a_data.get("voice_detected", False),
        "voice_status": a_data.get("voice_status", "Silent / Ambient"),
        "audio_source": a_data.get("source", "LIVE LAPTOP MICROPHONE"),
    }


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
<title>Module 4: Multi-Modal Rule-Based Decision Fusion (No LLM)</title>
<style>
  :root {{ --bg:#090d16; --card:#131a29; --border:#223049; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; }}
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
  body {{ background:var(--bg); color:var(--text); padding:16px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; padding:14px 22px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; }}
  .header h1 {{ font-size:18px; color:var(--accent); display:flex; align-items:center; gap:8px; }}
  .badge {{ padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700; }}
  .badge-safe {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); }}
  .badge-warn {{ background:#3d2406; color:var(--warning); border:1px solid var(--warning); }}
  .badge-crit {{ background:#3b1111; color:var(--danger); border:1px solid var(--danger); animation:pulse 1s infinite; }}
  @keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.5}} 100%{{opacity:1}} }}
  .layout {{ display:grid; grid-template-columns: 1.8fr 1.2fr; gap:16px; }}
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
  .ladder-table {{ width:100%; border-collapse:collapse; font-size:11px; margin-top:8px; }}
  .ladder-table th {{ background:#0a101b; color:#94a3b8; padding:6px 8px; text-align:left; border:1px solid var(--border); font-weight:600; }}
  .ladder-table td {{ padding:6px 8px; border:1px solid var(--border); color:#cbd5e1; }}
  .ladder-row-active {{ background:#1c2d42 !important; font-weight:700; border-left:3px solid var(--accent) !important; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🛡️ MODULE 4: Multi-Modal Rule-Based Decision Fusion Engine (Deterministic If-Else Ladder)</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">Inputs: YOLOv8 (M1) • ConvLSTM Heatmap (M2) • Google YAMNet Audio AI (M3) • Zero LLM Overhead (&lt; 0.1ms)</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL MONITORING</span>
  </div>

  <div class="layout">
    <!-- Left Column: Video, Acoustic Channel & Live Inputs -->
    <div>
      <!-- Video Feed -->
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

      <!-- Live Inputs Summary JSON -->
      <div class="card">
        <div class="card-title">Live Inputs Fed into Fusion Ladder (YOLO + ConvLSTM + AST)</div>
        <div class="code-block" id="inputsSummary">Loading sensor inputs...</div>
      </div>
    </div>

    <!-- Right Column: Decision Engine State & Ladder -->
    <div>
      <!-- Multi-Modal Decision Matrix Card -->
      <div class="card">
        <div class="card-title">
          <span>Active Decision Rule (If-Else Ladder Output)</span>
          <span id="fusionMs" style="color:#10b981;font-size:11px;">⚡ 0.05ms (Deterministic)</span>
        </div>

        <div style="background:#0d131f;border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Rule Triggered</span>
            <span id="activeRuleBadge" style="font-family:monospace;font-weight:700;color:#38bdf8;font-size:13px;">R17_SECURE_AMBIENT_BASELINE</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
            <span style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Fused Threat Score</span>
            <span id="fusedScoreVal" style="font-size:24px;font-weight:800;color:var(--accent);">0.05</span>
          </div>
          <div style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px;">
            <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Automated Action</div>
            <div id="finalActionVal" style="font-size:14px;font-weight:700;color:#00e5a3;margin-top:2px;">MONITOR_ONLY</div>
          </div>
        </div>

        <div style="font-size:12px;background:#05080e;border:1px solid var(--border);padding:12px;border-radius:6px;color:#cbd5e1;line-height:1.5;" id="decisionReason">
          All multi-modal sensors report secure baseline status. Zero violations.
        </div>
      </div>

      <!-- Decision Ladder Priority Inspector -->
      <div class="card">
        <div class="card-title">
          <span>Hierarchical If-Else Ladder Priority</span>
          <span style="color:#94a3b8;font-size:11px;">Top-to-Bottom Priority</span>
        </div>
        <div style="max-height:220px;overflow-y:auto;">
          <table class="ladder-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Conditions (YOLO + ConvLSTM + AST)</th>
                <th>Threat</th>
              </tr>
            </thead>
            <tbody>
              <tr id="row_R01"><td>R01</td><td>Weapon + Gunfire</td><td style="color:#ef4444;font-weight:700;">1.00</td></tr>
              <tr id="row_R02"><td>R02</td><td>Gunfire / Explosion Audio</td><td style="color:#ef4444;font-weight:700;">0.98</td></tr>
              <tr id="row_R03"><td>R03</td><td>Weapon + Zone Intrusion</td><td style="color:#ef4444;font-weight:700;">0.95</td></tr>
              <tr id="row_R04"><td>R04</td><td>Weapon Brandished (Knife/Gun)</td><td style="color:#ef4444;font-weight:700;">0.92</td></tr>
              <tr id="row_R05"><td>R05</td><td>Glass Break + Zone Intrusion</td><td style="color:#ef4444;font-weight:700;">0.92</td></tr>
              <tr id="row_R06"><td>R06</td><td>Human Distress Scream in Zone</td><td style="color:#ef4444;font-weight:700;">0.90</td></tr>
              <tr id="row_R07"><td>R07</td><td>Security / Civil Siren Active</td><td style="color:#ef4444;font-weight:700;">0.85</td></tr>
              <tr id="row_R08"><td>R08</td><td>Intrusion + Sprinting / Running</td><td style="color:#ef4444;font-weight:700;">0.82</td></tr>
              <tr id="row_R09"><td>R09</td><td>Intrusion + Loitering &gt; 6s</td><td style="color:#f59e0b;font-weight:700;">0.75</td></tr>
              <tr id="row_R10"><td>R10</td><td>Glass Break Acoustic Event</td><td style="color:#f59e0b;font-weight:700;">0.70</td></tr>
              <tr id="row_R11"><td>R11</td><td>Unauthorized Zone Intrusion</td><td style="color:#f59e0b;font-weight:700;">0.60</td></tr>
              <tr id="row_R12"><td>R12</td><td>Perimeter Loitering &gt; 6s</td><td style="color:#f59e0b;font-weight:700;">0.50</td></tr>
              <tr id="row_R13"><td>R13</td><td>Distant Distress Scream Audio</td><td style="color:#f59e0b;font-weight:700;">0.45</td></tr>
              <tr id="row_R14"><td>R14</td><td>ConvLSTM Kinetic Motion Anomaly</td><td style="color:#f59e0b;font-weight:700;">0.40</td></tr>
              <tr id="row_R15"><td>R15</td><td>Routine Person + Voice Active</td><td style="color:#00e5a3;font-weight:700;">0.15</td></tr>
              <tr id="row_R16"><td>R16</td><td>Baseline Person Detected</td><td style="color:#00e5a3;font-weight:700;">0.10</td></tr>
              <tr id="row_R17"><td>R17</td><td>Secure Ambient Baseline</td><td style="color:#00e5a3;font-weight:700;">0.05</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Incident Forensic Evidence Log -->
      <div class="card" style="margin-bottom:0;">
        <div class="card-title">Incident Audit Log (Evidence Database)</div>
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

setInterval(async () => {{
  try {{
    const r = await fetch('/api/telemetry');
    const d = await r.json();
    if (!d.source) return;

    document.getElementById('srcLabel').innerText = d.source;
    document.getElementById('inputsSummary').innerText = JSON.stringify(d.inputs_summary, null, 2);
    document.getElementById('activeRuleBadge').innerText = d.rule_id;
    document.getElementById('fusedScoreVal').innerText = d.fused_score.toFixed(2);
    document.getElementById('finalActionVal').innerText = d.final_action;
    document.getElementById('decisionReason').innerText = d.decision_reason;
    document.getElementById('fusionMs').innerText = '⚡ ' + d.fusion_latency_ms + 'ms (Deterministic)';

    // Acoustic Channel updates
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

    // Threat Badge
    const tb = document.getElementById('threatBadge');
    if (d.threat_level === 'CRITICAL') {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 CRITICAL BREACH';
      document.getElementById('fusedScoreVal').style.color = '#ef4444';
      document.getElementById('finalActionVal').style.color = '#ef4444';
    }} else if (d.threat_level === 'WARNING') {{
      tb.className = 'badge badge-warn';
      tb.innerText = '⚠️ ELEVATED SUSPICION';
      document.getElementById('fusedScoreVal').style.color = '#f59e0b';
      document.getElementById('finalActionVal').style.color = '#f59e0b';
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = 'NORMAL MONITORING';
      document.getElementById('fusedScoreVal').style.color = 'var(--accent)';
      document.getElementById('finalActionVal').style.color = '#00e5a3';
    }}

    // Highlight active ladder row in table
    document.querySelectorAll('.ladder-table tr').forEach(r => r.classList.remove('ladder-row-active'));
    const shortId = d.rule_id.split('_')[0];
    const targetRow = document.getElementById('row_' + shortId);
    if (targetRow) {{
      targetRow.classList.add('ladder-row-active');
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
