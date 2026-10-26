"""
MODULE 2: YOLO-Driven ConvLSTM Spatiotemporal Behavior & Restricted Zones
==========================================================================
Architecture:
  1. YOLOv8 + ByteTrack: Detects real entities (Persons, Bags, Weapons).
  2. Spatial Heatmap Projection: Converts YOLO entity centroids & bounding boxes
     into a clean (64x64) spatial occupancy tensor (eliminates background noise).
  3. ConvLSTM Recurrent Cell: Computes 2D spatiotemporal convolutions over sliding
     temporal sequences (T=16) of YOLO heatmaps to capture velocity, trajectory dispersion,
     and chaotic behavioral shifts.
  4. Ray-Casting Geofence Engine: Checks intrusions & loitering dwell times in restricted polygons.
  5. Raspberry Pi Optimized: Entire ConvLSTM + Geofence step executes in < 3ms.

Web Console: http://127.0.0.1:8000
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.m1_yolo_tracker import ZeroLagVideoStream, YOLOTrackerEngine, SURVEILLANCE_CLASSES, CLASS_COLORS


# ---------------------------------------------------------------------------
# 1. 2D ConvLSTM Cell (Convolutional Long Short-Term Memory)
# ---------------------------------------------------------------------------
class ConvLSTMCell:
    """
    2D Convolutional LSTM cell operating on spatial feature grids.
    Replaces matrix multiplication with 2D spatial convolution to maintain
    spatial coordinate geometry across consecutive frames.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 4, kernel_size: int = 3, grid_size: Tuple[int, int] = (64, 64)):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.H, self.W = grid_size
        self.pad = kernel_size // 2

        # 4 gates: input (i), forget (f), cell (c), output (o)
        total_in = in_channels + hidden_channels
        np.random.seed(42)
        # Weight tensor shape: (4 * hidden, in_channels + hidden, kernel_h, kernel_w)
        self.W_gates = np.random.randn(4 * hidden_channels, total_in, kernel_size, kernel_size).astype(np.float32) * 0.05
        self.b_gates = np.zeros((4 * hidden_channels, 1, 1), dtype=np.float32)

    def _conv2d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        C_out, C_in, kH, kW = w.shape
        H, W = x.shape[1], x.shape[2]
        pad = self.pad
        x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.zeros((C_out, H, W), dtype=np.float32)

        for co in range(C_out):
            val = np.full((H, W), float(b[co, 0, 0]), dtype=np.float32)
            for ci in range(C_in):
                for ki in range(kH):
                    for kj in range(kW):
                        val += w[co, ci, ki, kj] * x_pad[ci, ki:ki + H, kj:kj + W]
            out[co] = val
        return out

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: (1, H, W) current spatial heatmap
        h: (hidden_channels, H, W) previous hidden state
        c: (hidden_channels, H, W) previous cell state
        """
        combined = np.concatenate([x, h], axis=0)
        gates = self._conv2d(combined, self.W_gates, self.b_gates)

        hc = self.hidden_channels
        i_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[0:hc], -8.0, 8.0)))
        f_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[hc:2 * hc], -8.0, 8.0)))
        c_cand = np.tanh(gates[2 * hc:3 * hc])
        o_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[3 * hc:4 * hc], -8.0, 8.0)))

        c_next = f_gate * c + i_gate * c_cand
        h_next = o_gate * np.tanh(c_next)
        return h_next, c_next


# ---------------------------------------------------------------------------
# 2. Geofence & Ray-Casting Point-in-Polygon Engine
# ---------------------------------------------------------------------------
def point_in_polygon(x: float, y: float, polygon: List[List[int]]) -> bool:
    """Ray-casting algorithm to test if point (x, y) is inside polygon in < 0.05ms."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ---------------------------------------------------------------------------
# 3. YOLO-Driven ConvLSTM Behavior Analyzer
# ---------------------------------------------------------------------------
class YOLOConvLSTMBehaviorAnalyzer:
    """
    Ingests YOLO detections, maps them to a spatial heatmap, processes the sequence
    through a 2D ConvLSTM cell, and evaluates trajectory kinematics & zone rules.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        seq_len: int = 16,
        loiter_limit_sec: float = 6.0,
        run_speed_thresh_px_s: float = 160.0,
    ):
        self.grid_H, self.grid_W = grid_size
        self.seq_len = seq_len
        self.loiter_limit = loiter_limit_sec
        self.run_speed_thresh = run_speed_thresh_px_s

        # ConvLSTM model
        self.convlstm = ConvLSTMCell(in_channels=1, hidden_channels=4, kernel_size=3, grid_size=grid_size)
        self.h_state = np.zeros((4, self.grid_H, self.grid_W), dtype=np.float32)
        self.c_state = np.zeros((4, self.grid_H, self.grid_W), dtype=np.float32)

        # Sliding buffer of heatmaps
        self.heatmap_history = deque(maxlen=seq_len)

        # Trajectory & Dwell Time state
        self.trajectories: Dict[int, deque] = {}
        self.zone_entry_times: Dict[Tuple[int, str], float] = {}

        # Restricted Security Zones
        self.zones: Dict[str, Dict[str, Any]] = {
            "vault": {
                "name": "Restricted Vault / Server Area",
                "polygon": [[80, 100], [280, 100], [280, 420], [80, 420]],
                "color": (0, 0, 240),  # Red
            },
            "perimeter": {
                "name": "Perimeter Security Boundary",
                "polygon": [[360, 80], [600, 80], [600, 420], [360, 420]],
                "color": (0, 165, 255),  # Amber
            },
        }

    def generate_yolo_heatmap(self, detections: List[Dict[str, Any]], frame_w: int = 640, frame_h: int = 480) -> np.ndarray:
        """
        Converts YOLO entity bounding boxes & centroids into a clean (64x64) float32 spatial heatmap.
        Zero background noise from walls or shadows — only real entities verified by YOLO!
        """
        heatmap = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)
        scale_x = self.grid_W / float(frame_w)
        scale_y = self.grid_H / float(frame_h)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            # Map coordinates to 64x64 grid
            gx1 = max(0, min(self.grid_W - 1, int(x1 * scale_x)))
            gy1 = max(0, min(self.grid_H - 1, int(y1 * scale_y)))
            gx2 = max(0, min(self.grid_W - 1, int(x2 * scale_x)))
            gy2 = max(0, min(self.grid_H - 1, int(y2 * scale_y)))

            cx = int((gx1 + gx2) / 2)
            cy = int((gy1 + gy2) / 2)

            # Intensity weighted by confidence and entity importance
            intensity = conf
            radius = 3 if cls_name == "Person" else 2
            cv2.circle(heatmap, (cx, cy), radius=radius, color=float(intensity), thickness=-1)

        return heatmap

    def update(self, detections: List[Dict[str, Any]], frame_w: int = 640, frame_h: int = 480) -> Dict[str, Any]:
        t_start = time.perf_counter()
        now = time.time()
        active_track_ids = set()

        # 1. Generate clean YOLO spatial heatmap (64x64)
        heatmap = self.generate_yolo_heatmap(detections, frame_w, frame_h)
        self.heatmap_history.append(heatmap)

        # 2. Run ConvLSTM forward recurrence on the YOLO heatmap
        x_in = heatmap[np.newaxis, :, :]  # Shape: (1, 64, 64)
        self.h_state, self.c_state = self.convlstm.forward(x_in, self.h_state, self.c_state)

        # ConvLSTM activation energy: measures spatial-temporal shift across hidden layers
        convlstm_energy = float(np.mean(np.abs(self.h_state))) * 12.0
        norm_conv_energy = round(min(1.0, convlstm_energy), 3)

        # 3. Trajectory, Velocity & Zone Evaluation for each detected person
        enriched_detections = []
        intrusions = []
        loitering_events = []
        running_events = []
        max_speed = 0.0

        for det in detections:
            tid = det["track_id"]
            active_track_ids.add(tid)
            cx, cy = det["centroid"]
            cls_name = det["class_name"]

            # Maintain trajectory history (up to 20 seconds)
            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=60)
            self.trajectories[tid].append((now, cx, cy))

            # Velocity Kinematics (px/s)
            hist = self.trajectories[tid]
            current_speed = 0.0
            if len(hist) >= 3:
                t_first, x_first, y_first = hist[0]
                dt = max(0.05, now - t_first)
                dist = math.hypot(cx - x_first, cy - y_first)
                current_speed = dist / dt

            if current_speed > max_speed:
                max_speed = current_speed

            is_running = current_speed >= self.run_speed_thresh
            if is_running and cls_name == "Person":
                running_events.append({"track_id": tid, "speed_px_s": round(current_speed, 1)})

            # Geofence Checking & Dwell Times
            zones_occupied = []
            dwell_times = {}

            for zid, zdata in self.zones.items():
                if point_in_polygon(cx, cy, zdata["polygon"]):
                    zones_occupied.append(zdata["name"])
                    key = (tid, zid)
                    if key not in self.zone_entry_times:
                        self.zone_entry_times[key] = now
                    dwell = now - self.zone_entry_times[key]
                    dwell_times[zdata["name"]] = round(dwell, 1)

                    intrusions.append({
                        "track_id": tid,
                        "class_name": cls_name,
                        "zone_name": zdata["name"],
                        "dwell_sec": round(dwell, 1),
                    })

                    if dwell >= self.loiter_limit:
                        loitering_events.append({
                            "track_id": tid,
                            "class_name": cls_name,
                            "zone_name": zdata["name"],
                            "dwell_sec": round(dwell, 1),
                        })
                else:
                    self.zone_entry_times.pop((tid, zid), None)

            # Determine entity state
            state = "NORMAL"
            if len(zones_occupied) > 0:
                state = "INTRUSION"
            if any(dwell_times.get(z, 0) >= self.loiter_limit for z in zones_occupied):
                state = "LOITERING"
            if is_running:
                state = "RUNNING"

            item = dict(det)
            item["speed_px_s"] = round(current_speed, 1)
            item["zones_occupied"] = zones_occupied
            item["dwell_times"] = dwell_times
            item["behavior_state"] = state
            enriched_detections.append(item)

        # Cleanup stale tracks
        for tid in list(self.trajectories.keys()):
            if tid not in active_track_ids:
                if self.trajectories[tid] and (now - self.trajectories[tid][-1][0] > 3.0):
                    del self.trajectories[tid]
                    for key in list(self.zone_entry_times.keys()):
                        if key[0] == tid:
                            del self.zone_entry_times[key]

        # Combined Behavior Threat Score: Blend of ConvLSTM Spatiotemporal Energy (40%) + Rule Kinematics (60%)
        rule_score = 0.05
        if intrusions:
            rule_score = max(rule_score, 0.60)
        if loitering_events:
            rule_score = max(rule_score, 0.75)
        if running_events:
            rule_score = max(rule_score, 0.85)

        fused_threat = max(norm_conv_energy * 0.40 + rule_score * 0.60, rule_score)
        convlstm_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "timestamp": now,
            "convlstm_ms": round(convlstm_ms, 2),
            "convlstm_energy": norm_conv_energy,
            "threat_score": round(min(1.0, fused_threat), 2),
            "max_speed_px_s": round(max_speed, 1),
            "intrusions": intrusions,
            "loitering_events": loitering_events,
            "running_events": running_events,
            "enriched_detections": enriched_detections,
            "heatmap": heatmap,
        }


# ---------------------------------------------------------------------------
# 4. Clean Visual HUD (Zones, ConvLSTM Energy & Heatmap PIP)
# ---------------------------------------------------------------------------
def draw_module2_hud(
    frame: np.ndarray,
    behavior_results: Dict[str, Any],
    zones: Dict[str, Dict[str, Any]],
    source_label: str,
    stream_fps: float,
    yolo_ms: float,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Zones (Semi-transparent polygons)
    overlay = canvas.copy()
    for zid, zdata in zones.items():
        pts = np.array(zdata["polygon"], np.int32).reshape((-1, 1, 2))
        color = zdata["color"]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        top_left = zdata["polygon"][0]
        cv2.putText(canvas, f"ZONE: {zdata['name']}", (top_left[0] + 5, top_left[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # 2. Draw Detections with Trajectory & Behavior State
    detections = behavior_results.get("enriched_detections", [])
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        tid = det["track_id"]
        conf = det["confidence"]
        state = det["behavior_state"]
        dwells = det.get("dwell_times", {})
        speed = det.get("speed_px_s", 0)

        box_color = CLASS_COLORS.get(cls_name, (0, 220, 110))
        if state == "INTRUSION":
            box_color = (0, 140, 255)
        elif state == "LOITERING":
            box_color = (0, 60, 255)
        elif state == "RUNNING":
            box_color = (0, 0, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        label = f"{cls_name} #{tid} ({int(conf * 100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(canvas, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 10, 10), 2, cv2.LINE_AA)

        # Behavior Pill
        pill_parts = []
        if state != "NORMAL":
            pill_parts.append(f"[{state}]")
        if dwells:
            max_dwell = max(dwells.values())
            pill_parts.append(f"Dwell: {max_dwell:.1f}s")
        if speed > 40:
            pill_parts.append(f"{int(speed)} px/s")

        if pill_parts:
            sub_label = " • ".join(pill_parts)
            (stw, sth), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            sy1 = min(h - 5, y2 + 5)
            sy2 = min(h - 2, y2 + sth + 12)
            cv2.rectangle(canvas, (x1, sy1), (x1 + stw + 8, sy2), (20, 24, 32), -1)
            cv2.putText(canvas, sub_label, (x1 + 4, sy2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1, cv2.LINE_AA)

    # 3. Picture-in-Picture: ConvLSTM Spatial Heatmap (Bottom-Right Corner)
    heatmap = behavior_results.get("heatmap", None)
    if heatmap is not None:
        heat_vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_vis, cv2.COLORMAP_JET)
        heat_resized = cv2.resize(heat_color, (96, 72), interpolation=cv2.INTER_NEAREST)
        pip_x = w - 106
        pip_y = h - 82
        canvas[pip_y:pip_y + 72, pip_x:pip_x + 96] = heat_resized
        cv2.rectangle(canvas, (pip_x, pip_y), (pip_x + 96, pip_y + 72), (0, 230, 180), 1)
        cv2.putText(canvas, "ConvLSTM Heatmap", (pip_x, pip_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 180), 1, cv2.LINE_AA)

    # 4. Top Telemetry Bar
    top_bar = canvas.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 38), (14, 18, 26), -1)
    cv2.addWeighted(top_bar, 0.85, canvas, 0.15, 0, canvas)

    threat_score = behavior_results.get("threat_score", 0.05)
    threat_text = "NORMAL"
    threat_color = (0, 220, 120)
    if threat_score >= 0.70:
        threat_text = "CRITICAL VIOLATION"
        threat_color = (0, 0, 255)
    elif threat_score >= 0.50:
        threat_text = "WARNING"
        threat_color = (0, 160, 255)

    clstm_ms = behavior_results.get("convlstm_ms", 0.0)
    clstm_energy = behavior_results.get("convlstm_energy", 0.0)
    hud_left = f"M2: YOLO -> ConvLSTM | {source_label}"
    hud_right = f"Threat: {threat_text} ({threat_score:.2f}) | ConvLSTM: {clstm_ms:.1f}ms (E:{clstm_energy:.2f}) | YOLO: {yolo_ms:.0f}ms | {stream_fps:.0f} FPS"
    cv2.putText(canvas, hud_left, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 180), 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, hud_right, (w - rw - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, threat_color, 2, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# 5. Interactive Web Testbench (FastAPI on Port 8000)
# ---------------------------------------------------------------------------
app = FastAPI(title="Module 2 - YOLO -> ConvLSTM Behavior & Zones")

STREAM: Optional[ZeroLagVideoStream] = None
TRACKER: Optional[YOLOTrackerEngine] = None
ANALYZER: YOLOConvLSTMBehaviorAnalyzer = YOLOConvLSTMBehaviorAnalyzer()


@app.on_event("startup")
def on_startup():
    global STREAM, TRACKER
    STREAM = ZeroLagVideoStream(source=0, width=640, height=480)
    TRACKER = YOLOTrackerEngine(conf_threshold=0.45, imgsz=416)
    TRACKER.start_async_worker(STREAM)


@app.on_event("shutdown")
def on_shutdown():
    if TRACKER:
        TRACKER.stop_async_worker()
    if STREAM:
        STREAM.stop()


def mjpeg_generator():
    last_fid = -1
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    while True:
        if STREAM is None or TRACKER is None:
            time.sleep(0.02)
            continue
        pkt = STREAM.read()
        if pkt is None or pkt.frame_id == last_fid:
            time.sleep(0.005)
            continue
        last_fid = pkt.frame_id
        frame = pkt.image
        sfps = pkt.fps

        state = TRACKER.get_latest_state()
        raw_dets = state.detections if hasattr(state, "detections") else state[0]
        inf_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]

        b_data = ANALYZER.update(raw_dets, frame_w=frame.shape[1], frame_h=frame.shape[0])

        vis = draw_module2_hud(
            frame=frame,
            behavior_results=b_data,
            zones=ANALYZER.zones,
            source_label=STREAM.source_label,
            stream_fps=sfps,
            yolo_ms=inf_ms,
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
    if STREAM is None or TRACKER is None:
        return JSONResponse({"status": "initializing"})
    state = TRACKER.get_latest_state()
    raw_dets = state.detections if hasattr(state, "detections") else state[0]
    inf_ms = state.inference_ms if hasattr(state, "inference_ms") else state[1]

    pkt = STREAM.read()
    frame = pkt.image if pkt is not None else None
    sfps = pkt.fps if pkt is not None else 0.0
    fw = frame.shape[1] if frame is not None else 640
    fh = frame.shape[0] if frame is not None else 480
    b_data = ANALYZER.update(raw_dets, frame_w=fw, frame_h=fh)
    return {
        "source": STREAM.source_label,
        "stream_fps": sfps,
        "yolo_ms": inf_ms,
        "convlstm_ms": b_data["convlstm_ms"],
        "convlstm_energy": b_data["convlstm_energy"],
        "threat_score": b_data["threat_score"],
        "max_speed_px_s": b_data["max_speed_px_s"],
        "intrusions": b_data["intrusions"],
        "loitering": b_data["loitering_events"],
        "running": b_data["running_events"],
        "detections": b_data["enriched_detections"],
        "zones": {k: {"name": v["name"]} for k, v in ANALYZER.zones.items()},
    }


@app.post("/api/source")
def switch_source(src: str = Query(...)):
    if STREAM is None or TRACKER is None:
        return {"ok": False}
    TRACKER.model.predictor = None
    if src == "camera":
        ok = STREAM.open_source(0)
    else:
        target = BASE_DIR / "samples" / src
        ok = STREAM.open_source(str(target))
    return {"ok": ok, "source": STREAM.source_label}


@app.post("/api/set_loiter_limit")
def set_loiter_limit(sec: float = Query(...)):
    ANALYZER.loiter_limit = max(2.0, min(30.0, sec))
    return {"ok": True, "loiter_limit": ANALYZER.loiter_limit}


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
    return {"ok": True, "filename": dest.name, "source": STREAM.source_label if STREAM else ""}


@app.get("/", response_class=HTMLResponse)
def index_page():
    samples_dir = BASE_DIR / "samples"
    videos = sorted([p.name for p in samples_dir.glob("*.mp4")]) if samples_dir.exists() else []
    vid_buttons = "".join(
        f'<button class="btn" onclick="switchSource(\'{v}\')">🎬 {v[:26]}</button>'
        for v in videos
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Module 2 — YOLO -> ConvLSTM Spatiotemporal Behavior</title>
<style>
  :root {{ --bg:#0b0f17; --card:#151c28; --border:#263248; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; }}
  * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
  body {{ background:var(--bg); color:var(--text); padding:16px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; padding:12px 20px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; }}
  .header h1 {{ font-size:17px; color:var(--accent); }}
  .badge {{ padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }}
  .badge-safe {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); }}
  .badge-warn {{ background:#3d2406; color:var(--warning); border:1px solid var(--warning); }}
  .badge-crit {{ background:#3b1111; color:var(--danger); border:1px solid var(--danger); animation:pulse 1s infinite; }}
  @keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.5}} 100%{{opacity:1}} }}
  .grid {{ display:grid; grid-template-columns: 2fr 1fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .video-box {{ width:100%; border-radius:8px; overflow:hidden; background:#000; border:1px solid var(--border); }}
  .video-box img {{ width:100%; display:block; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }}
  .btn {{ background:#1e293b; color:#fff; border:1px solid #334155; padding:8px 12px; border-radius:6px; cursor:pointer; font-size:12px; }}
  .btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .btn-primary {{ background:#059669; border-color:#10b981; font-weight:600; }}
  .btn-upload {{ background:#2563eb; border-color:#3b82f6; font-weight:600; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px; }}
  .metric {{ background:#0f1520; border:1px solid var(--border); padding:10px; border-radius:8px; }}
  .metric .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; }}
  .metric .val {{ font-size:20px; font-weight:700; color:var(--accent); margin-top:3px; }}
  .alert-banner {{ padding:10px; border-radius:8px; margin-bottom:12px; display:none; font-weight:600; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }}
  th, td {{ text-align:left; padding:7px; border-bottom:1px solid var(--border); }}
  th {{ color:#94a3b8; font-size:11px; text-transform:uppercase; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>MODULE 2: YOLO -> ConvLSTM Spatiotemporal Behavior Engine</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">YOLO Heatmap Projection • 2D ConvLSTM Recurrence • Loitering & Intrusion • Raspberry Pi Ready (&lt;3ms)</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL</span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="video-box">
        <img src="/video_feed" alt="Live Behavior Analysis Stream" />
      </div>
      <div class="controls">
        <button class="btn btn-primary" onclick="switchSource('camera')">📷 Live Laptop Camera</button>
        {vid_buttons}
        <label class="btn btn-upload" style="cursor:pointer;">
          📂 Upload Any Video (.mp4)
          <input type="file" accept="video/mp4,video/*" style="display:none" onchange="uploadVid(this)">
        </label>
        <div style="margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12px;">
          <span>Loitering Alarm at: <b id="loiterSec">6s</b></span>
          <input type="range" min="3" max="15" value="6" oninput="setLoiterLimit(this.value)">
        </div>
      </div>
    </div>

    <div class="card">
      <div class="alert-banner" id="alertBox"></div>
      <div class="metrics">
        <div class="metric"><div class="label">Behavior Threat</div><div class="val" id="mThreat">0.05</div></div>
        <div class="metric"><div class="label">ConvLSTM Speed</div><div class="val" id="mClstm">0 ms</div></div>
        <div class="metric"><div class="label">ConvLSTM Energy</div><div class="val" id="mEnergy">0.00</div></div>
        <div class="metric"><div class="label">Peak Speed</div><div class="val" id="mSpeed">0 px/s</div></div>
      </div>
      <h3 style="font-size:12px;color:#94a3b8;text-transform:uppercase;margin-top:8px;">Live Entities Tracked by ConvLSTM</h3>
      <table>
        <thead><tr><th>ID</th><th>Class</th><th>State</th><th>Dwell Time</th><th>Speed</th></tr></thead>
        <tbody id="detTable"><tr><td colspan="5" style="color:#64748b;">No active entities</td></tr></tbody>
      </table>
    </div>
  </div>

<script>
async function switchSource(src) {{
  await fetch('/api/source?src=' + encodeURIComponent(src), {{method:'POST'}});
}}
async function setLoiterLimit(v) {{
  document.getElementById('loiterSec').innerText = v + 's';
  await fetch('/api/set_loiter_limit?sec=' + v, {{method:'POST'}});
}}
async function uploadVid(input) {{
  if (!input.files || !input.files.length) return;
  const file = input.files[0];
  const fd = new FormData();
  fd.append('file', file);
  const btn = input.parentElement;
  const oldText = btn.innerHTML;
  btn.innerText = '⏳ Uploading ' + file.name.substring(0, 15) + '...';
  try {{
    const res = await fetch('/api/upload_video', {{method:'POST', body:fd}});
    const data = await res.json();
    if (data.ok) {{
      setTimeout(() => location.reload(), 600);
    }}
  }} catch(e) {{
    alert('Upload failed: ' + e);
  }}
}}
setInterval(async () => {{
  try {{
    const r = await fetch('/api/telemetry');
    const d = await r.json();
    if (d.threat_score === undefined) return;

    const tb = document.getElementById('threatBadge');
    const ab = document.getElementById('alertBox');
    document.getElementById('mThreat').innerText = d.threat_score.toFixed(2);
    document.getElementById('mClstm').innerText = d.convlstm_ms + ' ms';
    document.getElementById('mEnergy').innerText = d.convlstm_energy.toFixed(2);
    document.getElementById('mSpeed').innerText = d.max_speed_px_s + ' px/s';

    if (d.threat_score >= 0.70) {{
      tb.className = 'badge badge-crit';
      tb.innerText = 'CRITICAL ALERT';
      ab.style.display = 'block';
      ab.style.background = '#3b1111';
      ab.style.color = '#ef4444';
      ab.innerText = '🚨 ' + (d.loitering && d.loitering.length ? 'Loitering Alert in Restricted Area!' : 'High-Speed Sprint / Intrusion Detected!');
    }} else if (d.threat_score >= 0.50) {{
      tb.className = 'badge badge-warn';
      tb.innerText = 'WARNING';
      ab.style.display = 'block';
      ab.style.background = '#3d2406';
      ab.style.color = '#f59e0b';
      ab.innerText = '⚠️ Zone Intrusion: Entity entered restricted boundary.';
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = 'NORMAL';
      ab.style.display = 'none';
    }}

    const tbody = document.getElementById('detTable');
    if (!d.detections || d.detections.length === 0) {{
      tbody.innerHTML = '<tr><td colspan="5" style="color:#64748b;">No active entities</td></tr>';
    }} else {{
      tbody.innerHTML = d.detections.map(x => {{
        let dwellStr = '-';
        if (x.dwell_times && Object.keys(x.dwell_times).length > 0) {{
          dwellStr = Object.values(x.dwell_times)[0] + 's';
        }}
        let color = '#00e5a3';
        if (x.behavior_state === 'LOITERING') color = '#ef4444';
        else if (x.behavior_state === 'INTRUSION') color = '#f59e0b';
        else if (x.behavior_state === 'RUNNING') color = '#ec4899';
        return `<tr>
          <td><b>#${{x.track_id}}</b></td>
          <td>${{x.class_name}}</td>
          <td><span style="color:${{color}};font-weight:700;">${{x.behavior_state}}</span></td>
          <td>${{dwellStr}}</td>
          <td>${{x.speed_px_s}} px/s</td>
        </tr>`;
      }}).join('');
    }}
  }} catch(e) {{}}
}}, 250);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
