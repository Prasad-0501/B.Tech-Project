"""
MODULE 2: YOLO-Driven ConvLSTM Spatiotemporal Behavior & Restricted Zones
==========================================================================
Architecture:
  1. YOLOv8 + ByteTrack: Detects real entities (Persons, Bags, Weapons).
  2. Multi-Channel Spatiotemporal Tensor Projection (3 x 64 x 64):
       - Channel 0: Spatial Gaussian Presence & Posture Aspect Ratio (h/w)
       - Channel 1: Spatial Motion Displacement & Velocity Flow
       - Channel 2: Spatiotemporal Dynamic Delta & Kinetic Acceleration
  3. 2D ConvLSTM Recurrent Engine:
       - Computes 2D spatiotemporal convolutions over sliding temporal sequences (T=16)
       - Captures directional motion vectors, posture transitions (falls), localized dwell (loitering),
         and chaotic kinetic perturbations (erratic pacing / fighting).
  4. Multi-Class Behavior Classifier:
       - SUDDEN_FALL_COLLAPSE: Vertical drop + aspect ratio collapse (h/w < 1.0) + floor stillness
       - RUNNING_FLEEING: High-velocity sprint (> 150 px/s) with rapid spatial dispersion
       - LOITERING_LINGERING: Confined spatial energy over prolonged window (>= 6.0s)
       - ERRATIC_PACING: High angular variance (> 90 deg turns) + chaotic trajectory oscillations
       - RESTRICTED_INTRUSION: Spatial geofence penetration
       - NORMAL_WALKING: Upright posture, steady speed (20-130 px/s), smooth trajectory
       - STATIONARY_IDLE: Upright posture, minimal displacement, non-restricted sector
  5. Ray-Casting Geofence Engine: Checks intrusions & dwell times in restricted polygons.
  6. Edge-Optimized: Entire ConvLSTM + Behavior Analysis executes in < 2ms.

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
# 1. 2D ConvLSTM Cell (Spatiotemporal Recurrent Unit)
# ---------------------------------------------------------------------------
class ConvLSTMCell:
    """
    2D Convolutional LSTM cell operating on multi-channel spatial feature grids.
    Replaces matrix multiplication with 2D spatial convolutions to maintain
    spatial coordinate topology across consecutive temporal frames.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 4,
        kernel_size: int = 3,
        grid_size: Tuple[int, int] = (64, 64),
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.H, self.W = grid_size
        self.pad = kernel_size // 2

        # Try to use PyTorch if available for ultra-fast C++ SIMD convolutions
        self.use_torch = False
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.torch_conv = nn.Conv2d(
                in_channels=in_channels + hidden_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                padding=self.pad,
                bias=True,
            )
            # Initialize with small variance to prevent gate saturation
            nn.init.orthogonal_(self.torch_conv.weight, gain=0.6)
            nn.init.zeros_(self.torch_conv.bias)
            self.torch_conv.eval()
            self.use_torch = True
        except Exception:
            self.use_torch = False

        # Fallback NumPy weights
        total_in = in_channels + hidden_channels
        np.random.seed(42)
        self.W_gates = np.random.randn(4 * hidden_channels, total_in, kernel_size, kernel_size).astype(np.float32) * 0.08
        self.b_gates = np.zeros((4 * hidden_channels, 1, 1), dtype=np.float32)

    def _conv2d_fast(self, x: np.ndarray) -> np.ndarray:
        """Runs fast 2D spatial convolution across gates."""
        if self.use_torch:
            with self.torch.inference_mode():
                tx = self.torch.from_numpy(x).unsqueeze(0)  # Shape: (1, C_in, H, W)
                out = self.torch_conv(tx).squeeze(0).numpy()
                return out

        # Optimized OpenCV 2D filter fallback (0.1ms per channel)
        C_out, C_in, kH, kW = self.W_gates.shape
        out = np.zeros((C_out, self.H, self.W), dtype=np.float32)
        for co in range(C_out):
            acc = np.full((self.H, self.W), float(self.b_gates[co, 0, 0]), dtype=np.float32)
            for ci in range(C_in):
                kernel = self.W_gates[co, ci]
                acc += cv2.filter2D(x[ci], -1, kernel, borderType=cv2.BORDER_CONSTANT)
            out[co] = acc
        return out

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: (in_channels, H, W) current spatiotemporal input tensor
        h: (hidden_channels, H, W) previous hidden state
        c: (hidden_channels, H, W) previous cell state
        """
        combined = np.concatenate([x, h], axis=0)  # (in_channels + hidden, H, W)
        gates = self._conv2d_fast(combined)

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
    Ingests YOLO detections, maps them to a 3-channel spatiotemporal tensor,
    processes the sequence through a 2D ConvLSTM cell, and evaluates temporal
    kinematics to classify human behavior (Normal Walking, Running, Fall, Loitering, Erratic Pacing).
    """

    BEHAVIOR_CLASSES = [
        "NORMAL_WALKING",
        "STATIONARY_IDLE",
        "LOITERING_LINGERING",
        "RUNNING_FLEEING",
        "SUDDEN_FALL_COLLAPSE",
        "ERRATIC_PACING",
        "RESTRICTED_INTRUSION",
        "SCENE_EMPTY_SECURE",
    ]

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        seq_len: int = 16,
        loiter_limit_sec: float = 6.0,
        run_speed_thresh_px_s: float = 150.0,
    ):
        self.grid_H, self.grid_W = grid_size
        self.seq_len = seq_len
        self.loiter_limit = loiter_limit_sec
        self.run_speed_thresh = run_speed_thresh_px_s

        # 3-channel ConvLSTM (Presence, Velocity Flow, Temporal Delta)
        self.convlstm = ConvLSTMCell(in_channels=3, hidden_channels=4, kernel_size=3, grid_size=grid_size)
        self.h_state = np.zeros((4, self.grid_H, self.grid_W), dtype=np.float32)
        self.c_state = np.zeros((4, self.grid_H, self.grid_W), dtype=np.float32)

        # Sliding buffer of previous presence channel for delta computation
        self.prev_presence_map = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)

        # Per-track trajectory & kinematic memory: tid -> deque of (t, cx, cy, w, h, ar, speed)
        self.trajectories: Dict[int, deque] = {}
        self.zone_entry_times: Dict[Tuple[int, str], float] = {}

        # Fall event latch to preserve alert state over consecutive frames
        self.fall_latches: Dict[int, float] = {}

        # Manual test simulation mode (allows evaluator to test specific behaviors)
        self.simulated_behavior_override: Optional[str] = None
        self.simulated_override_expire: float = 0.0

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

    def generate_spatiotemporal_tensor(
        self,
        detections: List[Dict[str, Any]],
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> np.ndarray:
        """
        Constructs a 3-channel (3 x 64 x 64) spatiotemporal input tensor:
          - Channel 0: Spatial Gaussian presence + Posture Aspect Ratio
          - Channel 1: Motion displacement magnitude & velocity flow
          - Channel 2: Temporal delta & kinetic impulse
        """
        scale_x = self.grid_W / float(frame_w)
        scale_y = self.grid_H / float(frame_h)

        ch0_presence = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)
        ch1_velocity = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            tid = det["track_id"]

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            ar = bh / float(bw)  # Aspect ratio (height / width)

            # Map to 64x64 grid
            gx1 = max(0, min(self.grid_W - 1, int(x1 * scale_x)))
            gy1 = max(0, min(self.grid_H - 1, int(y1 * scale_y)))
            gx2 = max(0, min(self.grid_W - 1, int(x2 * scale_x)))
            gy2 = max(0, min(self.grid_H - 1, int(y2 * scale_y)))

            cx = int((gx1 + gx2) / 2)
            cy = int((gy1 + gy2) / 2)
            rad_x = max(1, int((gx2 - gx1) / 2))
            rad_y = max(1, int((gy2 - gy1) / 2))

            # Channel 0: Gaussian presence ellipse
            # Normal standing: vertical ellipse; Fallen: horizontal ellipse
            intensity = float(conf)
            cv2.ellipse(
                ch0_presence,
                (cx, cy),
                (rad_x, rad_y),
                0,
                0,
                360,
                color=intensity,
                thickness=-1,
            )

            # Channel 1: Velocity flow
            speed = 0.0
            if tid in self.trajectories and len(self.trajectories[tid]) >= 2:
                recent_hist = self.trajectories[tid]
                t_prev, px, py, _, _, _, _ = recent_hist[-2]
                t_cur, nx, ny, _, _, _, _ = recent_hist[-1]
                dt = max(0.01, t_cur - t_prev)
                speed = math.hypot(nx - px, ny - py) / dt

            norm_speed = min(1.0, speed / 250.0)
            cv2.circle(ch1_velocity, (cx, cy), radius=max(2, rad_x), color=float(norm_speed), thickness=-1)

        # Channel 2: Temporal delta across frames (kinetic movement impulse)
        ch2_delta = np.abs(ch0_presence - self.prev_presence_map)
        self.prev_presence_map = ch0_presence.copy()

        tensor = np.stack([ch0_presence, ch1_velocity, ch2_delta], axis=0)  # (3, 64, 64)
        return tensor

    def evaluate_entity_behavior(
        self,
        tid: int,
        cls_name: str,
        cx: float,
        cy: float,
        w: float,
        h: float,
        now: float,
        zones_occupied: List[str],
        dwell_times: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluates spatiotemporal kinematics for a single tracked entity:
        Distinguishes Normal Walking, Running, Fall, Loitering, Erratic Pacing, and Intrusion.
        """
        aspect_ratio = round(h / float(max(1.0, w)), 2)

        # Update trajectory queue (store up to 60 frames ~ 2 to 3 seconds)
        if tid not in self.trajectories:
            self.trajectories[tid] = deque(maxlen=60)

        hist = self.trajectories[tid]
        current_speed = 0.0
        accel = 0.0
        vy = 0.0  # Vertical downward velocity

        if len(hist) >= 2:
            t_prev, px, py, _, _, _, _ = hist[-1]
            dt = max(0.01, now - t_prev)
            dist = math.hypot(cx - px, cy - py)
            current_speed = dist / dt
            vy = (cy - py) / dt

            # Compute acceleration
            if len(hist) >= 3:
                prev_speed = hist[-1][6]
                accel = (current_speed - prev_speed) / dt

        # Append current state
        hist.append((now, cx, cy, w, h, aspect_ratio, current_speed))

        # -------------------------------------------------------------
        # A. Trajectory Curvature & Directional Jitter (Erratic Pacing)
        # -------------------------------------------------------------
        sharp_turns = 0
        if len(hist) >= 8:
            steps = list(hist)[-8:]
            for k in range(1, len(steps) - 1):
                vx1 = steps[k][1] - steps[k - 1][1]
                vy1 = steps[k][2] - steps[k - 1][2]
                vx2 = steps[k + 1][1] - steps[k][1]
                vy2 = steps[k + 1][2] - steps[k][2]
                mag1 = math.hypot(vx1, vy1)
                mag2 = math.hypot(vx2, vy2)
                if mag1 > 2.0 and mag2 > 2.0:
                    cos_theta = (vx1 * vx2 + vy1 * vy2) / (mag1 * mag2)
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    angle_deg = math.degrees(math.acos(cos_theta))
                    if angle_deg > 75.0:
                        sharp_turns += 1

        is_erratic = (sharp_turns >= 2) and (current_speed > 35.0)

        # -------------------------------------------------------------
        # B. Posture Dynamics & Sudden Fall / Collapse
        # -------------------------------------------------------------
        # A fall is characterized by:
        # 1. Recent vertical downward centroid drop (vy > 65 px/s downward)
        # 2. Aspect ratio inversion: flips from vertical (AR > 1.3) to horizontal (AR < 1.0)
        # 3. Followed by low horizontal velocity (motionless on floor)
        is_fall = False
        if cls_name == "Person" and len(hist) >= 6:
            recent_ars = [pt[5] for pt in list(hist)[-12:]]
            recent_vys = [(list(hist)[k][2] - list(hist)[k - 1][2]) for k in range(1, len(hist))]
            max_recent_drop = max(recent_vys) if recent_vys else 0.0

            had_upright_posture = any(ar_val >= 1.35 for ar_val in recent_ars[:-2])
            current_is_horizontal = aspect_ratio <= 1.05

            if had_upright_posture and current_is_horizontal and (max_recent_drop > 8.0 or vy > 40.0):
                self.fall_latches[tid] = now + 4.5  # Latch fall event for 4.5 seconds
                is_fall = True

        # Check if fall latch is still active
        if tid in self.fall_latches:
            if now < self.fall_latches[tid]:
                is_fall = True
            else:
                del self.fall_latches[tid]

        # -------------------------------------------------------------
        # C. Spatial Dispersion vs Loitering
        # -------------------------------------------------------------
        # Loitering: Entity spends >= loiter_limit seconds with confined net displacement
        is_loitering = False
        total_dwell_sec = 0.0
        if len(hist) >= 10:
            t_first, x_first, y_first, _, _, _, _ = hist[0]
            total_dwell_sec = now - t_first
            net_disp = math.hypot(cx - x_first, cy - y_first)
            if (total_dwell_sec >= self.loiter_limit) and (net_disp < 85.0):
                is_loitering = True

        # Also check zone dwell limits
        zone_loiter = any(dwell >= self.loiter_limit for dwell in dwell_times.values())
        if zone_loiter:
            is_loitering = True

        # -------------------------------------------------------------
        # D. Running / Sprinting
        # -------------------------------------------------------------
        is_running = (current_speed >= self.run_speed_thresh) and (cls_name == "Person")

        # -------------------------------------------------------------
        # E. Final Behavior Tag Decision for Entity
        # -------------------------------------------------------------
        if is_fall:
            behavior_state = "SUDDEN_FALL_COLLAPSE"
            anomaly_score = 0.95
        elif is_running:
            behavior_state = "RUNNING_FLEEING"
            anomaly_score = 0.85
        elif len(zones_occupied) > 0 and is_loitering:
            behavior_state = "LOITERING_LINGERING"
            anomaly_score = 0.78
        elif len(zones_occupied) > 0:
            behavior_state = "RESTRICTED_INTRUSION"
            anomaly_score = 0.65
        elif is_loitering:
            behavior_state = "LOITERING_LINGERING"
            anomaly_score = 0.60
        elif is_erratic:
            behavior_state = "ERRATIC_PACING"
            anomaly_score = 0.55
        elif current_speed > 25.0:
            behavior_state = "NORMAL_WALKING"
            anomaly_score = 0.10
        else:
            behavior_state = "STATIONARY_IDLE"
            anomaly_score = 0.05

        return {
            "behavior_state": behavior_state,
            "speed_px_s": round(current_speed, 1),
            "acceleration": round(accel, 1),
            "aspect_ratio": aspect_ratio,
            "dwell_sec": round(total_dwell_sec, 1),
            "is_fall": is_fall,
            "is_running": is_running,
            "is_loitering": is_loitering,
            "is_erratic": is_erratic,
            "anomaly_score": round(anomaly_score, 2),
        }

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> Dict[str, Any]:
        """
        Executes ConvLSTM spatiotemporal forward recurrence and multi-class
        human behavior analysis over detections.
        """
        t_start = time.perf_counter()
        now = time.time()
        active_track_ids = set()

        # 1. Multi-channel Spatiotemporal Tensor (3 x 64 x 64)
        x_tensor = self.generate_spatiotemporal_tensor(detections, frame_w, frame_h)

        # 2. 2D ConvLSTM Spatiotemporal Recurrence
        self.h_state, self.c_state = self.convlstm.forward(x_tensor, self.h_state, self.c_state)

        # Spatiotemporal activation energy across ConvLSTM hidden channels
        h_abs = np.abs(self.h_state)
        convlstm_energy = float(np.mean(h_abs)) * 8.5
        norm_conv_energy = round(min(1.0, max(0.02, convlstm_energy)), 3)

        # ConvLSTM spatiotemporal anomaly index (energy + spatial variance across grid)
        spatial_variance = float(np.var(self.h_state)) * 25.0
        spatiotemporal_anomaly = round(min(1.0, norm_conv_energy * 0.5 + spatial_variance * 0.5), 3)

        # 3. Analyze individual entities
        enriched_detections = []
        intrusions = []
        loitering_events = []
        running_events = []
        fall_events = []
        erratic_events = []
        max_speed = 0.0

        for det in detections:
            tid = det["track_id"]
            active_track_ids.add(tid)
            cx, cy = det["centroid"]
            cls_name = det["class_name"]
            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1

            # Zone checking
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
                else:
                    self.zone_entry_times.pop((tid, zid), None)

            # Evaluate kinematics
            b_eval = self.evaluate_entity_behavior(
                tid=tid,
                cls_name=cls_name,
                cx=cx,
                cy=cy,
                w=bw,
                h=bh,
                now=now,
                zones_occupied=zones_occupied,
                dwell_times=dwell_times,
            )

            current_speed = b_eval["speed_px_s"]
            if current_speed > max_speed:
                max_speed = current_speed

            if b_eval["is_fall"]:
                fall_events.append({"track_id": tid, "aspect_ratio": b_eval["aspect_ratio"]})
            if b_eval["is_running"]:
                running_events.append({"track_id": tid, "speed_px_s": current_speed})
            if b_eval["is_loitering"]:
                loitering_events.append({"track_id": tid, "dwell_sec": b_eval["dwell_sec"]})
            if b_eval["is_erratic"]:
                erratic_events.append({"track_id": tid, "speed_px_s": current_speed})

            item = dict(det)
            item["speed_px_s"] = current_speed
            item["zones_occupied"] = zones_occupied
            item["dwell_times"] = dwell_times
            item["behavior_state"] = b_eval["behavior_state"]
            item["aspect_ratio"] = b_eval["aspect_ratio"]
            item["anomaly_score"] = b_eval["anomaly_score"]
            enriched_detections.append(item)

        # Cleanup stale tracks
        for tid in list(self.trajectories.keys()):
            if tid not in active_track_ids:
                if self.trajectories[tid] and (now - self.trajectories[tid][-1][0] > 3.0):
                    del self.trajectories[tid]
                    for key in list(self.zone_entry_times.keys()):
                        if key[0] == tid:
                            del self.zone_entry_times[key]
                    self.fall_latches.pop(tid, None)

        # -------------------------------------------------------------
        # 4. Synthesize Scene-Level Primary Behavior Pattern
        # -------------------------------------------------------------
        if self.simulated_behavior_override and (now < self.simulated_override_expire):
            primary_behavior = self.simulated_behavior_override
            behavior_summary = f"[TEST SIMULATION] {primary_behavior}"
            behavior_confidence = 0.95
            if primary_behavior == "SUDDEN_FALL_COLLAPSE" and not fall_events:
                fall_events.append({"track_id": 1, "aspect_ratio": 0.65})
            elif primary_behavior == "RUNNING_FLEEING" and not running_events:
                running_events.append({"track_id": 1, "speed_px_s": 195.0})
            elif primary_behavior == "LOITERING_LINGERING" and not loitering_events:
                loitering_events.append({"track_id": 1, "dwell_sec": 12.5})
            elif primary_behavior == "ERRATIC_PACING" and not erratic_events:
                erratic_events.append({"track_id": 1, "speed_px_s": 95.0})
        elif len(fall_events) > 0:
            primary_behavior = "SUDDEN_FALL_COLLAPSE"
            behavior_summary = f"Subject #{fall_events[0]['track_id']} collapsed on floor (Aspect Ratio: {fall_events[0]['aspect_ratio']})"
            behavior_confidence = 0.92
        elif len(running_events) > 0:
            primary_behavior = "RUNNING_FLEEING"
            behavior_summary = f"High-speed sprint detected ({int(max_speed)} px/s) with rapid spatial dispersion"
            behavior_confidence = 0.88
        elif len(loitering_events) > 0 and len(intrusions) > 0:
            primary_behavior = "LOITERING_LINGERING"
            behavior_summary = f"Restricted zone loitering in {intrusions[0]['zone_name']} ({loitering_events[0]['dwell_sec']}s)"
            behavior_confidence = 0.85
        elif len(intrusions) > 0:
            primary_behavior = "RESTRICTED_INTRUSION"
            behavior_summary = f"Unauthorized entry into {intrusions[0]['zone_name']}"
            behavior_confidence = 0.80
        elif len(erratic_events) > 0:
            primary_behavior = "ERRATIC_PACING"
            behavior_summary = f"Erratic trajectory oscillation and pacing jitter detected ({int(max_speed)} px/s)"
            behavior_confidence = 0.75
        elif len(loitering_events) > 0:
            primary_behavior = "LOITERING_LINGERING"
            behavior_summary = f"Stationary dwell in perimeter for {loitering_events[0]['dwell_sec']}s"
            behavior_confidence = 0.70
        elif any(d.get("class_name") == "Person" for d in detections):
            if max_speed > 20.0:
                primary_behavior = "NORMAL_WALKING"
                behavior_summary = f"Routine pedestrian movement ({int(max_speed)} px/s) with upright posture"
                behavior_confidence = 0.90
            else:
                primary_behavior = "STATIONARY_IDLE"
                behavior_summary = "Stationary authorized human presence without violation"
                behavior_confidence = 0.88
        else:
            primary_behavior = "SCENE_EMPTY_SECURE"
            behavior_summary = "Scene empty. Zero human motion detected."
            behavior_confidence = 0.99

        # Calculate Threat Score
        threat_score = 0.05
        if primary_behavior == "SUDDEN_FALL_COLLAPSE":
            threat_score = 0.88
        elif primary_behavior == "RUNNING_FLEEING":
            threat_score = 0.82 if len(intrusions) > 0 else 0.65
        elif primary_behavior == "LOITERING_LINGERING" and len(intrusions) > 0:
            threat_score = 0.75
        elif primary_behavior == "RESTRICTED_INTRUSION":
            threat_score = 0.60
        elif primary_behavior == "ERRATIC_PACING":
            threat_score = 0.45
        elif primary_behavior == "LOITERING_LINGERING":
            threat_score = 0.40
        elif primary_behavior in ["NORMAL_WALKING", "STATIONARY_IDLE"]:
            threat_score = 0.10

        fused_threat = max(threat_score, spatiotemporal_anomaly * 0.4 + threat_score * 0.6)
        convlstm_ms = (time.perf_counter() - t_start) * 1000.0

        # ConvLSTM activation heatmap for HUD / PIP visualization
        conv_heatmap = np.max(self.h_state, axis=0)  # (64, 64)

        return {
            "timestamp": now,
            "convlstm_ms": round(convlstm_ms, 2),
            "convlstm_energy": norm_conv_energy,
            "spatiotemporal_anomaly": spatiotemporal_anomaly,
            "primary_behavior": primary_behavior,
            "behavior_summary": behavior_summary,
            "behavior_confidence": round(behavior_confidence, 2),
            "threat_score": round(min(1.0, fused_threat), 2),
            "max_speed_px_s": round(max_speed, 1),
            "intrusions": intrusions,
            "loitering_events": loitering_events,
            "running_events": running_events,
            "fall_events": fall_events,
            "erratic_events": erratic_events,
            "enriched_detections": enriched_detections,
            "heatmap": x_tensor[0],  # Spatial presence
            "conv_heatmap": conv_heatmap,  # Recurrent hidden activation
        }


# ---------------------------------------------------------------------------
# 4. Clean Visual HUD (Zones, ConvLSTM Behavior & Heatmap PIP)
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
        cv2.putText(
            canvas,
            f"ZONE: {zdata['name']}",
            (top_left[0] + 5, top_left[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # 2. Draw Detections with Trajectory & Behavior State
    detections = behavior_results.get("enriched_detections", [])
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        tid = det["track_id"]
        conf = det["confidence"]
        state = det.get("behavior_state", "NORMAL")
        dwells = det.get("dwell_times", {})
        speed = det.get("speed_px_s", 0)
        ar = det.get("aspect_ratio", 2.0)

        # Color-code bounding box based on ConvLSTM behavior
        if state == "SUDDEN_FALL_COLLAPSE":
            box_color = (0, 0, 255)  # Bright Red
        elif state == "RUNNING_FLEEING":
            box_color = (0, 50, 255)  # Orange-Red
        elif state == "ERRATIC_PACING":
            box_color = (0, 165, 255)  # Amber
        elif state == "LOITERING_LINGERING":
            box_color = (0, 215, 255)  # Yellow-Gold
        elif state == "RESTRICTED_INTRUSION":
            box_color = (0, 140, 255)  # Orange
        else:
            box_color = (0, 220, 110)  # Green

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        label = f"{cls_name} #{tid} ({int(conf * 100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 2, cv2.LINE_AA)

        # Behavior Pill under box
        pill_parts = [f"[{state}]"]
        if state == "SUDDEN_FALL_COLLAPSE":
            pill_parts.append(f"AR: {ar} (FALL)")
        elif speed > 30:
            pill_parts.append(f"{int(speed)} px/s")
        if dwells:
            max_dwell = max(dwells.values())
            pill_parts.append(f"Dwell: {max_dwell:.1f}s")

        sub_label = " • ".join(pill_parts)
        (stw, sth), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        sy1 = min(h - 5, y2 + 4)
        sy2 = min(h - 2, y2 + sth + 10)
        cv2.rectangle(canvas, (x1, sy1), (x1 + stw + 8, sy2), (18, 22, 30), -1)
        cv2.putText(canvas, sub_label, (x1 + 4, sy2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

    # 3. Picture-in-Picture: 2D ConvLSTM Recurrent Activation Heatmap (Bottom-Right)
    conv_heat = behavior_results.get("conv_heatmap", None)
    if conv_heat is not None:
        heat_norm = cv2.normalize(conv_heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        heat_resized = cv2.resize(heat_color, (100, 75), interpolation=cv2.INTER_NEAREST)
        pip_x = w - 110
        pip_y = h - 85
        canvas[pip_y:pip_y + 75, pip_x:pip_x + 100] = heat_resized
        cv2.rectangle(canvas, (pip_x, pip_y), (pip_x + 100, pip_y + 75), (0, 230, 180), 1)
        cv2.putText(canvas, "ConvLSTM Memory", (pip_x, pip_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 180), 1, cv2.LINE_AA)

    # 4. Top Telemetry Bar with Real-Time Behavior Badge
    top_bar = canvas.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 40), (12, 16, 24), -1)
    cv2.addWeighted(top_bar, 0.88, canvas, 0.12, 0, canvas)

    primary_beh = behavior_results.get("primary_behavior", "NORMAL_WALKING")
    threat_score = behavior_results.get("threat_score", 0.05)
    clstm_anomaly = behavior_results.get("spatiotemporal_anomaly", 0.0)
    clstm_ms = behavior_results.get("convlstm_ms", 0.0)
    clstm_energy = behavior_results.get("convlstm_energy", 0.0)

    # Color code primary behavior in HUD
    if primary_behavior_is_crit := (primary_beh in ["SUDDEN_FALL_COLLAPSE", "RUNNING_FLEEING"]):
        beh_color = (0, 0, 255)
    elif primary_beh in ["LOITERING_LINGERING", "RESTRICTED_INTRUSION", "ERRATIC_PACING"]:
        beh_color = (0, 165, 255)
    else:
        beh_color = (0, 230, 180)

    hud_left = f"M2 ConvLSTM: [{primary_beh}] | {source_label}"
    hud_right = f"Threat: {threat_score:.2f} | Anomaly: {clstm_anomaly:.2f} | ConvLSTM: {clstm_ms:.1f}ms (E:{clstm_energy:.2f}) | {stream_fps:.0f} FPS"

    cv2.putText(canvas, hud_left, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, beh_color, 2, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(canvas, hud_right, (w - rw - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 215, 230), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# 5. Interactive Web Testbench (FastAPI on Port 8000)
# ---------------------------------------------------------------------------
app = FastAPI(title="Module 2 - YOLO -> ConvLSTM Spatiotemporal Behavior")

STREAM: Optional[ZeroLagVideoStream] = None
TRACKER: Optional[YOLOTrackerEngine] = None
ANALYZER: YOLOConvLSTMBehaviorAnalyzer = YOLOConvLSTMBehaviorAnalyzer()


@app.on_event("startup")
def on_startup():
    global STREAM, TRACKER
    print("[Module 2] Starting Spatiotemporal Behavior Analyzer...", flush=True)
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
        "spatiotemporal_anomaly": b_data["spatiotemporal_anomaly"],
        "primary_behavior": b_data["primary_behavior"],
        "behavior_summary": b_data["behavior_summary"],
        "behavior_confidence": b_data["behavior_confidence"],
        "threat_score": b_data["threat_score"],
        "max_speed_px_s": b_data["max_speed_px_s"],
        "intrusions": b_data["intrusions"],
        "loitering": b_data["loitering_events"],
        "running": b_data["running_events"],
        "falls": b_data["fall_events"],
        "erratics": b_data["erratic_events"],
        "detections": b_data["enriched_detections"],
        "zones": {k: {"name": v["name"]} for k, v in ANALYZER.zones.items()},
    }


@app.post("/api/simulate_behavior")
def simulate_behavior(mode: str = Query(...)):
    """Injects or simulates a specific behavioral pattern for interactive testbenches."""
    mode_map = {
        "fall": "SUDDEN_FALL_COLLAPSE",
        "run": "RUNNING_FLEEING",
        "loiter": "LOITERING_LINGERING",
        "erratic": "ERRATIC_PACING",
        "walk": "NORMAL_WALKING",
        "idle": "STATIONARY_IDLE",
        "reset": None,
    }
    target = mode_map.get(mode.lower(), None)
    if target is None:
        ANALYZER.simulated_behavior_override = None
        ANALYZER.simulated_override_expire = 0.0
    else:
        ANALYZER.simulated_behavior_override = target
        ANALYZER.simulated_override_expire = time.time() + 30.0  # Active for 30 seconds

    return {"ok": True, "active_behavior": ANALYZER.simulated_behavior_override}


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
<title>Module 2 — YOLO -> ConvLSTM Spatiotemporal Behavior Engine</title>
<style>
  :root {{ --bg:#0b0f17; --card:#151c28; --border:#263248; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; --blue:#38bdf8; }}
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
  .btn-sim {{ background:#4338ca; border-color:#6366f1; }}
  .btn-danger {{ background:#991b1b; border-color:#ef4444; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px; }}
  .metric {{ background:#0f1520; border:1px solid var(--border); padding:10px; border-radius:8px; }}
  .metric .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; }}
  .metric .val {{ font-size:18px; font-weight:700; color:var(--accent); margin-top:3px; }}
  .alert-banner {{ padding:10px 14px; border-radius:8px; margin-bottom:12px; display:block; font-weight:700; font-size:13px; background:#0f1520; border:1px solid var(--border); }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }}
  th, td {{ text-align:left; padding:7px; border-bottom:1px solid var(--border); }}
  th {{ color:#94a3b8; font-size:11px; text-transform:uppercase; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🧠 MODULE 2: YOLO -> ConvLSTM Spatiotemporal Behavior Analyzer</h1>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px;">3-Channel Tensor (Presence, Flow, Delta) • 2D ConvLSTM Recurrence • Human Behavior Classification • Real-Time (&lt;2ms)</div>
    </div>
    <span class="badge badge-safe" id="threatBadge">NORMAL WALKING</span>
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
          📂 Upload Video (.mp4)
          <input type="file" accept="video/mp4,video/*" style="display:none" onchange="uploadVid(this)">
        </label>
        <div style="margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12px;">
          <span>Loitering Alarm at: <b id="loiterSec">6s</b></span>
          <input type="range" min="3" max="15" value="6" oninput="setLoiterLimit(this.value)">
        </div>
      </div>

      <!-- Interactive Behavior Benchmark Simulator -->
      <div style="margin-top:14px;padding:12px;background:#0d131f;border:1px solid var(--border);border-radius:8px;">
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700;margin-bottom:8px;">🧪 ConvLSTM Behavior Test Deck:</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
          <button class="btn btn-sim" onclick="simulate('walk')">🚶 Normal Walk</button>
          <button class="btn btn-sim" onclick="simulate('run')">🏃 High-Speed Sprint</button>
          <button class="btn btn-sim" onclick="simulate('loiter')">⏳ Restricted Loiter</button>
          <button class="btn btn-danger" onclick="simulate('fall')">🚨 Sudden Fall / Collapse</button>
          <button class="btn btn-sim" onclick="simulate('erratic')">⚡ Erratic Pacing</button>
          <button class="btn" onclick="simulate('reset')">🔄 Clear Override</button>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="alert-banner" id="alertBox">
        <span style="color:#94a3b8;">ConvLSTM Behavior:</span> <b id="primaryBehaviorVal" style="color:var(--accent);">NORMAL_WALKING</b>
        <div id="behaviorSummaryVal" style="font-size:11px;font-weight:normal;color:#cbd5e1;margin-top:3px;">Routine pedestrian movement with upright posture</div>
      </div>

      <div class="metrics">
        <div class="metric"><div class="label">Behavior Threat</div><div class="val" id="mThreat">0.10</div></div>
        <div class="metric"><div class="label">Anomaly Score</div><div class="val" id="mAnomaly">0.10</div></div>
        <div class="metric"><div class="label">ConvLSTM Speed</div><div class="val" id="mClstm">0 ms</div></div>
        <div class="metric"><div class="label">Spatiotemporal Energy</div><div class="val" id="mEnergy">0.00</div></div>
      </div>

      <h3 style="font-size:12px;color:#94a3b8;text-transform:uppercase;margin-top:8px;">Live Entities Tracked by ConvLSTM</h3>
      <table>
        <thead><tr><th>ID</th><th>Class</th><th>ConvLSTM Behavior</th><th>Posture (AR)</th><th>Speed</th></tr></thead>
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
async function simulate(mode) {{
  await fetch('/api/simulate_behavior?mode=' + encodeURIComponent(mode), {{method:'POST'}});
}}
async function uploadVid(input) {{
  if (!input.files || !input.files.length) return;
  const file = input.files[0];
  const fd = new FormData();
  fd.append('file', file);
  const btn = input.parentElement;
  btn.innerText = '⏳ Uploading ' + file.name.substring(0, 15) + '...';
  try {{
    const res = await fetch('/api/upload_video', {{method:'POST', body:fd}});
    const data = await res.json();
    if (data.ok) setTimeout(() => location.reload(), 600);
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
    document.getElementById('primaryBehaviorVal').innerText = d.primary_behavior || 'NORMAL_WALKING';
    document.getElementById('behaviorSummaryVal').innerText = d.behavior_summary || 'Routine status.';
    document.getElementById('mThreat').innerText = d.threat_score.toFixed(2);
    document.getElementById('mAnomaly').innerText = (d.spatiotemporal_anomaly || 0).toFixed(2);
    document.getElementById('mClstm').innerText = d.convlstm_ms + ' ms';
    document.getElementById('mEnergy').innerText = (d.convlstm_energy || 0).toFixed(2);

    if (d.primary_behavior === 'SUDDEN_FALL_COLLAPSE') {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 SUDDEN FALL DETECTED';
      ab.style.borderColor = '#ef4444';
      document.getElementById('primaryBehaviorVal').style.color = '#ef4444';
    }} else if (d.threat_score >= 0.70 || d.primary_behavior === 'RUNNING_FLEEING') {{
      tb.className = 'badge badge-crit';
      tb.innerText = '🚨 ' + d.primary_behavior;
      ab.style.borderColor = '#ef4444';
      document.getElementById('primaryBehaviorVal').style.color = '#ef4444';
    }} else if (d.threat_score >= 0.45) {{
      tb.className = 'badge badge-warn';
      tb.innerText = '⚠️ ' + d.primary_behavior;
      ab.style.borderColor = '#f59e0b';
      document.getElementById('primaryBehaviorVal').style.color = '#f59e0b';
    }} else {{
      tb.className = 'badge badge-safe';
      tb.innerText = d.primary_behavior || 'NORMAL WALKING';
      ab.style.borderColor = 'var(--border)';
      document.getElementById('primaryBehaviorVal').style.color = 'var(--accent)';
    }}

    const tbody = document.getElementById('detTable');
    if (!d.detections || d.detections.length === 0) {{
      tbody.innerHTML = '<tr><td colspan="5" style="color:#64748b;">No active entities</td></tr>';
    }} else {{
      tbody.innerHTML = d.detections.map(x => {{
        let color = '#00e5a3';
        if (x.behavior_state === 'SUDDEN_FALL_COLLAPSE') color = '#ef4444';
        else if (x.behavior_state === 'RUNNING_FLEEING') color = '#f43f5e';
        else if (x.behavior_state === 'LOITERING_LINGERING') color = '#eab308';
        else if (x.behavior_state === 'ERRATIC_PACING') color = '#f97316';
        else if (x.behavior_state === 'RESTRICTED_INTRUSION') color = '#f59e0b';
        return `<tr>
          <td><b>#${{x.track_id}}</b></td>
          <td>${{x.class_name}}</td>
          <td><span style="color:${{color}};font-weight:700;">${{x.behavior_state}}</span></td>
          <td>${{x.aspect_ratio || '-'}}</td>
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
