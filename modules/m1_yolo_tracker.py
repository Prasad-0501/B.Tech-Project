# """
# MODULE 1: Real-Time YOLOv8 Object Detection & Multi-Object Tracking Engine
# ==========================================================================
# Dataset Backbone : MS COCO 2017 (80 classes, filtered to surveillance targets)
# Model            : Ultralytics YOLOv8n + ByteTrack Kalman Filter Tracking
# Architecture     : Decoupled 3-Stage Pipeline:
#                    1. ZeroLagVideoStream (DirectShow MJPG background grabber)
#                    2. YOLOTrackerEngine (Async YOLOv8 + ByteTrack thread)
#                    3. Fast MJPEG + WebSocket / JSON Telemetry Console (Port 8000)
# """

# import os
# import sys
# import time
# import threading
# import webbrowser
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple

# import cv2
# import numpy as np
# import uvicorn
# from fastapi import FastAPI, Query, UploadFile, File
# from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
# from ultralytics import YOLO

# BASE_DIR = Path(__file__).resolve().parent.parent
# if str(BASE_DIR) not in sys.path:
#     sys.path.insert(0, str(BASE_DIR))


# # ---------------------------------------------------------------------------
# # MS COCO Surveillance Target Classes
# # ---------------------------------------------------------------------------
# SURVEILLANCE_CLASSES: Dict[int, str] = {
#     0: "Person",
#     1: "Bicycle",
#     2: "Car",
#     3: "Motorcycle",
#     5: "Bus",
#     7: "Truck",
#     24: "Backpack",
#     26: "Handbag",
#     28: "Suitcase",
#     34: "Baseball Bat",
#     39: "Bottle",
#     43: "Knife",
#     63: "Laptop",
#     67: "Cell Phone",
#     76: "Scissors",
# }

# CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
#     "Person": (0, 220, 110),        # Bright Green
#     "Knife": (0, 0, 255),           # Red (Weapon)
#     "Scissors": (0, 60, 255),       # Orange-Red
#     "Baseball Bat": (0, 0, 255),    # Red (Weapon)
#     "Backpack": (0, 180, 255),      # Amber (Luggage)
#     "Handbag": (0, 180, 255),       # Amber
#     "Suitcase": (0, 180, 255),      # Amber
#     "Cell Phone": (255, 200, 0),    # Cyan-Blue
#     "Laptop": (255, 200, 0),        # Cyan-Blue
# }
# DEFAULT_COLOR = (255, 160, 50)


# # ---------------------------------------------------------------------------
# # 1. Zero-Lag Threaded Video / Camera Stream
# # ---------------------------------------------------------------------------
# class ZeroLagVideoStream:
#     """
#     Continuously grabs frames in a background thread so the display stream
#     and YOLO inference always read the newest frame with zero buffer delay.
#     """

#     def __init__(self, source: Any = 0, width: int = 640, height: int = 480):
#         self.width = width
#         self.height = height
#         self.source = source
#         self.cap: Optional[cv2.VideoCapture] = None
#         self.is_camera = False

#         self.lock = threading.Lock()
#         self.latest_frame: Optional[np.ndarray] = None
#         self.frame_id: int = 0
#         self.stream_fps: float = 0.0
#         self.running = False
#         self.thread: Optional[threading.Thread] = None
#         self.source_label = ""
#         self.video_fps = 30.0

#         self.open_source(source)

#     def open_source(self, source: Any) -> bool:
#         self.stop()

#         self.source = source
#         self.is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

#         if self.is_camera:
#             cam_idx = int(source)
#             self.source_label = f"LIVE LAPTOP CAMERA (Index {cam_idx})"
#             cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
#             if not cap.isOpened():
#                 cap = cv2.VideoCapture(cam_idx)
#             if cap.isOpened():
#                 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#                 cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
#                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
#                 cap.set(cv2.CAP_PROP_FPS, 30)
#         else:
#             path_str = str(source)
#             self.source_label = f"VIDEO: {Path(path_str).name[:38]}"
#             cap = cv2.VideoCapture(path_str)
#             if cap.isOpened():
#                 fps = cap.get(cv2.CAP_PROP_FPS)
#                 self.video_fps = fps if (fps and 5 <= fps <= 120) else 25.0

#         if not cap or not cap.isOpened():
#             print(f"[Module 1] WARNING: Could not open video source: {source}", flush=True)
#             return False

#         self.cap = cap
#         self.running = True
#         self.thread = threading.Thread(target=self._reader_loop, daemon=True)
#         self.thread.start()
#         print(f"[Module 1] Connected -> {self.source_label}", flush=True)
#         return True

#     def _reader_loop(self):
#         frame_interval = 1.0 / self.video_fps if not self.is_camera else 0.0
#         t_prev = time.perf_counter()
#         fps_smooth = 30.0

#         while self.running and self.cap is not None:
#             t_start = time.perf_counter()
#             ret, frame = self.cap.read()

#             if not ret or frame is None:
#                 if not self.is_camera:
#                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     continue
#                 else:
#                     time.sleep(0.01)
#                     continue

#             if frame.shape[1] != self.width or frame.shape[0] != self.height:
#                 frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

#             now = time.perf_counter()
#             dt = max(1e-4, now - t_prev)
#             t_prev = now
#             fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

#             with self.lock:
#                 self.latest_frame = frame
#                 self.frame_id += 1
#                 self.stream_fps = round(fps_smooth, 1)

#             if not self.is_camera and frame_interval > 0:
#                 elapsed = time.perf_counter() - t_start
#                 sleep_time = frame_interval - elapsed
#                 if sleep_time > 0:
#                     time.sleep(sleep_time)

#     def read(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
#         with self.lock:
#             if self.latest_frame is None:
#                 return False, None, 0, 0.0
#             return True, self.latest_frame.copy(), self.frame_id, self.stream_fps

#     def stop(self):
#         self.running = False
#         if self.thread and self.thread.is_alive():
#             self.thread.join(timeout=1.0)
#         if self.cap is not None:
#             try:
#                 self.cap.release()
#             except Exception:
#                 pass
#             self.cap = None


# # ---------------------------------------------------------------------------
# # 2. YOLOv8 + ByteTrack Detection Engine
# # ---------------------------------------------------------------------------
# class YOLOTrackerEngine:
#     def __init__(
#         self,
#         model_path: str = str(BASE_DIR / "yolov8n.pt"),
#         conf_threshold: float = 0.45,
#         iou_threshold: float = 0.50,
#         imgsz: int = 416,
#     ):
#         self.model_path = model_path
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.imgsz = imgsz
#         self.allowed_class_ids = list(SURVEILLANCE_CLASSES.keys())

#         print(f"[Module 1] Loading YOLOv8 model ({Path(model_path).name})...", flush=True)
#         self.model = YOLO(self.model_path)
#         dummy = np.zeros((320, 320, 3), dtype=np.uint8)
#         self.model.predict(dummy, imgsz=320, verbose=False)
#         print("[Module 1] YOLOv8 + ByteTrack warmed up and ready!", flush=True)

#         self.lock = threading.Lock()
#         self.latest_detections: List[Dict[str, Any]] = []
#         self.inference_ms: float = 0.0
#         self.inference_fps: float = 0.0
#         self._async_running = False
#         self._async_thread: Optional[threading.Thread] = None

#     def detect_frame(self, frame_bgr: np.ndarray, use_tracking: bool = True) -> List[Dict[str, Any]]:
#         t0 = time.perf_counter()

#         if use_tracking:
#             results = self.model.track(
#                 frame_bgr,
#                 imgsz=self.imgsz,
#                 conf=self.conf_threshold,
#                 iou=self.iou_threshold,
#                 classes=self.allowed_class_ids,
#                 persist=True,
#                 tracker="bytetrack.yaml",
#                 verbose=False,
#             )
#         else:
#             results = self.model.predict(
#                 frame_bgr,
#                 imgsz=self.imgsz,
#                 conf=self.conf_threshold,
#                 iou=self.iou_threshold,
#                 classes=self.allowed_class_ids,
#                 verbose=False,
#             )

#         dt_ms = (time.perf_counter() - t0) * 1000.0
#         detections: List[Dict[str, Any]] = []

#         if results and len(results) > 0:
#             r = results[0]
#             if r.boxes is not None and len(r.boxes) > 0:
#                 boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
#                 confs = r.boxes.conf.cpu().numpy()
#                 cls_ids = r.boxes.cls.cpu().numpy().astype(int)
#                 track_ids = (
#                     r.boxes.id.cpu().numpy().astype(int)
#                     if r.boxes.id is not None
#                     else list(range(1, len(boxes_xyxy) + 1))
#                 )

#                 for xyxy, conf, cls_id, tid in zip(boxes_xyxy, confs, cls_ids, track_ids):
#                     x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
#                     w, h = x2 - x1, y2 - y1
#                     if w < 15 or h < 15:
#                         continue

#                     cls_name = SURVEILLANCE_CLASSES.get(int(cls_id), f"Object_{cls_id}")
#                     detections.append({
#                         "track_id": int(tid),
#                         "class_id": int(cls_id),
#                         "class_name": cls_name,
#                         "confidence": round(float(conf), 2),
#                         "bbox": [x1, y1, x2, y2],
#                         "centroid": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
#                     })

#         with self.lock:
#             self.latest_detections = detections
#             self.inference_ms = round(dt_ms, 1)
#             self.inference_fps = round(1000.0 / max(1.0, dt_ms), 1)

#         return detections

#     def start_async_worker(self, stream: ZeroLagVideoStream):
#         self.stop_async_worker()
#         self._async_running = True
#         self._async_thread = threading.Thread(
#             target=self._async_loop, args=(stream,), daemon=True
#         )
#         self._async_thread.start()

#     def _async_loop(self, stream: ZeroLagVideoStream):
#         last_processed_id = -1
#         while self._async_running:
#             ok, frame, fid, _ = stream.read()
#             if not ok or frame is None or fid == last_processed_id:
#                 time.sleep(0.004)
#                 continue
#             last_processed_id = fid
#             try:
#                 self.detect_frame(frame, use_tracking=True)
#             except Exception as exc:
#                 print(f"[Module 1] Inference error: {exc}", flush=True)
#                 time.sleep(0.05)

#     def stop_async_worker(self):
#         self._async_running = False
#         if self._async_thread and self._async_thread.is_alive():
#             self._async_thread.join(timeout=1.0)

#     def get_latest_state(self) -> Tuple[List[Dict[str, Any]], float, float]:
#         with self.lock:
#             return list(self.latest_detections), self.inference_ms, self.inference_fps


# # ---------------------------------------------------------------------------
# # 3. Clean Frame Annotation
# # ---------------------------------------------------------------------------
# def draw_module1_hud(
#     frame: np.ndarray,
#     detections: List[Dict[str, Any]],
#     source_label: str,
#     display_fps: float,
#     inference_ms: float,
#     conf_thresh: float,
# ) -> np.ndarray:
#     canvas = frame.copy()
#     h, w = canvas.shape[:2]

#     person_count = sum(1 for d in detections if d["class_name"] == "Person")
#     other_count = len(detections) - person_count

#     for det in detections:
#         x1, y1, x2, y2 = det["bbox"]
#         cls_name = det["class_name"]
#         conf = det["confidence"]
#         tid = det["track_id"]
#         cx, cy = det["centroid"]

#         color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

#         cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
#         cv2.circle(canvas, (cx, cy), 4, color, -1)

#         label = f"{cls_name} #{tid} ({int(conf * 100)}%)"
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
#         ly1 = max(0, y1 - th - 10)
#         ly2 = max(th + 10, y1)
#         cv2.rectangle(canvas, (x1, ly1), (x1 + tw + 10, ly2), color, -1)
#         cv2.putText(
#             canvas,
#             label,
#             (x1 + 5, ly2 - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.54,
#             (10, 10, 10),
#             2,
#             cv2.LINE_AA,
#         )

#     overlay = canvas.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 36), (15, 18, 26), -1)
#     cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

#     hud_left = f"M1: YOLOv8n + ByteTrack | {source_label}"
#     hud_right = (
#         f"Persons: {person_count} | Objects: {other_count} | "
#         f"YOLO: {inference_ms:.0f}ms | Stream: {display_fps:.0f} FPS"
#     )
#     cv2.putText(canvas, hud_left, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 180), 1, cv2.LINE_AA)
#     (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
#     cv2.putText(canvas, hud_right, (w - rw - 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)

#     return canvas


# # ---------------------------------------------------------------------------
# # 4. Fast Interactive Web Testbench (http://127.0.0.1:8000)
# # ---------------------------------------------------------------------------
# app = FastAPI(title="Module 1 - YOLOv8 Object Detection & Tracking")

# STREAM: Optional[ZeroLagVideoStream] = None
# TRACKER: Optional[YOLOTrackerEngine] = None


# @app.on_event("startup")
# def on_startup():
#     global STREAM, TRACKER
#     STREAM = ZeroLagVideoStream(source=0, width=640, height=480)
#     TRACKER = YOLOTrackerEngine(conf_threshold=0.45, imgsz=416)
#     TRACKER.start_async_worker(STREAM)


# @app.on_event("shutdown")
# def on_shutdown():
#     if TRACKER:
#         TRACKER.stop_async_worker()
#     if STREAM:
#         STREAM.stop()


# def mjpeg_generator():
#     last_sent_id = -1
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 82]
#     while True:
#         if STREAM is None or TRACKER is None:
#             time.sleep(0.02)
#             continue
#         ok, frame, fid, sfps = STREAM.read()
#         if not ok or frame is None or fid == last_sent_id:
#             time.sleep(0.005)
#             continue
#         last_sent_id = fid

#         detections, inf_ms, _ = TRACKER.get_latest_state()
#         vis = draw_module1_hud(
#             frame=frame,
#             detections=detections,
#             source_label=STREAM.source_label,
#             display_fps=sfps,
#             inference_ms=inf_ms,
#             conf_thresh=TRACKER.conf_threshold,
#         )
#         ret, jpg = cv2.imencode(".jpg", vis, encode_param)
#         if not ret:
#             continue
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
#         )


# @app.get("/video_feed")
# def video_feed():
#     return StreamingResponse(
#         mjpeg_generator(),
#         media_type="multipart/x-mixed-replace; boundary=frame",
#     )


# @app.get("/api/telemetry")
# def get_telemetry():
#     if STREAM is None or TRACKER is None:
#         return JSONResponse({"status": "initializing"})
#     detections, inf_ms, inf_fps = TRACKER.get_latest_state()
#     _, _, fid, sfps = STREAM.read()
#     return {
#         "source": STREAM.source_label,
#         "stream_fps": sfps,
#         "inference_ms": inf_ms,
#         "inference_fps": inf_fps,
#         "conf_threshold": TRACKER.conf_threshold,
#         "total_count": len(detections),
#         "person_count": sum(1 for d in detections if d["class_name"] == "Person"),
#         "detections": detections,
#     }


# @app.post("/api/source")
# def switch_source(src: str = Query(...)):
#     if STREAM is None or TRACKER is None:
#         return {"ok": False}
#     TRACKER.model.predictor = None  # Reset tracker IDs cleanly
#     if src == "camera":
#         ok = STREAM.open_source(0)
#     else:
#         target = BASE_DIR / "samples" / src
#         ok = STREAM.open_source(str(target))
#     return {"ok": ok, "source": STREAM.source_label}


# @app.post("/api/confidence")
# def set_confidence(val: float = Query(...)):
#     if TRACKER is not None:
#         TRACKER.conf_threshold = max(0.20, min(0.90, round(val, 2)))
#     return {"ok": True, "conf_threshold": TRACKER.conf_threshold if TRACKER else 0.45}


# @app.post("/api/upload_video")
# async def upload_video(file: UploadFile = File(...)):
#     samples_dir = BASE_DIR / "samples"
#     samples_dir.mkdir(parents=True, exist_ok=True)
#     dest = samples_dir / f"uploaded_{file.filename}"
#     content = await file.read()
#     with open(dest, "wb") as f:
#         f.write(content)
#     if STREAM and TRACKER:
#         TRACKER.model.predictor = None
#         STREAM.open_source(str(dest))
#     return {"ok": True, "filename": dest.name, "source": STREAM.source_label if STREAM else ""}


# @app.get("/", response_class=HTMLResponse)
# def index_page():
#     samples_dir = BASE_DIR / "samples"
#     videos = sorted([p.name for p in samples_dir.glob("*.mp4")]) if samples_dir.exists() else []
#     vid_buttons = "".join(
#         f'<button class="btn" onclick="switchSource(\'{v}\')">🎬 {v[:28]}</button>'
#         for v in videos
#     )
#     return f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Module 1 — YOLOv8 Object Detection & ByteTrack</title>
# <style>
#   :root {{ --bg:#0b0f17; --card:#151c28; --border:#263248; --accent:#00e5a3; --text:#e2e8f0; }}
#   * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }}
#   body {{ background:var(--bg); color:var(--text); padding:18px; }}
#   .header {{ display:flex; justify-content:space-between; align-items:center; padding:14px 20px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:16px; }}
#   .header h1 {{ font-size:18px; color:var(--accent); }}
#   .badge {{ background:#0f2922; color:var(--accent); border:1px solid var(--accent); padding:5px 12px; border-radius:20px; font-size:12px; font-weight:600; }}
#   .grid {{ display:grid; grid-template-columns: 2fr 1fr; gap:16px; }}
#   .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }}
#   .video-box {{ width:100%; border-radius:8px; overflow:hidden; background:#000; border:1px solid var(--border); }}
#   .video-box img {{ width:100%; display:block; }}
#   .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }}
#   .btn {{ background:#1e293b; color:#fff; border:1px solid #334155; padding:8px 14px; border-radius:6px; cursor:pointer; font-size:13px; transition:0.15s; }}
#   .btn:hover {{ border-color:var(--accent); color:var(--accent); }}
#   .btn-primary {{ background:#059669; border-color:#10b981; font-weight:600; }}
#   .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px; }}
#   .metric {{ background:#0f1520; border:1px solid var(--border); padding:12px; border-radius:8px; }}
#   .metric .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; }}
#   .metric .val {{ font-size:22px; font-weight:700; color:var(--accent); margin-top:4px; }}
#   table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
#   th, td {{ text-align:left; padding:8px; border-bottom:1px solid var(--border); }}
#   th {{ color:#94a3b8; font-size:11px; text-transform:uppercase; }}
# </style>
# </head>
# <body>
#   <div class="header">
#     <div>
#       <h1>MODULE 1: Real-Time YOLOv8 Object Detection & ByteTrack</h1>
#       <div style="font-size:12px;color:#94a3b8;margin-top:3px;">Dataset: MS COCO 2017 (Filtered Surveillance Classes) • Zero-Lag 3-Thread Engine</div>
#     </div>
#     <span class="badge" id="srcBadge">LIVE CAMERA</span>
#   </div>

#   <div class="grid">
#     <div class="card">
#       <div class="video-box">
#         <img src="/video_feed" alt="Live YOLOv8 Stream" />
#       </div>
#       <div class="controls">
#         <button class="btn btn-primary" onclick="switchSource('camera')">📷 Live Laptop Camera</button>
#         {vid_buttons}
#         <label class="btn" style="cursor:pointer;">
#           📂 Upload MP4 <input type="file" accept="video/mp4,video/*" style="display:none" onchange="uploadVid(this)">
#         </label>
#         <div style="margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;">
#           <span>Confidence: <b id="confVal">45%</b></span>
#           <input type="range" min="25" max="85" value="45" step="5" oninput="setConf(this.value)">
#         </div>
#       </div>
#     </div>

#     <div class="card">
#       <div class="metrics">
#         <div class="metric"><div class="label">Persons Detected</div><div class="val" id="mPersons">0</div></div>
#         <div class="metric"><div class="label">Total Objects</div><div class="val" id="mTotal">0</div></div>
#         <div class="metric"><div class="label">Stream Speed</div><div class="val" id="mFps">0 FPS</div></div>
#         <div class="metric"><div class="label">YOLO Latency</div><div class="val" id="mMs">0 ms</div></div>
#       </div>
#       <h3 style="font-size:13px;color:#94a3b8;text-transform:uppercase;">Live Tracked Entities (ByteTrack)</h3>
#       <table>
#         <thead><tr><th>Track ID</th><th>Class</th><th>Confidence</th><th>Centroid</th></tr></thead>
#         <tbody id="detTable"><tr><td colspan="4" style="color:#64748b;">No objects in frame</td></tr></tbody>
#       </table>
#     </div>
#   </div>

# <script>
# async function switchSource(src) {{
#   await fetch('/api/source?src=' + encodeURIComponent(src), {{method:'POST'}});
# }}
# async function setConf(v) {{
#   document.getElementById('confVal').innerText = v + '%';
#   await fetch('/api/confidence?val=' + (v/100.0), {{method:'POST'}});
# }}
# async function uploadVid(input) {{
#   if (!input.files.length) return;
#   const fd = new FormData();
#   fd.append('file', input.files[0]);
#   await fetch('/api/upload_video', {{method:'POST', body:fd}});
#   location.reload();
# }}
# setInterval(async () => {{
#   try {{
#     const r = await fetch('/api/telemetry');
#     const d = await r.json();
#     if (!d.source) return;
#     document.getElementById('srcBadge').innerText = d.source;
#     document.getElementById('mPersons').innerText = d.person_count;
#     document.getElementById('mTotal').innerText = d.total_count;
#     document.getElementById('mFps').innerText = d.stream_fps + ' FPS';
#     document.getElementById('mMs').innerText = d.inference_ms + ' ms';
#     const tb = document.getElementById('detTable');
#     if (!d.detections || d.detections.length === 0) {{
#       tb.innerHTML = '<tr><td colspan="4" style="color:#64748b;">No objects in frame</td></tr>';
#     }} else {{
#       tb.innerHTML = d.detections.map(x =>
#         `<tr><td><b>#${{x.track_id}}</b></td><td>${{x.class_name}}</td><td>${{Math.round(x.confidence*100)}}%</td><td>(${{x.centroid[0]}}, ${{x.centroid[1]}})</td></tr>`
#       ).join('');
#     }}
#   }} catch(e) {{}}
# }}, 250);
# </script>
# </body>
# </html>"""


# if __name__ == "__main__":
#     threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")



"""
MODULE 1: Real-Time YOLOv8 Object Detection & Multi-Object Tracking Engine (tuned)
==================================================================================
Dataset Backbone : MS COCO 2017 (80 classes, filtered to surveillance targets)
                   Custom fine-tuned models (any class list) are auto-detected.
Model            : Ultralytics YOLOv8/YOLO11 + ByteTrack
Targets          : Laptop GPU (.pt / TensorRT .engine) and Raspberry Pi (NCNN / ONNX)

Pipeline         : 1. ZeroLagVideoStream  - background grabber, latest-frame-only
                   2. YOLOTrackerEngine   - async detection + tracking thread
                   3. MJPEG + JSON console on http://127.0.0.1:8000

Run examples
    python -m modules.m1_yolo_tracker
    python -m modules.m1_yolo_tracker --model yolov8n_ncnn_model --imgsz 640 --device cpu
    python -m modules.m1_yolo_tracker --source rtsp://user:pass@192.168.1.10/stream1

Export for Raspberry Pi (FP16 NCNN, no meaningful mAP loss):
    yolo export model=yolov8n.pt format=ncnn imgsz=640 half=True
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

SAMPLES_DIR = BASE_DIR / "samples"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
SHUTDOWN = threading.Event()

# ByteTrack needs low-confidence boxes for its 2nd association pass, so the
# detector always runs at this floor and the user threshold is applied afterwards.
TRACK_FLOOR_CONF = 0.25

# ---------------------------------------------------------------------------
# MS COCO Surveillance Target Classes
# ---------------------------------------------------------------------------
SURVEILLANCE_CLASSES: Dict[int, str] = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck",
    24: "Backpack", 26: "Handbag", 28: "Suitcase", 34: "Baseball Bat",
    39: "Bottle", 43: "Knife", 63: "Laptop", 67: "Cell Phone", 76: "Scissors",
}

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {  # BGR
    "Person": (0, 220, 110),
    "Knife": (0, 0, 255),
    "Scissors": (0, 60, 255),
    "Baseball Bat": (0, 0, 255),
    "Backpack": (0, 180, 255),
    "Handbag": (0, 180, 255),
    "Suitcase": (0, 180, 255),
    "Cell Phone": (255, 200, 0),
    "Laptop": (255, 200, 0),
}
DEFAULT_COLOR = (255, 160, 50)


# ---------------------------------------------------------------------------
# Settings (env-driven so CLI flags, env vars and uvicorn reloads all agree)
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    model: str
    imgsz: int
    stride: int
    conf: float
    device: str
    source: str
    width: int
    height: int
    display_fps: int

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            model=e("M1_MODEL", str(BASE_DIR / "yolov8n.pt")),
            imgsz=int(e("M1_IMGSZ", "640")),
            stride=int(e("M1_STRIDE", "1")),
            conf=float(e("M1_CONF", "0.45")),
            device=e("M1_DEVICE", "auto"),
            source=e("M1_SOURCE", "0"),
            width=int(e("M1_WIDTH", "640")),
            height=int(e("M1_HEIGHT", "480")),
            display_fps=int(e("M1_DISPLAY_FPS", "20")),
        )


def parse_cli(description: str) -> argparse.Namespace:
    """Shared CLI for module 1 and 2. Flags are exported as env vars."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", help="yolov8n.pt | yolov8n_ncnn_model | model.onnx | model.engine")
    p.add_argument("--imgsz", type=int, help="inference size (default 640)")
    p.add_argument("--stride", type=int, help="process every Nth frame (CPU duty-cycle limiter)")
    p.add_argument("--conf", type=float, help="display confidence threshold")
    p.add_argument("--device", help="auto | cpu | 0")
    p.add_argument("--source", help="camera index, video path, rtsp:// URL or GStreamer pipeline")
    p.add_argument("--display-fps", type=int, dest="display_fps", help="max MJPEG fps")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    for key in ("model", "imgsz", "stride", "conf", "device", "source", "display_fps"):
        val = getattr(args, key)
        if val is not None:
            os.environ[f"M1_{key.upper()}"] = str(val)
    return args


def resolve_device(pref: str) -> Any:
    if pref != "auto":
        return int(pref) if pref.isdigit() else pref
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FramePacket(NamedTuple):
    image: np.ndarray   # treat as READ-ONLY (shared between threads, never copied)
    frame_id: int
    fps: float
    ts: float           # time.monotonic() at capture


class DetectionState(NamedTuple):
    detections: List[Dict[str, Any]]
    inference_ms: float
    inference_fps: float
    version: int
    frame_ts: float
    frame_size: Tuple[int, int]  # (width, height)


def list_samples() -> List[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def safe_sample_path(name: str) -> Optional[Path]:
    """Resolve a sample filename without allowing path traversal."""
    p = (SAMPLES_DIR / Path(name).name).resolve()
    if p.parent != SAMPLES_DIR.resolve() or p.suffix.lower() not in VIDEO_EXTS or not p.is_file():
        return None
    return p


def blend_top_bar(canvas: np.ndarray, height: int = 36, color=(15, 18, 26), alpha: float = 0.82) -> None:
    roi = canvas[:height]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0, dst=roi)


def draw_label(canvas: np.ndarray, text: str, x1: int, y1: int, color, scale: float = 0.52) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(0, y1 - th - 10)
    bottom = max(th + 10, y1)
    cv2.rectangle(canvas, (x1, top), (x1 + tw + 10, bottom), color, -1)
    cv2.putText(canvas, text, (x1 + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (10, 10, 10), 2, cv2.LINE_AA)


def no_signal_frame(w: int, h: int, label: str) -> np.ndarray:
    img = np.full((h, w, 3), (18, 22, 30), np.uint8)
    text = f"NO SIGNAL - {label}"[:60]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, text, (max(10, (w - tw) // 2), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 160, 255), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 1. Zero-Lag Threaded Video / Camera Stream
# ---------------------------------------------------------------------------
class ZeroLagVideoStream:
    """
    Grabs frames in a background thread; consumers always get the newest frame.
    Supports USB/laptop cameras, video files (looped, real-time paced), RTSP/HTTP
    URLs (auto-reconnect) and GStreamer pipelines (e.g. libcamerasrc on a Pi).
    """

    def __init__(self, source: Any = 0, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.source_label = "NO SOURCE"
        self.is_live = False

        self._cond = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._frame_id = 0
        self._open_lock = threading.Lock()
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

        self.open_source(source)

    # -- source handling -----------------------------------------------------
    @staticmethod
    def _normalize(source: Any) -> Any:
        if isinstance(source, str):
            s = source.strip()
            return int(s) if s.isdigit() else s
        return source

    def _open_capture(self, src: Any) -> Tuple[Optional[cv2.VideoCapture], str, bool, float]:
        if isinstance(src, int):
            system = platform.system()
            backend = (cv2.CAP_DSHOW if system == "Windows"
                       else cv2.CAP_V4L2 if system == "Linux" else cv2.CAP_ANY)
            cap = cv2.VideoCapture(src, backend)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
            return cap, f"LIVE CAMERA (index {src})", True, 30.0

        src = str(src)
        if "!" in src:  # GStreamer pipeline string
            return cv2.VideoCapture(src, cv2.CAP_GSTREAMER), "GSTREAMER PIPELINE", True, 30.0
        if "://" in src:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            host = src.split("://", 1)[1].split("/")[0].split("@")[-1]  # never show credentials
            return cap, f"STREAM: {host}", True, 30.0

        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        vfps = fps if 5 <= fps <= 120 else 25.0
        return cap, f"VIDEO: {Path(src).name[:38]}", False, vfps

    def open_source(self, source: Any) -> bool:
        with self._open_lock:
            self._stop_reader()
            src = self._normalize(source)
            cap, label, live, vfps = self._open_capture(src)
            with self._cond:
                self._latest = None  # never show a frame from the previous source

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self.source_label = "NO SIGNAL"
                print(f"[Module 1] WARNING: could not open source: {source}", flush=True)
                return False

            self.source_label = label
            self.is_live = live
            stop_evt = threading.Event()
            self._stop_evt = stop_evt
            self._thread = threading.Thread(
                target=self._reader_loop, args=(cap, src, live, vfps, stop_evt), daemon=True)
            self._thread.start()
            print(f"[Module 1] Connected -> {label}", flush=True)
            return True

    # -- reader thread (owns the capture object) ----------------------------------
    def _reader_loop(self, cap, src, live: bool, vfps: float, stop_evt: threading.Event) -> None:
        interval = 0.0 if live else 1.0 / vfps
        fps_ema, t_prev, fails = 0.0, time.perf_counter(), 0
        try:
            while not stop_evt.is_set():
                t0 = time.perf_counter()
                ok, frame = cap.read()

                if not ok or frame is None:
                    fails += 1
                    if not live:                       # loop video files
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if fails > 5:
                            stop_evt.wait(0.2)         # broken file: don't spin
                    else:
                        stop_evt.wait(0.02)
                        if fails >= 150:               # ~3 s without frames -> reconnect
                            cap.release()
                            stop_evt.wait(1.0)
                            cap = self._open_capture(src)[0]
                            fails = 0
                    continue
                fails = 0

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                now = time.perf_counter()
                inst = 1.0 / max(1e-4, now - t_prev)
                t_prev = now
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst

                with self._cond:
                    self._frame_id += 1
                    self._latest = FramePacket(frame, self._frame_id, round(fps_ema, 1), time.monotonic())
                    self._cond.notify_all()

                if interval:
                    rest = interval - (time.perf_counter() - t0)
                    if rest > 0:
                        stop_evt.wait(rest)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    # -- consumers --------------------------------------------------------------
    def read(self) -> Optional[FramePacket]:
        with self._cond:
            return self._latest

    def wait_new(self, last_id: int, timeout: float = 1.0) -> Optional[FramePacket]:
        """Block until a frame newer than last_id exists (no busy-wait polling)."""
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None and self._latest.frame_id != last_id, timeout)
            pkt = self._latest
            return pkt if pkt is not None and pkt.frame_id != last_id else None

    def _stop_reader(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stop_evt = None
        self._thread = None

    def stop(self) -> None:
        with self._open_lock:
            self._stop_reader()


# ---------------------------------------------------------------------------
# 2. YOLO + ByteTrack Detection Engine
# ---------------------------------------------------------------------------
class YOLOTrackerEngine:
    def __init__(
        self,
        model_path: str = str(BASE_DIR / "yolov8n.pt"),
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        stride: int = 1,
        device: str = "auto",
        min_box_px: int = 15,
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.stride = max(1, stride)
        self.min_box_px = min_box_px

        suffix = Path(self.model_path).suffix.lower()
        dev = resolve_device(device)
        self._dev_kw: Dict[str, Any] = {}
        if suffix in (".pt", ".engine"):
            self._dev_kw["device"] = dev
            if suffix == ".pt" and dev != "cpu":
                self._dev_kw["half"] = True  # FP16 on GPU: ~2x faster, no real mAP loss

        print(f"[Module 1] Loading {Path(self.model_path).name} (imgsz={imgsz}, device={dev})...", flush=True)
        self.model = YOLO(self.model_path, task="detect")

        names = self.model.names
        self.is_coco = len(names) == 80
        if self.is_coco:
            self.class_names: Dict[int, str] = dict(SURVEILLANCE_CLASSES)
            self.allowed_class_ids: Optional[List[int]] = list(SURVEILLANCE_CLASSES)
        else:  # custom fine-tuned model (e.g. weapons): keep every class it knows
            self.class_names = {int(k): str(v).title() for k, v in names.items()}
            self.allowed_class_ids = None

        self._infer_lock = threading.Lock()
        self._cond = threading.Condition()
        self._version = 0
        self._ms_ema = 0.0
        self._state = DetectionState([], 0.0, 0.0, 0, 0.0, (0, 0))
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz,
                           conf=TRACK_FLOOR_CONF, verbose=False, **self._dev_kw)
        print("[Module 1] Model warmed up and ready.", flush=True)

    # -- configuration ----------------------------------------------------------
    def set_conf(self, val: float) -> float:
        self.conf_threshold = max(0.20, min(0.90, round(val, 2)))
        return self.conf_threshold

    def reset_tracking(self) -> None:
        """Safely reset ByteTrack IDs (waits for any in-flight inference)."""
        with self._infer_lock:
            self.model.predictor = None
        with self._cond:
            self._version += 1
            self._state = DetectionState([], self._state.inference_ms, self._state.inference_fps,
                                         self._version, time.monotonic(), self._state.frame_size)
            self._cond.notify_all()

    # -- inference --------------------------------------------------------------
    def detect_frame(self, frame_bgr: np.ndarray, use_tracking: bool = True,
                     frame_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        kwargs = dict(
            imgsz=self.imgsz,
            conf=min(self.conf_threshold, TRACK_FLOOR_CONF),
            iou=self.iou_threshold,
            classes=self.allowed_class_ids,
            verbose=False,
            **self._dev_kw,
        )
        with self._infer_lock:
            t0 = time.perf_counter()
            if use_tracking:
                results = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **kwargs)
            else:
                results = self.model.predict(frame_bgr, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._parse(results, self.conf_threshold, frame_bgr.shape)

        self._ms_ema = dt_ms if self._ms_ema == 0.0 else 0.8 * self._ms_ema + 0.2 * dt_ms
        h, w = frame_bgr.shape[:2]
        with self._cond:
            self._version += 1
            self._state = DetectionState(
                detections, round(self._ms_ema, 1), round(1000.0 / max(1.0, self._ms_ema), 1),
                self._version, frame_ts if frame_ts is not None else time.monotonic(), (w, h))
            self._cond.notify_all()
        return detections

    def _parse(self, results, conf_min: float, shape) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        H, W = shape[:2]

        for i, (b, c, k) in enumerate(zip(xyxy, confs, cls_ids)):
            if c < conf_min:
                continue
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(W - 1, int(b[2])), min(H - 1, int(b[3]))
            if x2 - x1 < self.min_box_px or y2 - y1 < self.min_box_px:
                continue
            out.append({
                "track_id": int(ids[i]) if ids is not None else -(i + 1),
                "class_id": int(k),
                "class_name": self.class_names.get(int(k), f"Object_{k}"),
                "confidence": round(float(c), 2),
                "bbox": [x1, y1, x2, y2],
                "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
            })
        return out

    # -- async worker -----------------------------------------------------------
    def start_async_worker(self, stream: ZeroLagVideoStream) -> None:
        self.stop_async_worker()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._async_loop, args=(stream, self._stop_evt), daemon=True)
        self._thread.start()

    def _async_loop(self, stream: ZeroLagVideoStream, stop_evt: threading.Event) -> None:
        last_id, seen = -1, 0
        while not stop_evt.is_set():
            pkt = stream.wait_new(last_id, timeout=0.2)
            if pkt is None:
                continue
            last_id = pkt.frame_id
            seen += 1
            if self.stride > 1 and seen % self.stride:
                continue
            try:
                self.detect_frame(pkt.image, use_tracking=True, frame_ts=pkt.ts)
            except Exception as exc:
                print(f"[Module 1] Inference error: {exc}", flush=True)
                stop_evt.wait(0.05)

    def stop_async_worker(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stop_evt = None
        self._thread = None

    # -- consumers --------------------------------------------------------------
    def get_latest_state(self) -> DetectionState:
        with self._cond:
            return self._state

    def wait_for_detections(self, last_version: int, timeout: float = 0.5) -> Optional[DetectionState]:
        with self._cond:
            self._cond.wait_for(lambda: self._state.version != last_version, timeout)
            return self._state if self._state.version != last_version else None


def change_source(stream: ZeroLagVideoStream, engine: YOLOTrackerEngine, src: str) -> bool:
    """Switch to the local camera or a file in samples/. Remote URLs are CLI-only."""
    if src == "camera":
        target: Any = 0
    else:
        path = safe_sample_path(src)
        if path is None:
            return False
        target = str(path)
    ok = stream.open_source(target)
    engine.reset_tracking()
    return ok


# ---------------------------------------------------------------------------
# 3. MJPEG streaming helper (shared with Module 2)
# ---------------------------------------------------------------------------
def mjpeg_frames(stream: ZeroLagVideoStream, render: Callable[[FramePacket], np.ndarray],
                 max_fps: int = 20, quality: int = 80) -> Iterator[bytes]:
    last_id, last_sent, min_dt = -1, 0.0, 1.0 / max(1, max_fps)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    while not SHUTDOWN.is_set():
        pkt = stream.wait_new(last_id, timeout=1.0)
        if pkt is None:
            # keep-alive placeholder: lets the server notice closed browser tabs
            frame = no_signal_frame(stream.width, stream.height, stream.source_label)
        else:
            last_id = pkt.frame_id
            now = time.monotonic()
            if now - last_sent < min_dt:
                continue
            last_sent = now
            frame = render(pkt)
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"


# ---------------------------------------------------------------------------
# 4. Frame annotation
# ---------------------------------------------------------------------------
def draw_module1_hud(frame: np.ndarray, detections: List[Dict[str, Any]], source_label: str,
                     display_fps: float, inference_ms: float) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    persons = sum(1 for d in detections if d["class_name"] == "Person")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class_name"], DEFAULT_COLOR)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, tuple(det["centroid"]), 4, color, -1)
        draw_label(canvas, f"{det['class_name']} #{det['track_id']} ({int(det['confidence'] * 100)}%)",
                   x1, y1, color, 0.54)

    blend_top_bar(canvas, 36)
    left = f"M1: YOLO + ByteTrack | {source_label}"
    right = (f"Persons: {persons} | Objects: {len(detections) - persons} | "
             f"YOLO: {inference_ms:.0f}ms | Stream: {display_fps:.0f} FPS")
    cv2.putText(canvas, left, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 180), 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(canvas, right, (max(10, w - rw - 10), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 5. Web testbench
# ---------------------------------------------------------------------------
DASHBOARD_CSS = """
:root { --bg:#0b0f17; --card:#151c28; --border:#263248; --accent:#00e5a3; --danger:#ef4444; --warning:#f59e0b; --text:#e2e8f0; }
* { box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }
body { background:var(--bg); color:var(--text); padding:16px; }
.header { display:flex; justify-content:space-between; align-items:center; padding:12px 20px; background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; gap:12px; }
.header h1 { font-size:17px; color:var(--accent); }
.sub { font-size:12px; color:#94a3b8; margin-top:2px; }
.badge { padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; background:#0f2922; color:var(--accent); border:1px solid var(--accent); white-space:nowrap; }
.badge-safe { background:#0f2922; color:var(--accent); border-color:var(--accent); }
.badge-warn { background:#3d2406; color:var(--warning); border-color:var(--warning); }
.badge-crit { background:#3b1111; color:var(--danger); border-color:var(--danger); animation:pulse 1s infinite; }
@keyframes pulse { 0%{opacity:1} 50%{opacity:.5} 100%{opacity:1} }
.grid { display:grid; grid-template-columns:2fr 1fr; gap:16px; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
.video-box { position:relative; width:100%; border-radius:8px; overflow:hidden; background:#000; border:1px solid var(--border); }
.video-box img { width:100%; display:block; }
.controls { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }
.btn { background:#1e293b; color:#fff; border:1px solid #334155; padding:7px 12px; border-radius:6px; cursor:pointer; font-size:12px; }
.btn:hover { border-color:var(--accent); color:var(--accent); }
.btn-primary { background:#059669; border-color:#10b981; font-weight:600; }
.metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px; }
.metric { background:#0f1520; border:1px solid var(--border); padding:10px; border-radius:8px; }
.metric .label { font-size:11px; color:#94a3b8; text-transform:uppercase; }
.metric .val { font-size:20px; font-weight:700; color:var(--accent); margin-top:3px; }
.alert-banner { padding:10px; border-radius:8px; margin-bottom:12px; display:none; font-weight:600; font-size:13px; }
h3 { font-size:12px; color:#94a3b8; text-transform:uppercase; margin-top:8px; }
table { width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }
th, td { text-align:left; padding:7px; border-bottom:1px solid var(--border); }
th { color:#94a3b8; font-size:11px; text-transform:uppercase; }
.muted { color:#64748b; }
"""

M1_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Module 1 - YOLO Object Detection & ByteTrack</title>
<style>__CSS__</style></head>
<body>
<div class="header">
  <div><h1>MODULE 1: Real-Time Object Detection & ByteTrack</h1>
  <div class="sub">MS COCO surveillance classes - latest-frame pipeline - Edge/Pi ready</div></div>
  <span class="badge" id="srcBadge">CONNECTING</span>
</div>
<div class="grid">
  <div class="card">
    <div class="video-box"><img src="/video_feed" alt="Live detection stream"></div>
    <div class="controls">
      <button class="btn btn-primary" onclick="switchSource('camera')">📷 Live Camera</button>
      <span id="samples" style="display:contents"></span>
      <label class="btn" style="cursor:pointer;">📂 Upload video
        <input type="file" accept="video/*" style="display:none" onchange="uploadVid(this)"></label>
      <div style="margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;">
        <span>Confidence: <b id="confVal">45%</b></span>
        <input type="range" min="25" max="85" value="45" step="5" oninput="setConf(this.value)">
      </div>
    </div>
  </div>
  <div class="card">
    <div class="metrics">
      <div class="metric"><div class="label">Persons</div><div class="val" id="mPersons">0</div></div>
      <div class="metric"><div class="label">Total Objects</div><div class="val" id="mTotal">0</div></div>
      <div class="metric"><div class="label">Stream Speed</div><div class="val" id="mFps">0 FPS</div></div>
      <div class="metric"><div class="label">YOLO Latency</div><div class="val" id="mMs">0 ms</div></div>
    </div>
    <h3>Live Tracked Entities</h3>
    <table><thead><tr><th>ID</th><th>Class</th><th>Conf</th><th>Centroid</th></tr></thead>
    <tbody id="detTable"><tr><td colspan="4" class="muted">No objects in frame</td></tr></tbody></table>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url, opts) { const r = await fetch(url, opts); return r.json(); }
async function switchSource(src) { await api('/api/source?src=' + encodeURIComponent(src), {method:'POST'}); }
async function setConf(v) { $('confVal').innerText = v + '%'; await api('/api/confidence?val=' + (v/100), {method:'POST'}); }
async function loadSamples() {
  const d = await api('/api/samples'); const box = $('samples'); box.replaceChildren();
  d.videos.forEach(v => { const b = document.createElement('button'); b.className = 'btn';
    b.textContent = '🎬 ' + v.slice(0, 26); b.onclick = () => switchSource(v); box.appendChild(b); });
}
async function uploadVid(input) {
  if (!input.files.length) return;
  const fd = new FormData(); fd.append('file', input.files[0]);
  await api('/api/upload_video', {method:'POST', body:fd}); loadSamples();
}
let busy = false;
setInterval(async () => {
  if (busy) return; busy = true;
  try {
    const d = await api('/api/telemetry'); if (!d.source) return;
    $('srcBadge').innerText = d.source; $('mPersons').innerText = d.person_count;
    $('mTotal').innerText = d.total_count; $('mFps').innerText = d.stream_fps + ' FPS';
    $('mMs').innerText = d.inference_ms + ' ms';
    $('detTable').innerHTML = (d.detections && d.detections.length) ? d.detections.map(x =>
      `<tr><td><b>#${x.track_id}</b></td><td>${esc(x.class_name)}</td><td>${Math.round(x.confidence*100)}%</td><td>(${x.centroid[0]}, ${x.centroid[1]})</td></tr>`
    ).join('') : '<tr><td colspan="4" class="muted">No objects in frame</td></tr>';
  } catch (e) {} finally { busy = false; }
}, 300);
loadSamples();
</script></body></html>"""

RT = SimpleNamespace(stream=None, engine=None, settings=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings.from_env()
    RT.settings = s
    RT.stream = ZeroLagVideoStream(s.source, s.width, s.height)
    RT.engine = YOLOTrackerEngine(s.model, conf_threshold=s.conf, imgsz=s.imgsz,
                                  stride=s.stride, device=s.device)
    RT.engine.start_async_worker(RT.stream)
    yield
    SHUTDOWN.set()
    RT.engine.stop_async_worker()
    RT.stream.stop()


app = FastAPI(title="Module 1 - YOLO Object Detection & Tracking", lifespan=lifespan)


@app.get("/video_feed")
def video_feed():
    def render(pkt: FramePacket) -> np.ndarray:
        st = RT.engine.get_latest_state()
        return draw_module1_hud(pkt.image, st.detections, RT.stream.source_label, pkt.fps, st.inference_ms)

    return StreamingResponse(mjpeg_frames(RT.stream, render, RT.settings.display_fps),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/telemetry")
def get_telemetry():
    if RT.stream is None or RT.engine is None:
        return JSONResponse({"status": "initializing"})
    st = RT.engine.get_latest_state()
    pkt = RT.stream.read()
    return {
        "source": RT.stream.source_label,
        "stream_fps": pkt.fps if pkt else 0.0,
        "inference_ms": st.inference_ms,
        "inference_fps": st.inference_fps,
        "conf_threshold": RT.engine.conf_threshold,
        "total_count": len(st.detections),
        "person_count": sum(1 for d in st.detections if d["class_name"] == "Person"),
        "detections": st.detections,
    }


@app.get("/api/samples")
def get_samples():
    return {"videos": list_samples()}


@app.post("/api/source")
def switch_source(src: str = Query(...)):
    ok = change_source(RT.stream, RT.engine, src)
    return {"ok": ok, "source": RT.stream.source_label}


@app.post("/api/confidence")
def set_confidence(val: float = Query(...)):
    return {"ok": True, "conf_threshold": RT.engine.set_conf(val)}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(VIDEO_EXTS)}")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAMPLES_DIR / f"uploaded_{name}"
    written = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (limit 300 MB)")
            f.write(chunk)
    ok = change_source(RT.stream, RT.engine, dest.name)
    return {"ok": ok, "filename": dest.name, "source": RT.stream.source_label}


@app.get("/", response_class=HTMLResponse)
def index_page():
    return M1_PAGE.replace("__CSS__", DASHBOARD_CSS)


def main() -> None:
    args = parse_cli("Module 1 - YOLO detection & ByteTrack")
    if not args.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()