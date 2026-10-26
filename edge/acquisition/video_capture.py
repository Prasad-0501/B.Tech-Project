"""
Video Stream Ingestion Manager
Captures frames from Webcam (0), RTSP CCTV feeds, pre-recorded video files,
or a high-fidelity synthetic CCTV generator with moving entities for testing.
"""
import time
import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Tuple, Optional
from config.settings import CONFIG

class VideoCaptureManager:
    def __init__(self, source: str = CONFIG.DEFAULT_VIDEO_SOURCE):
        self.source = source
        self.cap = None
        self.use_cv2 = False
        self.frame_count = 0
        self.start_time = time.time()
        self.simulated_entities = [
            {"x": 100, "y": 150, "vx": 2.5, "vy": 1.0, "class": "person", "id": 101},
            {"x": 420, "y": 280, "vx": -1.8, "vy": 0.8, "class": "person", "id": 102}
        ]

        # Check cv2 availability
        try:
            # pyrefly: ignore [missing-import]
            import cv2
            self.cv2 = cv2
            self.use_cv2 = True
            # Open source if not demo
            if source.isdigit():
                self.cap = cv2.VideoCapture(int(source))
            elif Path(source).exists():
                self.cap = cv2.VideoCapture(source)
            if self.cap and not self.cap.isOpened():
                self.cap = None
        except Exception:
            self.use_cv2 = False

    def generate_synthetic_cctv_frame(self, trigger_intrusion: bool = False, trigger_erratic: bool = False) -> np.ndarray:
        """
        Generates realistic 640x480 surveillance camera feed:
        Displays timestamp, CCTV overlays, geofenced zones, and dynamic walking pedestrians.
        """
        W, H = CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT
        self.frame_count += 1
        t = (time.time() - self.start_time)

        # Base security camera background: corridor / entrance with perspective
        img = Image.new("RGB", (W, H), color=(30, 35, 42))
        draw = ImageDraw.Draw(img)

        # Architectural floor grid lines
        draw.polygon([(0, H), (W, H), (W * 0.75, H * 0.45), (W * 0.25, H * 0.45)], fill=(45, 52, 60))
        for x in range(0, W, 80):
            draw.line([(x, H), (W * 0.25 + (x / W) * (W * 0.5), H * 0.45)], fill=(55, 62, 70), width=1)
        draw.line([(0, H * 0.45), (W, H * 0.45)], fill=(70, 80, 90), width=2)

        # Draw Geofenced Restricted Zones
        # Zone Alpha: Server & Vault Area
        draw.rectangle([50, 60, 240, 260], outline=(239, 68, 68), width=2)
        draw.text((55, 65), "RESTRICTED: VAULT/SERVER", fill=(239, 68, 68))
        
        # Zone Bravo: East Corridor Boundary
        draw.rectangle([380, 50, 610, 240], outline=(245, 158, 11), width=2)
        draw.text((385, 55), "PERIMETER BOUNDARY", fill=(245, 158, 11))

        # Dynamic Entity Movement
        for ent in self.simulated_entities:
            speed_mult = 3.5 if trigger_erratic else 1.0
            ent["x"] += ent["vx"] * speed_mult
            ent["y"] += ent["vy"] * speed_mult

            # Boundary bounce
            if ent["x"] < 30 or ent["x"] > W - 70:
                ent["vx"] *= -1
            if ent["y"] < H * 0.40 or ent["y"] > H - 120:
                ent["vy"] *= -1

            if trigger_intrusion:
                # Steer entity 101 directly into restricted server zone
                ent["x"] = 120 + 30 * np.sin(t * 1.5)
                ent["y"] = 140 + 20 * np.cos(t * 1.5)

            ex, ey = int(ent["x"]), int(ent["y"])
            # Draw human silhouette
            draw.ellipse([ex + 15, ey, ex + 35, ey + 25], fill=(210, 215, 220)) # Head
            draw.rectangle([ex + 10, ey + 25, ex + 40, ey + 80], fill=(60, 100, 150)) # Torso
            draw.rectangle([ex + 12, ey + 80, ex + 22, ey + 110], fill=(40, 40, 40)) # Leg L
            draw.rectangle([ex + 28, ey + 80, ex + 38, ey + 110], fill=(40, 40, 40)) # Leg R
            # Draw baggage/backpack on entity 101
            if ent["id"] == 101:
                draw.rectangle([ex + 5, ey + 35, ex + 15, ey + 65], fill=(160, 82, 45))

        # CCTV OSD (On-Screen Display)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        draw.text((15, 15), f"CAM-01 [HQ ENTRANCE] | {timestamp_str} | REC [●]", fill=(0, 255, 128))
        draw.text((W - 140, 15), f"FPS: 25.0 | AI: ACTIVE", fill=(0, 255, 128))

        # Subtle noise for CCTV sensor realism
        arr = np.array(img)
        noise = np.random.randint(0, 15, arr.shape, dtype=np.uint8)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return arr

    def read_frame(self, trigger_intrusion: bool = False, trigger_erratic: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Returns:
        - success: bool
        - frame_rgb: np.ndarray (H, W, 3)
        - frame_gray: np.ndarray (H, W)
        """
        if self.cap is not None and self.cap.isOpened():
            ret, cv_frame = self.cap.read()
            if ret:
                frame_rgb = self.cv2.cvtColor(cv_frame, self.cv2.COLOR_BGR2RGB)
                frame_gray = self.cv2.cvtColor(cv_frame, self.cv2.COLOR_BGR2GRAY)
                return True, frame_rgb, frame_gray

        # Synthetic CCTV Generator
        frame_rgb = self.generate_synthetic_cctv_frame(trigger_intrusion, trigger_erratic)
        # Fast luminance grayscale: 0.299 R + 0.587 G + 0.114 B
        frame_gray = (0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]).astype(np.uint8)
        return True, frame_rgb, frame_gray

    def encode_jpeg(self, frame_rgb: np.ndarray) -> bytes:
        """Encodes RGB frame to JPEG byte buffer for web streaming."""
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def save_snapshot(self, frame_rgb: np.ndarray, filename_prefix: str = "alert") -> str:
        """Saves alert frame snapshot to disk."""
        filename = f"{filename_prefix}_{int(time.time())}.jpg"
        filepath = CONFIG.SNAPSHOTS_DIR / filename
        img = Image.fromarray(frame_rgb)
        img.save(filepath, format="JPEG")
        return str(filepath)

VIDEO_CAPTURE = VideoCaptureManager()
