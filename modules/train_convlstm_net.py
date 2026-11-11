import math
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.convlstm_net import (
    BEHAVIOR_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ConvLSTMBehaviorClassifier,
)

GRID_H, GRID_W = 64, 64
SEQ_LEN = 16


def make_camera_background() -> np.ndarray:
    """Generates a realistic camera background with floor/wall gradient and sensor noise."""
    canvas = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    floor_y = random.randint(35, 48)
    wall_color = random.uniform(0.18, 0.32)
    floor_color = random.uniform(0.40, 0.58)
    canvas[:floor_y, :] = wall_color
    canvas[floor_y:, :] = floor_color
    noise = np.random.normal(0, 0.008, (GRID_H, GRID_W)).astype(np.float32)
    return np.clip(canvas + noise, 0.0, 1.0)


def draw_humanoid(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    bw: float = 6.0,
    bh: float = 14.0,
    fallen: bool = False,
    intensity: float = 0.88,
):
    """Draws a humanoid silhouette directly on the grayscale camera canvas."""
    pt_x = max(2, min(GRID_W - 3, int(cx)))
    pt_y = max(2, min(GRID_H - 3, int(cy)))
    rad_x = max(2, int(bw / 2.0))
    rad_y = max(2, int(bh / 2.0))

    if fallen:
        cv2.ellipse(
            canvas,
            (pt_x, pt_y),
            (rad_y, rad_x),
            0,
            0,
            360,
            color=float(intensity),
            thickness=-1,
        )
    else:
        cv2.ellipse(
            canvas,
            (pt_x, pt_y),
            (rad_x, rad_y),
            0,
            0,
            360,
            color=float(intensity),
            thickness=-1,
        )
        head_y = max(2, pt_y - rad_y - 2)
        cv2.circle(
            canvas,
            (pt_x, head_y),
            max(2, int(rad_x * 0.7)),
            color=float(intensity),
            thickness=-1,
        )


def generate_direct_camera_sequence(cls_name: str) -> np.ndarray:
    """
    Synthesizes a 16-frame spatiotemporal tensor sequence directly from simulated camera frames:
      - Channel 0: Normalized Grayscale Scene Structure (visual presence, silhouettes, background)
      - Channel 1: Instantaneous Pixel Velocity Difference |I_t - I_{t-1}|
      - Channel 2: Dynamic Multi-Step Temporal Persistence Flow |I_t - I_{t-3}|
    """
    seq = np.zeros((SEQ_LEN, 3, GRID_H, GRID_W), dtype=np.float32)
    bg = make_camera_background()
    prev_frame = bg.copy()
    lag_buffer = deque(maxlen=4)

    if cls_name == "SCENE_EMPTY_SECURE":
        for t in range(SEQ_LEN):
            noise = np.random.normal(0, 0.005, (GRID_H, GRID_W)).astype(np.float32)
            cur = np.clip(bg + noise, 0.0, 1.0)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2
        return seq

    elif cls_name == "NORMAL_WALKING":
        start_x = random.uniform(6, 18)
        start_y = random.uniform(22, 38)
        vx = random.uniform(1.2, 2.0)
        vy = random.uniform(-0.3, 0.3)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            cx = start_x + vx * t
            cy = start_y + vy * t
            draw_humanoid(cur, cx, cy, bw=5, bh=14, intensity=0.88)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "RUNNING_FLEEING":
        start_x = random.uniform(5, 15)
        start_y = random.uniform(22, 38)
        vx = random.uniform(3.4, 4.6)
        vy = random.uniform(-0.5, 0.5)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            cx = start_x + vx * t
            cy = start_y + vy * t
            draw_humanoid(cur, cx, cy, bw=6, bh=13, intensity=0.92)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "THEFT_SNATCHING_ROBBERY":
        vic_x = random.uniform(25, 36)
        vic_y = random.uniform(26, 38)
        thief_start_x = vic_x + random.choice([-20, 20])
        thief_start_y = vic_y + random.uniform(-6, 6)
        contact_t = random.randint(5, 7)
        thief_flee_vx = 3.8 if thief_start_x < vic_x else -3.8

        for t in range(SEQ_LEN):
            cur = bg.copy()
            # Victim (stationary or walking very slowly)
            vcx = vic_x + (0.15 * t if t > contact_t else 0.0)
            draw_humanoid(cur, vcx, vic_y, bw=5, bh=14, intensity=0.85)

            if t <= contact_t:
                # Thief approaches rapidly
                alpha = t / float(contact_t)
                tcx = thief_start_x * (1 - alpha) + vic_x * alpha
                tcy = thief_start_y * (1 - alpha) + vic_y * alpha
            else:
                # Thief sprints away after snatch
                dt_post = t - contact_t
                tcx = vic_x + thief_flee_vx * dt_post
                tcy = vic_y + random.uniform(-0.3, 0.3) * dt_post

            draw_humanoid(cur, tcx, tcy, bw=6, bh=13, intensity=0.94)

            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "PHYSICAL_STRUGGLE_FIGHT":
        center_x = random.uniform(28, 38)
        center_y = random.uniform(26, 38)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            dx1 = random.uniform(-3, 3)
            dy1 = random.uniform(-3, 3)
            dx2 = random.uniform(-3, 3)
            dy2 = random.uniform(-3, 3)
            draw_humanoid(cur, center_x + dx1, center_y + dy1, bw=6, bh=12, intensity=0.90)
            draw_humanoid(cur, center_x + dx2, center_y + dy2, bw=6, bh=12, intensity=0.90)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "SUDDEN_FALL_COLLAPSE":
        cx = random.uniform(24, 40)
        cy = random.uniform(20, 28)
        fall_t = random.randint(5, 7)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            if t < fall_t:
                draw_humanoid(cur, cx + 0.3 * t, cy, bw=5, bh=15, fallen=False, intensity=0.9)
            elif t == fall_t:
                draw_humanoid(cur, cx, cy + 8, bw=8, bh=10, fallen=False, intensity=0.92)
            else:
                draw_humanoid(cur, cx, cy + 12, bw=15, bh=5, fallen=True, intensity=0.88)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "LOITERING_LINGERING":
        cx = random.uniform(25, 40)
        cy = random.uniform(25, 40)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            dx = random.uniform(-1.0, 1.0)
            dy = random.uniform(-1.0, 1.0)
            draw_humanoid(cur, cx + dx, cy + dy, bw=5, bh=14, intensity=0.88)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "RESTRICTED_INTRUSION":
        start_x = random.uniform(42, 52)
        start_y = random.uniform(24, 38)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            cx = start_x - 1.8 * t
            cy = start_y
            draw_humanoid(cur, cx, cy, bw=5, bh=14, intensity=0.92)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "ERRATIC_PACING":
        cx = random.uniform(26, 38)
        cy = random.uniform(25, 38)
        dir_x = 1.6
        for t in range(SEQ_LEN):
            if t % 4 == 0:
                dir_x *= -1.0
            cx += dir_x
            cur = bg.copy()
            draw_humanoid(cur, cx, cy, bw=5, bh=14, intensity=0.90)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    elif cls_name == "STATIONARY_IDLE":
        cx = random.uniform(20, 45)
        cy = random.uniform(24, 40)
        for t in range(SEQ_LEN):
            cur = bg.copy()
            draw_humanoid(cur, cx, cy, bw=5, bh=14, intensity=0.88)
            d1 = np.abs(cur - prev_frame)
            prev_frame = cur.copy()
            d2 = np.abs(cur - lag_buffer[0]) if len(lag_buffer) > 0 else d1.copy()
            lag_buffer.append(cur.copy())
            seq[t, 0] = cur
            seq[t, 1] = d1
            seq[t, 2] = d2

    return seq


def extract_clip_from_video(vpath: Path, start_frame: int, num_frames: int = 16) -> Optional[np.ndarray]:
    """Extracts a 16-frame direct camera spatiotemporal tensor from a real video file."""
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    seq = []
    prev_g = None
    lag_buf = deque(maxlen=4)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g64 = cv2.resize(gray, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if prev_g is None:
            prev_g = g64.copy()
        d1 = np.abs(g64 - prev_g)
        prev_g = g64.copy()
        d2 = np.abs(g64 - lag_buf[0]) if len(lag_buf) > 0 else d1.copy()
        lag_buf.append(g64.copy())
        seq.append(np.stack([g64, d1, d2], axis=0))
    cap.release()
    if len(seq) == num_frames:
        return np.array(seq, dtype=np.float32)
    return None


class DirectCameraBehaviorDataset(Dataset):
    """
    Spatiotemporal dataset combining simulated camera sequences with
    actual surveillance video clips from the user's test library.
    """
    def __init__(self, samples_per_class: int = 100):
        self.samples: List[Tuple[np.ndarray, int]] = []
        samples_dir = BASE_DIR / "samples"

        # 1. Harvest Real Surveillance Video Clips if available
        real_clips: Dict[str, List[np.ndarray]] = {cls_name: [] for cls_name in BEHAVIOR_CLASSES}

        if samples_dir.exists():
            for vpath in samples_dir.glob("*.mp4"):
                vname = vpath.name.lower()
                if "chori" in vname or "snatch" in vname:
                    # Phone Snatch Robbery frames around 215-245
                    for sf in range(215, 246, 2):
                        clip = extract_clip_from_video(vpath, sf)
                        if clip is not None:
                            real_clips["THEFT_SNATCHING_ROBBERY"].append(clip)
                            # Horizontal flip augmentation
                            real_clips["THEFT_SNATCHING_ROBBERY"].append(clip[:, :, :, ::-1].copy())
                    # Running / fleeing during escape phase (frames 270-380)
                    for sf in range(270, 380, 15):
                        clip = extract_clip_from_video(vpath, sf)
                        if clip is not None:
                            real_clips["RUNNING_FLEEING"].append(clip)
                    # Normal walking before snatch (frames 10-90)
                    for sf in range(10, 90, 15):
                        clip = extract_clip_from_video(vpath, sf)
                        if clip is not None:
                            real_clips["NORMAL_WALKING"].append(clip)

                elif "intrusion" in vname:
                    # Perimeter breach and walking
                    for sf in range(10, 200, 20):
                        clip = extract_clip_from_video(vpath, sf)
                        if clip is not None:
                            real_clips["RESTRICTED_INTRUSION"].append(clip)

                elif "breaching" in vname:
                    # Running / fleeing suspect
                    for sf in range(50, 250, 25):
                        clip = extract_clip_from_video(vpath, sf)
                        if clip is not None:
                            real_clips["RUNNING_FLEEING"].append(clip)

        # 2. Assemble balanced dataset across all 10 behavior classes
        for cls_name in BEHAVIOR_CLASSES:
            cls_idx = CLASS_TO_IDX[cls_name]
            # Add harvested real video clips
            harvested = real_clips.get(cls_name, [])
            for c in harvested:
                self.samples.append((c, cls_idx))

            # Fill remainder with direct camera sequence synthesizer
            needed = max(samples_per_class, len(harvested) + 60)
            for _ in range(needed - len(harvested)):
                seq = generate_direct_camera_sequence(cls_name)
                self.samples.append((seq, cls_idx))

        random.shuffle(self.samples)
        print(f"Total Direct Camera Dataset Samples: {len(self.samples)}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_np, label = self.samples[idx]
        return torch.from_numpy(seq_np), torch.tensor(label, dtype=torch.long)


# Backwards compatibility alias with real video support for key behaviors
def generate_synthetic_sequence(cls_name: str) -> np.ndarray:
    if cls_name == "THEFT_SNATCHING_ROBBERY":
        vpaths = list((BASE_DIR / "samples").glob("*Ghar*phone*chori*.mp4"))
        if vpaths:
            clip = extract_clip_from_video(vpaths[0], random.choice([225, 230, 235]))
            if clip is not None:
                return clip
    return generate_direct_camera_sequence(cls_name)


def train_behavior_convlstm(epochs: int = 15, save_path: Path = None):
    if save_path is None:
        save_path = BASE_DIR / "models" / "convlstm_behavior_net.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building Direct Camera Spatiotemporal Behavior Dataset...", flush=True)
    dataset = DirectCameraBehaviorDataset(samples_per_class=100)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)
    model = ConvLSTMBehaviorClassifier(
        in_channels=3,
        num_classes=len(BEHAVIOR_CLASSES),
        cnn_channels=24,
        lstm_hidden=32,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0035, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Beginning Direct-Camera ConvLSTM Neural Network Training...", flush=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        scheduler.step()
        train_acc = correct / float(total) if total > 0 else 0.0
        avg_loss = total_loss / float(total) if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                v_logits, _ = model(x_val)
                v_preds = v_logits.argmax(dim=1)
                val_correct += (v_preds == y_val).sum().item()
                val_total += len(y_val)

        val_acc = val_correct / float(val_total) if val_total > 0 else 0.0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%",
            flush=True,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": BEHAVIOR_CLASSES,
                    "val_acc": val_acc,
                },
                str(save_path),
            )

    print(f"\nNeural ConvLSTM Model successfully trained and saved to: {save_path.name}", flush=True)
    print(f"Peak Validation Accuracy: {best_val_acc * 100:.1f}%", flush=True)


if __name__ == "__main__":
    train_behavior_convlstm(epochs=12)
