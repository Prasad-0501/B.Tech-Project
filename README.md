# 🛡️ SmartGuard: Multi-Modal Edge AI Surveillance System

An intelligent, multi-modal edge surveillance system designed for low-latency on-device processing (targeting Raspberry Pi 4/5 and local edge hardware).

---

## 🌟 Key Features & Architecture

The system is organized into modular edge components:

### 1. Vision & Tracking (`modules/m1_yolo_tracker.py`)
- **Model**: YOLOv8n (`yolov8n.pt`) with real-time ByteTrack Kalman tracking.
- **Performance**: Zero-lag asynchronous frame acquisition pipeline (`ZeroLagVideoStream`).
- **Classes**: Tracks MS COCO surveillance entities (Persons, Backpacks, Suitcases, Knives, etc.).

### 2. Spatiotemporal ConvLSTM Behavior Analyzer & Geofencing (`modules/m2_behavior_zones.py`)
- **3-Channel Spatiotemporal Tensor (3 x 64 x 64)**: Encodings for Gaussian spatial presence + posture aspect ratio ($h/w$), velocity flow vectors, and temporal frame-difference kinetic deltas.
- **2D ConvLSTM Recurrent Engine**: Maintains spatiotemporal memory ($T=16$ sequence) executing 2D spatial convolutions in $< 2\text{ms}$ to capture trajectory continuity and posture shifts.
- **Multi-Class Human Behavior Classification**:
  - `NORMAL_WALKING`: Upright posture ($AR \ge 1.4$), smooth trajectory, nominal velocity ($20-130\text{ px/s}$).
  - `RUNNING_FLEEING`: High velocity surge ($> 150\text{ px/s}$) with rapid spatial dispersion.
  - `SUDDEN_FALL_COLLAPSE`: Rapid vertical drop followed by aspect ratio inversion ($h/w < 1.0$) and floor stillness (critical medical alert).
  - `LOITERING_LINGERING`: Confined localized energy density with prolonged dwell ($\ge 6.0\text{s}$) in monitored/restricted sectors.
  - `ERRATIC_PACING`: Chaotic trajectory oscillations ($> 90^\circ$ sharp turns) and agitated jitter.
  - `RESTRICTED_INTRUSION`: Geofence boundary violation via ray-casting point-in-polygon.
- **Interactive Behavior Test Deck**: Instant testbench simulation controls for all behavioral modes in the dashboard.

### 3. Acoustic AI (`modules/m3_audio_ast.py`)
- **Model**: Google YAMNet ONNX (AudioSet 521 audio classes).
- **Execution**: Real-time microphone capture or pre-recorded audio events (Glass breaking, Gunshots, Screams, Sirens) with ~12ms inference.

### 4. Rule-Based Multi-Modal Fusion (`modules/m4_rule_fusion.py`)
- **Deterministic 17-Rule Ladder**: Pure deterministic if-else fusion combining YOLOv8 vision, ConvLSTM behavior, and YAMNet audio in 0.02ms.
- **Threat Scoring**: Generates multi-modal threat scores (0.05 to 1.00) and discrete actions without LLM dependency.

### 5. LLM Cognitive Dispatcher & Human-Like Reasoner (`modules/m5_llm_dispatcher.py`)
- **Structured JSON Telemetry**: Ingests structured telemetry JSON from M1, M2, and M3 alongside M4 threat scores.
- **On-Device LLM**: Local distilled `Qwen/Qwen2.5-0.5B-Instruct` generates human-like situational reasoning, proportional tactical actions, and security dispatch briefs.
- **Decoupled Architecture**: 30 FPS video pipeline runs completely decoupled from background neural reasoning (zero video lag).
- **Evidence Logging**: SQLite incident audit trail with snapshot captures.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Prasad-0501/B.Tech-Project.git
cd intelligent-surveillance-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows PowerShell / CMD
# source venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full System (Module 5: End-to-End LLM Cognitive Dispatcher)
```bash
python modules/m5_llm_dispatcher.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 3. Run Deterministic Rule-Based Fusion (Module 4)
```bash
python modules/m4_rule_fusion.py
```

---

## 📁 Repository Structure
```
├── modules/
│   ├── m1_yolo_tracker.py     # Vision & ByteTrack Kalman tracking
│   ├── m2_behavior_zones.py   # ConvLSTM behavior & restricted zones
│   ├── m3_audio_ast.py        # Google YAMNet audio classification & VAD
│   ├── m4_rule_fusion.py      # Deterministic 17-step rule-based fusion
│   └── m5_llm_dispatcher.py   # Dedicated LLM cognitive dispatcher (Full System)
├── models/
│   ├── yamnet.onnx            # Google YAMNet ONNX weights
│   └── yamnet_class_map.csv   # 521 AudioSet class mapping
├── samples/
│   ├── suspicious_intrusion_cctv.mp4
│   └── audio/*.wav            # Benchmark audio samples
├── yolov8n.pt                 # YOLOv8 nano model
├── run_surveillance.py        # Unified surveillance runner
└── README.md
```
