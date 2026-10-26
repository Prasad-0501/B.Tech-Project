# 🛡️ SmartGuard: Multi-Modal Edge AI Surveillance System

An intelligent, multi-modal edge surveillance system designed for low-latency on-device processing (targeting Raspberry Pi 4/5 and local edge hardware).

---

## 🌟 Key Features & Architecture

The system is organized into modular edge components:

### 1. Vision & Tracking (`modules/m1_yolo_tracker.py`)
- **Model**: YOLOv8n (`yolov8n.pt`) with real-time ByteTrack Kalman tracking.
- **Performance**: Zero-lag asynchronous frame acquisition pipeline (`ZeroLagVideoStream`).
- **Classes**: Tracks MS COCO surveillance entities (Persons, Backpacks, Suitcases, Knives, etc.).

### 2. Spatiotemporal Behavior & Zones (`modules/m2_behavior_zones.py`)
- **ConvLSTM Recurrent Engine**: Generates a 2D $64 \times 64$ spatial occupancy tensor from YOLO detections and processes it through a 2D ConvLSTM cell (< 3ms latency) to detect kinetic anomalies without background noise.
- **Geofencing**: Ray-casting point-in-polygon engine for restricted zones (e.g. Server Vault, Perimeter).
- **Kinematics**: Dwell-time loitering tracking and movement velocity estimation.

### 3. Acoustic AI (`modules/m3_audio_ast.py`)
- **Model**: Google YAMNet ONNX (AudioSet 521 audio classes).
- **Execution**: Real-time microphone capture or pre-recorded audio events (Glass breaking, Gunshots, Screams, Sirens) with ~12ms inference.

### 4. Multi-Modal Cognitive Core (`modules/m4_fusion_llm.py`)
- **Structured JSON Telemetry**: Serializes multi-modal inputs (vision detections, behavior state, acoustic classes) into a structured payload.
- **On-Device LLM**: Integrates local distilled `Qwen2.5-0.5B-Instruct` for incident reasoning and automated security dispatch.
- **Decoupled Architecture**: High-speed MJPEG video generator (30 FPS) runs completely decoupled from LLM inference loops to ensure zero stream freezing.
- **Evidence Logging**: SQLite incident audit trail with snapshot captures.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd intelligent-surveillance-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Mission Control
```bash
python modules/m4_fusion_llm.py
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 📁 Repository Structure
```
├── modules/
│   ├── m1_yolo_tracker.py     # Vision & ByteTrack Kalman tracking
│   ├── m2_behavior_zones.py   # ConvLSTM behavior & restricted zones
│   ├── m3_audio_ast.py        # Google YAMNet audio classification
│   ├── m4_fusion_engine.py    # Math fusion engine
│   └── m4_fusion_llm.py       # Full mission control with on-device LLM
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
