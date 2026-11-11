import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from modules.m1_yolo_tracker import YOLOTrackerEngine
from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer
from modules.m4_rule_fusion import RuleBasedFusionEngine
from modules.m5_llm_dispatcher import LLMCognitiveDispatcher

def test_uploaded_video():
    vpath = list(Path("samples").glob("*Ghar*phone*chori*.mp4"))[0]
    print(f"Testing on video: {vpath.name}")
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = YOLOTrackerEngine(conf_threshold=0.20, imgsz=416)
    analyzer = YOLOConvLSTMBehaviorAnalyzer()
    fusion = RuleBasedFusionEngine()
    dispatcher = LLMCognitiveDispatcher()

    frame_idx = 0
    snatch_frames = []
    first_snatch_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        # 1. Vision & Tracking
        dets = tracker.detect_frame(frame, use_tracking=True)

        # 2. ConvLSTM Spatiotemporal Behavior Engine - Fed directly with camera frame!
        b_data = analyzer.update(camera_frame=frame, detections=dets, frame_w=w, frame_h=h, timestamp=t_sec)

        # 3. Dummy Audio (silent background for video test)
        a_data = {
            "top_yamnet_class": "Silence",
            "surveillance_category": "Normal Ambient",
            "threat_score": 0.05,
            "db_level": -65.0,
            "dominant_hz": 0.0,
            "voice_detected": False,
        }

        # 4. Multi-Modal Rule Fusion
        y_data = {"detections": dets}
        f_res = fusion.evaluate(y_data, b_data, a_data)

        if b_data["primary_behavior"] == "THEFT_SNATCHING_ROBBERY":
            snatch_frames.append((t_sec, frame_idx, b_data, f_res))
            if first_snatch_result is None:
                first_snatch_result = (t_sec, frame_idx, b_data, f_res)

    cap.release()

    print("\n" + "=" * 70)
    print("TESTBENCH VALIDATION SUMMARY FOR PHONE SNATCH VIDEO")
    print("=" * 70)
    print(f"Total video frames analyzed: {frame_idx}")
    print(f"Frames where ConvLSTM detected THEFT_SNATCHING_ROBBERY: {len(snatch_frames)}")

    assert len(snatch_frames) > 0, "FAILED: ConvLSTM did not classify THEFT_SNATCHING_ROBBERY!"

    t_sec, f_idx, b_data, f_res = first_snatch_result
    print(f"\nFirst Detection Timestamp: {t_sec:.2f}s (Frame #{f_idx})")
    print(f"  Primary Behavior:    {b_data['primary_behavior']}")
    print(f"  Behavior Summary:    {b_data['behavior_summary']}")
    print(f"  Spatiotemporal Anom: {b_data['spatiotemporal_anomaly']}")
    print(f"  Fused Threat Level:  {f_res['threat_level']}")
    print(f"  Fused Threat Score:  {f_res['fused_score']}")
    print(f"  Triggered Rule:      {f_res['rule_id']}")
    print(f"  Rule Action:         {f_res['final_action']}")

    # 5. Test Cognitive LLM Reasoning on the snatch telemetry
    print("\nRunning Cognitive Dispatcher Reasoning on detected snatch...")
    telemetry = {
        "yolo_persons": sum(1 for d in b_data["enriched_detections"] if d.get("class_name") == "Person"),
        "yolo_weapons": [],
        "convlstm_behavior": b_data["primary_behavior"],
        "behavior_summary": b_data["behavior_summary"],
        "spatiotemporal_anomaly": b_data["spatiotemporal_anomaly"],
        "convlstm_energy": b_data["convlstm_energy"],
        "zone_breach": "None",
        "dwell_time_sec": 0.0,
        "audio_event": "Silence",
        "audio_category": "Normal Ambient",
        "dominant_hz": 0.0,
        "audio_db": -65.0,
    }
    decision, lat_ms = dispatcher.reason(telemetry, f_res["fused_score"], f_res["rule_id"])
    print(f"  LLM Decision Latency: {lat_ms:.1f}ms")
    print(f"  Incident Type:        {decision.get('incident_type')}")
    print(f"  Human Reasoning:      {decision.get('human_reasoning')}")
    print(f"  Decided Action:       {decision.get('decided_action')}")
    print(f"  Dispatch Brief:       {decision.get('dispatch_brief')}")

    assert "THEFT" in decision.get("incident_type", "").upper() or "SNATCH" in decision.get("incident_type", "").upper(), "Incident type should mention THEFT/SNATCH"
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_uploaded_video()
