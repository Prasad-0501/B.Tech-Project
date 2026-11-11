import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from modules.convlstm_net import (
    BEHAVIOR_CLASSES,
    ConvLSTMBehaviorClassifier,
)
from modules.train_convlstm_net import generate_synthetic_sequence
from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer
from modules.m4_rule_fusion import RuleBasedFusionEngine
from modules.m5_llm_dispatcher import LLMCognitiveDispatcher

def test_neural_convlstm_direct():
    print("=" * 70)
    print("TEST 1: Direct Neural ConvLSTM Sequence Inference & Softmax Distribution")
    print("=" * 70)

    model_path = BASE_DIR / "models" / "convlstm_behavior_net.pt"
    assert model_path.exists(), f"Model not found at {model_path}"

    model = ConvLSTMBehaviorClassifier(in_channels=3, num_classes=len(BEHAVIOR_CLASSES), cnn_channels=24, lstm_hidden=32)
    ckpt = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_classes = [
        "THEFT_SNATCHING_ROBBERY",
        "PHYSICAL_STRUGGLE_FIGHT",
        "SUDDEN_FALL_COLLAPSE",
        "RUNNING_FLEEING",
        "NORMAL_WALKING",
        "SCENE_EMPTY_SECURE",
    ]

    for target_class in test_classes:
        seq = generate_synthetic_sequence(target_class)  # (16, 3, 64, 64)
        pred = model.predict_sequence(seq)
        pred_cls = pred["predicted_class"]
        conf = pred["confidence"]
        print(f"Target: {target_class:25s} -> Predicted: {pred_cls:25s} (Conf: {conf * 100:.1f}%)")
        assert pred_cls == target_class, f"Expected {target_class}, but neural network predicted {pred_cls}"
        assert conf >= 0.25, f"Confidence too low: {conf}"

    print(">>> Direct Neural ConvLSTM Sequence Tests Passed 100%! <<<\n")

def test_m2_analyzer_integration():
    print("=" * 70)
    print("TEST 2: Module 2 YOLOConvLSTMBehaviorAnalyzer Neural Pipeline Integration")
    print("=" * 70)

    analyzer = YOLOConvLSTMBehaviorAnalyzer()
    fusion = RuleBasedFusionEngine()
    dispatcher = LLMCognitiveDispatcher()

    # Test Snatch behavior simulation through the neural pipeline
    analyzer.simulated_behavior_override = "THEFT_SNATCHING_ROBBERY"
    analyzer.simulated_override_expire = 9999999999.0

    dets = [
        {"track_id": 1, "class_id": 0, "class_name": "Person", "confidence": 0.95, "bbox": [100, 100, 180, 300], "centroid": [140, 200]},
        {"track_id": 2, "class_id": 0, "class_name": "Person", "confidence": 0.92, "bbox": [190, 110, 270, 310], "centroid": [230, 210]},
    ]
    b_data = analyzer.update(dets, frame_w=640, frame_h=480)
    print(f"Primary Behavior:    {b_data['primary_behavior']}")
    print(f"Behavior Summary:    {b_data['behavior_summary']}")
    print(f"Confidence:          {b_data['behavior_confidence']}")
    print(f"Threat Score:        {b_data['threat_score']}")

    assert b_data["primary_behavior"] == "THEFT_SNATCHING_ROBBERY"

    # Evaluate Rule Fusion
    a_data = {"top_yamnet_class": "Silence", "surveillance_category": "Normal", "threat_score": 0.05, "db_level": -65.0, "dominant_hz": 0.0}
    f_res = fusion.evaluate({"detections": dets}, b_data, a_data)
    print(f"Rule Triggered:      {f_res['rule_id']}")
    print(f"Action Executed:     {f_res['final_action']}")
    assert f_res["rule_id"] == "R02B_THEFT_SNATCHING_ROBBERY"

    # Evaluate Cognitive LLM
    dec, lat = dispatcher.reason(f_res["inputs_summary"], f_res["fused_score"], f_res["rule_id"])
    print(f"LLM Incident:        {dec.get('incident_type')}")
    print(f"LLM Action:          {dec.get('decided_action')}")
    print(f"LLM Dispatch Brief:  {dec.get('dispatch_brief')}")

    print("\n>>> ALL NEURAL CONVLSTM PIPELINE TESTS PASSED 100%! <<<")

if __name__ == "__main__":
    test_neural_convlstm_direct()
    test_m2_analyzer_integration()
