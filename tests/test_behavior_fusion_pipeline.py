import sys
import time
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.m2_behavior_zones import YOLOConvLSTMBehaviorAnalyzer
from modules.m4_rule_fusion import RuleBasedFusionEngine
from modules.m5_llm_dispatcher import LLMCognitiveDispatcher

def test_pipeline():
    analyzer = YOLOConvLSTMBehaviorAnalyzer()
    fusion = RuleBasedFusionEngine()
    dispatcher = LLMCognitiveDispatcher()

    # 1. Test Fall Behavior
    analyzer.simulated_behavior_override = "SUDDEN_FALL_COLLAPSE"
    analyzer.simulated_override_expire = time.time() + 10.0
    dets = [{'track_id': 1, 'class_name': 'Person', 'confidence': 0.88, 'centroid': (300, 300), 'bbox': (250, 280, 350, 320)}]
    b_data = analyzer.update(dets, 640, 480)
    yolo_data = {"detections": dets, "vis_ms": 15.0}
    a_data = {"top_yamnet_class": "Silence", "surveillance_category": "Normal Ambient", "threat_score": 0.05, "db_level": -55.0, "dominant_hz": 0.0}

    f_res = fusion.evaluate(yolo_data, b_data, a_data)
    print("\n--- TEST: SUDDEN FALL COLLAPSE ---")
    print("Rule Triggered:", f_res["rule_id"])
    print("Threat Level:", f_res["threat_level"])
    print("Fused Score:", f_res["fused_score"])
    print("Action:", f_res["final_action"])
    assert f_res["rule_id"] == "R07A_SUDDEN_FALL_COLLAPSE_DISTRESS"
    assert f_res["fused_score"] >= 0.85

    dec, lat = dispatcher.reason(f_res["inputs_summary"], f_res["fused_score"], f_res["rule_id"])
    print("Dispatcher Incident:", dec.get("incident_type"))
    print("Dispatcher Action:", dec.get("decided_action"))
    print("Dispatcher Reasoning:", dec.get("human_reasoning"))
    assert "COLLAPSE" in dec.get("incident_type", "") or "fall" in dec.get("human_reasoning", "").lower()

    # 2. Test Running Behavior
    analyzer.simulated_behavior_override = "RUNNING_FLEEING"
    analyzer.simulated_override_expire = time.time() + 10.0
    b_data = analyzer.update(dets, 640, 480)
    f_res = fusion.evaluate(yolo_data, b_data, a_data)
    print("\n--- TEST: RUNNING FLEEING ---")
    print("Rule Triggered:", f_res["rule_id"])
    print("Threat Level:", f_res["threat_level"])
    print("Action:", f_res["final_action"])
    assert "RUNNING" in f_res["rule_id"] or "FLEEING" in f_res["rule_id"]

    # 3. Test Normal Walking
    analyzer.simulated_behavior_override = None
    analyzer.simulated_override_expire = 0.0
    # Provide upright person in safe zone
    normal_dets = [{'track_id': 2, 'class_name': 'Person', 'confidence': 0.90, 'centroid': (300, 20), 'bbox': (280, 5, 320, 60)}]
    b_data = analyzer.update(normal_dets, 640, 480)
    f_res = fusion.evaluate({"detections": normal_dets, "vis_ms": 15.0}, b_data, a_data)
    print("\n--- TEST: NORMAL BASELINE ---")
    print("Rule Triggered:", f_res["rule_id"])
    print("Threat Level:", f_res["threat_level"])
    assert f_res["threat_level"] in ["NORMAL", "WARNING"]

    print("\n>>> ALL BEHAVIOR FUSION PIPELINE TESTS PASSED 100%! <<<")

if __name__ == "__main__":
    test_pipeline()
