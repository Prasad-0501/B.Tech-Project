"""
MODULE 4: Multi-Modal Decision Fusion & Security Reasoner Engine
================================================================
Combines:
  - Visual Score S_visual (from Module 1 YOLOv8 + ByteTrack)
  - Spatiotemporal Behavior Score S_behavior (from Module 2 ConvLSTM + Geofences)
  - Acoustic Score S_audio (from Module 3 Google YAMNet Audio AI)

Formulas:
  S_fused = 0.45 * S_visual + 0.35 * S_audio + 0.20 * S_behavior + Synergy_Boost
  Cross-Modal Synergy: Boosts threat if visual zone breach + acoustic anomaly happen together.
"""

import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "surveillance_events.db"
SNAPSHOTS_DIR = BASE_DIR / "database" / "snapshots"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    """Initializes clean SQLite tables for incidents."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            threat_level TEXT NOT NULL,
            fused_score REAL NOT NULL,
            visual_score REAL NOT NULL,
            audio_score REAL NOT NULL,
            behavior_type TEXT NOT NULL,
            audio_class TEXT NOT NULL,
            detected_entities TEXT NOT NULL,
            llm_summary TEXT NOT NULL,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    conn.close()


init_database()


class MultiModalFusionEngine:
    def __init__(
        self,
        w_visual: float = 0.45,
        w_audio: float = 0.35,
        w_behavior: float = 0.20,
        elevated_thresh: float = 0.40,
        critical_thresh: float = 0.68,
    ):
        self.w_visual = w_visual
        self.w_audio = w_audio
        self.w_behavior = w_behavior
        self.elevated_thresh = elevated_thresh
        self.critical_thresh = critical_thresh
        self.last_incident_time = 0.0
        self.incident_cooldown_sec = 4.0  # Prevent spamming alerts

    def fuse(
        self,
        visual_data: Dict[str, Any],
        behavior_data: Dict[str, Any],
        audio_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes the fused threat score and generates incident records.
        """
        now = time.time()

        # 1. Component scores
        # Visual threat: based on dangerous detected objects (knife, weapon) + person presence
        detections = visual_data.get("detections", [])
        has_weapon = any(d.get("class_name") in ["Knife", "Scissors", "Baseball Bat"] for d in detections)
        has_person = any(d.get("class_name") == "Person" for d in detections)

        s_visual = 0.05
        if has_person:
            s_visual = 0.20
        if has_weapon:
            s_visual = 0.90

        # Behavior threat: from ConvLSTM + Zone Intrusion + Loitering
        s_behavior = float(behavior_data.get("threat_score", 0.05))

        # Audio threat: from Google YAMNet
        s_audio = float(audio_data.get("threat_score", 0.05))
        top_sound = audio_data.get("top_yamnet_class", "Silence")
        audio_cat = audio_data.get("surveillance_category", "Normal Ambient")

        # 2. Adaptive Synergy Boost (Cross-modal correlation)
        # If visual breach/weapon happens AT THE SAME TIME as a loud acoustic anomaly, add synergy bonus
        synergy_boost = 0.0
        if (s_visual > 0.40 or s_behavior > 0.50) and s_audio > 0.60:
            synergy_boost = 0.15

        # 3. Compute weighted fusion
        fused_raw = (self.w_visual * s_visual) + (self.w_audio * s_audio) + (self.w_behavior * s_behavior) + synergy_boost
        fused_score = round(min(1.0, max(0.05, fused_raw)), 2)

        # 4. Threat level classification
        if fused_score >= self.critical_thresh:
            threat_level = "CRITICAL"
        elif fused_score >= self.elevated_thresh:
            threat_level = "WARNING"
        else:
            threat_level = "NORMAL"

        # 5. Extract primary behavior summary
        intrusions = behavior_data.get("intrusions", [])
        loiterings = behavior_data.get("loitering_events", [])
        runnings = behavior_data.get("running_events", [])

        behavior_desc = "Normal Movement"
        if loiterings:
            behavior_desc = f"Loitering in {loiterings[0]['zone_name']} ({loiterings[0]['dwell_sec']}s)"
        elif intrusions:
            behavior_desc = f"Zone Intrusion in {intrusions[0]['zone_name']}"
        elif runnings:
            behavior_desc = f"High-Speed Running ({runnings[0]['speed_px_s']} px/s)"
        elif has_weapon:
            behavior_desc = "Armed Entity Detected"

        # 6. Generate natural language incident summary (Edge Rule Reasoner)
        should_record = False
        summary = ""
        if threat_level in ["WARNING", "CRITICAL"]:
            if now - self.last_incident_time > self.incident_cooldown_sec:
                should_record = True
                self.last_incident_time = now

                summary = self._generate_security_summary(
                    threat_level=threat_level,
                    fused_score=fused_score,
                    behavior_desc=behavior_desc,
                    top_sound=top_sound,
                    audio_cat=audio_cat,
                    detections=detections,
                )

        return {
            "timestamp": now,
            "threat_level": threat_level,
            "fused_score": fused_score,
            "visual_score": round(s_visual, 2),
            "audio_score": round(s_audio, 2),
            "behavior_score": round(s_behavior, 2),
            "synergy_boost": round(synergy_boost, 2),
            "behavior_desc": behavior_desc,
            "top_sound": top_sound,
            "audio_cat": audio_cat,
            "should_record": should_record,
            "summary": summary,
        }

    def _generate_security_summary(
        self,
        threat_level: str,
        fused_score: float,
        behavior_desc: str,
        top_sound: str,
        audio_cat: str,
        detections: List[Dict[str, Any]],
    ) -> str:
        entities = [f"{d['class_name']} #{d.get('track_id', 1)}" for d in detections[:3]]
        ent_str = ", ".join(entities) if entities else "Unidentified Entity"

        if threat_level == "CRITICAL":
            return (
                f"CRITICAL BREACH: Multi-modal correlation confirmed (Score: {fused_score:.2f}). "
                f"{ent_str} engaged in {behavior_desc} while acoustic sensor detected '{top_sound}' ({audio_cat}). "
                f"Immediate perimeter dispatch recommended."
            )
        else:
            return (
                f"SECURITY WARNING: Elevated suspicion (Score: {fused_score:.2f}). "
                f"{ent_str} flagged for {behavior_desc}. Acoustic status: '{top_sound}'. Operator review advised."
            )

    def log_incident_to_db(self, incident_data: Dict[str, Any], snapshot_path: Optional[str] = None):
        """Saves incident into SQLite."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    timestamp, threat_level, fused_score, visual_score, audio_score,
                    behavior_type, audio_class, detected_entities, llm_summary, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(incident_data["timestamp"])),
                incident_data["threat_level"],
                incident_data["fused_score"],
                incident_data["visual_score"],
                incident_data["audio_score"],
                incident_data["behavior_desc"],
                incident_data["top_sound"],
                json.dumps(incident_data.get("detections", [])),
                incident_data["summary"],
                snapshot_path,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Module 4] DB log error: {e}", flush=True)

    def get_recent_incidents(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieves recent incidents from SQLite."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []
