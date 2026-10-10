"""Deterministic 200-scene catalog requested on 2026-10-07.

Every catalog block contains two related scenarios. Mirror directions and the
two longitudinal variants at one speed are therefore assigned together.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random


SEED = 20261007
HERE = Path(__file__).resolve().parent
OLD_CATALOG = HERE.parent / "ce_gmr_aggressive_grid_v2/manifests/aggressive_scenario_catalog.json"

TYPE_TOTALS = {"N1": 10, "N2": 10, "N3": 10, "N4": 10,
               "A1": 32, "A2": 32, "A3": 24, "A4": 24,
               "A5": 24, "A6": 16, "A7": 8}
TYPE_SPLITS = {
    "N1": (6, 2, 2), "N2": (6, 2, 2), "N3": (6, 2, 2), "N4": (6, 2, 2),
    "A1": (20, 6, 6), "A2": (20, 6, 6), "A3": (14, 6, 4),
    "A4": (14, 4, 6), "A5": (14, 4, 6), "A6": (10, 4, 2), "A7": (4, 2, 2),
}
SPEED_COUNTS = {
    "N1": {10: 2, 15: 2, 20: 2, 25: 2, 30: 2},
    "N2": {15: 2, 20: 4, 25: 2, 30: 2},
    "N3": {15: 2, 20: 2, 25: 2, 30: 2, 35: 2},
    "N4": {10: 2, 15: 2, 20: 2, 25: 2, 30: 2},
    "A1": {20: 4, 25: 6, 30: 6, 35: 8, 40: 8},
    "A2": {15: 4, 20: 4, 25: 4, 30: 6, 35: 8, 40: 6},
    "A3": {15: 2, 20: 4, 25: 6, 30: 6, 35: 6},
    "A4": {20: 2, 25: 6, 30: 4, 35: 6, 40: 6},
    "A5": {20: 2, 25: 4, 30: 6, 35: 6, 40: 6},
    "A6": {20: 4, 25: 4, 30: 4, 35: 4},
    "A7": {15: 2, 25: 2, 40: 4},
}

A_PARAMS = {
    "A1": [(0.55, 1.0), (0.65, 0.8), (0.75, 0.8), (0.85, 1.0)],
    "A2": [(0.50, 1.0), (0.65, 0.8), (0.75, 0.7), (0.80, 0.6)],
    "A3": [(0.30, 0.60, 2.0), (0.65, 0.30, 2.0), (0.30, 0.80, 3.0)],
    "A4": [(0.30, 0.50, 0.8, "pre_turn"),
           (0.40, 0.55, 0.0, "turn_in"),
           (0.45, 0.60, 0.4, "pre_turn")],
    "A5": [(0.30, 0.55, 1.0), (0.35, 0.65, 1.0), (0.45, 0.60, 1.5)],
    "A6": [(0.25, 0.50, 0.8, "first_turn"),
           (0.30, 0.60, 0.8, "reversal")],
}


def _base(scene_type: str, speed: int, group: int, variant: int,
          direction: str, **kw) -> dict:
    stable = scene_type.startswith("N")
    return {
        "scene_type": scene_type,
        "source_group": "nominal" if stable else "aggressive",
        "design_class": "stable" if stable else "aggressive",
        "speed_mps": float(speed),
        "parameter_group": int(group),
        "variant_index": int(variant),
        "direction": direction,
        "stable_lead_s": 3.0,
        "recovery_s": 2.0,
        "controller": "FoundationBridge_F1_single_query_corrected_QP",
        **kw,
    }


def _blocks() -> list[list[dict]]:
    blocks: list[list[dict]] = []
    for speed in (10, 15, 20, 25, 30):
        blocks.append([
            _base("N1", speed, 1, 1, "S", target_ax_g=+0.10,
                  longitudinal_action="mild_acceleration", accel_hold_s=2.0),
            _base("N1", speed, 2, 2, "S", target_ax_g=-0.10,
                  longitudinal_action="mild_deceleration", accel_hold_s=2.0),
        ])
    for speed in (15, 20, 25, 30):
        blocks.append([_base("N2", speed, 1, i+1, d, target_ay_g=0.15,
                    curvature_rise_s=2.0, curve_hold_s=3.0) for i, d in enumerate(("L", "R"))])
    blocks.append([_base("N2", 20, 2, i+1, d, target_ay_g=0.25,
                curvature_rise_s=2.0, curve_hold_s=3.0) for i, d in enumerate(("L", "R"))])
    for speed in (15, 20, 25, 30, 35):
        blocks.append([_base("N3", speed, 1, i+1, d, target_ay_g=0.25,
                    curvature_rise_s=1.5, reversal_s=2.0) for i, d in enumerate(("LR", "RL"))])
    for speed in (10, 20, 30):
        blocks.append([
            _base("N4", speed, 1, 1, "S", target_ax_g=-0.10, target_ay_g=0.0,
                  brake_hold_s=1.5, brake_phase="straight"),
            _base("N4", speed, 2, 2, "S", target_ax_g=-0.20, target_ay_g=0.0,
                  brake_hold_s=1.5, brake_phase="straight"),
        ])
    for speed in (15, 25):
        blocks.append([_base("N4", speed, 1, i+1, d, target_ax_g=-0.15,
                    target_ay_g=0.20, brake_hold_s=1.5, brake_phase="gentle_curve")
                    for i, d in enumerate(("L", "R"))])

    for scene_type in ("A1", "A2", "A3", "A4", "A5", "A6"):
        for speed, count in SPEED_COUNTS[scene_type].items():
            for pidx in range(count // 2):
                p = A_PARAMS[scene_type][pidx]
                directions = ("LR", "RL") if scene_type in {"A2", "A6"} else ("L", "R")
                pair = []
                for didx, direction in enumerate(directions):
                    if scene_type == "A1":
                        kw = dict(target_ay_g=p[0], curvature_rise_s=p[1], curve_hold_s=3.0)
                    elif scene_type == "A2":
                        kw = dict(target_ay_g=p[0], reversal_s=p[1], curvature_rise_s=1.0)
                    elif scene_type == "A3":
                        kw = dict(target_ay_start_g=p[0], target_ay_end_g=p[1],
                                  target_ay_g=max(p[:2]), curvature_change_s=p[2])
                    elif scene_type == "A4":
                        kw = dict(target_ax_g=-p[0], target_ay_g=p[1],
                                  brake_lead_s=p[2], brake_phase=p[3], brake_hold_s=1.0)
                    elif scene_type == "A5":
                        kw = dict(target_ax_g=-p[0], target_ay_g=p[1],
                                  brake_hold_s=p[2], brake_phase="in_curve")
                    else:
                        kw = dict(target_ax_g=-p[0], target_ay_g=p[1], reversal_s=p[2],
                                  brake_phase=p[3], brake_hold_s=1.0)
                    pair.append(_base(scene_type, speed, pidx+1, didx+1, direction, **kw))
                blocks.append(pair)
    for speed, values in ((15, (0.50, 0.65)), (25, (0.50, 0.65)),
                          (40, (0.45, 0.55, 0.65, 0.75))):
        for pair_index in range(0, len(values), 2):
            blocks.append([
                _base("A7", speed, pair_index//2+1, j+1, "S",
                      target_ax_g=-values[pair_index+j], target_ay_g=0.0,
                      brake_phase="straight", brake_hold_s=1.0)
                for j in range(2)
            ])
    return blocks


def _assign_splits(blocks: list[list[dict]]) -> None:
    rng = random.Random(SEED)
    by_type: dict[str, list[list[dict]]] = defaultdict(list)
    for block in blocks:
        by_type[block[0]["scene_type"]].append(block)
    best = None
    for _ in range(30000):
        complete = {s: [] for s in ("train", "validation", "test")}
        family_assignment = {}
        for scene_type, family_blocks in by_type.items():
            quotas = dict(zip(("train", "validation", "test"),
                              (x // 2 for x in TYPE_SPLITS[scene_type])))
            trial = family_blocks[:]; rng.shuffle(trial); pos = 0
            family_assignment[scene_type] = {}
            for split in ("validation", "test", "train"):
                chosen = trial[pos:pos+quotas[split]]; pos += quotas[split]
                family_assignment[scene_type][split] = chosen; complete[split].extend(chosen)
        score = 0.0
        overall_mean = sum(b[0]["speed_mps"] for b in blocks)/len(blocks)
        for split, chosen in complete.items():
            speeds=[b[0]["speed_mps"] for b in chosen]
            score += abs(sum(speeds)/len(speeds)-overall_mean)
            # 15--40 m/s are the principal speed bands and should all occur in
            # every partition when the fixed type quotas permit it.
            score += 100.0*sum(v not in speeds for v in (15,20,25,30,35,40))
        if best is None or score < best[0]: best=(score,family_assignment)
    for assigned in best[1].values():
        for split, chosen in assigned.items():
            for block in chosen:
                for row in block: row["split"] = split


def _old_identity(row: dict) -> dict:
    if not OLD_CATALOG.exists():
        return {"prior_identity": "none", "prior_scenario_id": "",
                "identity_handling": "new_catalog_configuration"}
    mapping = {"A1": "HSC", "A2": "RSB", "A3": "TGT", "A4": "BIT",
               "A5": "CCB", "A6": "BSB", "A7": "SEB"}
    old = json.loads(OLD_CATALOG.read_text(encoding="utf-8"))["scenarios"]
    candidates = [x for x in old if x["family_code"] == mapping.get(row["scene_type"])
                  and float(x["initial_speed_mps"]) == row["speed_mps"]
                  and x["direction"] == row["direction"]]
    if not candidates:
        return {"prior_identity": "none", "prior_scenario_id": "",
                "identity_handling": "new_catalog_configuration"}
    ay, ax = abs(float(row.get("target_ay_g", 0.0))), abs(float(row.get("target_ax_g", 0.0)))
    nearest = min(candidates, key=lambda x: abs(abs(float(x["target_ay_g"]))-ay)
                  + abs(abs(float(x["target_ax_g"]))-ax))
    distance = abs(abs(float(nearest["target_ay_g"]))-ay) + abs(abs(float(nearest["target_ax_g"]))-ax)
    identity = "near_repeat_previously_accessed" if distance <= 0.10 else "same_family_speed_previously_accessed"
    return {"prior_identity": identity, "prior_scenario_id": nearest["scenario_id"],
            "prior_split": nearest["split"], "blind_test_eligible": False,
            "identity_handling": "keep requested split but retain prior identity; never claim as a new blind scene"}


@lru_cache(maxsize=1)
def build_catalog() -> list[dict]:
    blocks = _blocks()
    _assign_splits(blocks)
    counters = Counter()
    rows = []
    for block_index, block in enumerate(blocks, 1):
        for row in block:
            counters[row["scene_type"]] += 1
            row["scenario_id"] = f"F1C200_{row['scene_type']}_{counters[row['scene_type']]:03d}"
            row["mirror_block_id"] = f"B{block_index:03d}"
            row["run_order"] = len(rows) + 1
            row.update(_old_identity(row))
            row.setdefault("blind_test_eligible", row["prior_identity"] == "none")
            rows.append(row)
    validate_catalog(rows)
    return rows


def validate_catalog(rows: list[dict]) -> None:
    assert len(rows) == 200
    assert Counter(x["scene_type"] for x in rows) == Counter(TYPE_TOTALS)
    assert Counter((x["scene_type"], int(x["speed_mps"])) for x in rows) == Counter(
        (t, v) for t, counts in SPEED_COUNTS.items() for v, n in counts.items() for _ in range(n))
    assert Counter(x["split"] for x in rows) == {"train": 120, "validation": 40, "test": 40}
    assert Counter((x["design_class"], x["split"]) for x in rows) == {
        ("stable", "train"): 24, ("stable", "validation"): 8, ("stable", "test"): 8,
        ("aggressive", "train"): 96, ("aggressive", "validation"): 32,
        ("aggressive", "test"): 32,
    }
    for block in {x["mirror_block_id"] for x in rows}:
        members = [x for x in rows if x["mirror_block_id"] == block]
        assert len(members) == 2 and len({x["split"] for x in members}) == 1


if __name__ == "__main__":
    print(json.dumps(build_catalog(), ensure_ascii=False, indent=2))
