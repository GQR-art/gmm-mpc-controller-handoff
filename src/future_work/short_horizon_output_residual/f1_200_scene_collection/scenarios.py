"""Smooth reference paths and speed profiles for the frozen 200-scene catalog."""
from __future__ import annotations

import numpy as np

from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import SpeedProfile
from future_work.short_horizon_output_residual.stanford_feature_driven_chrono_final.scenarios import CurvatureTemplatePath

G = 9.81
REFERENCE_DT = 0.02
MIN_SPEED_MPS = 2.0


def _smooth01(x):
    z = np.clip(np.asarray(x, float), 0.0, 1.0)
    return z*z*(3.0-2.0*z)


def _transition(t, start, duration, a, b):
    return a + (b-a)*_smooth01((t-start)/max(float(duration), 1e-6))


def _pulse(t, start, rise, hold, fall, amplitude):
    y = np.zeros_like(t)
    r1, h1, f1 = start+rise, start+rise+hold, start+rise+hold+fall
    m = (t >= start) & (t < r1); y[m] = _transition(t[m], start, rise, 0.0, amplitude)
    m = (t >= r1) & (t < h1); y[m] = amplitude
    m = (t >= h1) & (t <= f1); y[m] = _transition(t[m], h1, fall, amplitude, 0.0)
    return y


def _signed(direction: str) -> float:
    return 1.0 if direction in {"L", "LR"} else -1.0


def _duration(spec: dict) -> float:
    lead, recovery, typ = spec["stable_lead_s"], spec["recovery_s"], spec["scene_type"]
    if typ == "N1": action = 0.5 + 2.0 + 0.5
    elif typ in {"N2", "A1"}: action = 2*spec["curvature_rise_s"] + spec["curve_hold_s"]
    elif typ in {"N3", "A2"}: action = 2*spec["curvature_rise_s"] + spec["reversal_s"] + 1.0
    elif typ == "A3": action = 1.0 + 0.5 + spec["curvature_change_s"] + 0.75 + 1.0
    elif typ == "N4": action = 4.5 if spec["direction"] != "S" else 0.5+spec["brake_hold_s"]+0.5
    elif typ == "A4": action = max(4.0, spec["brake_hold_s"]+spec["brake_lead_s"]+2.0)
    elif typ == "A5": action = 1.0+1.0+spec["brake_hold_s"]+0.5+1.0+1.0
    elif typ == "A6": action = 1.0+0.5+spec["reversal_s"]+0.5+1.0
    elif typ == "A7": action = 0.25+spec["brake_hold_s"]+0.5
    else: raise ValueError(typ)
    return float(lead + action + recovery)


def _curvature(spec: dict, t: np.ndarray) -> np.ndarray:
    typ, start, v = spec["scene_type"], float(spec["stable_lead_s"]), float(spec["speed_mps"])
    sign = _signed(spec["direction"])
    if typ in {"N1", "A7"} or (typ == "N4" and spec["direction"] == "S"):
        return np.zeros_like(t)
    peak = float(spec.get("target_ay_g", 0.0))*G/(v*v)
    if typ in {"N2", "A1"}:
        return _pulse(t, start, spec["curvature_rise_s"], spec["curve_hold_s"],
                      spec["curvature_rise_s"], sign*peak)
    if typ in {"N3", "A2", "A6"}:
        rise, reversal = float(spec.get("curvature_rise_s", 1.0)), float(spec["reversal_s"])
        y = np.zeros_like(t); a = sign*peak
        marks = (start, start+rise, start+rise+0.5, start+rise+0.5+reversal,
                 start+rise+1.0+reversal, start+2*rise+1.0+reversal)
        m=(t>=marks[0])&(t<marks[1]); y[m]=_transition(t[m],marks[0],rise,0,a)
        m=(t>=marks[1])&(t<marks[2]); y[m]=a
        m=(t>=marks[2])&(t<marks[3]); y[m]=_transition(t[m],marks[2],reversal,a,-a)
        m=(t>=marks[3])&(t<marks[4]); y[m]=-a
        m=(t>=marks[4])&(t<=marks[5]); y[m]=_transition(t[m],marks[4],rise,-a,0)
        return y
    if typ == "A3":
        p0 = sign*spec["target_ay_start_g"]*G/(v*v)
        p1 = sign*spec["target_ay_end_g"]*G/(v*v)
        change = float(spec["curvature_change_s"]); y=np.zeros_like(t)
        marks=(start,start+1.0,start+1.5,start+1.5+change,start+2.25+change,start+3.25+change)
        m=(t>=marks[0])&(t<marks[1]); y[m]=_transition(t[m],marks[0],1,0,p0)
        m=(t>=marks[1])&(t<marks[2]); y[m]=p0
        m=(t>=marks[2])&(t<marks[3]); y[m]=_transition(t[m],marks[2],change,p0,p1)
        m=(t>=marks[3])&(t<marks[4]); y[m]=p1
        m=(t>=marks[4])&(t<=marks[5]); y[m]=_transition(t[m],marks[4],1,p1,0)
        return y
    if typ == "N4":
        return _pulse(t,start,1.5,1.5,1.5,sign*peak)
    if typ == "A4":
        return _pulse(t,start,1.0,2.0,1.0,sign*peak)
    if typ == "A5":
        hold=1.0+spec["brake_hold_s"]+0.5+1.0
        return _pulse(t,start,1.0,hold,1.0,sign*peak)
    raise ValueError(typ)


def _acceleration(spec: dict, t: np.ndarray) -> np.ndarray:
    typ, start = spec["scene_type"], float(spec["stable_lead_s"])
    ax = float(spec.get("target_ax_g", 0.0))*G
    if abs(ax) < 1e-12:
        return np.zeros_like(t)
    if typ == "N1":
        return _pulse(t,start,0.5,spec["accel_hold_s"],0.5,ax)
    if typ in {"N4", "A7"}:
        return _pulse(t,start,0.25,spec["brake_hold_s"],0.5,ax)
    if typ == "A4":
        brake_start = start-float(spec["brake_lead_s"])
        return _pulse(t,brake_start,0.25,spec["brake_hold_s"],0.5,ax)
    if typ == "A5":
        return _pulse(t,start+2.0,0.25,spec["brake_hold_s"],0.5,ax)
    if typ == "A6":
        brake_start = start if spec["brake_phase"] == "first_turn" else start+1.5
        return _pulse(t,brake_start,0.25,spec["brake_hold_s"],0.5,ax)
    return np.zeros_like(t)


def build_scene(spec: dict):
    total = _duration(spec)
    t = np.arange(0.0, total+REFERENCE_DT/2, REFERENCE_DT)
    ax = _acceleration(spec, t)
    speed = np.empty_like(t); speed[0] = float(spec["speed_mps"])
    for i in range(1, len(t)):
        speed[i] = max(MIN_SPEED_MPS, speed[i-1] + 0.5*(ax[i-1]+ax[i])*REFERENCE_DT)
    station = np.zeros_like(t)
    station[1:] = np.cumsum(0.5*(speed[:-1]+speed[1:])*REFERENCE_DT)
    curvature = _curvature(spec, t)
    # CurvatureTemplatePath and SpeedProfile both interpolate these frozen,
    # continuous reference arrays; no realized vehicle state alters them.
    path = CurvatureTemplatePath(station, curvature)
    profile = SpeedProfile(station, speed, spec["scenario_id"])
    lead_index = int(round(float(spec["stable_lead_s"])/REFERENCE_DT))
    formal_start = float(station[lead_index])
    formal_end = float(station[-1])
    metadata = {
        "total_reference_time_s": total,
        "formal_start_station_m": formal_start,
        "formal_end_station_m": formal_end,
        "path_length_m": float(path.length_m),
        "minimum_reference_speed_mps": float(speed.min()),
        "maximum_reference_speed_mps": float(speed.max()),
        "peak_abs_curvature_per_m": float(np.max(np.abs(curvature))),
        "reference_samples": int(len(t)),
        "reference_dt_s": REFERENCE_DT,
        "curvature_construction_speed_mps": float(spec["speed_mps"]),
    }
    return path, profile, formal_start, formal_end, metadata
