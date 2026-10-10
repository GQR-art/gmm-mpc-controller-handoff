"""Pre-registered FT03 station-based speed profiles."""

from __future__ import annotations
import numpy as np
import pandas as pd

from .variable_speed_mpc import SpeedProfile

G = 9.81
AY_ALLOW_MPS2 = 0.8 * G
DIAGNOSTIC_ACCEL_MPS2 = 1.5
DIAGNOSTIC_DECEL_MPS2 = 2.5


def source_stations(frame: pd.DataFrame) -> np.ndarray:
    e, n = frame.posE_m.to_numpy(float), frame.posN_m.to_numpy(float)
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(e), np.diff(n)))]


def constant_profile(frame: pd.DataFrame, speed=15.0) -> SpeedProfile:
    s = source_stations(frame)
    return SpeedProfile(s, np.full_like(s, float(speed)), "fixed_15")


def stanford_profile(frame: pd.DataFrame) -> SpeedProfile:
    s = source_stations(frame)
    return SpeedProfile(s, np.maximum(frame.Vx_mps.to_numpy(float), 1.0), "stanford_original_station")


def curvature_limited_profile(frame: pd.DataFrame) -> SpeedProfile:
    """Frozen 0.8g cap with conservative, explicitly assumed accel/decel bounds."""
    s = source_stations(frame)
    vx = np.maximum(frame.Vx_mps.to_numpy(float), 1.0)
    ground = np.maximum(np.hypot(frame.Ve_mps.to_numpy(float), frame.Vn_mps.to_numpy(float)), 1.0)
    kappa = frame.yawRate_radps.to_numpy(float) / ground
    cap = np.sqrt(AY_ALLOW_MPS2 / np.maximum(np.abs(kappa), 1e-6))
    plan = np.minimum(vx, cap)
    # Forward acceleration and backward braking feasibility passes in v^2 space.
    for i in range(1, len(plan)):
        ds = max(s[i] - s[i-1], 0.0)
        plan[i] = min(plan[i], np.sqrt(max(0.0, plan[i-1]**2 + 2*DIAGNOSTIC_ACCEL_MPS2*ds)))
    for i in range(len(plan)-2, -1, -1):
        ds = max(s[i+1] - s[i], 0.0)
        plan[i] = min(plan[i], np.sqrt(max(0.0, plan[i+1]**2 + 2*DIAGNOSTIC_DECEL_MPS2*ds)))
    return SpeedProfile(s, np.maximum(plan, 1.0), "curvature_limited_0p8g")

