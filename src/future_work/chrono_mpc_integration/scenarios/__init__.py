"""Minimal scenario metadata around frozen ``later_mpc`` reference paths."""

from __future__ import annotations

from dataclasses import dataclass

from future_work.later_mpc.reference_path import ReferencePath


@dataclass(frozen=True, slots=True)
class ChronoMPCScenario:
    initial_x_m: float
    initial_y_m: float
    initial_psi_rad: float
    initial_speed_mps: float
    duration_s: float
    reference_path: ReferencePath
