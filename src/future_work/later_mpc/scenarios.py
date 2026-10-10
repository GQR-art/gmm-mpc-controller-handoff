"""Baseline nominal-MPC validation scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from .reference_path import CircularPath, DoubleLaneChangePath, ReferencePath, StraightPath
from .types import VehicleState


@dataclass(frozen=True)
class NominalMPCScenario:
    """All inputs needed for one closed-loop nominal-MPC experiment."""

    name: str
    reference_path: ReferencePath
    initial_state: VehicleState
    duration_s: float
    description: str


def straight_offset_recovery() -> NominalMPCScenario:
    return NominalMPCScenario(
        name="s1_straight_offset_recovery",
        reference_path=StraightPath(length_m=300.0),
        initial_state=VehicleState(0.0, 1.0, 0.0, 0.0, 0.0),
        duration_s=6.0,
        description="Straight path, one-metre left lateral-offset recovery.",
    )


def constant_radius_turn() -> NominalMPCScenario:
    return NominalMPCScenario(
        name="s2_constant_radius_turn",
        reference_path=CircularPath(radius_m=80.0, arc_length_m=180.0),
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0, 0.0),
        duration_s=8.0,
        description="Left-hand constant-curvature turn with radius 80 m.",
    )


def double_lane_change() -> NominalMPCScenario:
    return NominalMPCScenario(
        name="s3_double_lane_change",
        reference_path=DoubleLaneChangePath(manoeuvre_length_m=90.0, heading_peak_rad=0.12),
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0, 0.0),
        duration_s=5.5,
        description="Smooth curvature-continuous S-shaped double lane change.",
    )


def baseline_scenarios() -> tuple[NominalMPCScenario, ...]:
    return straight_offset_recovery(), constant_radius_turn(), double_lane_change()
