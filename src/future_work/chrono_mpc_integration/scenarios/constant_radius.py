"""Constant-radius left turn using the frozen circular path implementation."""

from __future__ import annotations

from future_work.later_mpc.reference_path import CircularPath

from . import ChronoMPCScenario


RADIUS_M = 80.0


def make_constant_radius_left_scenario() -> ChronoMPCScenario:
    """Return the frozen Phase 3.4 circular-path scenario."""
    return ChronoMPCScenario(
        initial_x_m=0.0,
        initial_y_m=0.0,
        initial_psi_rad=0.0,
        initial_speed_mps=15.0,
        duration_s=8.0,
        reference_path=CircularPath(radius_m=RADIUS_M, arc_length_m=160.0),
    )
