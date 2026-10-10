"""One-metre offset recovery using the frozen straight reference path."""

from __future__ import annotations

from future_work.later_mpc.reference_path import StraightPath

from . import ChronoMPCScenario


def make_offset_recovery_scenario() -> ChronoMPCScenario:
    """Return the frozen Phase 3.3 initial condition and path object."""
    return ChronoMPCScenario(
        initial_x_m=0.0,
        initial_y_m=1.0,
        initial_psi_rad=0.0,
        initial_speed_mps=15.0,
        duration_s=6.0,
        reference_path=StraightPath(length_m=300.0),
    )
