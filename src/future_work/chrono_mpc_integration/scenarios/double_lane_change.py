"""Frozen Double Lane Change scenario adapter."""

from __future__ import annotations

from future_work.later_mpc.scenarios import double_lane_change

from . import ChronoMPCScenario


def make_double_lane_change_scenario() -> ChronoMPCScenario:
    """Reuse the frozen later_mpc path object and experiment duration verbatim."""
    frozen = double_lane_change()
    return ChronoMPCScenario(
        initial_x_m=frozen.initial_state.X_m,
        initial_y_m=frozen.initial_state.Y_m,
        initial_psi_rad=frozen.initial_state.psi_rad,
        initial_speed_mps=15.0,
        duration_s=frozen.duration_s,
        reference_path=frozen.reference_path,
    )
