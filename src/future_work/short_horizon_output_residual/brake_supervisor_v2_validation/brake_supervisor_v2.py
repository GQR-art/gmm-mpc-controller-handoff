"""General four-wheel brake supervisor v2 with continuous low-speed recovery."""
from __future__ import annotations

from future_work.short_horizon_output_residual.ft03_load_aware_four_wheel_abs_physical_correction.brake_supervisor import (
    FourWheelBrakeSupervisor,
    WHEELS,
    WheelBrakeCommand,
    WheelBrakeState,
)


class FourWheelBrakeSupervisorV2(FourWheelBrakeSupervisor):
    """Preserve v1 allocation while making ABS release and low-speed exit continuous.

    The rule is identical for every wheel. Above the ABS minimum speed, a
    threshold crossing commands a 25 ms full-scale pressure-release slew.
    Below that speed, slip feedback is disabled but the current multiplier is
    retained and smoothly re-applied; it is never reset discontinuously.
    """

    def __init__(
        self,
        *,
        dt_s: float,
        terrain_mu: float = 1.0,
        abs_release_rate_per_s: float = 40.0,
        low_speed_reapply_rate_per_s: float = 0.5,
    ) -> None:
        super().__init__(
            mode="load_aware_abs",
            dt_s=dt_s,
            terrain_mu=terrain_mu,
            abs_release_rate_per_s=abs_release_rate_per_s,
        )
        self.low_speed_reapply_rate_per_s = float(low_speed_reapply_rate_per_s)

    def update(
        self,
        requested_braking: float,
        vehicle_speed_mps: float,
        states: tuple[WheelBrakeState, ...],
    ) -> tuple[WheelBrakeCommand, ...]:
        if tuple(state.wheel_id for state in states) != WHEELS:
            raise ValueError("Wheel states must be ordered FL, FR, RL, RR")
        request = min(1.0, max(0.0, float(requested_braking)))
        capacities = {state.wheel_id: self._capacity(state) for state in states}
        loads = {state.wheel_id: max(0.0, state.normal_load_N) for state in states}
        base = self._allocate(4.0 * self.max_wheel_torque_Nm * request, loads, capacities)
        commands: list[WheelBrakeCommand] = []
        for state in states:
            wheel = state.wheel_id
            braking_slip = max(0.0, -float(state.kinematic_longitudinal_slip))
            multiplier = self._multipliers[wheel]
            if request <= 0.0:
                multiplier = min(1.0, multiplier + self.abs_reapply_rate_per_s * self.dt_s)
            elif not state.contact:
                multiplier = max(0.0, multiplier - self.abs_release_rate_per_s * self.dt_s)
            elif abs(vehicle_speed_mps) >= self.abs_min_speed_mps:
                if braking_slip > self.abs_release_slip:
                    multiplier = max(0.0, multiplier - self.abs_release_rate_per_s * self.dt_s)
                elif braking_slip < self.abs_reapply_slip:
                    multiplier = min(1.0, multiplier + self.abs_reapply_rate_per_s * self.dt_s)
            else:
                multiplier = min(1.0, multiplier + self.low_speed_reapply_rate_per_s * self.dt_s)
            self._multipliers[wheel] = multiplier
            torque = min(capacities[wheel], base[wheel] * multiplier)
            commands.append(WheelBrakeCommand(
                wheel_id=wheel,
                normalized=min(1.0, max(0.0, torque / self.max_wheel_torque_Nm)),
                torque_Nm=torque,
                capacity_Nm=capacities[wheel],
                base_torque_Nm=base[wheel],
                abs_multiplier=multiplier,
                braking_slip=braking_slip,
            ))
        return tuple(commands)
