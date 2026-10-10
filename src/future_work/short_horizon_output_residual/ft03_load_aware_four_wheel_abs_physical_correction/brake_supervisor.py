"""Symmetric load-aware brake allocation and per-wheel ABS supervision."""
from __future__ import annotations

from dataclasses import dataclass
import math


WHEELS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True, slots=True)
class WheelBrakeState:
    wheel_id: str
    normal_load_N: float
    lateral_force_N: float
    rolling_radius_m: float
    kinematic_longitudinal_slip: float
    contact: bool


@dataclass(frozen=True, slots=True)
class WheelBrakeCommand:
    wheel_id: str
    normalized: float
    torque_Nm: float
    capacity_Nm: float
    base_torque_Nm: float
    abs_multiplier: float
    braking_slip: float


class FourWheelBrakeSupervisor:
    """Allocate requested total brake torque by load and prevent wheel lock.

    No wheel or turn direction is special-cased.  The input brake request has
    the same total nominal authority as four existing 2000 Nm BrakeSimple
    units.  Capacity uses the frozen TMeasy load curve with a conservative
    friction-ellipse reserve for simultaneous lateral force.
    """

    def __init__(
        self,
        *,
        mode: str,
        dt_s: float,
        terrain_mu: float = 1.0,
        tire_reference_mu: float = 0.8,
        max_wheel_torque_Nm: float = 2000.0,
        nominal_load_N: float = 3580.65,
        maximum_load_N: float = 12532.275,
        fx_peak_load_N: tuple[float, float] = (4043.26998, 7802.95248),
        fy_peak_load_N: tuple[float, float] = (3583.872585, 6543.064971),
        capacity_margin: float = 0.85,
        abs_release_slip: float = 0.20,
        abs_reapply_slip: float = 0.10,
        abs_release_rate_per_s: float = 8.0,
        abs_reapply_rate_per_s: float = 2.0,
        abs_min_speed_mps: float = 3.0,
    ) -> None:
        if mode not in {"load_aware", "load_aware_abs"}:
            raise ValueError(mode)
        self.mode = mode
        self.dt_s = float(dt_s)
        self.terrain_mu = float(terrain_mu)
        self.tire_reference_mu = float(tire_reference_mu)
        self.max_wheel_torque_Nm = float(max_wheel_torque_Nm)
        self.nominal_load_N = float(nominal_load_N)
        self.maximum_load_N = float(maximum_load_N)
        self.fx_peak_load_N = tuple(map(float, fx_peak_load_N))
        self.fy_peak_load_N = tuple(map(float, fy_peak_load_N))
        self.capacity_margin = float(capacity_margin)
        self.abs_release_slip = float(abs_release_slip)
        self.abs_reapply_slip = float(abs_reapply_slip)
        self.abs_release_rate_per_s = float(abs_release_rate_per_s)
        self.abs_reapply_rate_per_s = float(abs_reapply_rate_per_s)
        self.abs_min_speed_mps = float(abs_min_speed_mps)
        self._multipliers = {wheel: 1.0 for wheel in WHEELS}

    def reset(self) -> None:
        self._multipliers = {wheel: 1.0 for wheel in WHEELS}

    def _interp_q(self, load_N: float, values: tuple[float, float]) -> float:
        q = max(0.0, min(load_N, self.maximum_load_N)) / self.nominal_load_N
        w1, w2 = values
        return q * (2.0 * w1 - 0.5 * w2 - (w1 - 0.5 * w2) * q)

    def _capacity(self, state: WheelBrakeState) -> float:
        if not state.contact or state.normal_load_N <= 100.0:
            return 0.0
        muscale = self.terrain_mu / self.tire_reference_mu
        fx_peak = muscale * self._interp_q(state.normal_load_N, self.fx_peak_load_N)
        fy_peak = muscale * self._interp_q(state.normal_load_N, self.fy_peak_load_N)
        lateral_ratio = min(1.0, abs(state.lateral_force_N) / max(fy_peak, 1e-9))
        combined_reserve = math.sqrt(max(0.0, 1.0 - lateral_ratio * lateral_ratio))
        return min(
            self.max_wheel_torque_Nm,
            self.capacity_margin * state.rolling_radius_m * fx_peak * combined_reserve,
        )

    @staticmethod
    def _allocate(total_Nm: float, loads: dict[str, float], capacities: dict[str, float]) -> dict[str, float]:
        allocation = {wheel: 0.0 for wheel in WHEELS}
        active = {wheel for wheel in WHEELS if capacities[wheel] > 0.0 and loads[wheel] > 0.0}
        remaining = max(0.0, float(total_Nm))
        for _ in range(len(WHEELS)):
            if not active or remaining <= 1e-9:
                break
            weight_sum = sum(loads[wheel] for wheel in active)
            if weight_sum <= 0.0:
                break
            proposed = {wheel: remaining * loads[wheel] / weight_sum for wheel in active}
            saturated = {
                wheel for wheel in active
                if allocation[wheel] + proposed[wheel] >= capacities[wheel] - 1e-9
            }
            if not saturated:
                for wheel in active:
                    allocation[wheel] += proposed[wheel]
                remaining = 0.0
                break
            for wheel in saturated:
                add = max(0.0, capacities[wheel] - allocation[wheel])
                allocation[wheel] += add
                remaining -= add
            active -= saturated
        return allocation

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
        commands = []
        for state in states:
            wheel = state.wheel_id
            braking_slip = max(0.0, -float(state.kinematic_longitudinal_slip))
            multiplier = self._multipliers[wheel]
            if request <= 0.0 or abs(vehicle_speed_mps) < self.abs_min_speed_mps or not state.contact:
                multiplier = 1.0
            elif self.mode == "load_aware_abs":
                if braking_slip > self.abs_release_slip:
                    multiplier = max(0.0, multiplier - self.abs_release_rate_per_s * self.dt_s)
                elif braking_slip < self.abs_reapply_slip:
                    multiplier = min(1.0, multiplier + self.abs_reapply_rate_per_s * self.dt_s)
            else:
                multiplier = 1.0
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
