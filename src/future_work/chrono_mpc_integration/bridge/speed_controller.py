"""Longitudinal-only speed tracking for physical Chrono replay.

This module owns no steering, path-following, MPC, or vehicle state mutation.
It maps a scalar speed error to mutually exclusive throttle/brake commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class LongitudinalCommand:
    """Normalized powertrain commands; throttle and brake are exclusive."""

    throttle: float
    braking: float
    speed_error_mps: float
    unsaturated_effort: float
    integral_error_mps_s: float

    def __post_init__(self) -> None:
        values = (
            self.throttle,
            self.braking,
            self.speed_error_mps,
            self.unsaturated_effort,
            self.integral_error_mps_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Longitudinal command values must be finite.")
        if not 0.0 <= self.throttle <= 1.0 or not 0.0 <= self.braking <= 1.0:
            raise ValueError("Throttle and braking must be in [0, 1].")
        if self.throttle > 0.0 and self.braking > 0.0:
            raise ValueError("Throttle and braking must be mutually exclusive.")


class LongitudinalSpeedPI:
    """PI speed tracker with conditional-integration anti-windup.

    Positive effort commands throttle; negative effort commands braking.  The
    controller cannot write pose, velocity, steering, or lateral state.
    """

    def __init__(
        self,
        *,
        kp: float = 0.25,
        ki: float = 0.08,
        integral_limit_mps_s: float = 5.0,
        deadband_mps: float = 0.05,
    ) -> None:
        if kp < 0.0 or ki < 0.0:
            raise ValueError("PI gains must be non-negative.")
        if integral_limit_mps_s <= 0.0 or deadband_mps < 0.0:
            raise ValueError("Integral limit must be positive and deadband non-negative.")
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_limit_mps_s = float(integral_limit_mps_s)
        self.deadband_mps = float(deadband_mps)
        self._integral_error_mps_s = 0.0

    @property
    def integral_error_mps_s(self) -> float:
        return self._integral_error_mps_s

    def reset(self) -> None:
        self._integral_error_mps_s = 0.0

    def update(self, *, vx_reference_mps: float, vx_measured_mps: float, dt_s: float) -> LongitudinalCommand:
        if not all(math.isfinite(value) for value in (vx_reference_mps, vx_measured_mps, dt_s)):
            raise ValueError("Speed tracker inputs must be finite.")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        error = float(vx_reference_mps - vx_measured_mps)
        proposed_integral = max(
            -self.integral_limit_mps_s,
            min(self.integral_limit_mps_s, self._integral_error_mps_s + error * dt_s),
        )
        proposed_effort = self.kp * error + self.ki * proposed_integral
        # Freeze integral only when saturation and error push in the same
        # direction.  Otherwise allow it to unwind.
        saturating_outward = (proposed_effort > 1.0 and error > 0.0) or (
            proposed_effort < -1.0 and error < 0.0
        )
        if not saturating_outward:
            self._integral_error_mps_s = proposed_integral
        effort = self.kp * error + self.ki * self._integral_error_mps_s
        if abs(error) <= self.deadband_mps and abs(effort) < 0.02:
            effort = 0.0
        throttle = min(1.0, max(0.0, effort))
        braking = min(1.0, max(0.0, -effort))
        return LongitudinalCommand(
            throttle=throttle,
            braking=braking,
            speed_error_mps=error,
            unsaturated_effort=effort,
            integral_error_mps_s=self._integral_error_mps_s,
        )
