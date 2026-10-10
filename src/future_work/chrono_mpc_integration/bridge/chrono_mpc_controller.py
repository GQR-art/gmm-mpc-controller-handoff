"""Thin adapter from Chrono measurements to the frozen Nominal MPC."""

from __future__ import annotations

import math

from future_work.later_mpc.controller import ControllerOutput, NominalMPCController
from future_work.later_mpc.reference_path import ReferencePath

from .state_mapper import chrono_raw_to_vehicle_state
from .types import ChronoRawState


class NominalMPCBridge:
    """Own only the controller-boundary state and previous road-wheel angle.

    ``previous_delta_rad`` is always an MPC road-wheel angle in radians.  A
    Chrono normalized steering command never enters this adapter.
    """

    def __init__(
        self,
        *,
        reference_path: ReferencePath,
        controller: NominalMPCController | None = None,
        initial_previous_delta_rad: float = 0.0,
    ) -> None:
        if not isinstance(reference_path, ReferencePath):
            raise TypeError("reference_path must be a later_mpc ReferencePath.")
        if not math.isfinite(initial_previous_delta_rad):
            raise ValueError("initial_previous_delta_rad must be finite.")
        self.reference_path = reference_path
        self.controller = controller or NominalMPCController()
        self._previous_delta_rad = float(initial_previous_delta_rad)

    @property
    def previous_delta_rad(self) -> float:
        """Road-wheel command returned by the preceding MPC update."""
        return self._previous_delta_rad

    def reset(self, previous_delta_rad: float = 0.0) -> None:
        """Reset only the adapter's receding-horizon input history."""
        if not math.isfinite(previous_delta_rad):
            raise ValueError("previous_delta_rad must be finite.")
        self._previous_delta_rad = float(previous_delta_rad)

    def compute_control(self, raw_state: ChronoRawState) -> ControllerOutput:
        """Map one Chrono state and invoke the frozen controller exactly once."""
        vehicle_state = chrono_raw_to_vehicle_state(raw_state)
        output = self.controller.compute_control(
            vehicle_state=vehicle_state,
            reference_path=self.reference_path,
            previous_delta=self._previous_delta_rad,
        )
        if not math.isfinite(output.delta_rad):
            raise RuntimeError("Nominal MPC returned a non-finite road-wheel angle.")
        self._previous_delta_rad = float(output.delta_rad)
        return output
