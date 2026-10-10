"""Nominal linear bicycle prediction model for MPC internal rollouts.

This module has no dependency on ``DynamicBicyclePlant`` and never invokes the
paper model's simulation routine.  It derives the continuous linear model from
the same fitted physical parameters used by the baseline plant.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import expm

from .config import VehicleParameters, load_fitted_vehicle_parameters
from .prediction_model import ContinuousAffineModel, DiscreteAffineModel, PredictionModel
from .types import VehicleErrorState


class NominalBicyclePredictor(PredictionModel):
    """Four-state nominal bicycle predictor, ``x=[ey, epsi, vy, r]``."""

    def __init__(self, vehicle_parameters: VehicleParameters | None = None) -> None:
        self._parameters = vehicle_parameters or load_fitted_vehicle_parameters()

    @staticmethod
    def _state_vector(state: np.ndarray | Sequence[float] | VehicleErrorState) -> np.ndarray:
        if isinstance(state, VehicleErrorState):
            vector = state.as_vector()
        else:
            vector = np.asarray(state, dtype=float)
        if vector.shape != (4,):
            raise ValueError("Prediction state must have shape (4,) for [ey, epsi, vy, r].")
        return vector

    def linearize(
        self,
        state: np.ndarray | Sequence[float] | VehicleErrorState,
        vx_mps: float,
        reference_curvature_per_m: float,
    ) -> ContinuousAffineModel:
        """Build the fixed-speed linear bicycle error model at one path curvature.

        The present nominal model is linear time invariant for fixed ``vx``;
        ``state`` is validated to preserve the common interface needed by future
        state-dependent GMR predictors.
        """
        self._state_vector(state)
        if vx_mps <= 0.0:
            raise ValueError("vx_mps must be positive.")
        p = self._parameters
        m, Iz, a, b, Cf, Cr = p.m, p.Iz, p.a, p.b, p.Cf, p.Cr
        v = float(vx_mps)

        Ac = np.array(
            [
                [0.0, v, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -(Cf + Cr) / (m * v), (b * Cr - a * Cf) / (m * v) - v],
                [0.0, 0.0, (b * Cr - a * Cf) / (Iz * v), -(a * a * Cf + b * b * Cr) / (Iz * v)],
            ],
            dtype=float,
        )
        # The positive sign follows the paper equation a*Fyf - b*Fyr, where
        # Fyf contains +Cf*delta.  It is required for plant consistency.
        Bc = np.array([[0.0], [0.0], [Cf / m], [a * Cf / Iz]], dtype=float)
        gc = np.array([0.0, -v * float(reference_curvature_per_m), 0.0, 0.0], dtype=float)
        return ContinuousAffineModel(Ac=Ac, Bc=Bc, gc=gc)

    def discretize(
        self,
        continuous_model: ContinuousAffineModel,
        sample_time_s: float,
    ) -> DiscreteAffineModel:
        """Use one augmented matrix exponential for exact ZOH discretization."""
        if sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be positive.")
        Ac, Bc, gc = continuous_model.Ac, continuous_model.Bc, continuous_model.gc
        if Ac.shape != (4, 4) or Bc.shape != (4, 1) or gc.shape != (4,):
            raise ValueError("Expected Ac(4,4), Bc(4,1), and gc(4,).")
        augmented = np.zeros((6, 6), dtype=float)
        augmented[:4, :4] = Ac
        augmented[:4, 4:5] = Bc
        augmented[:4, 5] = gc
        transition = expm(augmented * float(sample_time_s))
        return DiscreteAffineModel(
            Ad=transition[:4, :4],
            Bd=transition[:4, 4:5],
            gd=transition[:4, 5],
        )

    def predict(
        self,
        state: np.ndarray | Sequence[float] | VehicleErrorState,
        control_sequence_rad: Sequence[float] | np.ndarray,
        reference_curvature_preview_per_m: Sequence[float] | np.ndarray,
        vx_mps: float,
        sample_time_s: float,
    ) -> np.ndarray:
        """Roll out an MPC-internal state sequence using per-step curvature preview."""
        initial_state = self._state_vector(state)
        controls = np.asarray(control_sequence_rad, dtype=float).reshape(-1)
        curvatures = np.asarray(reference_curvature_preview_per_m, dtype=float).reshape(-1)
        if controls.size < 1:
            raise ValueError("control_sequence_rad must contain at least one input.")
        if controls.size != curvatures.size:
            raise ValueError("Control and curvature preview lengths must match.")

        states = np.empty((controls.size + 1, 4), dtype=float)
        states[0] = initial_state
        for step_index, (delta_rad, curvature) in enumerate(zip(controls, curvatures)):
            continuous = self.linearize(states[step_index], vx_mps, float(curvature))
            discrete = self.discretize(continuous, sample_time_s)
            states[step_index + 1] = (
                discrete.Ad @ states[step_index] + discrete.Bd[:, 0] * delta_rad + discrete.gd
            )
        return states
