"""Plant-independent interfaces and data structures for MPC prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContinuousAffineModel:
    """Continuous affine model ``xdot = Ac x + Bc u + gc``."""

    Ac: np.ndarray
    Bc: np.ndarray
    gc: np.ndarray


@dataclass(frozen=True)
class DiscreteAffineModel:
    """ZOH-discrete affine model ``x+ = Ad x + Bd u + gd``."""

    Ad: np.ndarray
    Bd: np.ndarray
    gd: np.ndarray


class PredictionModel(ABC):
    """MPC-internal future-state model, intentionally separate from the plant.

    A later GMR-augmented predictor must implement this same contract without
    requiring changes to the optimizer, controller, simulator, or metrics.
    """

    state_dimension: int = 4
    input_dimension: int = 1

    @abstractmethod
    def linearize(
        self,
        state: np.ndarray,
        vx_mps: float,
        reference_curvature_per_m: float,
    ) -> ContinuousAffineModel:
        """Return the current continuous affine prediction model."""

    @abstractmethod
    def discretize(
        self,
        continuous_model: ContinuousAffineModel,
        sample_time_s: float,
    ) -> DiscreteAffineModel:
        """Return the zero-order-hold discrete prediction model."""

    @abstractmethod
    def predict(
        self,
        state: np.ndarray,
        control_sequence_rad: Sequence[float] | np.ndarray,
        reference_curvature_preview_per_m: Sequence[float] | np.ndarray,
        vx_mps: float,
        sample_time_s: float,
    ) -> np.ndarray:
        """Return predicted states from step zero through the horizon end."""
