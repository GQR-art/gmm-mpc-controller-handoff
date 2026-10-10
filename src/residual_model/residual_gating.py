"""Regime-dependent gating for residual compensation (inference only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualGatingConfig:
    """Smooth weighting of predicted residual by nominal lateral-acceleration magnitude."""

    enabled: bool = True
    ay_threshold_mps2: float = 2.75
    k: float = 2.5
    use_sigmoid: bool = True

    @classmethod
    def disabled(cls) -> ResidualGatingConfig:
        """Return a config that applies full residual (legacy ungated behavior)."""
        return cls(enabled=False)


DEFAULT_RESIDUAL_GATING_CONFIG = ResidualGatingConfig()


def compute_residual_gate_weight(
    ay_bike: np.ndarray,
    config: ResidualGatingConfig | None = None,
) -> np.ndarray:
    """
    Compute smooth gating weight ``w`` in ``[0, 1]`` from ``|ay_bike|``.

    Sigmoid (default): ``w = 1 / (1 + exp(-k * (|ay_bike| - ay_threshold)))``.
    Linear clip: ``w = clip(|ay_bike| / ay_threshold, 0, 1)``.
    """
    cfg = config if config is not None else DEFAULT_RESIDUAL_GATING_CONFIG
    ay_bike = np.asarray(ay_bike, dtype=float)
    if not cfg.enabled:
        return np.ones_like(ay_bike, dtype=float)

    mag = np.abs(ay_bike)
    if cfg.use_sigmoid:
        z = cfg.k * (mag - cfg.ay_threshold_mps2)
        return 1.0 / (1.0 + np.exp(-z))

    threshold = max(float(cfg.ay_threshold_mps2), 1e-9)
    return np.clip(mag / threshold, 0.0, 1.0)


def gate_residual_prediction(
    ay_bike: np.ndarray,
    residual_pred: np.ndarray,
    config: ResidualGatingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply gating: ``residual_gated = w * residual_pred``.

    Returns
    -------
    residual_gated
        Gate-scaled residual used for ``ay_corrected = ay_bike + residual_gated``.
    gate_weight
        Per-sample weights ``w`` in ``[0, 1]``.
    """
    ay_bike = np.asarray(ay_bike, dtype=float)
    residual_pred = np.asarray(residual_pred, dtype=float).reshape(-1)
    if len(residual_pred) != len(ay_bike):
        raise ValueError("residual_pred length must match ay_bike.")

    w = compute_residual_gate_weight(ay_bike, config)
    return w * residual_pred, w


def apply_gated_correction(
    ay_bike: np.ndarray,
    residual_pred: np.ndarray,
    config: ResidualGatingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gated hybrid lateral acceleration.

    Returns ``(ay_corrected, residual_gated, gate_weight)``.
    """
    residual_gated, w = gate_residual_prediction(ay_bike, residual_pred, config)
    ay_corrected = np.asarray(ay_bike, dtype=float) + residual_gated
    return ay_corrected, residual_gated, w
