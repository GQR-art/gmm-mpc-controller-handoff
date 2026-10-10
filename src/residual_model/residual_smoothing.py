"""Temporal smoothing for GMM-GMR predicted residuals (inference only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import signal


class ResidualSmoothingMethod(str, Enum):
    """Smoothing method applied to ``residual_pred`` only."""

    NONE = "none"
    SAVGOL = "savgol"
    MOVING_AVERAGE = "moving_average"


@dataclass(frozen=True)
class ResidualSmoothingConfig:
    """Configuration for temporal residual smoothing."""

    method: ResidualSmoothingMethod = ResidualSmoothingMethod.NONE
    window_length: int = 9
    polyorder: int = 3
    window_size: int = 5

    def label(self) -> str:
        """Short human-readable label for tables and plots."""
        if self.method == ResidualSmoothingMethod.NONE:
            return "raw"
        if self.method == ResidualSmoothingMethod.SAVGOL:
            return f"savgol_w{self.window_length}_p{self.polyorder}"
        return f"ma_w{self.window_size}"

    def to_dict(self) -> dict[str, str | int]:
        """Serialize for JSON export."""
        return {
            "method": self.method.value,
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "window_size": self.window_size,
        }


def _ensure_odd_window(window: int, n: int, polyorder: int) -> int:
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, polyorder + 2)
    if n < w:
        max_odd = n if n % 2 == 1 else max(n - 1, 1)
        return max(1, max_odd)
    return min(w, n if n % 2 == 1 else n - 1)


def smooth_residual_temporal(
    residual_pred: np.ndarray,
    config: ResidualSmoothingConfig,
) -> np.ndarray:
    """Smooth a 1-D predicted residual time series (does not modify inputs).

    Parameters
    ----------
    residual_pred
        Raw GMR residual predictions, time-ordered.
    config
        Smoothing method and parameters.

    Returns
    -------
    numpy.ndarray
        Smoothed residual, same shape as ``residual_pred``.
    """
    x = np.asarray(residual_pred, dtype=float).copy()
    n = len(x)
    if n < 2 or config.method == ResidualSmoothingMethod.NONE:
        return x

    if config.method == ResidualSmoothingMethod.MOVING_AVERAGE:
        w = max(1, int(config.window_size))
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(x, kernel, mode="same")

    if config.method == ResidualSmoothingMethod.SAVGOL:
        w = _ensure_odd_window(config.window_length, n, config.polyorder)
        if n < w:
            return x
        return signal.savgol_filter(x, w, int(config.polyorder), mode="interp")

    raise ValueError(f"Unknown smoothing method: {config.method!r}")


from residual_model.residual_gating import (
    DEFAULT_RESIDUAL_GATING_CONFIG,
    ResidualGatingConfig,
    apply_gated_correction,
)


def apply_residual_smoothing(
    residual_pred_raw: np.ndarray,
    ay_bike: np.ndarray,
    config: ResidualSmoothingConfig,
    *,
    gating: ResidualGatingConfig | None = DEFAULT_RESIDUAL_GATING_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(residual_smoothed, ay_corrected)`` with optional residual gating."""
    residual_smooth = smooth_residual_temporal(residual_pred_raw, config)
    ay_corrected, _, _ = apply_gated_correction(ay_bike, residual_smooth, gating)
    return residual_smooth, ay_corrected
