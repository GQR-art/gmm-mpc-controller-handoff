"""Steering-angle rate feature engineering for the residual GMM/GMR pipeline.

Computes ``steering_rate = d(steering_angle)/dt`` from timestamp-ordered samples,
with optional pre-differentiation smoothing (moving average or Savitzky–Golay).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from residual_model.config import (
    DEFAULT_STEERING_SAVGOL_POLYORDER,
    DEFAULT_STEERING_SAVGOL_WINDOW,
)

STEERING_ANGLE_COLUMN: str = "steering_angle"
STEERING_RATE_COLUMN: str = "steering_rate"
TIMESTAMP_COLUMN: str = "timestamp"

MIN_DT_S: float = 1e-9


class SmoothingMethod(str, Enum):
    """Pre-differentiation smoothing method for steering angle."""

    NONE = "none"
    MOVING_AVERAGE = "moving_average"
    SAVGOL = "savgol"


@dataclass(frozen=True)
class SteeringRateConfig:
    """Configuration for steering-rate computation and optional smoothing."""

    smoothing: SmoothingMethod = SmoothingMethod.SAVGOL
    smoothing_window: int = DEFAULT_STEERING_SAVGOL_WINDOW
    savgol_polyorder: int = DEFAULT_STEERING_SAVGOL_POLYORDER
    first_sample_mode: Literal["copy_second", "zero"] = "copy_second"

    def __post_init__(self) -> None:
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1.")
        if self.savgol_polyorder < 0:
            raise ValueError("savgol_polyorder must be >= 0.")


def _ensure_odd_window(window: int, n: int, polyorder: int) -> int:
    """Return an odd window length in ``[polyorder+2, n]`` (or ``n`` if smaller)."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, polyorder + 2)
    if n < w:
        max_odd = n if n % 2 == 1 else max(n - 1, 1)
        return max(1, max_odd)
    return min(w, n if n % 2 == 1 else n - 1)


def smooth_steering_angle(
    steering_angle: np.ndarray,
    *,
    method: SmoothingMethod,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Apply optional smoothing to steering angle before differentiation.

    Parameters
    ----------
    steering_angle
        Steering angle time series (rad).
    method
        ``NONE``, ``MOVING_AVERAGE``, or ``SAVGOL``.
    window
        Window length (odd for Savitzky–Golay; moving average uses this width).
    polyorder
        Polynomial order for Savitzky–Golay.

    Returns
    -------
    numpy.ndarray
        Smoothed steering angle (or copy if ``method`` is ``NONE``).
    """
    x = np.asarray(steering_angle, dtype=float).copy()
    n = len(x)
    if n < 2 or method == SmoothingMethod.NONE:
        return x

    if method == SmoothingMethod.MOVING_AVERAGE:
        w = max(1, int(window))
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(x, kernel, mode="same")

    if method == SmoothingMethod.SAVGOL:
        w = _ensure_odd_window(window, n, polyorder)
        if n < w:
            return x
        return signal.savgol_filter(x, w, polyorder, mode="interp")

    raise ValueError(f"Unknown smoothing method: {method!r}")


def compute_steering_rate(
    timestamp: np.ndarray,
    steering_angle: np.ndarray,
    *,
    first_sample_mode: Literal["copy_second", "zero"] = "copy_second",
) -> np.ndarray:
    """Compute steering rate from non-uniform timestamps.

    Uses forward differences: ``rate[i] = (delta[i-1]) / dt[i-1]`` for ``i >= 1``.
    The first sample is filled per ``first_sample_mode``.

    Parameters
    ----------
    timestamp
        Monotonically increasing time vector (s).
    steering_angle
        Steering angle (rad), same length as ``timestamp``.
    first_sample_mode
        ``copy_second`` sets ``rate[0] = rate[1]``; ``zero`` sets ``rate[0] = 0``.

    Returns
    -------
    numpy.ndarray
        Steering rate (rad/s), same length as inputs.
    """
    t = np.asarray(timestamp, dtype=float)
    delta = np.asarray(steering_angle, dtype=float)
    n = len(t)
    if n != len(delta):
        raise ValueError(
            f"timestamp and steering_angle must have the same length; "
            f"got {n} vs {len(delta)}."
        )
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)

    dt = np.diff(t)
    ddelta = np.diff(delta)
    safe_dt = np.where(np.abs(dt) >= MIN_DT_S, dt, np.nan)
    rate = np.full(n, np.nan, dtype=float)
    rate[1:] = ddelta / safe_dt

    # Replace invalid interior samples with zero (divide-by-zero guard).
    bad = ~np.isfinite(rate[1:])
    if np.any(bad):
        rate[1:][bad] = 0.0

    if first_sample_mode == "copy_second":
        rate[0] = rate[1]
    elif first_sample_mode == "zero":
        rate[0] = 0.0
    else:
        raise ValueError(f"Unknown first_sample_mode: {first_sample_mode!r}")

    return rate


def compute_steering_rate_series(
    timestamp: np.ndarray,
    steering_angle: np.ndarray,
    *,
    config: SteeringRateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth steering angle (optional) and return ``(delta_smooth, steering_rate)``."""
    cfg = config or SteeringRateConfig()
    delta_smooth = smooth_steering_angle(
        steering_angle,
        method=cfg.smoothing,
        window=cfg.smoothing_window,
        polyorder=cfg.savgol_polyorder,
    )
    rate = compute_steering_rate(
        timestamp,
        delta_smooth,
        first_sample_mode=cfg.first_sample_mode,
    )
    return delta_smooth, rate


def add_steering_rate_column(
    df: pd.DataFrame,
    *,
    config: SteeringRateConfig | None = None,
    sort_by_time: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add or refresh ``steering_rate`` on a frame with ``timestamp`` and ``steering_angle``.

    Parameters
    ----------
    df
        Input table.
    config
        Smoothing and boundary handling for differentiation.
    sort_by_time
        If True, sort by ``timestamp`` (stable) before computing.
    overwrite
        If False and ``steering_rate`` exists, leave column unchanged.

    Returns
    -------
    pandas.DataFrame
        Copy with ``steering_rate`` column.
    """
    if TIMESTAMP_COLUMN not in df.columns or STEERING_ANGLE_COLUMN not in df.columns:
        raise KeyError(
            f"DataFrame must contain {TIMESTAMP_COLUMN!r} and {STEERING_ANGLE_COLUMN!r}."
        )
    if STEERING_RATE_COLUMN in df.columns and not overwrite:
        return df

    out = df.copy()
    if sort_by_time:
        out = out.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    t = out[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    delta = out[STEERING_ANGLE_COLUMN].to_numpy(dtype=float)
    _, rate = compute_steering_rate_series(t, delta, config=config)
    out[STEERING_RATE_COLUMN] = rate
    return out


def plot_steering_rate_diagnostics(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: SteeringRateConfig | None = None,
    prefix: str = "",
) -> None:
    """Save steering-angle / steering-rate diagnostic figures (A–D).

    A. steering_angle vs time
    B. steering_rate vs time
    C. histogram of steering_rate
    D. raw vs smoothed steering_rate
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""

    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    delta = df[STEERING_ANGLE_COLUMN].to_numpy(dtype=float)
    cfg = config or SteeringRateConfig()

    rate_raw = compute_steering_rate(t, delta, first_sample_mode=cfg.first_sample_mode)
    delta_smooth, rate_smooth = compute_steering_rate_series(t, delta, config=cfg)

    # A: steering angle
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, delta, linewidth=0.8, color="C0")
    ax.set_title("Steering angle vs time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("steering_angle [rad]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}steering_angle_vs_time.png", dpi=150)
    plt.close(fig)

    # B: steering rate (pipeline column if present, else smoothed path)
    rate_plot = (
        df[STEERING_RATE_COLUMN].to_numpy(dtype=float)
        if STEERING_RATE_COLUMN in df.columns
        else rate_smooth
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, rate_plot, linewidth=0.8, color="C1")
    ax.set_title("Steering rate vs time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("steering_rate [rad/s]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}steering_rate_vs_time.png", dpi=150)
    plt.close(fig)

    # C: histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rate_plot[np.isfinite(rate_plot)], bins=80, density=True, alpha=0.85)
    ax.set_title("Histogram of steering_rate")
    ax.set_xlabel("steering_rate [rad/s]")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}steering_rate_histogram.png", dpi=150)
    plt.close(fig)

    # D: raw vs smoothed steering_rate
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, rate_raw, label="raw (no pre-smooth)", linewidth=0.7, alpha=0.75)
    ax.plot(t, rate_smooth, label=f"smoothed ({cfg.smoothing.value})", linewidth=0.8)
    ax.set_title("Raw vs smoothed steering_rate")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("steering_rate [rad/s]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}steering_rate_raw_vs_smoothed.png", dpi=150)
    plt.close(fig)
