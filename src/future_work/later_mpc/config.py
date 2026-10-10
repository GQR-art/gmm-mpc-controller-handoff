"""Configuration for the staged nominal MPC baseline.

The module reads the existing fitted bicycle parameters but never alters the
source JSON or any paper code.  Numerical controller implementation is deferred
to later stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_FITTED_PARAMETERS_PATH = PROJECT_ROOT / "FITTED_MODEL_PARAMETERS.json"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "outputs"


@dataclass(frozen=True)
class VehicleParameters:
    """Physical parameters shared by nominal prediction and plant construction."""

    m: float
    Iz: float
    a: float
    b: float
    Cf: float
    Cr: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "VehicleParameters":
        """Validate and convert an existing fitted-parameter mapping."""
        required = ("m", "Iz", "a", "b", "Cf", "Cr")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Missing fitted bicycle parameters: {missing}")
        values = {key: float(raw[key]) for key in required}
        if any(value <= 0.0 for value in values.values()):
            raise ValueError("All fitted bicycle parameters must be positive.")
        return cls(**values)

    def as_dict(self) -> dict[str, float]:
        """Return the parameter representation expected by DynamicBicycleModel."""
        return {
            "m": self.m,
            "Iz": self.Iz,
            "a": self.a,
            "b": self.b,
            "Cf": self.Cf,
            "Cr": self.Cr,
        }


def load_fitted_vehicle_parameters(
    path: str | Path = DEFAULT_FITTED_PARAMETERS_PATH,
) -> VehicleParameters:
    """Read fitted bicycle parameters without modifying their source file."""
    parameter_path = Path(path)
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Fitted parameter file not found: {parameter_path}")
    with parameter_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Fitted parameter JSON must contain an object mapping.")
    # The established fitted-parameter artifact stores its vehicle parameters
    # below this key together with descriptive metadata and fit metrics.
    parameter_mapping = raw.get("fitted_model_parameters", raw)
    if not isinstance(parameter_mapping, Mapping):
        raise ValueError("fitted_model_parameters must contain an object mapping.")
    return VehicleParameters.from_mapping(parameter_mapping)


@dataclass(frozen=True)
class StateWeights:
    """Quadratic penalties for [ey, epsi, vy, r] and terminal state."""

    ey: float = 40.0
    epsi: float = 12.0
    vy: float = 2.0
    r: float = 2.0
    terminal_scale: float = 2.0


@dataclass(frozen=True)
class InputWeights:
    """Quadratic penalties for steering magnitude and steering increments."""

    delta: float = 0.5
    delta_rate: float = 8.0


@dataclass(frozen=True)
class SteeringConstraints:
    """Road-wheel steering and per-sample steering-increment bounds."""

    delta_min_rad: float = -0.35
    delta_max_rad: float = 0.35
    delta_rate_min_rad_per_sample: float = -0.03
    delta_rate_max_rad_per_sample: float = 0.03

    def __post_init__(self) -> None:
        if self.delta_min_rad >= self.delta_max_rad:
            raise ValueError("delta_min_rad must be less than delta_max_rad.")
        if self.delta_rate_min_rad_per_sample >= self.delta_rate_max_rad_per_sample:
            raise ValueError("Steering increment lower bound must be below upper bound.")


@dataclass(frozen=True)
class MPCConfig:
    """Immutable baseline configuration; future stages consume this object."""

    sample_time_s: float = 0.05
    horizon_steps: int = 20
    longitudinal_speed_mps: float = 15.0
    state_weights: StateWeights = field(default_factory=StateWeights)
    input_weights: InputWeights = field(default_factory=InputWeights)
    steering_constraints: SteeringConstraints = field(default_factory=SteeringConstraints)

    def __post_init__(self) -> None:
        if self.sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be positive.")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be at least one.")
        if self.longitudinal_speed_mps <= 0.0:
            raise ValueError("longitudinal_speed_mps must be positive.")
