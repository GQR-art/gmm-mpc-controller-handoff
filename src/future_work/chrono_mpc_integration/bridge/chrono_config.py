"""Chrono vehicle-model paths and deterministic integration timing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Literal


VehicleModel = Literal["Sedan", "GolfLike", "GolfLike3D_v1"]
SUPPORTED_VEHICLE_MODELS: tuple[str, ...] = (
    "Sedan",
    "GolfLike",
    "GolfLike3D_v1",
)


def _path_from_environment(name: str) -> Path | None:
    """Return an optional path configured by the release environment."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


@dataclass(frozen=True, slots=True)
class ChronoSedanConfig:
    """Immutable configuration for the shared headless vehicle bridge.

    Set ``CHRONO_MPC_CHRONO_RELEASE_DIR`` and ``CHRONO_MPC_CHRONO_DATA_DIR``
    before constructing a plant, or pass both paths explicitly.  This keeps
    the established interface while avoiding developer-machine path defaults.
    ``Sedan`` remains the default; ``GolfLike`` selects the planar
    Bicycle-equivalent baseline and ``GolfLike3D_v1`` its independently
    versioned 3-D physical completion.
    """

    chrono_release_dir: Path | None = field(
        default_factory=lambda: _path_from_environment("CHRONO_MPC_CHRONO_RELEASE_DIR")
    )
    data_dir: Path | None = field(
        default_factory=lambda: _path_from_environment("CHRONO_MPC_CHRONO_DATA_DIR")
    )
    physics_dt_s: float = 0.002
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_z_m: float = 0.5
    initial_yaw_rad: float = 0.0
    initial_forward_speed_mps: float = 0.0
    vehicle_model: VehicleModel = "Sedan"

    def __post_init__(self) -> None:
        if self.vehicle_model not in SUPPORTED_VEHICLE_MODELS:
            raise ValueError(
                "vehicle_model must be one of "
                f"{SUPPORTED_VEHICLE_MODELS}, got {self.vehicle_model!r}."
            )
        if not math.isfinite(self.initial_yaw_rad):
            raise ValueError("initial_yaw_rad must be finite.")

    @property
    def vehicle_data_dir(self) -> Path:
        if self.data_dir is None:
            raise RuntimeError(
                "Chrono data directory is not configured. Set "
                "CHRONO_MPC_CHRONO_DATA_DIR or pass data_dir explicitly."
            )
        return self.data_dir / "vehicle"

    @property
    def sedan_vehicle_json(self) -> Path:
        return self.vehicle_data_dir / "sedan" / "vehicle" / "Sedan_Vehicle.json"

    @property
    def sedan_engine_json(self) -> Path:
        return self.vehicle_data_dir / "sedan" / "powertrain" / "Sedan_EngineSimpleMap.json"

    @property
    def sedan_transmission_json(self) -> Path:
        return (
            self.vehicle_data_dir
            / "sedan"
            / "powertrain"
            / "Sedan_AutomaticTransmissionSimpleMap.json"
        )

    @property
    def sedan_tire_json(self) -> Path:
        return self.vehicle_data_dir / "sedan" / "tire" / "Sedan_TMeasyTire.json"

    @property
    def golf_like_model_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "vehicle_models" / "GolfLike"

    @property
    def golf_like_vehicle_json(self) -> Path:
        return self.golf_like_model_dir / "GolfLike_Vehicle.json"

    @property
    def golf_like_chassis_json(self) -> Path:
        return self.golf_like_model_dir / "GolfLike_Chassis.json"

    @property
    def golf_like_front_tire_json(self) -> Path:
        return self.golf_like_model_dir / "GolfLike_TMeasy_Front.json"

    @property
    def golf_like_rear_tire_json(self) -> Path:
        return self.golf_like_model_dir / "GolfLike_TMeasy_Rear.json"

    @property
    def golf_like_3d_model_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "vehicle_models" / "GolfLike3D_v1"

    @property
    def golf_like_3d_vehicle_json(self) -> Path:
        return self.golf_like_3d_model_dir / "GolfLike3D_v1_Vehicle.json"

    @property
    def golf_like_3d_chassis_json(self) -> Path:
        return self.golf_like_3d_model_dir / "GolfLike3D_v1_Chassis.json"

    @property
    def golf_like_3d_suspension_json(self) -> Path:
        return self.golf_like_3d_model_dir / "GolfLike3D_v1_DoubleWishbone.json"

    @property
    def golf_like_3d_front_antirollbar_json(self) -> Path:
        return self.golf_like_3d_model_dir / "GolfLike3D_v1_AntirollBar_Front.json"

    @property
    def golf_like_3d_rear_antirollbar_json(self) -> Path:
        return self.golf_like_3d_model_dir / "GolfLike3D_v1_AntirollBar_Rear.json"

    @property
    def golf_like_3d_front_tire_json(self) -> Path:
        # Phase 2.14 deliberately reuses the frozen Phase 2.8 tire file.
        return self.golf_like_front_tire_json

    @property
    def golf_like_3d_rear_tire_json(self) -> Path:
        # Phase 2.14 deliberately reuses the frozen Phase 2.8 tire file.
        return self.golf_like_rear_tire_json

    @property
    def vehicle_json(self) -> Path:
        if self.vehicle_model == "GolfLike3D_v1":
            return self.golf_like_3d_vehicle_json
        if self.vehicle_model == "GolfLike":
            return self.golf_like_vehicle_json
        return self.sedan_vehicle_json

    @property
    def front_tire_json(self) -> Path:
        if self.vehicle_model == "GolfLike3D_v1":
            return self.golf_like_3d_front_tire_json
        if self.vehicle_model == "GolfLike":
            return self.golf_like_front_tire_json
        return self.sedan_tire_json

    @property
    def rear_tire_json(self) -> Path:
        if self.vehicle_model == "GolfLike3D_v1":
            return self.golf_like_3d_rear_tire_json
        if self.vehicle_model == "GolfLike":
            return self.golf_like_rear_tire_json
        return self.sedan_tire_json

    def tire_json_for_axle(self, axle_index: int) -> Path:
        if axle_index == 0:
            return self.front_tire_json
        if axle_index == 1:
            return self.rear_tire_json
        raise IndexError(f"No tire configuration for axle index {axle_index}.")

    @property
    def rigid_terrain_json(self) -> Path:
        return self.vehicle_data_dir / "terrain" / "RigidPlane.json"

    def validate(self) -> None:
        if self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        if self.initial_forward_speed_mps < 0.0:
            raise ValueError("initial_forward_speed_mps must be non-negative.")
        if self.chrono_release_dir is None or self.data_dir is None:
            raise RuntimeError(
                "Chrono runtime paths are not configured. Set "
                "CHRONO_MPC_CHRONO_RELEASE_DIR and CHRONO_MPC_CHRONO_DATA_DIR, "
                "or pass chrono_release_dir and data_dir explicitly. See "
                "configs/chrono.example.env."
            )
        required_paths = (
            self.chrono_release_dir,
            self.data_dir,
            self.vehicle_json,
            *((self.golf_like_chassis_json,) if self.vehicle_model == "GolfLike" else ()),
            *(
                (
                    self.golf_like_3d_chassis_json,
                    self.golf_like_3d_suspension_json,
                    self.golf_like_3d_front_antirollbar_json,
                    self.golf_like_3d_rear_antirollbar_json,
                )
                if self.vehicle_model == "GolfLike3D_v1"
                else ()
            ),
            self.sedan_engine_json,
            self.sedan_transmission_json,
            self.front_tire_json,
            self.rear_tire_json,
            self.rigid_terrain_json,
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Chrono vehicle paths are missing: {missing}")
