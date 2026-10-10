"""Calibrated conversion between MPC road-wheel angle and Chrono input."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class SteeringCalibration:
    """Measured, monotonic normalized-steering/road-wheel-angle lookup table.

    The table is intentionally loaded from a calibration artifact instead of
    assuming a linear steering ratio.  Conversion uses piecewise-linear
    interpolation and endpoint saturation in both directions.
    """

    normalized_steering: tuple[float, ...]
    road_wheel_angle_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.normalized_steering) < 2:
            raise ValueError("A steering calibration needs at least two points.")
        if len(self.normalized_steering) != len(self.road_wheel_angle_rad):
            raise ValueError("Calibration input and output arrays must have equal length.")
        if any(
            right <= left
            for left, right in zip(self.normalized_steering, self.normalized_steering[1:])
        ):
            raise ValueError("Normalized steering calibration points must be strictly increasing.")
        if any(
            right <= left
            for left, right in zip(self.road_wheel_angle_rad, self.road_wheel_angle_rad[1:])
        ):
            raise ValueError("Road-wheel angle calibration points must be strictly increasing.")

    @classmethod
    def from_mapping_table(cls, rows: Iterable[dict[str, object]]) -> "SteeringCalibration":
        """Build the adapter from generated calibration JSON table rows."""
        ordered_rows = sorted(
            rows, key=lambda row: float(row["normalized_steering"])
        )
        return cls(
            normalized_steering=tuple(
                float(row["normalized_steering"]) for row in ordered_rows
            ),
            road_wheel_angle_rad=tuple(
                float(row["equivalent_road_wheel_angle_rad"]) for row in ordered_rows
            ),
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_vehicle_model: str | None = None,
    ) -> "SteeringCalibration":
        """Load a calibration artifact and optionally enforce its plant identity."""
        with Path(path).open(encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if expected_vehicle_model is not None:
            actual_vehicle_model = payload.get("vehicle_model")
            if actual_vehicle_model != expected_vehicle_model:
                raise ValueError(
                    "Steering calibration vehicle_model mismatch: expected "
                    f"{expected_vehicle_model!r}, got {actual_vehicle_model!r}."
                )
        table = payload.get("mapping_table")
        if not isinstance(table, list):
            raise ValueError("Calibration JSON must contain a mapping_table list.")
        return cls.from_mapping_table(table)

    def normalized_to_delta_rad(self, normalized_steering: float) -> float:
        """Convert Chrono normalized steering to calibrated road-wheel angle."""
        return self._interpolate(
            float(normalized_steering),
            self.normalized_steering,
            self.road_wheel_angle_rad,
        )

    def delta_rad_to_normalized(self, delta_rad: float) -> float:
        """Convert MPC road-wheel angle to calibrated Chrono normalized input."""
        return self._interpolate(
            float(delta_rad),
            self.road_wheel_angle_rad,
            self.normalized_steering,
        )

    @staticmethod
    def _interpolate(value: float, x_values: Sequence[float], y_values: Sequence[float]) -> float:
        if value <= x_values[0]:
            return y_values[0]
        if value >= x_values[-1]:
            return y_values[-1]

        upper_index = bisect_left(x_values, value)
        lower_index = upper_index - 1
        x0, x1 = x_values[lower_index], x_values[upper_index]
        y0, y1 = y_values[lower_index], y_values[upper_index]
        ratio = (value - x0) / (x1 - x0)
        return y0 + ratio * (y1 - y0)
