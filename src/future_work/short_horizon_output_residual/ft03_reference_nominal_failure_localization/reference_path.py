"""GNSS-backed Stanford reference paths for the isolated FT03 audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd

from future_work.later_mpc.reference_path import ReferencePath
from future_work.later_mpc.types import ReferencePoint


RAW_FILE = Path(__file__).resolve().parents[3] / "VehicleDynamicsDataset" / "Nov2023" / "2023-11_11.csv"
SAMPLE_TIME_S = 0.05


@dataclass(frozen=True)
class PathBuild:
    path: "StanfordGNSSPath"
    frame: pd.DataFrame
    formal_start_s: float
    formal_end_s: float
    source_formal_start_s: float
    source_formal_end_s: float


class StanfordGNSSPath(ReferencePath):
    """Finite path using measured GNSS position and measured yaw geometry.

    East/North coordinates are rigidly transformed into a local frame whose
    origin and +X tangent are the first source sample. Curvature is the
    measured yaw rate divided by GNSS ground speed. No vehicle state is used
    to initialize Chrono from this object.
    """

    name = "stanford_gnss_reference"

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {
            "posE_m", "posN_m", "yawAngle_rad", "yawRate_radps",
            "Ve_mps", "Vn_mps", "t_s",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing GNSS path columns: {missing}")
        if len(frame) < 3:
            raise ValueError("GNSS path requires at least three samples")
        east = frame.posE_m.to_numpy(float)
        north = frame.posN_m.to_numpy(float)
        yaw = np.unwrap(frame.yawAngle_rad.to_numpy(float))
        # Stanford yaw is zero along +North. yaw + pi/2 is the conventional
        # EN-plane angle measured counter-clockwise from +East.
        heading0 = float(yaw[0] + math.pi / 2.0)
        de = east - east[0]
        dn = north - north[0]
        c, s = math.cos(heading0), math.sin(heading0)
        local_x = c * de + s * dn
        local_y = -s * de + c * dn
        station = np.r_[0.0, np.cumsum(np.hypot(np.diff(east), np.diff(north)))]
        keep = np.r_[True, np.diff(station) > 1e-8]
        self._s = station[keep]
        self._x = local_x[keep]
        self._y = local_y[keep]
        self._psi = (yaw - yaw[0])[keep]
        ground_speed = np.hypot(
            frame.Ve_mps.to_numpy(float), frame.Vn_mps.to_numpy(float)
        )[keep]
        self._kappa = frame.yawRate_radps.to_numpy(float)[keep] / np.maximum(ground_speed, 1.0)
        if not np.all(np.diff(self._s) > 0.0):
            raise ValueError("GNSS path station must be strictly increasing")
        if not np.isfinite(np.c_[self._s, self._x, self._y, self._psi, self._kappa]).all():
            raise ValueError("GNSS path contains non-finite values")
        count = max(1001, int(math.ceil(self.length_m / 0.05)) + 1)
        self._projection_s = np.linspace(0.0, self.length_m, count)
        self._projection_x = np.interp(self._projection_s, self._s, self._x)
        self._projection_y = np.interp(self._projection_s, self._s, self._y)

    @property
    def length_m(self) -> float:
        return float(self._s[-1])

    def sample(self, s_m: float) -> ReferencePoint:
        station = float(np.clip(s_m, 0.0, self.length_m))
        return ReferencePoint(
            s_m=station,
            X_m=float(np.interp(station, self._s, self._x)),
            Y_m=float(np.interp(station, self._s, self._y)),
            psi_rad=float(np.interp(station, self._s, self._psi)),
            curvature_per_m=float(np.interp(station, self._s, self._kappa)),
        )

    def project(self, X_m: float, Y_m: float) -> ReferencePoint:
        d2 = (self._projection_x - X_m) ** 2 + (self._projection_y - Y_m) ** 2
        return self.sample(float(self._projection_s[int(np.argmin(d2))]))


def read_raw_session(path: Path = RAW_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=2)
    frame = frame.sort_values("t_s").drop_duplicates("t_s").reset_index(drop=True)
    return frame


def resample_source(frame: pd.DataFrame, start_s: float, stop_s: float) -> pd.DataFrame:
    grid = np.arange(start_s, stop_s + 0.25 * SAMPLE_TIME_S, SAMPLE_TIME_S)
    numeric = [
        "posE_m", "posN_m", "yawAngle_rad", "yawRate_radps", "Vx_mps", "Vy_mps",
        "Ve_mps", "Vn_mps", "roadWheelAngle_rad", "ayCG_mps2", "axCG_mps2",
    ]
    source_t = frame.t_s.to_numpy(float)
    data: dict[str, np.ndarray] = {"t_s": grid}
    for column in numeric:
        values = frame[column].to_numpy(float)
        if column == "yawAngle_rad":
            values = np.unwrap(values)
        data[column] = np.interp(grid, source_t, values)
    result = pd.DataFrame(data)
    result["delta_dot_radps"] = np.gradient(
        result.roadWheelAngle_rad.to_numpy(float), SAMPLE_TIME_S, edge_order=2
    )
    return result


def build_path(
    formal_start_s: float,
    formal_end_s: float,
    *,
    lead_in_s: float = 5.0,
    trailing_preview_s: float = 2.0,
) -> PathBuild:
    raw = read_raw_session()
    start = formal_start_s - lead_in_s
    stop = formal_end_s + trailing_preview_s
    frame = resample_source(raw, start, stop)
    path = StanfordGNSSPath(frame)
    index_start = int(np.argmin(np.abs(frame.t_s.to_numpy(float) - formal_start_s)))
    index_end = int(np.argmin(np.abs(frame.t_s.to_numpy(float) - formal_end_s)))
    east = frame.posE_m.to_numpy(float)
    north = frame.posN_m.to_numpy(float)
    stations = np.r_[0.0, np.cumsum(np.hypot(np.diff(east), np.diff(north)))]
    return PathBuild(
        path=path,
        frame=frame,
        formal_start_s=float(stations[index_start]),
        formal_end_s=float(stations[index_end]),
        source_formal_start_s=formal_start_s,
        source_formal_end_s=formal_end_s,
    )

