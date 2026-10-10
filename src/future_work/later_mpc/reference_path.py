"""Smooth reference paths for staged lateral-MPC validation.

The paths are parameterized by travelled arc length ``s``.  A positive lateral
axis is the left side of the path tangent, consistent with the error convention
used by :mod:`error_state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import atan2, cos, pi, sin

import numpy as np
from scipy.optimize import minimize_scalar

from .types import ReferencePoint, ReferenceTrajectory


class ReferencePath(ABC):
    """Common arc-length path interface used by controller-side code."""

    name: str = "reference_path"

    @property
    @abstractmethod
    def length_m(self) -> float:
        """Finite path length available for projection and preview."""

    @abstractmethod
    def sample(self, s_m: float) -> ReferencePoint:
        """Return position, heading, and curvature at arc-length station ``s_m``."""

    def project(self, X_m: float, Y_m: float) -> ReferencePoint:
        """Find the closest point on this finite reference path.

        A coarse grid selects a robust local interval before bounded scalar
        minimization.  Analytic paths may override this method when useful.
        """
        stations = np.linspace(0.0, self.length_m, 401)
        distances_sq = np.array(
            [(self.sample(s).X_m - X_m) ** 2 + (self.sample(s).Y_m - Y_m) ** 2 for s in stations]
        )
        index = int(np.argmin(distances_sq))
        lower = float(stations[max(0, index - 1)])
        upper = float(stations[min(len(stations) - 1, index + 1)])
        if lower == upper:
            return self.sample(lower)

        def squared_distance(s_m: float) -> float:
            point = self.sample(s_m)
            return (point.X_m - X_m) ** 2 + (point.Y_m - Y_m) ** 2

        result = minimize_scalar(squared_distance, bounds=(lower, upper), method="bounded")
        return self.sample(float(result.x))

    def generate_horizon(
        self,
        current_position: float,
        N: int,
        Ts: float,
        vx: float,
    ) -> ReferenceTrajectory:
        """Generate an ``N+1`` point constant-speed path preview.

        ``current_position`` is the current reference-path station in metres.
        The returned first point is at that station and the following points are
        separated by ``vx * Ts``.  Stations are clipped at the finite path end.
        """
        if N < 1:
            raise ValueError("N must be at least one.")
        if Ts <= 0.0 or vx <= 0.0:
            raise ValueError("Ts and vx must be positive.")
        stations = np.clip(
            float(current_position) + np.arange(N + 1, dtype=float) * Ts * vx,
            0.0,
            self.length_m,
        )
        points = [self.sample(float(s)) for s in stations]
        return ReferenceTrajectory(
            X_m=np.array([point.X_m for point in points]),
            Y_m=np.array([point.Y_m for point in points]),
            psi_rad=np.array([point.psi_rad for point in points]),
            curvature_per_m=np.array([point.curvature_per_m for point in points]),
            arc_length_m=stations,
            name=self.name,
        )


class StraightPath(ReferencePath):
    """Straight reference: ``X=s, Y=0, psi=0, kappa=0``."""

    name = "straight"

    def __init__(self, length_m: float = 300.0) -> None:
        if length_m <= 0.0:
            raise ValueError("length_m must be positive.")
        self._length_m = float(length_m)

    @property
    def length_m(self) -> float:
        return self._length_m

    def sample(self, s_m: float) -> ReferencePoint:
        s = float(np.clip(s_m, 0.0, self.length_m))
        return ReferencePoint(s, s, 0.0, 0.0, 0.0)

    def project(self, X_m: float, Y_m: float) -> ReferencePoint:
        return self.sample(float(np.clip(X_m, 0.0, self.length_m)))


class CircularPath(ReferencePath):
    """Constant-curvature left-hand circular arc, beginning at the origin.

    The circle centre is ``(0, radius_m)``.  Consequently the initial tangent
    points along +X and the curvature is ``1 / radius_m``.
    """

    name = "circular"

    def __init__(self, radius_m: float, arc_length_m: float = 150.0) -> None:
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive for this left-hand arc.")
        if arc_length_m <= 0.0:
            raise ValueError("arc_length_m must be positive.")
        self.radius_m = float(radius_m)
        self._length_m = float(arc_length_m)

    @property
    def length_m(self) -> float:
        return self._length_m

    def sample(self, s_m: float) -> ReferencePoint:
        s = float(np.clip(s_m, 0.0, self.length_m))
        theta = s / self.radius_m
        return ReferencePoint(
            s_m=s,
            X_m=self.radius_m * sin(theta),
            Y_m=self.radius_m * (1.0 - cos(theta)),
            psi_rad=theta,
            curvature_per_m=1.0 / self.radius_m,
        )


class DoubleLaneChangePath(ReferencePath):
    """Smooth S-shaped path with zero curvature at both straight joins.

    The heading is a compact smooth function on the manoeuvre interval.  Its
    sign reverses halfway through the interval, so integrating it in arc length
    yields a smooth out-and-back double-lane-change geometry rather than a
    piecewise-linear path.
    """

    name = "double_lane_change"

    def __init__(self, manoeuvre_length_m: float = 80.0, heading_peak_rad: float = 0.12) -> None:
        if manoeuvre_length_m <= 0.0:
            raise ValueError("manoeuvre_length_m must be positive.")
        if heading_peak_rad <= 0.0:
            raise ValueError("heading_peak_rad must be positive.")
        self.manoeuvre_length_m = float(manoeuvre_length_m)
        self.heading_peak_rad = float(heading_peak_rad)
        self._integration_stations = np.linspace(0.0, self.manoeuvre_length_m, 2001)
        headings = np.array([self._heading(s) for s in self._integration_stations])
        station_step = self._integration_stations[1] - self._integration_stations[0]
        self._X_lookup = np.concatenate(
            ([0.0], np.cumsum(0.5 * station_step * (np.cos(headings[:-1]) + np.cos(headings[1:]))))
        )
        self._Y_lookup = np.concatenate(
            ([0.0], np.cumsum(0.5 * station_step * (np.sin(headings[:-1]) + np.sin(headings[1:]))))
        )
        self._end_X_m, self._end_Y_m = float(self._X_lookup[-1]), float(self._Y_lookup[-1])

    @property
    def length_m(self) -> float:
        return self.manoeuvre_length_m

    def _heading(self, s_m: float) -> float:
        t = float(np.clip(s_m / self.manoeuvre_length_m, 0.0, 1.0))
        return self.heading_peak_rad * sin(2.0 * pi * t) * sin(pi * t) ** 2

    def _curvature(self, s_m: float) -> float:
        t = float(np.clip(s_m / self.manoeuvre_length_m, 0.0, 1.0))
        # d/ds [theta_peak sin(2*pi*t) sin(pi*t)^2]
        derivative_t = self.heading_peak_rad * (
            2.0 * pi * cos(2.0 * pi * t) * sin(pi * t) ** 2
            + 2.0 * pi * sin(2.0 * pi * t) * sin(pi * t) * cos(pi * t)
        )
        return derivative_t / self.manoeuvre_length_m

    def _integrate_position(self, s_m: float) -> tuple[float, float]:
        s = float(np.clip(s_m, 0.0, self.length_m))
        return (
            float(np.interp(s, self._integration_stations, self._X_lookup)),
            float(np.interp(s, self._integration_stations, self._Y_lookup)),
        )

    def sample(self, s_m: float) -> ReferencePoint:
        s = float(np.clip(s_m, 0.0, self.length_m))
        X_m, Y_m = self._integrate_position(s)
        return ReferencePoint(s, X_m, Y_m, self._heading(s), self._curvature(s))


# Short alias for manuscripts or scripts preferring the conventional S-path name.
SPath = DoubleLaneChangePath
