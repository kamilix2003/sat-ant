from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class MultilaterationResult:
    """Result returned by multilateration.solve_multilateration."""

    position_km: np.ndarray
    residuals_km: np.ndarray
    success: bool
    cost: float
    message: str
    n_stations: int


def solve_multilateration(
    station_positions_km: Iterable[Iterable[float]],
    ranges_km: Iterable[float],
    initial_guess_km: Iterable[float] | None = None,
    weights: Iterable[float] | None = None,
) -> MultilaterationResult:
    """Estimate a 3D position from station positions and measured ranges.

    Parameters
    ----------
    station_positions_km:
        Iterable of shape (n, 3) containing station positions in ECEF km.
    ranges_km:
        Iterable of length n containing measured ranges in km.
    initial_guess_km:
        Optional initial estimate for the unknown position. If omitted, the
        centroid of the station positions is used.
    weights:
        Optional per-station weights. Larger values increase the influence of a
        station on the fit. If provided, the residuals are multiplied by these
        weights.

    Returns
    -------
    MultilaterationResult
        Estimated position, residuals, and solver metadata.
    """
    station_positions = np.asarray(station_positions_km, dtype=float)
    ranges = np.asarray(ranges_km, dtype=float)

    if station_positions.ndim != 2 or station_positions.shape[1] != 3:
        raise ValueError("station_positions_km must have shape (n, 3)")
    if ranges.ndim != 1:
        raise ValueError("ranges_km must be a 1D sequence")
    if station_positions.shape[0] != ranges.shape[0]:
        raise ValueError("station_positions_km and ranges_km must have the same length")
    if station_positions.shape[0] < 3:
        raise ValueError("multilateration needs at least 3 stations")

    if initial_guess_km is None:
        initial_guess = station_positions.mean(axis=0)
    else:
        initial_guess = np.asarray(initial_guess_km, dtype=float)
        if initial_guess.shape != (3,):
            raise ValueError("initial_guess_km must be a length-3 vector")

    if weights is None:
        weights_array = None
    else:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != (station_positions.shape[0],):
            raise ValueError("weights must have one value per station")

    def residuals(point_km: np.ndarray) -> np.ndarray:
        predicted = np.linalg.norm(station_positions - point_km, axis=1)
        raw = predicted - ranges
        if weights_array is not None:
            return raw * weights_array
        return raw

    result = least_squares(residuals, initial_guess)
    predicted = np.linalg.norm(station_positions - result.x, axis=1)
    residual_vector = predicted - ranges

    return MultilaterationResult(
        position_km=result.x,
        residuals_km=residual_vector,
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        n_stations=int(station_positions.shape[0]),
    )


def multilaterate(
    station_positions_km: Iterable[Iterable[float]],
    ranges_km: Iterable[float],
    initial_guess_km: Iterable[float] | None = None,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    """Convenience wrapper returning only the estimated position."""
    return solve_multilateration(
        station_positions_km=station_positions_km,
        ranges_km=ranges_km,
        initial_guess_km=initial_guess_km,
        weights=weights,
    ).position_km
