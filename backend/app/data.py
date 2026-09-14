"""Surface-data access layer.

The synthetic provider mirrors the shape of an Xarray/Dask-backed satellite
window so it can be replaced by Copernicus Marine downloads without changing
model or API code.
"""
from dataclasses import dataclass
from datetime import date
import numpy as np
import xarray as xr

LAT_MIN, LAT_MAX = 5.0, 23.0
LON_MIN, LON_MAX = 80.0, 100.0
DEPTHS = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000])


@dataclass
class SurfaceWindow:
    values: dict[str, float]
    dataset: xr.Dataset


def validate_point(lat: float, lon: float) -> None:
    if not LAT_MIN <= lat <= LAT_MAX or not LON_MIN <= lon <= LON_MAX:
        raise ValueError(f"Point must be inside the Bay of Bengal box ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")


def synthetic_surface_window(lat: float, lon: float, observation_date: date) -> SurfaceWindow:
    """Create a small lazy-compatible observation window for the demo."""
    validate_point(lat, lon)
    day = observation_date.toordinal() % 365
    spatial = np.sin(np.radians(lat * 7 + lon * 2))
    seasonal = np.cos(day / 58)
    values = {
        "sst": 28.2 + 1.6 * spatial + 0.9 * seasonal,
        "sss": 33.7 + 0.8 * np.cos(np.radians(lon * 5)) - 0.3 * spatial,
        "ssh": 0.12 * np.sin(np.radians(lat * 4)) + 0.05 * seasonal,
        "u": 0.45 * np.cos(np.radians(lon * 8 + day)),
        "v": 0.35 * np.sin(np.radians(lat * 9 - day)),
    }
    times = np.arange(-6, 1)
    dataset = xr.Dataset(
        {name: (("time",), np.repeat(value, len(times))) for name, value in values.items()},
        coords={"time": times},
    ).chunk({"time": 1})
    return SurfaceWindow(values=values, dataset=dataset)


def copernicus_surface_window(*args, **kwargs):
    """Extension point for copernicusmarine.open_* downloads."""
    raise NotImplementedError("Set COPERNICUS_* credentials and implement the regional product query.")
