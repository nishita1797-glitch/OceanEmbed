"""Demo Multi-Modal Ocean Encoder inference implementation."""
from datetime import date
import numpy as np
from .data import DEPTHS, synthetic_surface_window

VARIABLES = ("SST", "SSS", "SSH/SLA", "Current U/V")


def predict_profile(lat: float, lon: float, observation_date: date) -> dict:
    surface = synthetic_surface_window(lat, lon, observation_date)
    values = surface.values
    normalized = np.array([
        (values["sst"] - 28) / 3,
        (values["sss"] - 34) / 2,
        values["ssh"] / 0.2,
        values["u"],
        values["v"],
    ])
    # A compact deterministic stand-in for the trained encoder and decoder.
    embedding_signal = float(normalized @ np.array([0.8, -0.25, 0.35, 0.12, -0.08]))
    thermocline = 135 + 35 * np.tanh(values["ssh"] * 4) + 12 * np.sin(np.radians(lat * 3))
    deep_temperature = 5.2 + 0.3 * np.cos(np.radians(lon * 4))
    temperatures = []
    uncertainties = []
    for depth in DEPTHS:
        decay = np.exp(-depth / 180)
        temperature = deep_temperature + (values["sst"] - deep_temperature) * np.exp(-depth / thermocline)
        temperature += 0.18 * embedding_signal * decay
        uncertainty = 0.22 + 0.00048 * depth + 0.05 * abs(np.sin(depth / 170))
        temperatures.append(round(float(temperature), 2))
        uncertainties.append(round(float(uncertainty), 2))
    weights = np.stack([
        np.full(len(DEPTHS), 0.38),
        np.full(len(DEPTHS), 0.18),
        np.full(len(DEPTHS), 0.22),
        np.full(len(DEPTHS), 0.22),
    ])
    weights[0] *= np.exp(-DEPTHS / 500)
    weights[2] *= 0.7 + DEPTHS / 1000
    weights[3] *= 0.8 + 0.3 * np.sin(DEPTHS / 240) ** 2
    weights /= weights.sum(axis=0)
    return {
        "lat": lat,
        "lon": lon,
        "date": observation_date.isoformat(),
        "depths": DEPTHS.tolist(),
        "temperatures": temperatures,
        "uncertainties": uncertainties,
        "attention": {variable: [round(float(v), 3) for v in row] for variable, row in zip(VARIABLES, weights)},
        "model": "OceanEmbed synthetic Multi-Modal Ocean Encoder",
    }


def heatmap(date_value: date, depth: int) -> dict:
    if depth not in DEPTHS:
        raise ValueError(f"Depth must be one of: {', '.join(map(str, DEPTHS))}")
    # Keep the demo field in the open-water core of the Bay, avoiding land tiles.
    lats = np.linspace(7, 21, 22)
    lons = np.linspace(84, 98, 23)
    points = [predict_profile(float(lat), float(lon), date_value) for lat in lats for lon in lons]
    index = int(np.where(DEPTHS == depth)[0][0])
    return {
        "date": date_value.isoformat(),
        "depth": depth,
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "points": [{"lat": p["lat"], "lon": p["lon"], "temperature": p["temperatures"][index]} for p in points],
        "min_temperature": min(p["temperatures"][index] for p in points),
        "max_temperature": max(p["temperatures"][index] for p in points),
    }
