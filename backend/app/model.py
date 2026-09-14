"""Demo Multi-Modal Ocean Encoder inference implementation."""
from datetime import date
from functools import lru_cache
import numpy as np
from .data import DEPTHS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, synthetic_surface_window

VARIABLES = ("SST", "SSS", "SSH/SLA", "Current U/V")
PARAMETERS = {
    "temperature": {"label": "Temperature (SST / SubSST)", "unit": "°C", "minimum": 0, "maximum": 32},
    "salinity": {"label": "Salinity (PSAL)", "unit": "PSU", "minimum": 30, "maximum": 38},
    "oxygen": {"label": "Dissolved Oxygen (DO)", "unit": "µmol/kg", "minimum": 0, "maximum": 260},
    "ph": {"label": "pH Levels", "unit": "pH", "minimum": 7.4, "maximum": 8.3},
    "turbidity": {"label": "Turbidity / Water Clarity (Kd490)", "unit": "Kd490", "minimum": 0.02, "maximum": 0.6},
}


def predict_profile(lat: float, lon: float, observation_date: date, parameter: str = "temperature") -> dict:
    if parameter not in PARAMETERS:
        raise ValueError(f"Parameter must be one of: {', '.join(PARAMETERS)}")
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
        if parameter == "salinity": value = values["sss"] + 0.9 * (1 - decay)
        elif parameter == "oxygen": value = 245 * decay + 62 * (1 - decay) + 6 * values["v"]
        elif parameter == "ph": value = 8.18 - 0.45 * (1 - decay) + 0.02 * values["u"]
        elif parameter == "turbidity": value = 0.06 + 0.28 * (1 - decay) + 0.02 * abs(values["v"])
        else: value = temperature
        temperatures.append(round(float(value), 2))
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
        "parameter": parameter,
        "unit": PARAMETERS[parameter]["unit"],
        "depths": DEPTHS.tolist(),
        "temperatures": temperatures,
        "uncertainties": uncertainties,
        "parameter_values": {"temperature": temperatures, "salinity": [round(float(values["sss"] + 0.9 * (1 - np.exp(-depth / 180))), 2) for depth in DEPTHS], "oxygen": [round(float(245 * np.exp(-depth / 180) + 62 * (1 - np.exp(-depth / 180)) + 6 * values["v"]), 2) for depth in DEPTHS], "ph": [round(float(8.18 - 0.45 * (1 - np.exp(-depth / 180)) + 0.02 * values["u"]), 2) for depth in DEPTHS], "turbidity": [round(float(0.06 + 0.28 * (1 - np.exp(-depth / 180)) + 0.02 * abs(values["v"])), 3) for depth in DEPTHS]},
        "attention": {variable: [round(float(v), 3) for v in row] for variable, row in zip(VARIABLES, weights)},
        "model": "OceanEmbed synthetic Multi-Modal Ocean Encoder",
    }


@lru_cache(maxsize=64)
def heatmap(date_value: date, depth: int, parameter: str = "temperature") -> dict:
    if depth not in DEPTHS or parameter not in PARAMETERS:
        raise ValueError(f"Depth must be one of: {', '.join(map(str, DEPTHS))}")
    # Dask remains the data-loading boundary; this grid is the cached demo output.
    lats = np.linspace(LAT_MIN, LAT_MAX, 71)
    lons = np.linspace(LON_MIN, LON_MAX, 111)
    points = [predict_profile(float(lat), float(lon), date_value, parameter) for lat in lats for lon in lons]
    index = int(np.where(DEPTHS == depth)[0][0])
    return {
        "date": date_value.isoformat(),
        "depth": depth,
        "parameter": parameter,
        "unit": PARAMETERS[parameter]["unit"],
        "bounds": [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        "points": [{"lat": p["lat"], "lon": p["lon"], "temperature": p["temperatures"][index]} for p in points],
        "min_temperature": min(p["temperatures"][index] for p in points),
        "max_temperature": max(p["temperatures"][index] for p in points),
    }
