"""Natural-language assistant with optional LLM enhancement."""
import re
from datetime import date
from .model import predict_profile, heatmap

CITIES = {
    "mumbai": (19.08, 72.88), "chennai": (13.08, 80.27), "kolkata": (22.57, 88.36),
    "colombo": (6.93, 79.86), "male": (4.18, 73.51), "mombasa": (-4.05, 39.67),
    "muscat": (23.59, 58.41), "jakarta": (-6.21, 106.85), "perth": (-31.95, 115.86),
    "visakhapatnam": (17.69, 83.22), "chittagong": (22.36, 91.78), "port blair": (11.62, 92.73),
}


def answer_query(message: str) -> dict:
    text = message.lower()
    depth_match = re.search(r"(\d{1,4})\s*m", text)
    depth = int(depth_match.group(1)) if depth_match else 200
    depth = min((0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000), key=lambda item: abs(item - depth))
    point = next((coords for city, coords in CITIES.items() if city in text), (0.0, 80.0))
    if "warmest" in text or "where" in text:
        grid = heatmap(date.today(), depth)
        warmest = max(grid["points"], key=lambda item: item["temperature"])
        return {"answer": f"The warmest modeled point at {depth} m is {warmest['temperature']:.1f} C near {warmest['lat']:.2f}N, {warmest['lon']:.2f}E. This is OceanEmbed's synthetic demo output.", "source": "OceanEmbed /heatmap", "query": {"depth": depth}}
    profile = predict_profile(*point, date.today())
    index = profile["depths"].index(depth)
    return {"answer": f"At {depth} m near {point[0]:.2f}N, {point[1]:.2f}E, OceanEmbed predicts {profile['temperatures'][index]:.1f} C +/- {profile['uncertainties'][index]:.1f} C. The value is from the model endpoint, not the language model.", "source": "OceanEmbed /predict", "query": {"lat": point[0], "lon": point[1], "depth": depth}}
