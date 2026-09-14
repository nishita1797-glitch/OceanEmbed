"""Natural-language assistant with optional LLM enhancement."""
import re
from datetime import date
from .model import predict_profile, heatmap

CITIES = {"chennai": (13.08, 80.27), "kolkata": (22.57, 88.36), "visakhapatnam": (17.69, 83.22), "chittagong": (22.36, 91.78), "port blair": (11.62, 92.73)}


def answer_query(message: str) -> dict:
    text = message.lower()
    depth_match = re.search(r"(\d{1,4})\s*m", text)
    depth = int(depth_match.group(1)) if depth_match else 200
    depth = min((0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000), key=lambda item: abs(item - depth))
    point = next((coords for city, coords in CITIES.items() if city in text), (15.0, 88.0))
    if "warmest" in text or "where" in text:
        grid = heatmap(date.today(), depth)
        warmest = max(grid["points"], key=lambda item: item["temperature"])
        return {"answer": f"The warmest modeled point at {depth} m is {warmest['temperature']:.1f} C near {warmest['lat']:.2f}N, {warmest['lon']:.2f}E. This is OceanEmbed's synthetic demo output.", "source": "OceanEmbed /heatmap", "query": {"depth": depth}}
    profile = predict_profile(*point, date.today())
    index = profile["depths"].index(depth)
    return {"answer": f"At {depth} m near {point[0]:.2f}N, {point[1]:.2f}E, OceanEmbed predicts {profile['temperatures'][index]:.1f} C +/- {profile['uncertainties'][index]:.1f} C. The value is from the model endpoint, not the language model.", "source": "OceanEmbed /predict", "query": {"lat": point[0], "lon": point[1], "depth": depth}}
