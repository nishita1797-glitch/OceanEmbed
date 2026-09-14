"""Natural-language assistant with optional LLM enhancement."""
import json
import os
import re
from datetime import date
from urllib.request import Request, urlopen
from .model import predict_profile, heatmap

CITIES = {
    "mumbai": (19.08, 72.88), "chennai": (13.08, 80.27), "kolkata": (22.57, 88.36),
    "colombo": (6.93, 79.86), "male": (4.18, 73.51), "mombasa": (-4.05, 39.67),
    "muscat": (23.59, 58.41), "jakarta": (-6.21, 106.85), "perth": (-31.95, 115.86),
    "visakhapatnam": (17.69, 83.22), "chittagong": (22.36, 91.78), "port blair": (11.62, 92.73),
}


def _llm_response(message: str, facts: str) -> str | None:
    """Use OpenAI only when configured; failures fall back to deterministic parsing."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    payload = json.dumps({
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Rewrite a concise answer using only the verified OceanEmbed facts supplied below. Never change or invent numbers. Mention uncertainty when present."},
            {"role": "user", "content": f"Question: {message}\nVerified OceanEmbed facts: {facts}"},
        ],
    }).encode()
    request = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=8) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"]
    except Exception:
        return None


def answer_query(message: str) -> dict:
    text = message.lower()
    depth_match = re.search(r"(\d{1,4})\s*m", text)
    depth = int(depth_match.group(1)) if depth_match else 200
    depth = min((0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000), key=lambda item: abs(item - depth))
    point = next((coords for city, coords in CITIES.items() if city in text), (0.0, 80.0))
    if "warmest" in text or "where" in text:
        grid = heatmap(date.today(), depth)
        warmest = max(grid["points"], key=lambda item: item["temperature"])
        fallback = f"The warmest modeled point at {depth} m is {warmest['temperature']:.1f} C near {warmest['lat']:.2f}N, {warmest['lon']:.2f}E. This is OceanEmbed's synthetic demo output."
        return {"answer": _llm_response(message, fallback) or fallback, "source": "OceanEmbed /heatmap", "query": {"depth": depth}}
    profile = predict_profile(*point, date.today())
    index = profile["depths"].index(depth)
    fallback = f"At {depth} m near {point[0]:.2f}N, {point[1]:.2f}E, OceanEmbed predicts {profile['temperatures'][index]:.1f} C +/- {profile['uncertainties'][index]:.1f} C. The value is from the model endpoint, not the language model."
    return {"answer": _llm_response(message, fallback) or fallback, "source": "OceanEmbed /predict", "query": {"lat": point[0], "lon": point[1], "depth": depth}}
