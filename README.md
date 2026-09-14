# OceanEmbed

OceanEmbed is a hackathon MVP for SIH26066, Ministry of Earth Sciences, India. It predicts a 15-level subsurface ocean temperature profile from surface satellite variables across the Bay of Bengal.

The first demo runs on deterministic synthetic observations so it is immediately usable without credentials or large datasets. The modular data layer is shaped for Xarray + Dask processing of Copernicus Marine NetCDF/Zarr products, while the compose stack includes PostGIS for the next persistence step.

## Stack

- **Backend:** FastAPI, Uvicorn, NumPy, Xarray, Dask, optional `copernicusmarine`
- **Frontend:** React + TypeScript + Vite, Tailwind CSS, Leaflet, Plotly
- **Infrastructure:** Docker Compose, PostgreSQL + PostGIS

## Run locally

1. Copy `.env.example` to `.env` and add `LLM_API_KEY` only if you configure an LLM provider.
2. Start the full stack:

   ```bash
   docker-compose up --build
   ```

3. Open http://localhost:8080. API docs are at http://localhost:8000/docs.

For fast backend-only iteration: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`. For frontend-only iteration: `cd frontend && npm install && npm run dev`.

## API

- `POST /predict` with `{ "lat": 13.08, "lon": 80.27, "date": "2026-09-14" }` returns depths, temperatures, uncertainty, and variable attention weights.
- `GET /heatmap?date=2026-09-14&depth=200` returns a color-map-ready grid over the supported region.
- `GET /export?lat=13.08&lon=80.27&date=2026-09-14` downloads a CSV profile.
- `POST /agent` with `{ "message": "What's the temperature at 200m near Chennai today?" }` queries the real prediction/heatmap functions and summarizes their output.
- `GET /health` reports service status.

Coordinates are validated against the MVP region: 5-23°N, 80-100°E. Supported depth levels are 0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, and 1000m.

## Model and data roadmap

`backend/app/model.py` is the inference boundary for the Multi-Modal Ocean Encoder: surface fusion, temporal-window signal, depth-aware decoder, smooth profile behavior, uncertainty, and interpretability are represented in the demo contract. Replace that implementation with trained ConvNeXt-style and attention modules without changing the API. `backend/app/data.py` contains the Xarray/Dask surface-window boundary and the Copernicus integration point. PostGIS can be added behind this same repository boundary for spatial point queries.

No GLORYS comparison is included in the product UI; OceanEmbed presents its own predictions only.
