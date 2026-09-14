# OceanEmbed

OceanEmbed is a hackathon MVP for SIH26066, Ministry of Earth Sciences, India. It predicts a 15-level subsurface ocean temperature profile from surface satellite variables across the full Indian Ocean basin.

The first demo runs on deterministic synthetic observations so it is immediately usable without credentials or large datasets. The modular data layer is shaped for Xarray + Dask processing of Copernicus Marine NetCDF/Zarr products, while the compose stack includes PostGIS for the next persistence step. Heatmap date/depth results are cached in a bounded in-process cache.

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

## Deploy on Vercel

Deploy this as two Vercel projects; the `services` JSON format is not supported for sibling Vite and FastAPI services.

1. Import the repository twice in Vercel.
2. Frontend project: set **Root Directory** to `frontend`. Vercel uses `frontend/vercel.json` and `npm run build` automatically.
3. Backend project: set **Root Directory** to `backend`. Vercel uses `backend/api/index.py` and `backend/vercel.json` to serve FastAPI.
4. In the frontend project environment variables, set `VITE_API_URL` to the deployed backend URL, for example `https://oceanembed-api.vercel.app`.
5. Redeploy the frontend after setting the variable.

The backend's `LLM_API_KEY` is optional for the current graceful fallback agent. `DATABASE_URL` can be set to an external PostgreSQL/PostGIS service when persistence is enabled.

## API

- `POST /predict` with `{ "lat": 13.08, "lon": 80.27, "date": "2026-09-14" }` returns depths, temperatures, uncertainty, and variable attention weights.
- `GET /heatmap?date=2026-09-14&depth=200` returns a color-map-ready grid over the supported region.
- `GET /export?lat=13.08&lon=80.27&date=2026-09-14` downloads a CSV profile.
- `POST /agent` with `{ "message": "What's the temperature at 200m near Chennai today?" }` queries the real prediction/heatmap functions and summarizes their output.
- `GET /health` reports service status.

Coordinates are validated against the MVP region: 40°S-30°N, 20-130°E. Supported depth levels are 0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, and 1000m.

## Model and data roadmap

`backend/app/model.py` is the inference boundary for the Multi-Modal Ocean Encoder: surface fusion, temporal-window signal, depth-aware decoder, smooth profile behavior, uncertainty, and interpretability are represented in the demo contract. Replace that implementation with trained ConvNeXt-style and attention modules without changing the API. `backend/app/data.py` contains the Xarray/Dask surface-window boundary and the Copernicus integration point. PostGIS can be added behind this same repository boundary for spatial point queries.

No GLORYS comparison is included in the product UI; OceanEmbed presents its own predictions only.
