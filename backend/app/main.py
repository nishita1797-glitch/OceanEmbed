from datetime import date
import csv
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from .data import DEPTHS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from .model import PARAMETERS, heatmap, predict_profile
from .agent import answer_query

app = FastAPI(title="OceanEmbed API", version="0.1.0")

class StripVercelApiPrefixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.scope["path"].startswith("/api/"):
            request.scope["path"] = request.scope["path"][4:]
        return await call_next(request)

app.add_middleware(StripVercelApiPrefixMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX)
    lon: float = Field(..., ge=LON_MIN, le=LON_MAX)
    date: date
    parameter: str = "temperature"

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=500)

@app.get("/health")
def health():
    return {"status": "ok", "region": "Indian Ocean"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_profile(request.lat, request.lon, request.date, request.parameter)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@app.get("/heatmap")
def get_heatmap(date: date = Query(...), depth: int = Query(..., ge=0, le=1000), parameter: str = Query("temperature")):
    try:
        return heatmap(date, depth, parameter)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@app.get("/export")
def export_profile(lat: float = Query(..., ge=LAT_MIN, le=LAT_MAX), lon: float = Query(..., ge=LON_MIN, le=LON_MAX), date: date = Query(...), parameter: str = Query("temperature")):
    try:
        profile = predict_profile(lat, lon, date, parameter)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["latitude", "longitude", "date", "depth_m", "parameter", "value", "uncertainty"])
    writer.writerows([[lat, lon, date, depth, parameter, value, uncertainty] for depth, value, uncertainty in zip(profile["depths"], profile["temperatures"], profile["uncertainties"])])
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=oceanembed-profile.csv"})

@app.post("/agent")
def agent(request: AgentRequest):
    try:
        return answer_query(request.message)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Assistant unavailable: {error}") from error
