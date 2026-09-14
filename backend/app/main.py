from datetime import date
import csv
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .data import DEPTHS
from .model import heatmap, predict_profile
from .agent import answer_query

app = FastAPI(title="OceanEmbed API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=5, le=23)
    lon: float = Field(..., ge=80, le=100)
    date: date

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=500)

@app.get("/health")
def health():
    return {"status": "ok", "region": "Bay of Bengal"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_profile(request.lat, request.lon, request.date)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@app.get("/heatmap")
def get_heatmap(date: date = Query(...), depth: int = Query(..., ge=0, le=1000)):
    try:
        return heatmap(date, depth)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@app.get("/export")
def export_profile(lat: float = Query(..., ge=5, le=23), lon: float = Query(..., ge=80, le=100), date: date = Query(...)):
    try:
        profile = predict_profile(lat, lon, date)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["latitude", "longitude", "date", "depth_m", "temperature_c", "uncertainty_c"])
    writer.writerows([[lat, lon, date, depth, temperature, uncertainty] for depth, temperature, uncertainty in zip(profile["depths"], profile["temperatures"], profile["uncertainties"])])
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=oceanembed-profile.csv"})

@app.post("/agent")
def agent(request: AgentRequest):
    try:
        return answer_query(request.message)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Assistant unavailable: {error}") from error
