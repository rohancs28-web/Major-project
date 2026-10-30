from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Initialize FastAPI application for the Climate-Resilient Crop Yield Prediction project
app = FastAPI(
    title="Climate-Resilient Crop Yield Prediction API",
    description="Backend server powering spatial CNN, temporal LSTM, and XGBoost multimodal fusion inference[cite: 1].",
    version="1.0.0"
)

# Enable CORS to allow your frontend dashboard to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming prediction requests from the frontend UI
class PredictionRequest(BaseModel):
    region: str
    crop: str
    latitude: float = 28.4744   # Default coordinate reference (e.g., Greater Noida zone)
    longitude: float = 77.5040

@app.get("/")
def read_root():
    """
    Health check endpoint to verify backend status.
    """
    return {
        "status": "online",
        "project": "Climate-Resilient Crop Yield Prediction using Multimodal Remote Sensing Data[cite: 2]",
        "team": ["Rohan Kumar Singh", "Ayush Ranjan", "Sejal Sharma", "Raj Kumar Jha"][cite: 1]
    }

@app.post("/predict")
def predict_crop_yield(request: PredictionRequest):
    """
    Inference endpoint: Ingests region and crop parameters, runs the multimodal 
    pipeline (Sentinel-2 CNN + NASA LSTM + XGBoost), and returns yield forecasts[cite: 1].
    """
    try:
        # Simulated multimodal inference output (Replace with loaded model inference in production)
        # Yield values expressed in Tons per Hectare (T/ha)
        predicted_yield = round(float(np.random.uniform(6.2, 8.1)), 2)
        confidence_score = 92.4
        stress_factor = "Elevated Temperature Anomaly (+1.4°C)"
        ndvi_health = 0.78

        return {
            "status": "success",
            "region": request.region,
            "crop": request.crop,
            "metrics": {
                "projected_harvest_t_ha": predicted_yield,
                "confidence_percentage": confidence_score,
                "ndvi_health_index": ndvi_health,
                "primary_climate_stress": stress_factor
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)