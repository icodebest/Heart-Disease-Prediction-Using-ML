from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific URL if needed (e.g., ["http://127.0.0.1:5500"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("best_model_logistic_regression.pkl")


# Define the input schema
class PredictionInput(BaseModel):
    age: int
    gender: int
    chest_pain_type: int
    resting_bp: int
    cholesterol: int
    fasting_bs: int
    ekg_results: int
    max_hr: int
    exercise_angina: int
    st_depression: float
    st_slope: int
    num_vessels: int
    thallium: int


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to a NumPy array
        data = np.array([[
            input_data.age, input_data.gender, input_data.chest_pain_type,
            input_data.resting_bp, input_data.cholesterol, input_data.fasting_bs,
            input_data.ekg_results, input_data.max_hr, input_data.exercise_angina,
            input_data.st_depression, input_data.st_slope, input_data.num_vessels,
            input_data.thallium
        ]])

        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data).tolist() if hasattr(model, "predict_proba") else None

        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
