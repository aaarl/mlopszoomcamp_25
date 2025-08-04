from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import uvicorn

# FastAPI Setup
app = FastAPI(title="Breast Cancer Classifier", version="1.0")

# Input Schema
class PatientData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float

# Load model from MLflow
print("📦 Loading model from MLflow registry...")
MODEL_URI = "runs:/9f123abc456de789/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    return {
        "prediction": int(prediction[0]),
        "result": result
    }

# Optional for standalone execution
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
