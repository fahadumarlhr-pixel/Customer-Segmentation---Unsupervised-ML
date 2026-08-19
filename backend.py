from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Creating FastAPI app :-
app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments using K-Means clustering",
    version="1.0"
)


# Loading trained model :-
kmeans = joblib.load("kmeans_model.pkl")

# Loading scaler :-
scaler = joblib.load("scaler.pkl")

# Loading feature names :-
features = joblib.load("features.pkl")


# Creating input structure :-
class CustomerData(BaseModel):
    Age: float
    Annual_Income: float
    Spending_Score: float


# Home :-
@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running"
    }


# Prediction :-
@app.post("/predict")
def predict_cluster(data: CustomerData):
    # Creating input data :-
    input_data = np.array([[
        data.Age,
        data.Annual_Income,
        data.Spending_Score
    ]])

    # Scaling input data :-
    input_scaled = scaler.transform(input_data)

    # Predicting cluster :-
    cluster = kmeans.predict(input_scaled)[0]

    # Returning prediction :-
    return {
        "cluster": int(cluster)
    }