from fastapi import FastAPI, Request
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import time
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("grapevine_rf_visible_tuned.joblib")

FEATURES = [
    'r_400_409', 'r_410_419', 'r_420_429', 'r_430_439',
    'r_440_449', 'r_450_459', 'r_460_469', 'r_470_479',
    'r_480_489', 'r_490_499', 'r_500_509', 'r_510_519',
    'r_520_529', 'r_530_539', 'r_540_549', 'r_550_559',
    'r_560_569', 'r_570_579', 'r_580_589', 'r_590_599',
    'r_600_609', 'r_610_619', 'r_620_629', 'r_630_639',
    'r_640_649', 'r_650_659', 'r_660_669', 'r_670_679',
    'r_680_689', 'r_690_699'
]

@app.get("/")
def home():
    return {"message": "Grapevine Symptom Detection API is running"}


@app.post("/predict")
def predict(data: dict, request: Request):
    start_time = time.perf_counter()

    client_ip = request.headers.get(
    "x-forwarded-for",
    request.client.host
    ).split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "Unknown")

    X = pd.DataFrame([data], columns=FEATURES)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    result = "symptomatic" if prediction == 1 else "healthy"
    healthy_probability = float(probabilities[0])
    symptomatic_probability = float(probabilities[1])

    processing_time = (time.perf_counter() - start_time) * 1000

    print(
        f"\n"
        f"========== PREDICTION ==========\n"
        f"IP: {client_ip}\n"
        f"User-Agent: {user_agent}\n"
        f"Spectral values: {len(data)}\n"
        f"Result: {result}\n"
        f"Healthy: {healthy_probability:.2%}\n"
        f"Symptomatic: {symptomatic_probability:.2%}\n"
        f"Processing time: {processing_time:.2f} ms\n"
        f"=================================\n"
    )

    return {
        "prediction": result,
        "healthy_probability": healthy_probability,
        "symptomatic_probability": symptomatic_probability
    }
   