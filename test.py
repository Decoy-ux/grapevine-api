import joblib
import pandas as pd

model = joblib.load("grapevine_rf_visible_tuned.joblib")

features = [
    'r_400_409', 'r_410_419', 'r_420_429', 'r_430_439',
    'r_440_449', 'r_450_459', 'r_460_469', 'r_470_479',
    'r_480_489', 'r_490_499', 'r_500_509', 'r_510_519',
    'r_520_529', 'r_530_539', 'r_540_549', 'r_550_559',
    'r_560_569', 'r_570_579', 'r_580_589', 'r_590_599',
    'r_600_609', 'r_610_619', 'r_620_629', 'r_630_639',
    'r_640_649', 'r_650_659', 'r_660_669', 'r_670_679',
    'r_680_689', 'r_690_699'
]

# Temporary test spectrum
data = {feature: 0.5 for feature in features}

X = pd.DataFrame([data], columns=features)

prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0]

print("Prediction:", prediction)
print("Healthy probability:", probability[0])
print("Symptomatic probability:", probability[1])