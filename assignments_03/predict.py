import joblib
import json
import pandas as pd
from pathlib import Path

# Task 7: Load and Predict

model_folder = Path(__file__).parent / "models"
model_path = model_folder / "weather_classifier.pkl"
clf = joblib.load(model_path)
json_path = model_folder / "weather_classifier_metadata.json"
with open(json_path, "r") as f:
    metadata = json.load(f)
    
# Build new days using the exact feature names and order from the metadata
new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
        {"temperature_2m_max": 22, "temperature_2m_min": 13, "precipitation_sum": 9.0, "wind_speed_10m_max": 14},
        {"temperature_2m_max": 29, "temperature_2m_min": 15, "precipitation_sum": 10.0, "wind_speed_10m_max": 33},
        {"temperature_2m_max": 26, "temperature_2m_min": 0, "precipitation_sum": 0.1, "wind_speed_10m_max": 30},
    ]
)[metadata["feature_names"]]

print("\nCity:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test accuracy:", metadata["test_accuracy"])

preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i + 1}: {label}  (confidence: {prob:.2f})")
    
    
# Task 8: Reflect
"""
1. Borderline day (0.52):
 A probability of 0.52 is uncertain, not confident. The model predicts "good", but I would treat it as a borderline case and review it carefully.

2. Running predict.py before training:
 It would fail because weather_classifier.pkl does not exist yet, causing a FileNotFoundError. A better message would be:

Python
"Model not found. Run train_classifier.py first."

3. Cloud pipeline change:
 Replace manual user input with automatic input from a file, database, or API so predictions can run without human interaction.
"""