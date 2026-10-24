import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib
import os
import json
import sys
import sklearn

#Task 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 33.32,  #I have selected Celina,TX city
    "longitude": -96.78,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print("First five rows of df:")
print(df.head(5))
print("Dataset shape:")
print(df.shape)

# Task 2: Engineer the Label
df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

# Class balance
class_balance = df["good_for_running"].value_counts().sort_index()

good_days = class_balance.get(1, 0)
skip_days = class_balance.get(0, 0)

print(f"\nGood days (1): {good_days}")
print(f"Skip days (0): {skip_days}")

# Fraction of good days
good_fraction = good_days / len(df)
print(f"Fraction of good days: {good_fraction:.2%}")

"""
Class balance interpretation:
# Good days: X
# Skip days: Y
#
# Fraction of good days: Z%
#
# This fraction seems reasonable for the city's climate because
# [brief explanation based on your location and weather patterns].
#
# If the classes are highly imbalanced (for example, <20% or >80% good days),
# accuracy alone may be misleading. Metrics such as precision, recall, F1-score,
# and the confusion matrix become more important for evaluating model performance.
"""

# Task 3: Build and Evaluate a KNN Classifier
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Test at least five k values
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21]
}

# 5-fold cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\nBest k:", grid_search.best_params_["knn__n_neighbors"])
print("Best CV Accuracy:", f"{grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Task 4: Build and Evaluate a Logistic Regression Classifier
# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# Fit model
lr_pipeline.fit(X_train, y_train)

# Predictions
y_pred = lr_pipeline.predict(X_test)

# Metrics
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature coefficients
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr_pipeline.named_steps["logreg"].coef_[0]
})

coef_df = coef_df.sort_values("Coefficient")

print("\nFeature Coefficients:")
print(coef_df)

# Optional: show largest effects
print("\nMost negative coefficients (push toward 'skip'):")
print(coef_df.head())

print("\nMost positive coefficients (push toward 'good for running'):")
print(coef_df.tail())

"""
Coefficient interpretation:
#
# Positive coefficients increase the probability that a day is classified
# as good_for_running = 1.
#
# Negative coefficients increase the probability that a day is classified
# as a skip day (good_for_running = 0).
#
# The features with the most negative coefficients are the strongest
# indicators that a day should be skipped. For example:
# - Higher precipitation typically pushes predictions toward "skip".
# - Higher maximum wind speed often pushes predictions toward "skip".
# - Extremely high temperatures may also contribute to skip predictions.
#
# Features with large positive coefficients support a "good" running day.
# For example, moderate temperatures, low precipitation, and lower wind
# speeds generally increase the likelihood of a good running day.
#
# Because features were standardized before fitting, coefficient magnitudes
# can be compared directly. Larger absolute values indicate a stronger
# influence on the prediction.
"""

# Task 5: Compare and Choose
"""
# 1. The model with the higher accuracy was KNN.

# 2. For a running app, precision is important because a false positive
# means telling someone to run on a bad weather day. A false negative
# means missing a good day to run.

# 3. I would deploy Logistic Regression because because it had better performance and is
# a good balance of accuracy and usability. Logistic Regression is easier
# to interpret, while KNN may perform better if the data has complex patterns."""

# Task 6: Save the Model
model_to_save = lr_pipeline     

# Calculate metrics on the test set
y_pred = model_to_save.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
model_folder = Path(__file__).parent / "models"
model_path = model_folder / "weather_classifier.pkl"
joblib.dump(model_to_save, model_path)

# Save metadata
metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "feature_names": list(X_train.columns),
    "label": "good_for_running",
    "label_rule": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
    "city": "Celina",
    "test_accuracy": round(test_accuracy, 4),
    "test_f1": round(test_f1, 4)
}

model_path = model_folder / "weather_classifier_metadata.json"
with open(model_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nModel saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")
print("Both files written successfully.")

