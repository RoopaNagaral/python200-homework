import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
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

file_path = Path(__file__).parent / "weather_classification.csv"
#print(file_path)

df = pd.read_csv(file_path)
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column in the scaled training data
print("Column means of scaled X_train:")
print(X_train_scaled.mean(axis=0))

# Fit the scaler on the training data only to prevent information from the test set leaking into the training process.

# Q2
# One-hot encode the categorical 'season' column
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["season"]])
print("\nShape", encoded.shape)
print("Feature Names:")
print(encoder.get_feature_names_out())

# The single 'season' column became one column for each unique season category,
# because one-hot encoding creates a separate binary indicator column for every distinct category.

# Q3
df_columns = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max","season"]
X_pip = df[df_columns]
y_pip = df["good_for_running"]

X_pip_train, X_pip_test, y_pip_train, y_pip_test = train_test_split(
    X_pip, y_pip, test_size=0.2, random_state=42, stratify=y_pip
)

numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_pip_train, y_pip_train)      # fits the scaler, the encoder, AND the classifier
predictions = model.predict(X_pip_test)   # applies the same preprocessing, then predicts
accuracy = accuracy_score(y_pip_test, predictions)

print("\nTest Accuracy:", accuracy)

# The pipeline automatically applies the correct preprocessing (scaling and encoding)
# to the training and test data before fitting and predicting, so you do not have to do those steps manually.

# --- KNN ---
# Q1
knn_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)
knn_scaled.fit(X_train, y_train)
print("\nScaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))

y_pred = knn_scaled.predict(X_test)
print("Full classification report:")
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

# Q2
print("k mean Cv score:")
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

# I would choose k=11 because it had the best average validation performance across the 5 folds.

# Q3
# KNN without scaling
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)

y_pred_raw = knn_raw.predict(X_test)
raw_accuracy = accuracy_score(y_test, y_pred_raw)

# KNN with scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)
scaled_accuracy = accuracy_score(y_test, y_pred_scaled)

print("\nTest Accuracy without scaling:", raw_accuracy)
print("Test Accuracy with scaling:", scaled_accuracy)

# Scaling helped if the scaled accuracy is higher. Scaling matters for KNN because
# it uses distances between data points, and features with larger numeric ranges
# can dominate the distance calculation if the data is not scaled.
# Scaling helped because the accuracy increased from 0.925 to 0.9583333333333334.

# --- Logistic Regression ---
# Q1
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Print results
print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Q2
# Get the feature names
feature_names = X_train.columns

# Get the logistic regression coefficients
coefficients = pipeline.named_steps["logreg"].coef_[0]
print("")
# Print feature names with their coefficients
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# Features with the most negative coefficients push most strongly toward "skip"
# (assuming "good" is the positive class). This generally matches intuition if
# variables such as high wind, high humidity, or extreme temperatures make running
# conditions less comfortable and therefore increase the likelihood of skipping a run.

# Q3
# Get predicted probabilities for each class
y_prob = pipeline.predict_proba(X_test)

# Find the index of the "good" class
good_index = list(pipeline.classes_).index(1)
print("")
# Print the probability of "good" for the first five test samples
for i, prob in enumerate(y_prob[:5, good_index], start=1):
    print(f"Test day {i}: {prob:.2f}")

# A probability near 0.5 means the model is uncertain because it sees
# nearly equal evidence for both classes and is not strongly confident
# whether that day should be classified as "good" or not.

# --- Evaluation ---
# Q1
# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")

# Save the figure
img_path_folder = Path(__file__).parent / "outputs"
img_path = img_path_folder / "logreg_confusion_matrix.png"
os.makedirs("outputs", exist_ok=True)
plt.savefig(img_path, bbox_inches="tight")
plt.show()

# Extract confusion matrix values
tn, fp, fn, tp = cm.ravel()

print("\nFalse Positives:", fp)
print("False Negatives:", fn)

# False positives are days predicted as "good" when they were actually "skip".
# False negatives are days predicted as "skip" when they were actually "good".
# For a running app, false positives may matter more because the app could
# encourage a run on a poor-weather day, while a false negative mainly misses
# an opportunity to recommend a run on a suitable day.

# --- joblib ---
# Q1
model_folder = Path(__file__).parent / "models"
model_path = model_folder / "warmup_model.pkl"
joblib.dump(pipeline, model_path)
loaded = joblib.load(model_path)
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")

# If only the LogisticRegression model were saved without the scaler,
# predictions on raw data would be unreliable because the model was trained
# on scaled features and would receive input in a different scale at prediction time.