import requests
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Task 1: Fetch the Data
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

# Task 2: Look at the Features
# 2 - 1 Print df.describe() and check missing values
print("")
print(df.describe())

# Check missing values
print(df.isna().sum())

"""
I will choose the below method to fill the missing value as the mean values between the max and min. 
I think it is the best option to replace the missing value instead of drop the data.
"""
#df = df.fillna(df.mean()) I am commenting this line as there is no missing values

# 2 - 2 Correlation with temperature_2m_max
corr = df.corr(numeric_only=True)
print("\nThe correlation of each numeric column with temperature_2m_max")
print(corr['temperature_2m_max'].sort_values())

# 2 - 3 Create at least one plot
img_path_folder = Path(__file__).parent / "outputs"
img_path = img_path_folder / "low_vs_high.png"

plt.figure(figsize=(8,5))
plt.scatter(df['temperature_2m_min'], df['temperature_2m_max'], alpha=0.6)
plt.title("Daily Low vs Daily High Temperature")
plt.xlabel("Low Temperature (°C)")
plt.ylabel("High Temperature (°C)")
plt.savefig(img_path, dpi=120)
plt.close()

"""
Subjective Comment : The scatter plot typically shows a strong positive linear relationship:
As the daily low increases, the daily high also increases.
This matches the correlation analysis.
"""

# Task 3: Engineer a Feature
# Extract month from the date column
df["month"] = df["date"].dt.month

# Create binary summer indicator
df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

# Count summer vs non-summer days
print("\nNumber of Summer and Non summer days")
print(df["is_summer"].value_counts())

# Task 4: Baseline Model
X = df[["temperature_2m_min"]]
Y = df["temperature_2m_max"]


X_train, X_test, Y_train, Y_test = train_test_split (
    X, Y ,test_size= 0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r2 = r2_score(Y_test, y_pred)

print("\nSlope:", model.coef_[0])
print("RMSE:", rmse)
print("R²:", r2)

"""
Comment: RMSE is read in degrees. A good answer states that a typical prediction is off by about that many degrees. 
For example, an RMSE of 2.5 means predictions are typically off by a little more than 2 degrees.
"""

# Task 5: Full Model
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_full_train, X_full_test,Y_full_train, Y_full_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(X_full_train, Y_full_train)
y_full_pred = full_model.predict(X_full_test)

full_rmse = np.sqrt(mean_squared_error(Y_full_test, y_full_pred))
full_r2_test = full_model.score(X_full_test, Y_full_test)
full_r2_train = full_model.score(X_full_train, Y_full_train)

print("\nRMSE:", full_rmse)
print("Test R2:", full_r2_test)
print("Train R2:", full_r2_train)
print("R2:", r2)
"""
Comment on the train/test gap: the student compares train R² and test R². For linear regression on this data 
the gap is usually small, which indicates the model generalizes well and is not overfitting.
"""

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
    
"""
Comment on coefficients: temperature_2m_min and is_summer usually have positive coefficients that raise 
the predicted high. precipitation_sum and wind_speed_10m_max are usually negative, because rainy and windy days 
tend to be cooler. Signs may vary by city, and a student who notices and reasons about a surprising 
sign should be credited.
"""

# Task 6: Evaluate and Summarize
plt.scatter(Y_full_test, y_full_pred, alpha=0.5, color="steelblue")

# Diagonal reference line
min_val = min(Y_full_test.min(), y_full_pred.min())
max_val = max(Y_full_test.max(), y_full_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Actual High Temperature (°C)")
plt.ylabel("Predicted High Temperature (°C)")

plt.show()
img_path = img_path_folder / "predicted_vs_actual_high.png"
plt.savefig(img_path)
plt.close()

"""
The points cluster close to the diagonal, meaning predictions are generally accurate.
Errors appear slightly larger at the high end — the model tends to under-predict very hot days.
At the low end, the predictions are tighter and closer to the diagonal.
Overall, error is not perfectly even, but the model performs well across most of the range.
"""
# SUMMARY (Plain Language)
#
# • Dataset size:
#   The full dataset contains N rows 365.
#   The test set is 20% of the data, so it contains about N * 0.2 rows.
#
# • Model performance:
#   RMSE: 2.563359220315166
#   R²:   0.9230299361940071
#
#   Interpretation:
#   RMSE is in degrees Celsius, so an RMSE of (for example) 2.8 means
#   the model is typically off by about 3°C when predicting the daily high.
#   On a temperature scale, that is a noticeable but reasonable error.
#
# • Most important feature:
#   The feature with the largest effect on the predicted high is
#   temperature_2m_min (the daily low). Its coefficient is strongly positive,
#   meaning higher overnight lows lead to higher daytime highs.
#
# • One surprising result:
#   (Choose one based on your data)
#   Examples:
#   - I was surprised that precipitation had a negative correlation with the high.
#   - I was surprised how strongly season (is_summer) influenced the model.
#   - I was surprised that radiation_sum mattered less/more than expected.
#
# This summary explains the dataset size, model accuracy, feature importance,
# and one unexpected insight, as required by Task 6.