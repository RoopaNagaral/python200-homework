import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


# --- scikit-learn API ---
# Q1 - 

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)
#predicted = model.predict(temp_min)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Low 6:", model.predict([[6]])[0])
print("Low 18:", model.predict([[18]])[0])


# Q2 -
x = np.array([10, 20, 30, 40, 50])
print("\nOriginal:", x.shape)

x_2d = x.reshape(-1,1)
print("X_2d shape:", x_2d.shape)

"""
Subjective Comment: cikit-learn expects X as a 2D array of shape (n_samples, n_features). 
Each row is one sample and each column is one feature, so a single feature must still be a column rather than 
a flat 1D array. A 1D array raises an error when passed to .fit() or .predict().
"""

# --- Linear Regression ---
# Q1

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)



plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily low (C)")
plt.ylabel("Daily high (C)")

img_path_folder = Path(__file__).parent / "outputs"
img_path = img_path_folder / "high_vs_low.png"
plt.savefig(img_path)
plt.show()

"""
Subjective Comment: the student should notice two visible bands. Rainy days form a lower band 
and dry days form a higher band at the same low temperature. This suggests is_rainy carries real 
information about the high and is worth adding as a feature.
"""

# Q2
X = temp_min.reshape(-1,1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("\nShape of all 4 arrays")
print("X_Train", X_train.shape)
print("X_test", X_test.shape)
print("Y_Train", y_train.shape)
print("Y_test", y_test.shape)

# Q3
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R²:", r2)

"""
Subjective Comment: Slope interpretation: the predicted daily high rises by about one degree for each 
additional degree of the daily low. Any interpretation near "one degree of high per degree of low" is correct.
"""

# Q4
X_full = np.column_stack([temp_min, is_rainy])

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
y_full_pred = model_full.predict(X_full_test)

r2_full = model_full.score(X_full_test, y_full_test)

print("\nCompare r2 from Q2 and Q3:")
print("R2:", r2)
print("R2 full:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

"""
Subjective Comment: is_rainy coefficient interpretation: a rainy day is predicted to be about 3 degrees cooler than 
a dry day with the same low temperature, holding the low constant. Accept any answer that reads 
the negative sign as "rain lowers the predicted high" and connects the size to a few degrees.
"""

# Q5
plt.scatter(y_pred, y_test, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.show()
img_path = img_path_folder / "predicted_vs_actual_warmup.png"
plt.savefig(img_path)

"""
Subjective Comment: a point above the diagonal means the true high is greater than the prediction, 
so the model under-predicted that day. A point below the diagonal means the model over-predicted.
"""