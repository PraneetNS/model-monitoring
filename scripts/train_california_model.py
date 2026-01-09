from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import os

os.makedirs("models", exist_ok=True)

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/model_california.pkl")

print("✅ California model trained and saved")
