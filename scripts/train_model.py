import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Load cleaned reference data
# --------------------------------------------------
df = pd.read_csv("data/loan_ref_stable.csv")

# 🔑 KEEP ONLY NUMERIC COLUMNS
df = df.select_dtypes(include=["number"])

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Training data shape:", X.shape)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

# --------------------------------------------------
# Save model
# --------------------------------------------------
joblib.dump(model, "model_numeric.pkl")

print("✅ Numeric-only model saved as model_numeric.pkl")
