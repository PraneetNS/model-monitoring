import pandas as pd
from pathlib import Path

# -------- CONFIG --------
INPUT_CSV = "data/Loan_default.csv"
OUTPUT_DIR = Path("data")
SPLIT_RATIO = 0.7
TARGET_COLUMN = None  # set if you want to enforce target position
RANDOM_STATE = 42
SIMULATE_DRIFT = True
# ------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Shuffle (important if dataset is ordered)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Optional: move target column to the end
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    cols = [c for c in df.columns if c != TARGET_COLUMN] + [TARGET_COLUMN]
    df = df[cols]

# Split
split_idx = int(len(df) * SPLIT_RATIO)
reference_df = df.iloc[:split_idx].copy()
current_df = df.iloc[split_idx:].copy()

# Optional: simulate drift
if SIMULATE_DRIFT:
    numeric_cols = reference_df.select_dtypes(include="number").columns
    for col in numeric_cols[:2]:  # drift first 1–2 numeric features
        current_df[col] *= 1.2

# Save
reference_df.to_csv(OUTPUT_DIR / "reference.csv", index=False)
current_df.to_csv(OUTPUT_DIR / "current.csv", index=False)

print("✔ reference.csv and current.csv created")
print("Reference shape:", reference_df.shape)
print("Current shape:", current_df.shape)
