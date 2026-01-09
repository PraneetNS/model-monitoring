from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Shuffle for realism
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into reference / current
split = int(0.7 * len(df))

reference_df = df.iloc[:split]
current_df = df.iloc[split:]

# Save
reference_df.to_csv("data/reference_california.csv", index=False)
current_df.to_csv("data/current_california.csv", index=False)

print("✅ California Housing reference & current CSV created")
