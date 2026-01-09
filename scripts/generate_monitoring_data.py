import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load base dataset
# -----------------------------
df = pd.read_csv("data/Loan_default.csv")

TARGET_COL = df.columns[-1]

# -----------------------------
# Scenario 1: Stable (no drift)
# -----------------------------
ref_stable = df.sample(frac=0.7, random_state=1)
cur_stable = df.drop(ref_stable.index)

ref_stable.to_csv(OUTPUT_DIR / "loan_ref_stable.csv", index=False)
cur_stable.to_csv(OUTPUT_DIR / "loan_cur_stable.csv", index=False)

# -----------------------------
# Scenario 2: Feature drift
# -----------------------------
ref_drift = df.sample(frac=0.7, random_state=2)
cur_drift = df.drop(ref_drift.index).copy()

cur_drift["Age"] += 5
cur_drift["Income"] *= 1.25
cur_drift["CreditScore"] -= 30

ref_drift.to_csv(OUTPUT_DIR / "loan_ref_drift.csv", index=False)
cur_drift.to_csv(OUTPUT_DIR / "loan_cur_drift.csv", index=False)

# -----------------------------
# Scenario 3: Performance degradation
# -----------------------------
ref_perf = df.sample(frac=0.7, random_state=3)
cur_perf = df.drop(ref_perf.index).copy()

# Flip labels for 30% of rows
flip_idx = cur_perf.sample(frac=0.3, random_state=4).index
cur_perf.loc[flip_idx, TARGET_COL] = 1 - cur_perf.loc[flip_idx, TARGET_COL]

ref_perf.to_csv(OUTPUT_DIR / "loan_ref_perf.csv", index=False)
cur_perf.to_csv(OUTPUT_DIR / "loan_cur_perf.csv", index=False)

# -----------------------------
# Scenario 4: No labels (production-like)
# -----------------------------
ref_nolabel = ref_stable.copy()
cur_nolabel = cur_stable.drop(columns=[TARGET_COL])

ref_nolabel.to_csv(OUTPUT_DIR / "loan_ref_nolabel.csv", index=False)
cur_nolabel.to_csv(OUTPUT_DIR / "loan_cur_nolabel.csv", index=False)

print("✅ Monitoring datasets generated successfully")
