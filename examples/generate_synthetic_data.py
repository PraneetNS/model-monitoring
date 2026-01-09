"""
Generate Synthetic Regression Data for Drift and Monitoring demos.

Saves four CSVs under data/:
- reference_data.csv
- reference_targets.csv
- current_data.csv
- current_targets.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression


def main() -> None:
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_ref, y_ref = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        noise=10,
        random_state=42,
    )
    X_curr, y_curr = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=5,
        noise=15,
        random_state=123,
    )

    ref_df = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(8)])
    ref_t = pd.Series(y_ref, name="target")
    curr_df = pd.DataFrame(X_curr, columns=[f"feature_{i}" for i in range(8)])
    curr_t = pd.Series(y_curr, name="target")

    ref_df.to_csv(DATA_DIR / "reference_data.csv", index=False)
    ref_t.to_csv(DATA_DIR / "reference_targets.csv", index=False)
    curr_df.to_csv(DATA_DIR / "current_data.csv", index=False)
    curr_t.to_csv(DATA_DIR / "current_targets.csv", index=False)

    print("Saved synthetic data to data/:")
    for f in ["reference_data.csv", "reference_targets.csv", "current_data.csv", "current_targets.csv"]:
        print(f" - {DATA_DIR / f}")


if __name__ == "__main__":
    main()
