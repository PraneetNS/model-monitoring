"""
Run drift detection on saved synthetic data (no monitoring/alerts).

Prerequisite:
    python examples/generate_synthetic_data.py
"""

from pathlib import Path

import pandas as pd

from ml_monitor.drift_detection import MLBasedDriftDetector, StatisticalDriftDetector


DATA_DIR = Path("data")
REF_FEATURES = DATA_DIR / "reference_data.csv"
CURR_FEATURES = DATA_DIR / "current_data.csv"


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not REF_FEATURES.exists() or not CURR_FEATURES.exists():
        raise FileNotFoundError(
            "Data files missing. Run: python examples/generate_synthetic_data.py"
        )
    ref = pd.read_csv(REF_FEATURES)
    curr = pd.read_csv(CURR_FEATURES)
    return ref, curr


def main() -> None:
    ref, curr = load_features()
    print(f"Reference shape: {ref.shape}")
    print(f"Current shape:   {curr.shape}")

    detectors = {
        "KS Test": StatisticalDriftDetector(threshold=0.05, method="ks_test"),
        "PSI": StatisticalDriftDetector(threshold=0.05, method="psi"),
        "ML-Based": MLBasedDriftDetector(threshold=0.6),
    }

    print("\n" + "=" * 60)
    print("DRIFT DETECTION ONLY")
    print("=" * 60)

    for name, detector in detectors.items():
        print(f"\n--- {name} ---")
        detector.fit(ref)
        drift = detector.detect(curr)
        print(f"Drift Detected: {drift['drift_detected']}")
        print(f"Drift Score:    {drift['drift_score']:.4f}")
        fr = drift.get("feature_results", {})
        if fr:
            num = sum(1 for r in fr.values() if r.get("drift_detected"))
            print(f"Features with drift: {num}")


if __name__ == "__main__":
    main()
