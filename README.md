## ML Model Monitoring and Data Drift Detection

This project provides a small, focused toolkit for monitoring ML models in production: tracking performance, detecting data drift, and generating alerts when behaviour deviates from a reference baseline.

### Problem: Monitoring ML Models in Production

- **Inputs drift over time**: feature distributions change due to user behaviour, product changes, or upstream pipelines.
- **Targets and labels drift**: the relationship between features and targets changes, making a previously good model obsolete.
- **Performance can silently degrade**: accuracy/MSE metrics in training no longer represent production behaviour.
- **Static checks are insufficient**: you need continuous, automated evaluation on live (or recent) data.

This library standardizes how to:
- define a **reference window** (training or recent “good” data),
- compare new data windows against that reference,
- compute metrics, drift scores, and alerts in a reproducible way.

### Why Data Drift and Performance Degradation Matter

- **Data drift without label monitoring**: Even if you cannot get labels in real time, significant drift in feature distributions is a strong signal that the model may be miscalibrated.
- **Performance degradation with labels**: When labels are available (online, delayed, or batched), tracking metrics such as accuracy, F1, MSE, and R² indicates whether the model still solves the task adequately.
- **Operational risk**: Undetected drift and degradation can lead to incorrect business decisions, bad user experience, and cascading failures when downstream systems rely on model outputs.

This project is designed so that:
- drift detection and performance metrics are computed in a structured way,
- alerts are triggered automatically based on configurable thresholds,
- all evaluation logic lives in the monitoring module, not scattered throughout application code.

### How Drift Is Detected

Drift detection is implemented in `ml_monitor/drift_detection` with two main approaches:

- **Statistical methods (`StatisticalDriftDetector`)**
  - Uses **Kolmogorov–Smirnov (KS) test** for numerical features:
    - For each feature, compares reference vs current distributions.
    - Returns KS statistic and p-value.
    - Flags drift when p-value \< configured significance threshold.
  - Uses **Population Stability Index (PSI)**:
    - Bins reference and current values; computes PSI per feature.
    - Flags drift when PSI exceeds a configurable threshold.
  - Supports:
    - minimum sample size,
    - minimum number of features that must show drift,
    - confirmation over multiple windows to reduce false positives.

- **ML-based method (`MLBasedDriftDetector`)**
  - Trains a **binary classifier** to distinguish reference (class 0) vs current (class 1) samples using common feature columns.
  - Computes **AUC** of this classifier:
    - AUC \(\approx 0.5\): reference and current are similar.
    - AUC \(\gg 0.5\): classifier easily separates the two → strong evidence of drift.
  - Flags drift when AUC exceeds a configurable threshold and returns:
    - drift flag,
    - drift score (AUC),
    - feature importances from the classifier.

Both detectors expose:
- a legacy `fit(reference_df)` + `detect(current_df)` API, and
- a unified `detect(reference_df, current_df)` API returning a standardized result object.

### How Alerts Are Triggered

Alert logic lives entirely in `ml_monitor/monitoring/alerts.py`:

- **Alert levels** (`AlertLevel`):
  - `INFO`, `WARNING`, `CRITICAL` mapped to Python logging levels.
  - Can also consume custom string levels (e.g. `"error"`, `"debug"`) and map them to logging levels.

- **Threshold configuration** (`AlertThresholds`):
  - `accuracy`: minimum acceptable classification accuracy.
  - `r2_score`: minimum acceptable regression \(R^2\).
  - `prediction_shift`: absolute mean prediction shift vs reference.
  - `prediction_mean_threshold`: absolute prediction mean threshold for regression anomalies.
  - `drift_level`: alert level to use when drift is detected.

- **Alert manager** (`AlertManager`):
  - Evaluates a monitoring result dictionary produced by `ModelMonitor`.
  - Built-in checks:
    - **Data drift**: if any detector reports `drift_detected=True`, raises a `data_drift` alert.
    - **Performance degradation**:
      - accuracy \< `accuracy` → `CRITICAL` `performance_degradation` alert.
      - R² \< `r2_score` → `WARNING` `performance_degradation` alert.
    - **Prediction shift**:
      - absolute prediction mean shift \> `prediction_shift` → `WARNING` `prediction_shift` alert.
    - **Prediction mean anomaly** (regression):
      - absolute prediction mean \> `prediction_mean_threshold` → `CRITICAL` `prediction_mean_anomaly` alert.
  - Stores all alerts in an in-memory history (including timestamps).
  - Optionally logs alerts via Python `logging` at the appropriate severity.

Example of direct usage:

```python
from ml_monitor.monitoring.alerts import AlertManager, AlertThresholds

thresholds = AlertThresholds(accuracy=0.75, prediction_shift=0.3)
alert_manager = AlertManager(thresholds=thresholds)

alerts = alert_manager.check_alerts(monitoring_results)
```

### How Monitoring Works (`ModelMonitor`)

The `ModelMonitor` class in `ml_monitor/monitoring/monitor.py` orchestrates:

- **Reference setup**:
  - `set_reference(reference_data, reference_targets=None)`:
    - stores reference features and predictions,
    - fits the configured drift detector on reference data.

- **Monitoring step**:
  - `monitor(current_data, current_targets=None, metadata=None)`:
    - computes model predictions for `current_data`,
    - optionally computes performance metrics if `current_targets` is provided,
    - computes prediction statistics (mean, std, min, max),
    - computes prediction shift relative to reference predictions,
    - runs the configured drift detector,
    - calls `AlertManager.check_alerts` to produce alerts,
    - returns a result dict containing:
      - `timestamp`
      - `metadata`
      - `drift`
      - `performance`
      - `predictions`
      - `prediction_shift`
      - `alerts`
    - appends the full result to an in-memory history (`get_history`, `export_history`).

### Architecture (data → model → drift → metrics → alerts)

```
incoming data
     │
     ▼
 model.predict(X) ──► predictions
     │                   │
     │                   ├─► metrics (if y is available): accuracy, F1, MSE, R²
     │                   │
     │                   └─► prediction stats: mean, std, min, max, prediction_shift vs reference
     │
     ├─► drift detection (statistical KS/PSI or ML-based AUC)
     │       └─► drift flags + drift scores + feature-level results
     │
     └─► AlertManager (uses thresholds + levels)
             ├─ data_drift
             ├─ performance_degradation
             ├─ prediction_shift
             └─ prediction_mean_anomaly
                     │
                     ▼
                  alert history + optional logging
```

### Installation

From the project root:

```bash
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

### How to Run the Examples

From the project root (`project/`):

- **Basic monitoring + drift example**:

```bash
python examples/basic_example.py
```

This script:
- generates synthetic classification data,
- trains a classifier,
- runs statistical drift detection,
- computes metrics,
- prints alerts.

- **Train and save a model + reference statistics**:

```bash
python examples/simple_train_example.py
```

This script:
- trains a classification model,
- saves it with `joblib`,
- computes and saves reference feature statistics.

- **Load saved model and run monitoring**:

```bash
python examples/load_and_use_model.py
```

This script:
- loads a saved model and reference statistics,
- generates new data,
- sets up a `ModelMonitor` with a drift detector,
- runs monitoring and prints drift, performance metrics, and alerts.

- **Advanced drift and alert comparison**:

```bash
python examples/advanced_example.py
```

This script:
- generates regression data with controlled noise differences,
- trains a regression model,
- compares multiple drift detectors (KS, PSI, ML-based),
- configures `AlertManager` thresholds,
- runs monitoring and prints drift scores and alerts for each method.

### Project Structure (High-Level)

```
ml_monitor/
├── drift_detection/
│   ├── detector.py        # Base + standardized result types
│   ├── statistical.py     # KS / PSI-based drift detection
│   └── ml_based.py        # ML-based drift detection
├── monitoring/
│   ├── monitor.py         # ModelMonitor orchestration
│   ├── metrics.py         # Performance metrics
│   └── alerts.py          # Alert thresholds, levels, and logging
└── utils/
    ├── data_utils.py      # Data preprocessing / validation
    └── visualization.py   # Basic plotting utilities
```

Tests and additional examples are under `tests/` and `examples/`. Use them as reference implementations when integrating the toolkit into your own pipelines. 
