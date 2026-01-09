"""
Model Monitor

Main class for monitoring ML model performance and detecting issues.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml_monitor.drift_detection import DriftDetector
from ml_monitor.monitoring.alerts import AlertManager
from ml_monitor.monitoring.metrics import MetricsCollector


class ModelMonitor:
    def __init__(
        self,
        model,
        drift_detector: Optional[DriftDetector] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Args:
            model: Trained ML model or sklearn Pipeline
            drift_detector: Drift detector instance
            metrics_collector: Metrics collector
            alert_manager: Alert manager
        """
        self.model = model
        self.drift_detector = drift_detector
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.alert_manager = alert_manager or AlertManager()

        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_signal: Optional[np.ndarray] = None
        self.monitoring_history = []

    # --------------------------------------------------
    # Internal prediction helper
    # --------------------------------------------------
    def _predict(self, X: pd.DataFrame):
        preds = self.model.predict(X)

        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
            except Exception:
                proba = None
        elif hasattr(self.model, "decision_function"):
            try:
                scores = self.model.decision_function(X)
                proba = np.atleast_2d(scores)
            except Exception:
                proba = None

        return np.asarray(preds), proba

    # --------------------------------------------------
    # Reference setup
    # --------------------------------------------------
    def set_reference(
        self,
        reference_data: pd.DataFrame,
        reference_targets: Optional[pd.Series] = None,
    ):
        """
        Sets reference data and computes baseline prediction signal.
        """
        self.reference_data = reference_data.copy()

        preds, proba = self._predict(reference_data)

        if proba is not None:
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                self.reference_signal = proba[:, 1]
            else:
                self.reference_signal = np.max(proba, axis=1)
        else:
            self.reference_signal = preds

        if self.drift_detector:
            self.drift_detector.fit(reference_data)

    # --------------------------------------------------
    # Monitoring
    # --------------------------------------------------
    def monitor(
        self,
        current_data: pd.DataFrame,
        current_targets: Optional[pd.Series] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs monitoring on current data.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        preds, proba = self._predict(current_data)

        if proba is not None:
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                signal = proba[:, 1]
                signal_type = "probability"
            else:
                signal = np.max(proba, axis=1)
                signal_type = "confidence"
        else:
            signal = preds
            signal_type = "label"

        # Drift detection
        if self.drift_detector:
            results["drift"] = self.drift_detector.detect(current_data)

        # Performance metrics
        if current_targets is not None:
            results["performance"] = self.metrics_collector.calculate_metrics(
                y_true=current_targets,
                y_pred=preds,
            )

        # Prediction statistics
        results["predictions"] = {
            "type": signal_type,
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
        }

        # Prediction shift
        if self.reference_signal is not None:
            results["prediction_shift"] = float(
                np.mean(signal) - np.mean(self.reference_signal)
            )

        # Alerts
        results["alerts"] = self.alert_manager.check_alerts(results)

        self.monitoring_history.append(results)
        return results

    # --------------------------------------------------
    # History utilities
    # --------------------------------------------------
    def get_history(self, limit: Optional[int] = None):
        if limit:
            return self.monitoring_history[-limit:]
        return self.monitoring_history

    def export_history(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.monitoring_history, f, indent=2, default=str)
