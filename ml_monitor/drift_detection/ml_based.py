"""
ML-Based Drift Detection Methods

Uses machine learning models to detect data drift.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ml_monitor.drift_detection.detector import (
    BaseDriftDetector,
    DriftDetectionResult,
    DriftDetector,
)


class MLBasedDriftDetector(DriftDetector, BaseDriftDetector):
    """
    ML-based drift detector using a classifier to distinguish reference vs current data.
    
    If a classifier can easily distinguish between reference and current data,
    it indicates significant drift.
    """
    
    def __init__(self, threshold: float = 0.6, model=None):
        """
        Initialize ML-based drift detector.
        
        Args:
            threshold: AUC threshold above which drift is detected
            model: Sklearn classifier (default: RandomForestClassifier)
        """
        super().__init__(threshold)
        self.model = model or RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.fitted_model: Optional[RandomForestClassifier] = None
    
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector on reference data.

        Args:
            reference_data: Baseline data to use as reference
        """
        self.reference_data = reference_data.copy()
        self.is_fitted = True

    def _detect_with_fitted_current(
        self, current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Core ML-based detection logic assuming the detector has already been fitted.

        This preserves the original detection behavior and is used by both
        the legacy single-argument detect() API and the new BaseDriftDetector
        two-argument API.
        """
        self.check_fitted()

        # Prepare data
        reference = self.reference_data.copy()
        current = current_data.copy()

        # Ensure same columns
        common_cols = list(set(reference.columns) & set(current.columns))
        reference = reference[common_cols]
        current = current[common_cols]

        # Create labels: 0 for reference, 1 for current
        X_ref = reference.values
        X_curr = current.values
        y_ref = np.zeros(len(X_ref))
        y_curr = np.ones(len(X_curr))

        # Combine and shuffle
        X = np.vstack([X_ref, X_curr])
        y = np.hstack([y_ref, y_curr])

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Train classifier
        self.fitted_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
        )
        self.fitted_model.fit(X, y)

        # Calculate AUC score
        y_pred_proba = self.fitted_model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_pred_proba)

        # Drift is detected if AUC is above threshold
        # High AUC means the model can easily distinguish the datasets
        drift_detected = auc_score > self.threshold

        # Feature importance
        feature_importance = dict(
            zip(common_cols, self.fitted_model.feature_importances_)
        )

        return {
            'drift_detected': drift_detected,
            'drift_score': auc_score,
            'auc_score': auc_score,
            'feature_importance': feature_importance,
            'method': 'ml_based',
        }

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: Optional[pd.DataFrame] = None,
    ):
        """
        Detect drift using an ML classifier.

        Overloaded behavior:
        - detect(current_data): legacy API from DriftDetector; expects fit() has
          been called and returns the original dictionary result.
        - detect(reference_data, current_data): new BaseDriftDetector API that
          fits on reference_data and returns a DriftDetectionResult object.
        """
        # Legacy usage: detect(current_data) with reference already fitted.
        if current_data is None:
            return self._detect_with_fitted_current(reference_data)

        # New unified API: detect(reference_data, current_data)
        self.fit(reference_data)
        raw = self._detect_with_fitted_current(current_data)

        return DriftDetectionResult(
            drift_detected=bool(raw.get('drift_detected', False)),
            drift_score=float(raw.get('drift_score', 0.0)),
            method=str(raw.get('method', 'ml_based')),
            details=raw,
        )
