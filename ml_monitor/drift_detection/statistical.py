"""
Statistical Drift Detection Methods

Implements statistical tests for detecting data drift (KS test, PSI, etc.).
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ml_monitor.drift_detection.detector import (
    BaseDriftDetector,
    DriftDetectionResult,
    DriftDetector,
)


class StatisticalDriftDetector(DriftDetector, BaseDriftDetector):
    """
    Statistical drift detector using Kolmogorov-Smirnov test and PSI.
    
    Uses statistical tests to detect distribution shifts in numerical features.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        method: str = "ks_test",
        psi_threshold: float = 0.25,
        min_samples: int = 50,
        min_features_drift: int = 1,
        confirmation_windows: int = 1,
    ):
        """
        Initialize statistical drift detector.
        
        Args:
            threshold: Significance threshold for drift detection
            method: Method to use ('ks_test' or 'psi')
            psi_threshold: PSI value above which drift is flagged
            min_samples: Minimum rows in current_data to evaluate drift
            min_features_drift: Require at least this many features with drift
                                before raising overall drift
            confirmation_windows: Require this many consecutive drift detections
                                  (stateful) to confirm overall drift
        """
        super().__init__(threshold)
        self.method = method
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples
        self.min_features_drift = min_features_drift
        self.confirmation_windows = max(1, confirmation_windows)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self._drift_history: Deque[bool] = deque(maxlen=self.confirmation_windows)
    
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector on reference data.
        
        Args:
            reference_data: Baseline data to use as reference
        """
        self.reference_data = reference_data.copy()
        
        # Calculate statistics for each numerical feature
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'min': reference_data[col].min(),
                'max': reference_data[col].max(),
            }
        
        self.is_fitted = True
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                       bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference data distribution
            current: Current data distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate distributions
        ref_counts = pd.cut(reference, breakpoints).value_counts().sort_index()
        curr_counts = pd.cut(current, breakpoints).value_counts().sort_index()
        
        # Normalize to probabilities
        ref_probs = ref_counts / len(reference)
        curr_probs = curr_counts / len(current)
        
        # Handle zero probabilities
        ref_probs = ref_probs.replace(0, 0.0001)
        curr_probs = curr_probs.replace(0, 0.0001)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return psi
    
    def _ks_test_drift(self, reference: pd.Series, 
                       current: pd.Series) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for drift detection.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = p_value < self.threshold
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    
    def _detect_with_fitted_current(
        self, current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Core detection logic assuming the detector has already been fitted.

        This preserves the original detection behavior and is used by both
        the legacy single-argument detect() API and the new BaseDriftDetector
        two-argument API.
        """
        self.check_fitted()

        sample_count = len(current_data)
        if sample_count < self.min_samples:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'feature_results': {},
                'method': self.method,
                'reason': f'Insufficient samples ({sample_count} < {self.min_samples})',
                'samples': sample_count,
                'features_with_drift': 0,
                'min_features_required': self.min_features_drift,
                'confirmed_windows': list(self._drift_history),
                'confirmation_windows': self.confirmation_windows,
            }
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        drift_results = {}
        overall_drift_detected = False
        max_drift_score = 0.0
        features_with_drift = 0
        
        for col in numeric_cols:
            if col not in self.feature_stats:
                continue
            
            ref_data = self.reference_data[col]
            curr_data = current_data[col]
            
            if self.method == "ks_test":
                result = self._ks_test_drift(ref_data, curr_data)
                drift_score = result['statistic']
                p_value = result['p_value']
            elif self.method == "psi":
                psi = self._calculate_psi(ref_data, curr_data)
                drift_score = psi
                p_value = None
                # PSI threshold configurable
                drift_detected = psi > self.psi_threshold
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            if self.method == "ks_test":
                drift_detected = result['drift_detected']
            
            drift_results[col] = {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'p_value': p_value,
            }
            
            if drift_detected:
                overall_drift_detected = True
                features_with_drift += 1
            max_drift_score = max(max_drift_score, drift_score)

        # Apply feature-level voting rule
        if features_with_drift < self.min_features_drift:
            overall_drift_detected = False

        # Stateful confirmation across windows
        self._drift_history.append(overall_drift_detected)
        if self.confirmation_windows > 1:
            confirmed = len(self._drift_history) == self.confirmation_windows and all(
                list(self._drift_history)[-self.confirmation_windows :]
            )
        else:
            confirmed = overall_drift_detected

        return {
            'drift_detected': confirmed,
            'preliminary_drift_detected': overall_drift_detected,
            'drift_score': max_drift_score,
            'feature_results': drift_results,
            'method': self.method,
            'samples': sample_count,
            'features_with_drift': features_with_drift,
            'min_features_required': self.min_features_drift,
            'psi_threshold': self.psi_threshold if self.method == "psi" else None,
            'confirmation_windows': self.confirmation_windows,
            'confirmed_windows': list(self._drift_history),
        }

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: Optional[pd.DataFrame] = None,
    ):
        """
        Detect drift.

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
            method=str(raw.get('method', self.method)),
            details=raw,
        )
