"""
Base Drift Detector Classes

Abstract base classes and common result type for all drift detection methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DriftDetectionResult:
    """
    Standardized drift detection result object.

    This wraps detector-specific outputs in a common structure so callers
    can rely on a consistent interface.
    """

    drift_detected: bool
    drift_score: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftResult:
    """
    Feature-level drift result.

    Attributes:
        feature_name: Name of the feature evaluated.
        score: Drift score/statistic for the feature.
        threshold: Threshold used to flag drift for the feature.
        drift_detected: Whether drift was detected for this feature.
    """

    feature_name: str
    score: float
    threshold: float
    drift_detected: bool


class BaseDriftDetector(ABC):
    """
    Common abstract base class for drift detectors with a unified detect API.

    New implementations should prefer this interface:

        detect(reference_df, current_df) -> DriftDetectionResult
    """

    @abstractmethod
    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DriftDetectionResult:
        """
        Detect drift between reference and current data.

        Args:
            reference_data: Baseline (reference) feature data.
            current_data: Current feature data to compare against reference.

        Returns:
            DriftDetectionResult: Standardized result object.
        """
        raise NotImplementedError


class DriftDetector(ABC):
    """
    Legacy abstract base class for drift detection algorithms.

    Existing detectors use a two-step API:
      1) fit(reference_data)
      2) detect(current_data)
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize the drift detector.

        Args:
            threshold: Significance threshold for drift detection.
        """
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector on reference (baseline) data.

        Args:
            reference_data: Baseline data to use as reference.
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference data.

        Args:
            current_data: Current data to check for drift.

        Returns:
            Dictionary containing drift detection results with keys:
            - 'drift_detected': bool
            - 'drift_score': float
            - 'p_value': float (if applicable)
            - 'details': dict with additional information
              (implementation-specific fields may be added).
        """
        raise NotImplementedError

    def check_fitted(self) -> None:
        """Check if the detector has been fitted."""
        if not self.is_fitted:
            raise ValueError(
                "Detector must be fitted before calling detect(). "
                "Call fit() first with reference data."
            )
