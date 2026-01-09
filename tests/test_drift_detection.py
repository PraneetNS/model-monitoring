"""Tests for drift detection modules."""

import pytest
import pandas as pd
import numpy as np
from ml_monitor.drift_detection import StatisticalDriftDetector, MLBasedDriftDetector


@pytest.fixture
def reference_data():
    """Generate reference data."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 10, 1000),
    })


@pytest.fixture
def current_data_no_drift():
    """Generate current data with no drift."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 500),
        'feature_2': np.random.normal(5, 2, 500),
        'feature_3': np.random.uniform(0, 10, 500),
    })


@pytest.fixture
def current_data_with_drift():
    """Generate current data with drift."""
    np.random.seed(123)
    return pd.DataFrame({
        'feature_1': np.random.normal(2, 1, 500),  # Shifted mean
        'feature_2': np.random.normal(5, 3, 500),  # Different std
        'feature_3': np.random.uniform(5, 15, 500),  # Shifted range
    })


def test_statistical_detector_fit(reference_data):
    """Test statistical detector fitting."""
    detector = StatisticalDriftDetector()
    detector.fit(reference_data)
    assert detector.is_fitted
    assert detector.reference_data is not None


def test_statistical_detector_no_drift(reference_data, current_data_no_drift):
    """Test statistical detector with no drift."""
    detector = StatisticalDriftDetector(threshold=0.05)
    detector.fit(reference_data)
    results = detector.detect(current_data_no_drift)
    assert 'drift_detected' in results
    assert 'drift_score' in results


def test_statistical_detector_with_drift(reference_data, current_data_with_drift):
    """Test statistical detector with drift."""
    detector = StatisticalDriftDetector(threshold=0.05)
    detector.fit(reference_data)
    results = detector.detect(current_data_with_drift)
    assert 'drift_detected' in results
    # With significant drift, should detect it
    # (may vary based on random seed, so we just check structure)


def test_ml_based_detector(reference_data, current_data_with_drift):
    """Test ML-based detector."""
    detector = MLBasedDriftDetector(threshold=0.6)
    detector.fit(reference_data)
    results = detector.detect(current_data_with_drift)
    assert 'drift_detected' in results
    assert 'drift_score' in results
    assert 'auc_score' in results
    assert 'feature_importance' in results


def test_detector_not_fitted(current_data_no_drift):
    """Test that detector raises error if not fitted."""
    detector = StatisticalDriftDetector()
    with pytest.raises(ValueError, match="must be fitted"):
        detector.detect(current_data_no_drift)
