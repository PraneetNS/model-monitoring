"""Tests for monitoring modules."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml_monitor.monitoring import ModelMonitor, MetricsCollector, AlertManager
from ml_monitor.drift_detection import StatisticalDriftDetector


@pytest.fixture
def sample_model():
    """Create a simple trained model."""
    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def reference_data():
    """Generate reference data."""
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])


@pytest.fixture
def reference_targets():
    """Generate reference targets."""
    np.random.seed(42)
    return pd.Series((np.random.rand(100) > 0.5).astype(int))


def test_metrics_collector_classification():
    """Test metrics collector for classification."""
    collector = MetricsCollector()
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = collector.calculate_metrics(y_true, y_pred, task_type='classification')
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics


def test_metrics_collector_regression():
    """Test metrics collector for regression."""
    collector = MetricsCollector()
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    metrics = collector.calculate_metrics(y_true, y_pred, task_type='regression')
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2_score' in metrics


def test_alert_manager():
    """Test alert manager."""
    manager = AlertManager()
    
    # Test with drift results
    monitoring_results = {
        'drift': {
            'drift_detected': True,
            'drift_score': 0.8
        }
    }
    alerts = manager.check_alerts(monitoring_results)
    assert len(alerts) > 0
    assert any(alert['type'] == 'data_drift' for alert in alerts)


def test_model_monitor_set_reference(sample_model, reference_data, reference_targets):
    """Test setting reference data."""
    monitor = ModelMonitor(model=sample_model)
    monitor.set_reference(reference_data, reference_targets)
    assert monitor.reference_data is not None
    assert monitor.reference_predictions is not None


def test_model_monitor_monitor(sample_model, reference_data, reference_targets):
    """Test monitoring."""
    detector = StatisticalDriftDetector()
    monitor = ModelMonitor(model=sample_model, drift_detector=detector)
    monitor.set_reference(reference_data, reference_targets)
    
    current_data = pd.DataFrame(np.random.rand(50, 5), columns=[f'f{i}' for i in range(5)])
    current_targets = pd.Series((np.random.rand(50) > 0.5).astype(int))
    
    results = monitor.monitor(current_data, current_targets)
    assert 'timestamp' in results
    assert 'drift' in results
    assert 'performance' in results
    assert 'predictions' in results
    assert 'alerts' in results


def test_model_monitor_history(sample_model, reference_data, reference_targets):
    """Test monitoring history."""
    monitor = ModelMonitor(model=sample_model)
    monitor.set_reference(reference_data, reference_targets)
    
    current_data = pd.DataFrame(np.random.rand(50, 5), columns=[f'f{i}' for i in range(5)])
    
    monitor.monitor(current_data)
    monitor.monitor(current_data)
    
    history = monitor.get_history()
    assert len(history) == 2
    
    limited_history = monitor.get_history(limit=1)
    assert len(limited_history) == 1
