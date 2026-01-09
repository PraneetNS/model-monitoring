"""
Advanced Example: Custom Alerts and Multiple Drift Detectors

This example demonstrates advanced features including custom alerts
and comparing different drift detection methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from ml_monitor import ModelMonitor
from ml_monitor.drift_detection import StatisticalDriftDetector, MLBasedDriftDetector
from ml_monitor.monitoring.alerts import AlertManager, AlertThresholds


def main():
    """Run advanced monitoring example."""
    # Generate regression data
    print("Generating regression data...")
    X_ref, y_ref = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        noise=10,
        random_state=42
    )
    
    X_curr, y_curr = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=5,
        noise=15,  # Higher noise = potential drift
        random_state=123
    )
    
    reference_data = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(8)])
    reference_targets = pd.Series(y_ref)
    
    current_data = pd.DataFrame(X_curr, columns=[f"feature_{i}" for i in range(8)])
    current_targets = pd.Series(y_curr)
    
    # Train model
    print("Training regression model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(reference_data, reference_targets)
    
    # Compare different drift detectors
    detectors = {
        'KS Test': StatisticalDriftDetector(threshold=0.05, method="ks_test"),
        'PSI': StatisticalDriftDetector(threshold=0.05, method="psi"),
        'ML-Based': MLBasedDriftDetector(threshold=0.6),
    }
    
    # Set up alert manager with configured thresholds
    # All threshold logic lives in AlertManager - examples only configure via API
    thresholds = AlertThresholds(
        prediction_mean_threshold=1.5  # Configure threshold, but logic is in AlertManager
    )
    alert_manager = AlertManager(thresholds=thresholds)
    
    print("\n" + "="*60)
    print("COMPARING DRIFT DETECTION METHODS")
    print("="*60)
    
    for name, detector in detectors.items():
        print(f"\n--- {name} ---")
        
        monitor = ModelMonitor(
            model=model,
            drift_detector=detector,
            alert_manager=alert_manager
        )
        
        monitor.set_reference(reference_data, reference_targets)
        results = monitor.monitor(current_data, current_targets)
        
        drift = results['drift']
        print(f"Drift Detected: {drift['drift_detected']}")
        print(f"Drift Score: {drift['drift_score']:.4f}")
        
        if 'feature_results' in drift:
            print(f"Features with drift: {sum(1 for r in drift['feature_results'].values() if r['drift_detected'])}")
        
        print(f"Alerts: {len(results['alerts'])}")
        for alert in results['alerts']:
            print(f"  [{alert['level']}] {alert['type']}: {alert['message']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
