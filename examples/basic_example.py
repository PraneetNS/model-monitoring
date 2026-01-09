"""
Basic Example: Using ML Monitor for Drift Detection

This example demonstrates basic usage of the ML Monitor tool.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from ml_monitor import ModelMonitor
from ml_monitor.drift_detection import StatisticalDriftDetector


def main():
    """Run basic monitoring example."""
    # Generate sample data
    print("Generating sample data...")
    X_ref, y_ref = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Introduce drift in current data
    X_curr, y_curr = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=123  # Different seed = different distribution
    )
    # Add some shift
    X_curr = X_curr + np.random.normal(0, 0.5, X_curr.shape)
    
    # Convert to DataFrames
    reference_data = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(10)])
    reference_targets = pd.Series(y_ref)
    
    current_data = pd.DataFrame(X_curr, columns=[f"feature_{i}" for i in range(10)])
    current_targets = pd.Series(y_curr)
    
    # Train a simple model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(reference_data, reference_targets)
    
    # Initialize drift detector
    drift_detector = StatisticalDriftDetector(threshold=0.05, method="ks_test")
    
    # Initialize monitor
    monitor = ModelMonitor(
        model=model,
        drift_detector=drift_detector
    )
    
    # Set reference data
    print("Setting reference data...")
    monitor.set_reference(reference_data, reference_targets)
    
    # Monitor current data
    print("Monitoring current data...")
    results = monitor.monitor(current_data, current_targets)
    
    # Print results
    print("\n" + "="*50)
    print("MONITORING RESULTS")
    print("="*50)
    print(f"\nDrift Detected: {results['drift']['drift_detected']}")
    print(f"Drift Score: {results['drift']['drift_score']:.4f}")
    
    print(f"\nPerformance Metrics:")
    for metric, value in results['performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nPrediction Statistics:")
    for stat, value in results['predictions'].items():
        print(f"  {stat}: {value:.4f}")
    
    print(f"\nAlerts: {len(results['alerts'])}")
    for alert in results['alerts']:
        print(f"  [{alert['level'].upper()}] {alert['message']}")
    
    # Get history
    history = monitor.get_history()
    print(f"\nMonitoring history: {len(history)} entries")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
