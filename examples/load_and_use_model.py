"""
Example: Load Saved Model and Use Reference Statistics

This script shows how to:
1. Load a saved model using joblib
2. Load reference statistics
3. Use them for monitoring/drift detection
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.datasets import make_classification

from ml_monitor import ModelMonitor
from ml_monitor.drift_detection import StatisticalDriftDetector


def load_model_and_stats(model_path: str, stats_path: str):
    """Load model and statistics from files."""
    # Load model
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load statistics
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"✓ Statistics loaded from {stats_path}\n")
    
    return model, stats


def main():
    """Demonstrate loading and using saved model."""
    
    model_path = "models/trained_model.pkl"
    stats_path = "models/reference_stats.json"
    
    # Check if files exist
    if not Path(model_path).exists() or not Path(stats_path).exists():
        print("ERROR: Model or statistics file not found!")
        print("Please run 'python examples/simple_train_example.py' first.")
        return
    
    # Load model and statistics
    print("="*60)
    print("LOADING SAVED MODEL AND STATISTICS")
    print("="*60)
    model, reference_stats = load_model_and_stats(model_path, stats_path)
    
    # Display loaded statistics
    print("Reference Statistics Summary:")
    print("-" * 60)
    for feature, stats in list(reference_stats.items())[:3]:
        if feature != 'target':
            print(f"\n{feature}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Generate some test data (simulating production data)
    print("\n" + "="*60)
    print("TESTING MODEL ON NEW DATA")
    print("="*60)
    
    # Create test data with same structure
    X_test, y_test = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=123  # Different seed = different distribution
    )
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    # Make predictions
    predictions = model.predict(X_test_df)
    print(f"\nPredictions made: {len(predictions)}")
    print(f"Prediction distribution:\n{pd.Series(predictions).value_counts()}")
    
    # Set up monitoring with drift detection
    print("\n" + "="*60)
    print("SETTING UP MONITORING")
    print("="*60)
    
    # We need reference data for drift detection
    # In practice, you'd load this from your training data
    # For demo, we'll recreate it
    X_ref, y_ref = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42  # Same as training
    )
    X_ref_df = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(X_ref.shape[1])])
    
    # Initialize drift detector and monitor
    drift_detector = StatisticalDriftDetector(threshold=0.05)
    monitor = ModelMonitor(model=model, drift_detector=drift_detector)
    
    # Set reference data
    monitor.set_reference(X_ref_df, pd.Series(y_ref))
    print("✓ Reference data set for monitoring")
    
    # Monitor new data
    results = monitor.monitor(X_test_df, pd.Series(y_test))
    
    # Display results
    print("\n" + "="*60)
    print("MONITORING RESULTS")
    print("="*60)
    print(f"\nDrift Detected: {results['drift']['drift_detected']}")
    print(f"Drift Score: {results['drift']['drift_score']:.4f}")
    
    print(f"\nPerformance Metrics:")
    for metric, value in results['performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nAlerts: {len(results['alerts'])}")
    for alert in results['alerts']:
        print(f"  [{alert['level'].upper()}] {alert['message']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
