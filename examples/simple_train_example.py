"""
Simple Example: Train, Save Model, and Store Reference Statistics

This is a minimal example showing how to:
1. Train a scikit-learn classification model
2. Save it with joblib
3. Store reference feature statistics
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def train_and_save_model():
    """Train a model and save it with reference statistics."""
    
    # Step 1: Prepare data
    print("Generating training data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    
    print(f"Data shape: {X_df.shape}")
    print(f"Target distribution:\n{y_series.value_counts()}\n")
    
    # Step 2: Train model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_df, y_series)
    print("✓ Model trained successfully\n")
    
    # Step 3: Calculate reference feature statistics
    print("Calculating reference feature statistics...")
    reference_stats = {}
    
    for col in X_df.columns:
        reference_stats[col] = {
            'mean': float(X_df[col].mean()),
            'std': float(X_df[col].std()),
            'min': float(X_df[col].min()),
            'max': float(X_df[col].max()),
            'median': float(X_df[col].median()),
        }
    
    # Add target statistics
    reference_stats['target'] = {
        'class_distribution': y_series.value_counts().to_dict(),
        'n_classes': int(y_series.nunique()),
    }
    
    print("✓ Statistics calculated\n")
    
    # Step 4: Save model with joblib
    model_path = "models/trained_model.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("✓ Model saved\n")
    
    # Step 5: Save reference statistics
    stats_path = "models/reference_stats.json"
    print(f"Saving statistics to {stats_path}...")
    with open(stats_path, 'w') as f:
        json.dump(reference_stats, f, indent=2)
    print("✓ Statistics saved\n")
    
    # Display summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Statistics saved: {stats_path}")
    print(f"\nReference statistics for first 3 features:")
    for i, (feature, stats) in enumerate(list(reference_stats.items())[:3]):
        if feature != 'target':
            print(f"\n  {feature}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print(f"\n  Target class distribution:")
    for cls, count in reference_stats['target']['class_distribution'].items():
        print(f"    Class {cls}: {count}")


if __name__ == "__main__":
    train_and_save_model()
