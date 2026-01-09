"""
Train and Save Model Script

This script demonstrates how to:
1. Train a scikit-learn classification model
2. Save the model using joblib
3. Store reference feature statistics for drift detection
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

from ml_monitor.utils.data_utils import DataValidator


def calculate_feature_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive statistics for all features.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Dictionary containing statistics for each feature
    """
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'skewness': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis()),
            'missing_count': int(df[col].isna().sum()),
            'missing_percentage': float(df[col].isna().sum() / len(df) * 100),
        }
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts().to_dict()
        stats[col] = {
            'value_counts': {str(k): int(v) for k, v in value_counts.items()},
            'unique_count': int(df[col].nunique()),
            'most_frequent': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
            'missing_count': int(df[col].isna().sum()),
            'missing_percentage': float(df[col].isna().sum() / len(df) * 100),
        }
    
    return stats


def save_model_artifacts(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: str = "models/model.pkl",
    stats_path: str = "models/reference_stats.json",
    metadata_path: str = "models/model_metadata.json",
):
    """
    Save model and reference statistics.
    
    Args:
        model: Trained scikit-learn model
        X_train: Training features DataFrame
        y_train: Training targets Series
        model_path: Path to save the model
        stats_path: Path to save feature statistics
        metadata_path: Path to save model metadata
    """
    # Create directories if they don't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model with joblib
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"✓ Model saved successfully")
    
    # Calculate and save feature statistics
    print(f"\nCalculating reference feature statistics...")
    feature_stats = calculate_feature_statistics(X_train)
    
    # Add target statistics if numeric
    if y_train.dtype in [np.number]:
        target_stats = {
            'target': {
                'mean': float(y_train.mean()),
                'std': float(y_train.std()),
                'min': float(y_train.min()),
                'max': float(y_train.max()),
                'class_distribution': y_train.value_counts().to_dict(),
            }
        }
        feature_stats.update(target_stats)
    
    print(f"Saving statistics to {stats_path}...")
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"✓ Statistics saved successfully")
    
    # Save model metadata
    metadata = {
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        'training_date': datetime.now().isoformat(),
        'n_features': len(X_train.columns),
        'feature_names': list(X_train.columns),
        'n_samples': len(X_train),
        'n_classes': len(np.unique(y_train)) if hasattr(y_train, 'unique') else None,
        'class_names': list(np.unique(y_train)) if hasattr(y_train, 'unique') else None,
    }
    
    # Add model-specific metadata
    if hasattr(model, 'n_estimators'):
        metadata['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        metadata['max_depth'] = model.max_depth
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved successfully")
    
    return feature_stats, metadata


def main():
    """Main training function."""
    print("="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)
    
    # Generate or load your data
    print("\n1. Loading/preparing data...")
    # Option 1: Use synthetic data (for demonstration)
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame for better feature tracking
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Option 2: Load your own data
    # X_df = pd.read_csv("data/train_features.csv")
    # y_series = pd.read_csv("data/train_targets.csv").squeeze()
    
    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Features: {list(X_df.columns)}")
    print(f"   Target distribution:\n{y_series.value_counts()}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Validate data quality
    print("\n3. Validating data quality...")
    validator = DataValidator()
    quality_report = validator.check_data_quality(X_train)
    print(f"   Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"   Duplicate rows: {quality_report['duplicate_rows']}")
    
    # Train model
    print("\n4. Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    print("\n   Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Save model and statistics
    print("\n6. Saving model artifacts...")
    feature_stats, metadata = save_model_artifacts(
        model=model,
        X_train=X_train,
        y_train=y_train,
        model_path="models/classification_model.pkl",
        stats_path="models/reference_stats.json",
        metadata_path="models/model_metadata.json",
    )
    
    # Display summary statistics
    print("\n7. Reference Feature Statistics Summary:")
    print("-" * 60)
    for feature, stats in list(feature_stats.items())[:5]:  # Show first 5 features
        if 'mean' in stats:  # Numeric feature
            print(f"\n{feature}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nSaved files:")
    print(f"  - Model: models/classification_model.pkl")
    print(f"  - Statistics: models/reference_stats.json")
    print(f"  - Metadata: models/model_metadata.json")
    print("\nYou can now use these files for model monitoring and drift detection.")


if __name__ == "__main__":
    main()
