"""
Model Utilities

Helper functions for loading and saving models with reference statistics.
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def load_model_and_stats(
    model_path: str,
    stats_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load a saved model and its associated statistics.
    
    Args:
        model_path: Path to the saved model (.pkl file)
        stats_path: Path to reference statistics JSON file (optional)
        metadata_path: Path to model metadata JSON file (optional)
        
    Returns:
        Tuple of (model, stats_dict, metadata_dict)
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load statistics if provided
    stats = None
    if stats_path and Path(stats_path).exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    # Load metadata if provided
    metadata = None
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, stats, metadata


def save_model_with_stats(
    model: Any,
    reference_data: pd.DataFrame,
    reference_targets: Optional[pd.Series] = None,
    model_path: str = "models/model.pkl",
    stats_path: str = "models/reference_stats.json",
    metadata_path: str = "models/model_metadata.json",
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Save a model along with reference statistics.
    
    This is a convenience function that saves everything needed for monitoring.
    
    Args:
        model: Trained scikit-learn model
        reference_data: Reference feature data
        reference_targets: Reference target values (optional)
        model_path: Path to save the model
        stats_path: Path to save feature statistics
        metadata_path: Path to save model metadata
        additional_metadata: Additional metadata to include
        
    Returns:
        Tuple of (stats_dict, metadata_dict)
    """
    # Create directories
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Calculate statistics
    stats = calculate_feature_statistics(reference_data)
    
    # Add target statistics if provided
    if reference_targets is not None:
        if reference_targets.dtype in [np.number]:
            stats['target'] = {
                'mean': float(reference_targets.mean()),
                'std': float(reference_targets.std()),
                'min': float(reference_targets.min()),
                'max': float(reference_targets.max()),
            }
        stats['target']['class_distribution'] = reference_targets.value_counts().to_dict()
    
    # Save statistics
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create metadata
    metadata = {
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        'training_date': datetime.now().isoformat(),
        'n_features': len(reference_data.columns),
        'feature_names': list(reference_data.columns),
        'n_samples': len(reference_data),
    }
    
    if reference_targets is not None:
        metadata['n_classes'] = len(np.unique(reference_targets))
        metadata['class_names'] = list(np.unique(reference_targets))
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return stats, metadata


def calculate_feature_statistics(df: pd.DataFrame) -> Dict[str, Any]:
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


def compare_with_reference_stats(
    current_data: pd.DataFrame,
    reference_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare current data statistics with reference statistics.
    
    Args:
        current_data: Current data to compare
        reference_stats: Reference statistics dictionary
        
    Returns:
        Dictionary with comparison results
    """
    current_stats = calculate_feature_statistics(current_data)
    comparison = {}
    
    for feature in reference_stats.keys():
        if feature not in current_stats:
            comparison[feature] = {'status': 'missing', 'message': 'Feature not in current data'}
            continue
        
        ref_stat = reference_stats[feature]
        curr_stat = current_stats[feature]
        
        if 'mean' in ref_stat:  # Numeric feature
            mean_diff = abs(curr_stat['mean'] - ref_stat['mean'])
            std_diff = abs(curr_stat['std'] - ref_stat['std'])
            
            comparison[feature] = {
                'status': 'ok',
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'mean_shift_percentage': float(mean_diff / ref_stat['std'] * 100) if ref_stat['std'] > 0 else 0,
            }
        else:  # Categorical feature
            comparison[feature] = {
                'status': 'ok',
                'unique_count_diff': curr_stat['unique_count'] - ref_stat['unique_count'],
            }
    
    return comparison
