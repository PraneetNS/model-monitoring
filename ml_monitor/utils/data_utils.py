"""
Data Utilities

Helper functions for data preprocessing and validation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class DataPreprocessor:
    """Utility class for preprocessing data."""
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in dataframe.
        
        Args:
            df: Input dataframe
            strategy: Strategy to use ('mean', 'median', 'mode', 'drop')
            columns: Specific columns to process (None for all)
            
        Returns:
            Processed dataframe
        """
        df = df.copy()
        cols_to_process = columns or df.columns
        
        for col in cols_to_process:
            if df[col].isna().any():
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == "drop":
                    df.dropna(subset=[col], inplace=True)
        
        return df
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """
        Normalize features in dataframe.
        
        Args:
            df: Input dataframe
            method: Normalization method ('standard' or 'minmax')
            
        Returns:
            Normalized dataframe
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        elif method == "minmax":
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
                df[numeric_cols].max() - df[numeric_cols].min()
            )
        
        return df


class DataValidator:
    """Utility class for validating data."""
    
    @staticmethod
    def validate_schema(
        df: pd.DataFrame,
        expected_columns: List[str],
        allow_extra: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate dataframe schema.
        
        Args:
            df: Dataframe to validate
            expected_columns: List of expected column names
            allow_extra: Whether to allow extra columns
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'missing_columns': [],
            'extra_columns': [],
        }
        
        missing = set(expected_columns) - set(df.columns)
        if missing:
            results['valid'] = False
            results['missing_columns'] = list(missing)
        
        if not allow_extra:
            extra = set(df.columns) - set(expected_columns)
            if extra:
                results['valid'] = False
                results['extra_columns'] = list(extra)
        
        return results
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            df: Dataframe to check
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        }
