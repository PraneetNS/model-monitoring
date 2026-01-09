"""
Metrics Collector

Collects and calculates various performance metrics for ML models.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class MetricsCollector:
    """
    Collects and calculates model performance metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize metrics collector.

        Args:
            logger: Optional logger instance. If not provided, a module-level
                    logger will be created.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            task_type: Type of task ('classification' or 'regression').
                      If None, will be inferred from data.
            
        Returns:
            Dictionary of calculated metrics
        """
        # Infer task type if not provided
        if task_type is None:
            # Check if predictions are discrete (classification)
            unique_preds = len(np.unique(y_pred))
            unique_true = len(np.unique(y_true))
            if unique_preds <= 20 and unique_true <= 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        metrics = {}
        
        if task_type == 'classification':
            metrics = self._classification_metrics(y_true, y_pred)
        elif task_type == 'regression':
            metrics = self._regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return metrics

    def log_batch_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        batch_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Calculate and log batch-level classification metrics.

        Computes accuracy, precision, recall, and F1 score for a batch of
        predictions and logs them using the configured logger.

        Args:
            y_true: True labels for the batch.
            y_pred: Predicted labels for the batch.
            batch_id: Optional identifier for the batch (e.g., timestamp, index).
            extra: Optional extra context to include in the log record.

        Returns:
            Dictionary with the computed metrics.
        """
        metrics = self._classification_metrics(y_true, y_pred)

        log_context: Dict[str, Any] = {
            "batch_id": batch_id,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        }
        if extra:
            log_context.update(extra)

        # Log as a single structured line (use INFO level by default)
        self.logger.info(
            "Batch metrics: batch_id=%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            log_context.get("batch_id"),
            log_context["accuracy"],
            log_context["precision"],
            log_context["recall"],
            log_context["f1_score"],
            extra={"metrics": log_context},
        )

        return metrics
    
    def _classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Try to calculate ROC AUC if binary classification
        if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                pass
        
        return metrics
    
    def _regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2_score': float(r2_score(y_true, y_pred)),
        }
