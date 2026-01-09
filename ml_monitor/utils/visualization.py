"""
Visualization Utilities

Helper functions for visualizing drift detection and monitoring results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def plot_drift_results(
    drift_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot drift detection results.
    
    Args:
        drift_results: Results from drift detector
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    if 'feature_results' not in drift_results:
        print("No feature-level results to plot.")
        return
    
    feature_results = drift_results['feature_results']
    n_features = len(feature_results)
    
    if n_features == 0:
        print("No features to plot.")
        return
    
    fig, axes = plt.subplots(1, min(3, n_features), figsize=figsize)
    if n_features == 1:
        axes = [axes]
    
    for idx, (feature, results) in enumerate(list(feature_results.items())[:3]):
        ax = axes[idx] if n_features > 1 else axes[0]
        
        drift_score = results.get('drift_score', 0)
        drift_detected = results.get('drift_detected', False)
        
        color = 'red' if drift_detected else 'green'
        ax.bar([feature], [drift_score], color=color, alpha=0.7)
        ax.set_title(f"{feature}\nDrift: {drift_detected}")
        ax.set_ylabel('Drift Score')
        ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_monitoring_history(
    history: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> None:
    """
    Plot monitoring history over time.
    
    Args:
        history: List of monitoring results
        metrics: List of metrics to plot (optional)
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    if not history:
        print("No history to plot.")
        return
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Drift scores over time
    if 'drift' in df.columns:
        drift_scores = df['drift'].apply(
            lambda x: x.get('drift_score', 0) if isinstance(x, dict) else 0
        )
        axes[0].plot(drift_scores, marker='o')
        axes[0].set_title('Drift Score Over Time')
        axes[0].set_ylabel('Drift Score')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction statistics
    if 'predictions' in df.columns:
        pred_means = df['predictions'].apply(
            lambda x: x.get('mean', 0) if isinstance(x, dict) else 0
        )
        axes[1].plot(pred_means, marker='o', color='blue')
        axes[1].set_title('Mean Predictions Over Time')
        axes[1].set_ylabel('Mean Prediction')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics (if available)
    if 'performance' in df.columns and metrics:
        for metric in metrics:
            metric_values = df['performance'].apply(
                lambda x: x.get(metric, None) if isinstance(x, dict) else None
            )
            if metric_values.notna().any():
                axes[2].plot(metric_values, marker='o', label=metric)
        axes[2].set_title('Performance Metrics Over Time')
        axes[2].set_ylabel('Metric Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Alert count
    if 'alerts' in df.columns:
        alert_counts = df['alerts'].apply(len)
        axes[3].bar(range(len(alert_counts)), alert_counts, color='orange', alpha=0.7)
        axes[3].set_title('Alert Count Over Time')
        axes[3].set_ylabel('Number of Alerts')
        axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_feature_distribution(
    reference_data,
    current_data,
    feature_name,
    bins=30,
):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        reference_data[feature_name],
        bins=bins,
        alpha=0.6,
        label="Reference",
        density=True,
    )

    ax.hist(
        current_data[feature_name],
        bins=bins,
        alpha=0.6,
        label="Current",
        density=True,
    )

    ax.set_title(f"Distribution shift: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend()

    return fig
