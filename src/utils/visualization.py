"""
Visualization utilities for evaluation results.
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_calibration_curve(
    calibration_data: List[Dict],
    title: str = "Calibration Curve",
    save_path: Optional[str] = None
):
    """
    Plot calibration curve showing confidence vs accuracy.

    Args:
        calibration_data: List of bin data from compute_calibration_error
        title: Plot title
        save_path: Path to save figure
    """
    confidences = [bin_data['confidence'] for bin_data in calibration_data]
    accuracies = [bin_data['accuracy'] for bin_data in calibration_data]
    counts = [bin_data['count'] for bin_data in calibration_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot calibration curve
    ax.plot(confidences, accuracies, 'o-', label='Model', markersize=8)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')

    # Size points by bin count
    ax.scatter(confidences, accuracies, s=[c/10 for c in counts], alpha=0.3)

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_risk_coverage_curve(
    risk_coverage_data: Dict,
    title: str = "Risk-Coverage Curve",
    save_path: Optional[str] = None
):
    """
    Plot risk-coverage curve for selective prediction.

    Args:
        risk_coverage_data: Data from compute_risk_coverage
        title: Plot title
        save_path: Path to save figure
    """
    coverages = risk_coverage_data['coverages']
    risks = risk_coverage_data['risks']
    auc = risk_coverage_data['auc']

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, risks, 'o-', linewidth=2)
    ax.fill_between(coverages, risks, alpha=0.3)

    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('Risk (Error Rate)', fontsize=12)
    ax.set_title(f'{title}\nAUC = {auc:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Invert x-axis (higher coverage on left)
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confidence_distribution(
    confidences: np.ndarray,
    labels: np.ndarray,
    title: str = "Confidence Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of confidence scores for correct/incorrect predictions.

    Args:
        confidences: Confidence scores
        labels: Binary labels
        title: Plot title
        save_path: Path to save figure
    """
    correct_confidences = confidences[labels == 1]
    incorrect_confidences = confidences[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(correct_confidences, bins=20, alpha=0.6, label='Correct', color='green')
    ax.hist(incorrect_confidences, bins=20, alpha=0.6, label='Incorrect', color='red')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_method_comparison(
    comparison_data: Dict[str, Dict],
    metrics: List[str] = ['auc_roc', 'auc_pr', 'ece'],
    save_path: Optional[str] = None
):
    """
    Plot comparison of multiple methods across metrics.

    Args:
        comparison_data: Dictionary mapping method names to results
        metrics: Metrics to compare
        save_path: Path to save figure
    """
    methods = list(comparison_data.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [comparison_data[method].get(metric, 0) for method in methods]

        bars = ax.bar(range(len(methods)), values)

        # Color best method
        if metric in ['ece', 'mce']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_color('green')

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(metric.upper(), fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_ablation_results(
    ablation_results: Dict[str, Dict],
    metric: str = 'auc_roc',
    save_path: Optional[str] = None
):
    """
    Plot results of ablation study.

    Args:
        ablation_results: Dictionary mapping ablation names to results
        metric: Metric to visualize
        save_path: Path to save figure
    """
    ablations = list(ablation_results.keys())
    values = [ablation_results[abl].get(metric, 0) for abl in ablations]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(ablations)), values)

    # Highlight full model (if present)
    if 'full_model' in ablations:
        full_idx = ablations.index('full_model')
        bars[full_idx].set_color('green')

    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels(ablations, rotation=45, ha='right')
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'Ablation Study: {metric.upper()}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
