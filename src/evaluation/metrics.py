"""
Evaluation metrics for confidence scoring.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


def compute_auc_roc(
    confidences: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        confidences: Confidence scores (N,)
        labels: Binary labels (N,)

    Returns:
        AUC-ROC score
    """
    return roc_auc_score(labels, confidences)


def compute_auc_pr(
    confidences: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        confidences: Confidence scores (N,)
        labels: Binary labels (N,)

    Returns:
        AUC-PR score
    """
    return average_precision_score(labels, confidences)


def compute_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> Dict[str, float]:
    """
    Compute calibration error metrics.

    Args:
        confidences: Confidence scores (N,)
        labels: Binary labels (N,)
        n_bins: Number of bins for calibration
        strategy: Binning strategy ('uniform' or 'quantile')

    Returns:
        Dictionary with ECE, MCE, and calibration data
    """
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    else:  # quantile
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_boundaries = np.quantile(confidences, quantiles)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error

    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = labels[in_bin].mean()

            # Average confidence in bin
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Calibration error for this bin
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)

            # Update ECE (weighted by proportion in bin)
            ece += prop_in_bin * bin_error

            # Update MCE (max error)
            mce = max(mce, bin_error)

            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum(),
                'error': bin_error
            })

    return {
        'ece': ece,
        'mce': mce,
        'bin_data': bin_data
    }


def compute_ece(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Confidence scores
        labels: Binary labels
        n_bins: Number of bins

    Returns:
        ECE score
    """
    result = compute_calibration_error(confidences, labels, n_bins)
    return result['ece']


def compute_risk_coverage(
    confidences: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Compute risk-coverage curve for selective prediction.

    Risk is error rate on answered examples.
    Coverage is fraction of examples answered.

    Args:
        confidences: Confidence scores
        labels: Binary labels (1 = correct, 0 = incorrect)
        thresholds: Confidence thresholds to evaluate

    Returns:
        Dictionary with risk-coverage data
    """
    if thresholds is None:
        # Use percentiles as thresholds
        thresholds = np.percentile(confidences, np.arange(0, 101, 5))

    risks = []
    coverages = []

    for threshold in thresholds:
        # Select samples above threshold
        selected = confidences >= threshold
        coverage = selected.mean()

        if coverage > 0:
            # Compute accuracy on selected samples
            accuracy = labels[selected].mean()
            risk = 1 - accuracy
        else:
            risk = 0.0

        risks.append(risk)
        coverages.append(coverage)

    # Compute AUC of risk-coverage curve
    # Lower AUC is better (less risk at same coverage)
    if len(coverages) > 1:
        auc = np.trapz(risks, coverages)
    else:
        auc = 0.0

    return {
        'thresholds': thresholds,
        'risks': risks,
        'coverages': coverages,
        'auc': auc
    }


def compute_abstention_metrics(
    confidences: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict:
    """
    Compute metrics for abstention at a specific threshold.

    Args:
        confidences: Confidence scores
        labels: Binary labels
        threshold: Abstention threshold

    Returns:
        Dictionary with abstention metrics
    """
    # Samples above threshold (answered)
    answered = confidences >= threshold

    # Metrics
    coverage = answered.mean()
    accuracy_on_answered = labels[answered].mean() if coverage > 0 else 0.0
    accuracy_on_abstained = labels[~answered].mean() if coverage < 1.0 else 0.0

    # Overall accuracy (including abstained as errors)
    accuracy_overall = labels[answered].sum() / len(labels)

    return {
        'threshold': threshold,
        'coverage': coverage,
        'accuracy_on_answered': accuracy_on_answered,
        'accuracy_on_abstained': accuracy_on_abstained,
        'accuracy_overall': accuracy_overall,
        'num_answered': answered.sum(),
        'num_abstained': (~answered).sum()
    }


def compute_correlation_metrics(
    confidences: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    Compute correlation between confidence and correctness.

    Args:
        confidences: Confidence scores
        labels: Binary labels

    Returns:
        Dictionary with correlation metrics
    """
    from scipy.stats import pearsonr, spearmanr

    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(confidences, labels)

    # Spearman correlation (rank-based)
    spearman_corr, spearman_p = spearmanr(confidences, labels)

    return {
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p
    }


def evaluate_confidence_scores(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    abstention_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Comprehensive evaluation of confidence scores.

    Args:
        confidences: Confidence scores (N,)
        labels: Binary labels (N,)
        n_bins: Number of calibration bins
        abstention_thresholds: Thresholds for abstention analysis

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}

    # AUC metrics
    results['auc_roc'] = compute_auc_roc(confidences, labels)
    results['auc_pr'] = compute_auc_pr(confidences, labels)

    # Calibration
    calibration = compute_calibration_error(confidences, labels, n_bins)
    results['ece'] = calibration['ece']
    results['mce'] = calibration['mce']
    results['calibration_bins'] = calibration['bin_data']

    # Risk-coverage
    risk_coverage = compute_risk_coverage(confidences, labels)
    results['risk_coverage'] = risk_coverage

    # Abstention metrics
    if abstention_thresholds is None:
        abstention_thresholds = [0.3, 0.5, 0.7, 0.9]

    abstention_results = []
    for threshold in abstention_thresholds:
        abstention = compute_abstention_metrics(confidences, labels, threshold)
        abstention_results.append(abstention)
    results['abstention'] = abstention_results

    # Correlation
    correlation = compute_correlation_metrics(confidences, labels)
    results['correlation'] = correlation

    # Summary statistics
    results['mean_confidence'] = confidences.mean()
    results['std_confidence'] = confidences.std()
    results['mean_accuracy'] = labels.mean()

    return results


def compare_methods(
    method_results: Dict[str, Dict]
) -> Dict:
    """
    Compare multiple confidence estimation methods.

    Args:
        method_results: Dictionary mapping method names to their results

    Returns:
        Comparison summary
    """
    comparison = {}

    # Compare key metrics
    metrics_to_compare = ['auc_roc', 'auc_pr', 'ece', 'mce']

    for metric in metrics_to_compare:
        comparison[metric] = {
            name: results.get(metric, None)
            for name, results in method_results.items()
        }

        # Rank methods
        values = [(name, val) for name, val in comparison[metric].items() if val is not None]

        if metric in ['ece', 'mce']:
            # Lower is better
            values.sort(key=lambda x: x[1])
        else:
            # Higher is better
            values.sort(key=lambda x: x[1], reverse=True)

        comparison[f'{metric}_ranking'] = [name for name, _ in values]

    return comparison
