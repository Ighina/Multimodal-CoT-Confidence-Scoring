"""Evaluation metrics and framework."""

from .metrics import (
    compute_auc_roc,
    compute_auc_pr,
    compute_calibration_error,
    compute_ece,
    compute_risk_coverage,
    evaluate_confidence_scores
)
from .evaluator import ConfidenceEvaluator

__all__ = [
    "compute_auc_roc",
    "compute_auc_pr",
    "compute_calibration_error",
    "compute_ece",
    "compute_risk_coverage",
    "evaluate_confidence_scores",
    "ConfidenceEvaluator",
]
