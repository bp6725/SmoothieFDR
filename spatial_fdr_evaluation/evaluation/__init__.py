"""Evaluation metrics and analysis."""

from .metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_power_at_fdr,
    summarize_metrics,
    compare_methods,
    compute_relative_power_gain
)

__all__ = [
    'compute_confusion_matrix',
    'compute_metrics',
    'compute_power_at_fdr',
    'summarize_metrics',
    'compare_methods',
    'compute_relative_power_gain'
]
