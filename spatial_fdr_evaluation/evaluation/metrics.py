"""
Evaluation metrics for FDR control methods.

Computes power, FDR, TPR, precision, etc.
"""

import numpy as np
from typing import Dict


def compute_confusion_matrix(
    discoveries: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, int]:
    """
    Compute confusion matrix elements.
    
    Parameters
    ----------
    discoveries : np.ndarray, dtype=bool
        Boolean array indicating rejections (1 = reject, 0 = accept)
    true_labels : np.ndarray
        True labels (1 = H0 true/null, 0 = H1 true/alternative)
        
    Returns
    -------
    confusion : dict
        Dictionary with keys: 'TP', 'FP', 'TN', 'FN', 'n_discoveries', 'n_true_signals'
    """
    # Convert to boolean if needed
    discoveries = discoveries.astype(bool)
    
    # True positives: correctly reject H0 when H1 is true
    TP = int(np.sum((discoveries == True) & (true_labels == 0)))
    
    # False positives: incorrectly reject H0 when H0 is true
    FP = int(np.sum((discoveries == True) & (true_labels == 1)))
    
    # True negatives: correctly accept H0 when H0 is true
    TN = int(np.sum((discoveries == False) & (true_labels == 1)))
    
    # False negatives: incorrectly accept H0 when H1 is true
    FN = int(np.sum((discoveries == False) & (true_labels == 0)))
    
    n_discoveries = int(np.sum(discoveries))
    n_true_signals = int(np.sum(true_labels == 0))
    
    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'n_discoveries': n_discoveries,
        'n_true_signals': n_true_signals
    }


def compute_metrics(
    discoveries: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Parameters
    ----------
    discoveries : np.ndarray, dtype=bool
        Boolean array indicating rejections
    true_labels : np.ndarray
        True labels (1 = H0, 0 = H1)
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - TPR (True Positive Rate / Sensitivity / Recall / Power)
        - FDR (False Discovery Rate)
        - FPR (False Positive Rate)
        - Precision (Positive Predictive Value)
        - F1 (F1 score)
        - n_discoveries (number of rejections)
    """
    cm = compute_confusion_matrix(discoveries, true_labels)
    
    TP = cm['TP']
    FP = cm['FP']
    TN = cm['TN']
    FN = cm['FN']
    n_discoveries = cm['n_discoveries']
    n_true_signals = cm['n_true_signals']
    
    # True Positive Rate (Sensitivity / Recall / Power)
    TPR = TP / n_true_signals if n_true_signals > 0 else 0.0
    
    # False Discovery Rate
    FDR = FP / n_discoveries if n_discoveries > 0 else 0.0
    
    # False Positive Rate
    n_true_nulls = len(true_labels) - n_true_signals
    FPR = FP / n_true_nulls if n_true_nulls > 0 else 0.0
    
    # Precision (Positive Predictive Value)
    precision = TP / n_discoveries if n_discoveries > 0 else 0.0
    
    # F1 Score
    if precision + TPR > 0:
        F1 = 2 * (precision * TPR) / (precision + TPR)
    else:
        F1 = 0.0
    
    return {
        'TPR': TPR,
        'power': TPR,  # Alias
        'FDR': FDR,
        'FPR': FPR,
        'precision': precision,
        'F1': F1,
        'n_discoveries': n_discoveries,
        'n_true_signals': n_true_signals,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }


def compute_power_at_fdr(
    discoveries: np.ndarray,
    true_labels: np.ndarray,
    target_fdr: float = 0.1
) -> float:
    """
    Compute power (TPR) when FDR is controlled at target level.
    
    Parameters
    ----------
    discoveries : np.ndarray, dtype=bool
        Boolean array indicating rejections
    true_labels : np.ndarray
        True labels (1 = H0, 0 = H1)
    target_fdr : float, default=0.1
        Target FDR level
        
    Returns
    -------
    power : float
        Power (TPR) if FDR <= target_fdr, else 0.0
    """
    metrics = compute_metrics(discoveries, true_labels)
    
    if metrics['FDR'] <= target_fdr:
        return metrics['TPR']
    else:
        return 0.0


def summarize_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    Summarize metrics across multiple replications.
    
    Parameters
    ----------
    metrics_list : list of dict
        List of metric dictionaries from multiple replications
        
    Returns
    -------
    summary : dict
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}
    
    # Get metric names
    metric_names = metrics_list[0].keys()
    
    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        
        summary[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    return summary


def compare_methods(
    results_dict: Dict[str, list],
    metric: str = 'power'
) -> Dict[str, float]:
    """
    Compare multiple methods on a specific metric.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to lists of metric dicts
    metric : str, default='power'
        Metric to compare
        
    Returns
    -------
    comparison : dict
        Dictionary mapping method names to mean metric values
    """
    comparison = {}
    
    for method_name, metrics_list in results_dict.items():
        values = [m[metric] for m in metrics_list]
        comparison[method_name] = float(np.mean(values))
    
    return comparison


def compute_relative_power_gain(
    power_method: float,
    power_baseline: float
) -> float:
    """
    Compute relative power gain over baseline.
    
    gain = (power_method - power_baseline) / power_baseline
    
    Parameters
    ----------
    power_method : float
        Power of the method
    power_baseline : float
        Power of the baseline
        
    Returns
    -------
    gain : float
        Relative power gain (e.g., 0.5 = 50% gain)
    """
    if power_baseline == 0:
        return np.inf if power_method > 0 else 0.0
    
    return (power_method - power_baseline) / power_baseline
