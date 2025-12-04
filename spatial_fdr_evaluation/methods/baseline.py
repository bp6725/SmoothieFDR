"""
Baseline FDR control methods.

Implements standard Benjamini-Hochberg (BH) procedure.
"""

import numpy as np
from typing import Tuple


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.1,
    return_threshold: bool = False
) -> np.ndarray:
    """
    Benjamini-Hochberg (BH) procedure for FDR control.
    
    Controls FDR at level α for independent tests or under
    positive regression dependency (PRDS).
    
    Parameters
    ----------
    p_values : np.ndarray, shape (n_tests,)
        P-values for each test
    alpha : float, default=0.1
        Target FDR level (typically 0.05 or 0.1)
    return_threshold : bool, default=False
        If True, also return the threshold used
        
    Returns
    -------
    discoveries : np.ndarray, shape (n_tests,), dtype=bool
        Boolean array indicating which hypotheses are rejected
    threshold : float, optional
        The p-value threshold used (only if return_threshold=True)
    """
    n_tests = len(p_values)
    
    # Sort p-values in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    # Find largest k such that P_(k) <= (k/n) * α
    # Using BH procedure
    comparisons = sorted_pvals <= (np.arange(1, n_tests + 1) / n_tests) * alpha
    
    if np.any(comparisons):
        # Find largest k satisfying the condition
        k_max = np.where(comparisons)[0].max()
        threshold = sorted_pvals[k_max]
        
        # Reject all hypotheses with p-value <= threshold
        discoveries = p_values <= threshold
    else:
        # No rejections
        discoveries = np.zeros(n_tests, dtype=bool)
        threshold = 0.0
    
    if return_threshold:
        return discoveries, threshold
    else:
        return discoveries


def count_discoveries(discoveries: np.ndarray) -> int:
    """
    Count number of discoveries (rejections).
    
    Parameters
    ----------
    discoveries : np.ndarray, dtype=bool
        Boolean array indicating rejections
        
    Returns
    -------
    n_discoveries : int
        Number of discoveries
    """
    return int(np.sum(discoveries))


def estimate_pi0_storey(p_values: np.ndarray, lambda_val: float = 0.5) -> float:
    """
    Estimate proportion of true nulls using Storey's method.
    
    π₀ = #{p_i > λ} / ((1-λ) * n)
    
    Parameters
    ----------
    p_values : np.ndarray
        P-values
    lambda_val : float, default=0.5
        Threshold parameter (typically 0.5)
        
    Returns
    -------
    pi0 : float
        Estimated proportion of true nulls
    """
    n = len(p_values)
    n_above = np.sum(p_values > lambda_val)
    pi0 = n_above / ((1 - lambda_val) * n)
    
    # Ensure π₀ is in [0, 1]
    pi0 = np.clip(pi0, 0, 1)
    
    return pi0


def local_fdr_from_mixture(
    p_values: np.ndarray,
    alpha_values: np.ndarray,
    f0_values: np.ndarray,
    f1_values: np.ndarray
) -> np.ndarray:
    """
    Compute local FDR from mixture model parameters.
    
    lfdr(p) = α(loc) * f0(p) / [α(loc) * f0(p) + (1-α(loc)) * f1(p)]
    
    Parameters
    ----------
    p_values : np.ndarray
        P-values
    alpha_values : np.ndarray
        Prior null probabilities α(loc) for each location
    f0_values : np.ndarray
        Null density f0(p) evaluated at each p-value
    f1_values : np.ndarray
        Alternative density f1(p) evaluated at each p-value
        
    Returns
    -------
    lfdr : np.ndarray
        Local false discovery rate for each test
    """
    numerator = alpha_values * f0_values
    denominator = alpha_values * f0_values + (1 - alpha_values) * f1_values
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)
    
    lfdr = numerator / denominator
    
    return lfdr
