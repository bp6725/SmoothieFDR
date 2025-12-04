"""
Data loading utilities for ADbench anomaly detection datasets.
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Dict, Tuple


def load_from_ADbench(dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from ADbench.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., '25_musk')
    
    Returns
    -------
    dict
        Dictionary with keys 'X_train', 'X_test', 'y_train', 'y_test'
    """
    # TODO: Replace with actual ADbench loading logic
    # This is a placeholder - implement your actual loading here
    raise NotImplementedError("Implement ADbench loading logic")


def extract_spatial_structure(X: np.ndarray, 
                              bandwidth: float = 'scott',
                              kernel: str = 'gaussian',
                              random_state: int = 42) -> Tuple[np.ndarray, KernelDensity]:
    """
    Extract spatial structure from data using Kernel Density Estimation.
    
    This provides the spatial locations and their density for realistic
    spatial FDR evaluation.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data
    bandwidth : float or str, default='scott'
        Bandwidth for KDE. If 'scott', uses Scott's rule.
    kernel : str, default='gaussian'
        Kernel type for KDE
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial locations (same as X)
    kde : KernelDensity
        Fitted KDE model for sampling distribution p(loc)
    """
    n_samples, n_features = X.shape
    
    # Compute bandwidth using Scott's rule if needed
    if bandwidth == 'scott':
        bandwidth = n_samples ** (-1.0 / (n_features + 4))
    
    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(X)
    
    return X, kde


def subsample_locations(X: np.ndarray,
                        n_samples: int,
                        random_state: int = 42) -> np.ndarray:
    """
    Subsample actual locations from dataset.

    Parameters
    ----------
    X : np.ndarray
        Full dataset
    n_samples : int
        Number of samples to select
    random_state : int
        Random seed

    Returns
    -------
    locations : np.ndarray
        Subsampled locations (actual data points)
    """
    np.random.seed(random_state)

    # Subsample from actual data points
    if n_samples >= len(X):
        return X

    indices = np.random.choice(len(X), size=n_samples, replace=False)
    return X[indices]
