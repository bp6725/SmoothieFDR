"""
Kernel functions for RKHS-based spatial FDR.

Implements various kernels including Matérn, RBF, and graph-based kernels.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from typing import Literal, Optional


def matern_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    nu: float = 1.5,
    length_scale: float = 1.0
) -> np.ndarray:
    """
    Matérn kernel with explicit smoothness control.
    
    K(x, y) = (2^(1-ν)/Γ(ν)) * (√(2ν) * r / ℓ)^ν * K_ν(√(2ν) * r / ℓ)
    
    where r = ||x - y||, K_ν is modified Bessel function of second kind.
    
    Smoothness properties:
    - ν = 0.5: Exponential kernel (C^0, continuous but not differentiable)
    - ν = 1.5: Once differentiable (C^1)
    - ν = 2.5: Twice differentiable (C^2)
    - ν → ∞: Approaches RBF (C^∞, infinitely smooth)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        First set of locations
    Y : np.ndarray, shape (n_samples_Y, n_features), optional
        Second set of locations. If None, Y = X.
    nu : float, default=1.5
        Smoothness parameter (ν > 0)
    length_scale : float, default=1.0
        Length scale parameter (ℓ > 0)
        
    Returns
    -------
    K : np.ndarray, shape (n_samples_X, n_samples_Y)
        Kernel matrix
    """
    if Y is None:
        Y = X
        
    # Compute pairwise distances
    dists = cdist(X, Y, metric='euclidean')
    
    # Handle special cases
    if nu == 0.5:
        # Exponential kernel
        K = np.exp(-dists / length_scale)
    elif nu == 1.5:
        # Once differentiable
        scaled_dist = np.sqrt(3) * dists / length_scale
        K = (1 + scaled_dist) * np.exp(-scaled_dist)
    elif nu == 2.5:
        # Twice differentiable
        scaled_dist = np.sqrt(5) * dists / length_scale
        K = (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
    elif np.isinf(nu):
        # RBF kernel (infinite smoothness)
        K = np.exp(-0.5 * (dists / length_scale)**2)
    else:
        # General case using Bessel function
        scaled_dist = np.sqrt(2 * nu) * dists / length_scale
        
        # Avoid division by zero
        scaled_dist[scaled_dist == 0.0] = 1e-8
        
        tmp = 2 ** (1 - nu) / gamma(nu)
        K = tmp * (scaled_dist ** nu) * kv(nu, scaled_dist)
        
        # Handle the diagonal (distance = 0)
        K[dists == 0] = 1.0
    
    return K


def rbf_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    length_scale: float = 1.0
) -> np.ndarray:
    """
    Radial Basis Function (RBF) / Gaussian kernel.
    
    K(x, y) = exp(-||x - y||^2 / (2 * ℓ^2))
    
    This is the infinitely smooth (C^∞) case.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        First set of locations
    Y : np.ndarray, shape (n_samples_Y, n_features), optional
        Second set of locations. If None, Y = X.
    length_scale : float, default=1.0
        Length scale parameter (ℓ > 0)
        
    Returns
    -------
    K : np.ndarray, shape (n_samples_X, n_samples_Y)
        Kernel matrix
    """
    if Y is None:
        Y = X
    
    dists = cdist(X, Y, metric='sqeuclidean')
    K = np.exp(-dists / (2 * length_scale**2))
    
    return K


def compute_kernel_matrix(
    locations: np.ndarray,
    kernel_type: Literal['matern', 'rbf'] = 'matern',
    **kernel_params
) -> np.ndarray:
    """
    Compute kernel matrix for given locations.
    
    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial coordinates
    kernel_type : {'matern', 'rbf'}, default='matern'
        Type of kernel to use
    **kernel_params : dict
        Additional parameters for the kernel:
        - For Matérn: nu (default=1.5), length_scale (default=1.0)
        - For RBF: length_scale (default=1.0)
        
    Returns
    -------
    K : np.ndarray, shape (n_samples, n_samples)
        Kernel (Gram) matrix
    """
    if kernel_type == 'matern':
        nu = kernel_params.get('nu', 1.5)
        length_scale = kernel_params.get('length_scale', 1.0)
        K = matern_kernel(locations, nu=nu, length_scale=length_scale)
    elif kernel_type == 'rbf':
        length_scale = kernel_params.get('length_scale', 1.0)
        K = rbf_kernel(locations, length_scale=length_scale)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return K


def add_regularization(K: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Add small diagonal regularization to ensure numerical stability.
    
    K_reg = K + ε*I
    
    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix
    epsilon : float, default=1e-6
        Regularization strength
        
    Returns
    -------
    K_reg : np.ndarray, shape (n, n)
        Regularized kernel matrix
    """
    n = K.shape[0]
    return K + epsilon * np.eye(n)


def estimate_length_scale(locations: np.ndarray, 
                          method: Literal['median', 'scott'] = 'median') -> float:
    """
    Estimate reasonable length scale from data.
    
    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial coordinates
    method : {'median', 'scott'}, default='median'
        Method for estimation:
        - 'median': Median of pairwise distances
        - 'scott': Scott's rule (similar to KDE bandwidth)
        
    Returns
    -------
    length_scale : float
        Estimated length scale
    """
    if method == 'median':
        # Median heuristic: median of pairwise distances
        dists = pdist(locations, metric='euclidean')
        length_scale = np.median(dists)
    elif method == 'scott':
        # Scott's rule
        n_samples, n_features = locations.shape
        length_scale = n_samples ** (-1.0 / (n_features + 4))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return length_scale
