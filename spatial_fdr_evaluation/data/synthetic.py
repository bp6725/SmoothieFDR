"""
Synthetic data generation for spatial FDR evaluation.

This module generates spatially structured null/alternative labels
and corresponding p-values for evaluation.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Literal


def generate_spatial_clusters(
    locations: np.ndarray,
    n_clusters: int = 5,
    signal_cluster_fraction: float = 0.4,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spatial clustering structure.
    
    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial coordinates
    n_clusters : int, default=5
        Number of spatial clusters
    signal_cluster_fraction : float, default=0.4
        Fraction of clusters that are signal-rich
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    cluster_assignments : np.ndarray, shape (n_samples,)
        Cluster ID for each location
    is_signal_cluster : np.ndarray, shape (n_samples,)
        Boolean indicating if location is in signal-rich cluster
    """
    np.random.seed(random_state)
    
    # Spatial clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_assignments = kmeans.fit_predict(locations)
    
    # Select signal-rich clusters
    n_signal_clusters = max(1, int(n_clusters * signal_cluster_fraction))
    signal_cluster_ids = np.random.choice(
        n_clusters,
        size=n_signal_clusters,
        replace=False
    )
    
    is_signal_cluster = np.isin(cluster_assignments, signal_cluster_ids)
    
    return cluster_assignments, is_signal_cluster


def generate_clustered_labels(
    locations: np.ndarray,
    spatial_strength: Literal['none', 'weak', 'medium', 'strong'] = 'medium',
    n_clusters: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate true null/alternative labels with spatial clustering.
    
    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial coordinates
    spatial_strength : {'none', 'weak', 'medium', 'strong'}, default='medium'
        Strength of spatial clustering:
        - 'none': Random assignment (no spatial structure)
        - 'weak': Modest spatial clustering
        - 'medium': Moderate spatial clustering
        - 'strong': High spatial clustering
    n_clusters : int, default=5
        Number of spatial clusters
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    true_labels : np.ndarray, shape (n_samples,)
        True labels: 1 = H0 (null), 0 = H1 (alternative)
    """
    np.random.seed(random_state)
    N = len(locations)
    
    if spatial_strength == 'none':
        # Random assignment - no spatial structure
        π0 = 0.7  # 70% nulls
        true_labels = np.random.binomial(1, π0, size=N)
        
    else:
        # Cluster-based assignment
        cluster_assignments, is_signal_cluster = generate_spatial_clusters(
            locations,
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        # Define probabilities based on clustering strength
        strength_params = {
            'weak': (0.6, 0.8),      # (p_H0_signal, p_H0_null)
            'medium': (0.4, 0.9),
            'strong': (0.2, 0.95)
        }
        
        p_H0_in_signal_cluster, p_H0_in_null_cluster = strength_params[spatial_strength]
        
        # Generate labels based on cluster membership
        true_labels = np.zeros(N, dtype=int)
        for i in range(N):
            if is_signal_cluster[i]:
                true_labels[i] = np.random.binomial(1, p_H0_in_signal_cluster)
            else:
                true_labels[i] = np.random.binomial(1, p_H0_in_null_cluster)
    
    return true_labels


def generate_pvalues(
    true_labels: np.ndarray,
    effect_strength: Literal['weak', 'medium', 'strong'] = 'medium',
    random_state: int = 42
) -> np.ndarray:
    """
    Generate p-values based on true null/alternative labels.
    
    For H0 (true_labels=1): p ~ Uniform(0,1)
    For H1 (true_labels=0): p ~ Beta(α, 1) with small α (concentrated near 0)
    
    Parameters
    ----------
    true_labels : np.ndarray, shape (n_samples,)
        True labels: 1 = H0 (null), 0 = H1 (alternative)
    effect_strength : {'weak', 'medium', 'strong'}, default='medium'
        Strength of alternative signal (how close to 0)
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    p_values : np.ndarray, shape (n_samples,)
        Generated p-values
    """
    np.random.seed(random_state)
    N = len(true_labels)
    p_values = np.zeros(N)
    
    # Beta distribution parameters for different effect strengths
    effect_params = {
        'weak': 0.2,      # Moderate signal
        'medium': 0.05,   # Strong signal
        'strong': 0.01    # Very strong signal (very close to 0)
    }
    
    alpha = effect_params[effect_strength]
    
    for i in range(N):
        if true_labels[i] == 1:  # H0 is true
            p_values[i] = np.random.uniform(0, 1)
        else:  # H1 is true
            p_values[i] = np.random.beta(alpha, 1)
    
    return p_values


def generate_evaluation_data(
    locations: np.ndarray,
    spatial_strength: Literal['none', 'weak', 'medium', 'strong'] = 'medium',
    effect_strength: Literal['weak', 'medium', 'strong'] = 'medium',
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete evaluation dataset: true labels + p-values.
    
    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial coordinates
    spatial_strength : str, default='medium'
        Strength of spatial clustering
    effect_strength : str, default='medium'
        Strength of alternative signal
    n_clusters : int, default=5
        Number of spatial clusters
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    true_labels : np.ndarray, shape (n_samples,)
        True labels: 1 = H0, 0 = H1
    p_values : np.ndarray, shape (n_samples,)
        Generated p-values
    """
    true_labels = generate_clustered_labels(
        locations,
        spatial_strength=spatial_strength,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    p_values = generate_pvalues(
        true_labels,
        effect_strength=effect_strength,
        random_state=random_state + 1  # Different seed for p-values
    )
    
    return true_labels, p_values


def generate_square_data(
        n_samples: int = 200,
        effect_size: float = 2.0,
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate toy data with 4-square spatial structure."""
    from scipy import stats

    np.random.seed(random_state)
    locations = np.random.rand(n_samples, 2)
    in_top_right = (locations[:, 0] > 0.5) & (locations[:, 1] > 0.5)

    labels = np.zeros(n_samples, dtype=int)

    # Top-right: 90% alternatives
    top_right_indices = np.where(in_top_right)[0]
    n_alt_topright = int(0.9 * len(top_right_indices))
    null_topright = top_right_indices[n_alt_topright:]
    labels[null_topright] = 1

    # Other three quarters: 10% alternatives
    other_indices = np.where(~in_top_right)[0]
    n_alt_other = int(0.1 * len(other_indices))
    labels[other_indices] = 1
    labels[other_indices[:n_alt_other]] = 0

    # Generate p-values
    z_scores = np.random.randn(n_samples)
    z_scores[labels == 0] += effect_size
    p_values = 1 - stats.norm.cdf(z_scores)

    return locations, labels, p_values