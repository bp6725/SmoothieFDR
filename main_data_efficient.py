"""
Data-Efficient Testing via Optimal Experimental Design
File: main_data_efficient.py

This experiment demonstrates that we can save resources by:
1. Sampling only part of the p-values (70%)
2. Predicting alpha on the unseen points (30%)

Comparison:
- Full data learning (benchmark)
- Random 70% subsampling
- A-optimal 70% subsampling (smart selection minimizing prediction variance)

Based on Section: "Data-Efficient Testing via Optimal Experimental Design"
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.special import expit, logit
import pickle as pkl
import sys
import os

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# --- CACHE CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- IMPORTS ---
from spatial_fdr_evaluation.data.adbench_loader import load_from_ADbench

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASETS_TO_RUN = [
    '33_skin',
    '19_landsat',
    '31_satimage-2',
    '30_satellite',
    '41_Waveform',
    '25_musk',
    '4_breastw',
    '45_wine',
    '15_Hepatitis'
]

DATASET_CONFIG = {
    "33_skin": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "19_landsat": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "31_satimage-2": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0/2, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "30_satellite": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 50, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "41_Waveform": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0/2, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "25_musk": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "4_breastw": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "45_wine": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
    "15_Hepatitis": {'n_total': 200, 'n_clusters': 3, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.1, 'effect_strength': 'medium'},
}

# Optimization Parameters
OPT_PARAMS = {'lr': 0.0005, 'lambda_reg': 10.0, 'lambda_bound': 500.0, 'max_iter': 5000}

# Experiment Parameters
TRAIN_FRACTION = 0.7  # 70% for training, 30% for testing
RANDOM_STATE = 42
REGULARIZATION_LAMBDA = 10.0  # For A-optimal design


# ============================================================================
# A-OPTIMAL EXPERIMENTAL DESIGN
# ============================================================================

def compute_prior_variance(K_full, selected_indices, unselected_indices, lambda_reg):
    """
    Compute prior prediction variance at unselected locations.

    σ²_prior(loc | S) = K(loc,loc) - k(loc)^T (K_S + λI)^{-1} k(loc)

    Parameters
    ----------
    K_full : np.ndarray (M, M)
        Full kernel matrix
    selected_indices : list
        Indices of selected (observed) locations
    unselected_indices : list
        Indices of unselected locations
    lambda_reg : float
        Regularization parameter

    Returns
    -------
    variances : np.ndarray
        Prior variance at each unselected location
    """
    if len(selected_indices) == 0:
        # No observations: variance is just the diagonal
        return np.diag(K_full)[unselected_indices]

    S = list(selected_indices)
    K_S = K_full[np.ix_(S, S)]
    K_S_reg = K_S + lambda_reg * np.eye(len(S))

    # Cholesky factorization for efficient solving
    try:
        L = np.linalg.cholesky(K_S_reg)
    except np.linalg.LinAlgError:
        # Add more regularization if not positive definite
        K_S_reg = K_S + (lambda_reg + 1e-6) * np.eye(len(S))
        L = np.linalg.cholesky(K_S_reg)

    variances = []
    for i in unselected_indices:
        k_i = K_full[i, S]  # kernel vector from i to selected
        # Solve L @ v = k_i, then variance reduction = v^T v
        v = np.linalg.solve(L, k_i)
        var_reduction = np.dot(v, v)
        var_i = K_full[i, i] - var_reduction
        variances.append(max(0, var_i))  # Ensure non-negative

    return np.array(variances)


def greedy_a_optimal_selection(K_full, n_select, lambda_reg, verbose=True):
    """
    Greedy A-Optimal location selection.

    Iteratively selects locations that minimize average prediction variance
    at unselected locations.

    Parameters
    ----------
    K_full : np.ndarray (M, M)
        Full kernel matrix
    n_select : int
        Number of locations to select
    lambda_reg : float
        Regularization parameter
    verbose : bool
        Print progress

    Returns
    -------
    selected : list
        Indices of selected locations
    """
    M = K_full.shape[0]
    selected = []
    unselected = set(range(M))

    for k in range(n_select):
        best_idx = None
        best_avg_var = np.inf

        # Evaluate each candidate
        candidates = list(unselected)

        # For efficiency, sample candidates if too many
        if len(candidates) > 100 and k > 0:
            # Use a smarter heuristic: pick from current highest variance locations
            current_vars = compute_prior_variance(K_full, selected, candidates, lambda_reg)
            # Focus on high-variance candidates
            top_indices = np.argsort(current_vars)[-min(100, len(candidates)):]
            candidates = [candidates[i] for i in top_indices]

        for j in candidates:
            # Temporarily add j to selected
            temp_selected = selected + [j]
            temp_unselected = [i for i in unselected if i != j]

            if len(temp_unselected) == 0:
                avg_var = 0
            else:
                vars_j = compute_prior_variance(K_full, temp_selected, temp_unselected, lambda_reg)
                avg_var = np.mean(vars_j)

            if avg_var < best_avg_var:
                best_avg_var = avg_var
                best_idx = j

        selected.append(best_idx)
        unselected.remove(best_idx)

        if verbose and (k + 1) % 20 == 0:
            print(f"    A-optimal: selected {k+1}/{n_select}, avg_var={best_avg_var:.4f}")

    return selected


def random_selection(M, n_select, random_state=None):
    """Random location selection."""
    rng = np.random.RandomState(random_state)
    return list(rng.choice(M, n_select, replace=False))


# ============================================================================
# DATA LOADING AND PROCESSING (from main_allspace_cv.py)
# ============================================================================

def compute_kernel_matrix(X, length_scale=1.0):
    """Compute RBF kernel matrix."""
    dist_sq = cdist(X, X, 'sqeuclidean')
    return np.exp(-dist_sq / (2 * length_scale**2))


def estimate_length_scale(X, method='median'):
    """Estimate kernel length scale from data."""
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        dists = cdist(X[idx], X[idx], 'euclidean')
    else:
        dists = cdist(X, X, 'euclidean')
    return np.median(dists[dists > 0])


def generate_pvalues(labels, effect_strength='medium'):
    """Generate p-values based on true labels."""
    p_values = np.zeros(len(labels))
    alphas = {'weak': 0.5, 'medium': 0.05, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)
    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())
    p_values[labels == 0] = np.random.beta(a, 3, size=(labels == 0).sum())
    return np.clip(p_values, 1e-10, 1.0)


def hierarchical_cluster_selection_smart(K, n_total, n_clusters_to_pick, min_cluster_size):
    """Select points via hierarchical clustering."""
    N = K.shape[0]
    K_normalized = K / (np.sqrt(np.outer(np.diag(K), np.diag(K))) + 1e-10)
    distance_matrix = np.clip(1 - K_normalized, 0, None)
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')

    valid_clusterings = []
    for n_try in range(2, N // min_cluster_size + 1):
        labels = fcluster(Z, t=n_try, criterion='maxclust')
        unique_labels = np.unique(labels)
        cluster_info = {lbl: np.sum(labels == lbl) for lbl in unique_labels if np.sum(labels == lbl) >= min_cluster_size}
        if len(cluster_info) >= n_clusters_to_pick:
            valid_clusterings.append({
                'n_clusters': n_try, 'labels': labels,
                'valid_clusters': list(cluster_info.keys()), 'cluster_sizes': cluster_info
            })

    if not valid_clusterings:
        raise ValueError("Selection Failed")

    best_config = max(valid_clusterings, key=lambda x: len(x['valid_clusters']))
    labels = best_config['labels']
    valid_labels = best_config['valid_clusters']

    cluster_centroids = {lbl: K[labels == lbl, :].mean(axis=0) for lbl in valid_labels}
    selected_clusters = []
    first_cluster = max(valid_labels, key=lambda lbl: best_config['cluster_sizes'][lbl])
    selected_clusters.append(first_cluster)
    remaining = [c for c in valid_labels if c != first_cluster]

    while len(selected_clusters) < n_clusters_to_pick and len(remaining) > 0:
        best_dist = -np.inf
        best_cluster = None
        for candidate in remaining:
            min_dist = min(np.linalg.norm(cluster_centroids[candidate] - cluster_centroids[sel]) for sel in selected_clusters)
            if min_dist > best_dist:
                best_dist = min_dist
                best_cluster = candidate
        selected_clusters.append(best_cluster)
        remaining.remove(best_cluster)

    points_per_cluster = n_total // len(selected_clusters)
    selected_indices = []
    final_cluster_labels = []

    for cluster_idx, lbl in enumerate(selected_clusters):
        cluster_points = np.where(labels == lbl)[0]
        n_sample = min(points_per_cluster, len(cluster_points))
        sampled = np.random.choice(cluster_points, size=n_sample, replace=False)
        selected_indices.extend(sampled)
        final_cluster_labels.extend([cluster_idx] * n_sample)

    while len(selected_indices) < n_total:
        sizes = [np.sum(np.array(final_cluster_labels) == i) for i in range(len(selected_clusters))]
        lbl = selected_clusters[np.argmax(sizes)]
        cluster_points = np.where(labels == lbl)[0]
        available = [p for p in cluster_points if p not in selected_indices]
        if not available:
            break
        selected_indices.append(np.random.choice(available))
        final_cluster_labels.append(np.argmax(sizes))

    return np.array(selected_indices), np.array(final_cluster_labels)


def load_and_sample_hierarchical(dataset_name):
    """Load dataset and sample points via hierarchical clustering."""
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        return None

    try:
        data = load_from_ADbench(dataset_name)
        X_full = StandardScaler().fit_transform(data['X_train'])
    except Exception as e:
        print(f"  Failed to load {dataset_name}: {e}")
        return None

    if len(X_full) > config['candidate_pool']:
        X_pool = X_full[np.random.choice(len(X_full), config['candidate_pool'], replace=False)]
    else:
        X_pool = X_full

    sigma = estimate_length_scale(X_pool, method='median') * config['sigma_factor']
    K_pool = compute_kernel_matrix(X_pool, length_scale=sigma)

    try:
        sel_idx, group_ids = hierarchical_cluster_selection_smart(
            K_pool, config['n_total'], config['n_clusters'], config['min_cluster_size']
        )
    except Exception as e:
        print(f"  Clustering failed for {dataset_name}: {e}")
        return None

    true_labels = np.zeros(len(sel_idx), dtype=int)
    for gid in range(config['n_clusters']):
        mask = (group_ids == gid)
        n_g = mask.sum()
        if n_g == 0:
            continue
        if gid == 0:
            lbls = np.ones(n_g, dtype=int)
            n_corr = int(n_g * config['cluster_corruption'])
            if n_corr > 0:
                lbls[np.random.choice(n_g, n_corr, replace=False)] = 0
            true_labels[mask] = lbls
        elif gid == 1:
            lbls = np.zeros(n_g, dtype=int)
            n_corr = int(n_g * config['cluster_corruption'])
            if n_corr > 0:
                lbls[np.random.choice(n_g, n_corr, replace=False)] = 1
            true_labels[mask] = lbls
        else:
            true_labels[mask] = 0

    return {
        'X': X_pool[sel_idx],
        'p_values': generate_pvalues(true_labels, config['effect_strength']),
        'true_labels': true_labels,
        'sigma': sigma
    }


# ============================================================================
# OPTIMIZATION
# ============================================================================

def run_optimization_stage1(K_matrix, f0_vals, f1_vals, verbose=False):
    """Run Stage 1 point-wise optimization."""
    n = K_matrix.shape[0]
    c = np.ones(n) * (1.0 / (K_matrix.sum(axis=1).mean() + 1e-10))

    lr = OPT_PARAMS['lr']
    reg = OPT_PARAMS['lambda_reg']
    bnd = OPT_PARAMS['lambda_bound']

    for t in range(OPT_PARAMS['max_iter']):
        alpha = K_matrix @ c
        mix = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)
        grad_nll = -(f0_vals - f1_vals) / mix
        grad_bound = 2 * bnd * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * reg * c)
        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0:
            grad_nat = grad_nat * (5.0 / gnorm)
        c -= lr * grad_nat

    return c


def run_optimization_stage1_with_hessian(K_matrix, f0_vals, f1_vals, verbose=False):
    """
    Run Stage 1 optimization and return coefficients + Hessian for uncertainty.

    Returns
    -------
    c : np.ndarray
        Optimized coefficients
    H : np.ndarray
        Hessian matrix at MAP estimate (for posterior variance)
    alpha_final : np.ndarray
        Final alpha values at training points
    """
    n = K_matrix.shape[0]
    c = np.ones(n) * (1.0 / (K_matrix.sum(axis=1).mean() + 1e-10))

    lr = OPT_PARAMS['lr']
    reg = OPT_PARAMS['lambda_reg']
    bnd = OPT_PARAMS['lambda_bound']

    for t in range(OPT_PARAMS['max_iter']):
        alpha = K_matrix @ c
        mix = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)
        grad_nll = -(f0_vals - f1_vals) / mix
        grad_bound = 2 * bnd * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * reg * c)
        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0:
            grad_nat = grad_nat * (5.0 / gnorm)
        c -= lr * grad_nat

    # Compute final alpha
    alpha_final = K_matrix @ c

    # Compute Hessian at MAP estimate
    # H = H_data + 2*lambda*K + H_bound
    # H_data = K^T diag(h_i) K where h_i = (f0-f1)^2 / mix^2 (Fisher info)

    mix_final = np.clip(alpha_final * f0_vals + (1 - alpha_final) * f1_vals, 1e-12, None)

    # Fisher information per observation
    h_i = ((f0_vals - f1_vals) ** 2) / (mix_final ** 2)

    # Data Hessian: H_data = K^T diag(h_i) K
    # More efficient: H_data = (K * sqrt(h_i)).T @ (K * sqrt(h_i))
    sqrt_h = np.sqrt(h_i)
    K_weighted = K_matrix * sqrt_h[:, np.newaxis]  # K_ij * sqrt(h_i)
    H_data = K_weighted.T @ K_weighted

    # Regularization Hessian: 2 * lambda * K
    H_reg = 2 * reg * K_matrix

    # Boundary Hessian (zero at feasible solution where 0 <= alpha <= 1)
    # b_i = 2 if alpha_i > 1 or alpha_i < 0, else 0
    b_i = np.zeros(n)
    b_i[alpha_final > 1] = 2
    b_i[alpha_final < 0] = 2
    if np.any(b_i > 0):
        sqrt_b = np.sqrt(b_i)
        K_bound = K_matrix * sqrt_b[:, np.newaxis]
        H_bound = bnd * K_bound.T @ K_bound
    else:
        H_bound = np.zeros((n, n))

    H = H_data + H_reg + H_bound

    return c, H, alpha_final


def compute_posterior_variance(H, K_train, K_test_train):
    """
    Compute posterior variance at test locations using Laplace approximation.

    Parameters
    ----------
    H : np.ndarray (n_train, n_train)
        Hessian matrix at MAP estimate
    K_train : np.ndarray (n_train, n_train)
        Kernel matrix for training points
    K_test_train : np.ndarray (n_test, n_train)
        Cross-kernel matrix from test to training points

    Returns
    -------
    posterior_var : np.ndarray (n_test,)
        Posterior variance at each test location
    """
    n_test = K_test_train.shape[0]

    # Add small regularization for numerical stability
    H_reg = H + 1e-6 * np.eye(H.shape[0])

    try:
        # Cholesky factorization: H = L L^T
        L = np.linalg.cholesky(H_reg)

        # For each test point: var = k^T H^{-1} k
        # Solve L v = k, then var = v^T v
        posterior_var = np.zeros(n_test)
        for i in range(n_test):
            k_i = K_test_train[i, :]
            v = np.linalg.solve(L, k_i)
            posterior_var[i] = np.dot(v, v)

    except np.linalg.LinAlgError:
        # Fallback to direct solve if Cholesky fails
        print("    Warning: Cholesky failed, using direct solve")
        try:
            H_inv = np.linalg.inv(H_reg)
            posterior_var = np.array([K_test_train[i, :] @ H_inv @ K_test_train[i, :]
                                       for i in range(n_test)])
        except:
            # Last resort: return prior variance
            print("    Warning: Matrix inversion failed, returning prior variance")
            posterior_var = np.ones(n_test) * 0.25  # Default uncertainty

    return np.clip(posterior_var, 0, None)  # Ensure non-negative


class GlobalFDRRegressor:
    """Stage 2: Global Inference using Natural Gradient KLR."""

    def __init__(self, lambda_global=1.0, lr=0.005, max_iter=2000, tol=1e-5):
        self.lambda_global = lambda_global
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.c = None

    def fit(self, K_train, alpha_hat_stage1):
        n = K_train.shape[0]
        epsilon = 0.01
        alpha_clipped = np.clip(alpha_hat_stage1, epsilon, 1 - epsilon)
        target_logits = logit(alpha_clipped)

        # Warm Start
        self.c = np.linalg.solve(K_train + 1e-6 * np.eye(n), target_logits)

        # Optimize
        for i in range(self.max_iter):
            g = K_train @ self.c
            sigma = expit(g)
            grad = (sigma - alpha_clipped) + 2 * self.lambda_global * self.c
            self.c -= self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break
        return self

    def predict(self, K_test):
        if self.c is None:
            raise ValueError("Model not fitted")
        return expit(K_test @ self.c)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def compute_lfdr(p_values, f0_vals, f1_vals, alpha_vals):
    """
    Compute local FDR: lfdr = alpha * f0 / (alpha * f0 + (1-alpha) * f1)
    """
    mix = alpha_vals * f0_vals + (1 - alpha_vals) * f1_vals
    mix = np.clip(mix, 1e-12, None)
    lfdr = (alpha_vals * f0_vals) / mix
    return np.clip(lfdr, 0, 1)


def run_experiment(X, p_values, true_labels, sigma, train_indices, test_indices, method_name):
    """
    Run inference with given train/test split.

    Returns predictions on test set with confidence intervals.
    """
    X_tr = X[train_indices]
    X_te = X[test_indices]
    p_tr = p_values[train_indices]

    # Density estimation on training data
    def f0_func(p):
        return np.ones_like(p)

    z = stats.norm.ppf(1 - np.clip(p_tr, 1e-10, 1-1e-10))
    mask = p_tr < 0.2
    if mask.sum() > 10:
        mu, sig = np.mean(z[mask]), np.std(z[mask])
    else:
        mu, sig = 2.5, 1.0

    def f1_func(p):
        z_p = stats.norm.ppf(1 - np.clip(p, 1e-10, 1-1e-10))
        return np.clip(stats.norm.pdf(z_p, loc=mu, scale=sig) / stats.norm.pdf(z_p), 0, 2000.0)

    f0_tr = f0_func(p_tr)
    f1_tr = f1_func(p_tr)

    # Compute kernel matrices
    K_tr = compute_kernel_matrix(X_tr, length_scale=sigma)

    # Stage 1: Point-wise optimization WITH Hessian for uncertainty
    c_stage1, H_stage1, alpha_hat_tr = run_optimization_stage1_with_hessian(K_tr, f0_tr, f1_tr)
    alpha_hat_tr = np.clip(alpha_hat_tr, 0, 1)

    # Stage 2: Global inference
    klr = GlobalFDRRegressor(lambda_global=0.1, lr=0.005, max_iter=2000)
    klr.fit(K_tr, alpha_hat_tr)

    # Predict on test set
    dist_te = cdist(X_te, X_tr, 'sqeuclidean')
    K_te_tr = np.exp(-dist_te / (2 * sigma**2))
    alpha_pred_te = klr.predict(K_te_tr)

    # Compute posterior variance (confidence intervals) at test locations
    posterior_var = compute_posterior_variance(H_stage1, K_tr, K_te_tr)
    posterior_std = np.sqrt(posterior_var)

    # 95% credible intervals (approximately 2 standard deviations)
    alpha_ci_lower = np.clip(alpha_pred_te - 2 * posterior_std, 0, 1)
    alpha_ci_upper = np.clip(alpha_pred_te + 2 * posterior_std, 0, 1)

    # Also compute lfdr on test set for comparison
    f0_te = f0_func(p_values[test_indices])
    f1_te = f1_func(p_values[test_indices])
    lfdr_te = compute_lfdr(p_values[test_indices], f0_te, f1_te, alpha_pred_te)

    return {
        'alpha_pred': alpha_pred_te,
        'alpha_std': posterior_std,
        'alpha_ci_lower': alpha_ci_lower,
        'alpha_ci_upper': alpha_ci_upper,
        'posterior_var': posterior_var,
        'lfdr': lfdr_te,
        'test_indices': test_indices,
        'train_indices': train_indices,
        'method': method_name
    }


def process_dataset(dataset_name):
    """Process a single dataset with all three methods."""
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Processing...")
    print(f"{'='*60}")

    np.random.seed(RANDOM_STATE)

    # Load data
    data = load_and_sample_hierarchical(dataset_name)
    if data is None:
        return None

    X = data['X']
    p_values = data['p_values']
    true_labels = data['true_labels']
    sigma = data['sigma']

    if np.isnan(X).any():
        X = np.nan_to_num(X)

    M = len(X)
    n_train = int(M * TRAIN_FRACTION)
    n_test = M - n_train

    print(f"  Total points: {M}, Train: {n_train} (70%), Test: {n_test} (30%)")

    # Compute full kernel matrix (needed for A-optimal)
    K_full = compute_kernel_matrix(X, length_scale=sigma)

    # ========================================================================
    # Method 1: Full data (benchmark) - use all points for training
    # ========================================================================
    print(f"\n  [1/3] Full data benchmark...")
    all_indices = list(range(M))

    # For full data, we still need to evaluate on "test" points
    # Use same test indices as subsampled methods for fair comparison
    # First, get random test indices
    rng = np.random.RandomState(RANDOM_STATE)
    test_indices_random = sorted(rng.choice(M, n_test, replace=False))
    train_indices_full = [i for i in range(M) if i not in test_indices_random]

    # Run with ALL data as training (benchmark)
    result_full = run_experiment(X, p_values, true_labels, sigma,
                                  all_indices, test_indices_random, 'full')

    # ========================================================================
    # Method 2: Random 70% subsampling
    # ========================================================================
    print(f"\n  [2/3] Random subsampling ({TRAIN_FRACTION*100:.0f}%)...")
    train_indices_random = [i for i in range(M) if i not in test_indices_random]

    result_random = run_experiment(X, p_values, true_labels, sigma,
                                    train_indices_random, test_indices_random, 'random')

    # ========================================================================
    # Method 3: A-optimal 70% subsampling
    # ========================================================================
    print(f"\n  [3/3] A-optimal subsampling ({TRAIN_FRACTION*100:.0f}%)...")
    train_indices_aopt = greedy_a_optimal_selection(
        K_full, n_train, REGULARIZATION_LAMBDA, verbose=True
    )
    test_indices_aopt = [i for i in range(M) if i not in train_indices_aopt]

    result_aopt = run_experiment(X, p_values, true_labels, sigma,
                                  train_indices_aopt, test_indices_aopt, 'a_optimal')

    # ========================================================================
    # Compute ground truth lfdr using full data for reference
    # ========================================================================
    # Density estimation on all data
    def f0_func(p):
        return np.ones_like(p)

    z = stats.norm.ppf(1 - np.clip(p_values, 1e-10, 1-1e-10))
    mask = p_values < 0.2
    if mask.sum() > 10:
        mu, sig = np.mean(z[mask]), np.std(z[mask])
    else:
        mu, sig = 2.5, 1.0

    def f1_func(p):
        z_p = stats.norm.ppf(1 - np.clip(p, 1e-10, 1-1e-10))
        return np.clip(stats.norm.pdf(z_p, loc=mu, scale=sig) / stats.norm.pdf(z_p), 0, 2000.0)

    f0_all = f0_func(p_values)
    f1_all = f1_func(p_values)

    # Run full optimization to get "oracle" alpha
    c_oracle = run_optimization_stage1(K_full, f0_all, f1_all)
    alpha_oracle = np.clip(K_full @ c_oracle, 0, 1)
    lfdr_oracle = compute_lfdr(p_values, f0_all, f1_all, alpha_oracle)

    return {
        'dataset': dataset_name,
        'X': X,
        'p_values': p_values,
        'true_labels': true_labels,
        'sigma': sigma,
        'n_total': M,
        'n_train': n_train,
        'n_test': n_test,
        # Oracle (full data)
        'alpha_oracle': alpha_oracle,
        'lfdr_oracle': lfdr_oracle,
        # Results
        'result_full': result_full,
        'result_random': result_random,
        'result_aopt': result_aopt,
    }


if __name__ == "__main__":
    all_results = {}

    for ds in DATASETS_TO_RUN:
        try:
            result = process_dataset(ds)
            if result:
                all_results[ds] = result
                print(f"\n  [{ds}] Completed successfully!")
        except Exception as e:
            print(f"\n  [{ds}] Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results to cache
    cache_path = os.path.join(CACHE_DIR, "data_efficient_results.pkl")
    with open(cache_path, "wb") as f:
        pkl.dump(all_results, f)

    print(f"\n{'='*60}")
    print(f"Results saved to: {cache_path}")
    print(f"Processed {len(all_results)} datasets successfully.")
    print(f"Run 'python plot_data_efficient.py' to generate visualizations.")
    print(f"{'='*60}")
