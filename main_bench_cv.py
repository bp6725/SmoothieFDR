"""
End-to-End Pipeline Validation: Multi-Dataset Loop (Paper Grade)

Updates:
- FIX: Geometric Failure Analysis now has 2 Subplots (Isolation from H1, Isolation from H0).
- FIX: Combined Summary now generates two separate grids for H1/H0 isolation.
- Adheres to: Vanilla Optimizer, Auto Init, Fixed Hyperparameters.
- REFACTOR: Results saved to cache for separate visualization (use plot_bench_cv.py).
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import KFold
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
from spatial_fdr_evaluation.methods.kernels import compute_kernel_matrix, estimate_length_scale

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATASETS_TO_RUN = [
'30_satellite',
    '33_skin',
    '19_landsat',
    '31_satimage-2',
    '41_Waveform',
    '25_musk',
    '4_breastw',
    '45_wine',
    '15_Hepatitis'
]

# Dataset Config (Turn 40)
DATASET_CONFIG = {
    "33_skin": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "19_landsat": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "31_satimage-2": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0/2, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "30_satellite": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 50, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "41_Waveform": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0/2, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "25_musk": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 30, 'candidate_pool': 10000, 'sigma_factor': 1.0, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "4_breastw": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "45_wine": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
    "15_Hepatitis": {'n_total': [50,200,200], 'n_clusters': 2, 'min_cluster_size': 20, 'candidate_pool': 10000, 'sigma_factor': 0.5, 'cluster_corruption': 0.2, 'effect_strength': 'medium'},
}

# Optimization (Strict Adherence)
OPTIMIZER = 'vanilla'
OPT_PARAMS = {'c_init': "Auto", 'lr': 0.0005, 'lambda_reg': 10.0, 'lambda_bound': 500.0, 'max_iter': 5000,'lambda_sparse': 0.001}

# CV Grid
SIGMA_GRID = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_FOLDS = 5

RANDOM_STATE = 42

# ============================================================================
# 2. HIERARCHICAL DATA SELECTION
# ============================================================================
def hierarchical_cluster_selection_smart(K, n_total, n_clusters_to_pick, min_cluster_size):
    """
    Validates clusters via Ward linkage and picks the most distant ones.
    UPDATED: Supports specific sample counts per cluster.
    """
    N = K.shape[0]
    # Normalize Kernel to Distance
    K_normalized = K / (np.sqrt(np.outer(np.diag(K), np.diag(K))) + 1e-10)
    distance_matrix = np.clip(1 - K_normalized, 0, None)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical Clustering
    Z = linkage(condensed_dist, method='ward')

    # Find valid configuration
    valid_clusterings = []
    for n_try in range(2, N // min_cluster_size + 1):
        labels = fcluster(Z, t=n_try, criterion='maxclust')
        unique_labels = np.unique(labels)
        cluster_info = {lbl: np.sum(labels == lbl) for lbl in unique_labels if
                        np.sum(labels == lbl) >= min_cluster_size}

        if len(cluster_info) >= n_clusters_to_pick:
            valid_clusterings.append({
                'n_clusters': n_try, 'labels': labels,
                'valid_clusters': list(cluster_info.keys()), 'cluster_sizes': cluster_info
            })

    if not valid_clusterings:
        raise ValueError(f"Could not find {n_clusters_to_pick} clusters with size >= {min_cluster_size}")

    best_config = max(valid_clusterings, key=lambda x: len(x['valid_clusters']))
    labels = best_config['labels']
    valid_labels = best_config['valid_clusters']

    # Compute Centroids
    cluster_centroids = {}
    for lbl in valid_labels:
        cluster_centroids[lbl] = K[labels == lbl, :].mean(axis=0)

    # Greedy Selection of Distant Clusters
    selected_clusters = []
    first_cluster = max(valid_labels, key=lambda lbl: best_config['cluster_sizes'][lbl])
    selected_clusters.append(first_cluster)
    remaining = [c for c in valid_labels if c != first_cluster]

    while len(selected_clusters) < n_clusters_to_pick and len(remaining) > 0:
        best_dist = -np.inf;
        best_cluster = None
        for candidate in remaining:
            min_dist = min(
                np.linalg.norm(cluster_centroids[candidate] - cluster_centroids[sel]) for sel in selected_clusters)
            if min_dist > best_dist:
                best_dist = min_dist;
                best_cluster = candidate
        selected_clusters.append(best_cluster)
        remaining.remove(best_cluster)

    # --- SAMPLING LOGIC (UPDATED) ---
    selected_indices = [];
    final_cluster_labels = []

    # Determine targets per cluster
    if isinstance(n_total, list) or isinstance(n_total, np.ndarray):
        if len(n_total) != n_clusters_to_pick:
            raise ValueError(f"n_total list length ({len(n_total)}) must match n_clusters ({n_clusters_to_pick})")
        target_counts = n_total
    else:
        # Fallback to even split if integer
        target_counts = [n_total // n_clusters_to_pick] * n_clusters_to_pick

    # Sampling loop
    for cluster_idx, (lbl, target) in enumerate(zip(selected_clusters, target_counts)):
        cluster_points = np.where(labels == lbl)[0]

        if len(cluster_points) < target:
            print(f"Warning: Cluster {cluster_idx} has {len(cluster_points)} points, requested {target}. Taking all.")

        n_sample = min(target, len(cluster_points))
        sampled = np.random.choice(cluster_points, size=n_sample, replace=False)
        selected_indices.extend(sampled)
        final_cluster_labels.extend([cluster_idx] * n_sample)

    return np.array(selected_indices), np.array(final_cluster_labels)

# ============================================================================
# 3. DATA LOADING
# ============================================================================
def generate_pvalues(labels, effect_strength='medium'):
    p_values = np.zeros(len(labels))
    alphas = {'weak': 0.5, 'medium': 0.05, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)
    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())
    p_values[labels == 0] = np.random.beta(a, 3, size=(labels == 0).sum())
    return np.clip(p_values, 1e-5, 1.0)

def load_and_sample_hierarchical(dataset_name):
    config = DATASET_CONFIG.get(dataset_name)
    if not config: return None
    try:
        data = load_from_ADbench(dataset_name)
        X_full = StandardScaler().fit_transform(data['X_train'])
    except:
        return None

    # Subsample Pool
    if len(X_full) > config['candidate_pool']:
        X_pool = X_full[np.random.choice(len(X_full), config['candidate_pool'], replace=False)]
    else:
        X_pool = X_full

    sigma = estimate_length_scale(X_pool, method='median') * config['sigma_factor']
    K_pool = compute_kernel_matrix(X_pool, kernel_type='rbf', length_scale=sigma)

    # --- 1. CLUSTER SELECTION (Groups 0 & 1) ---
    # Parse list config: e.g., [80, 60, 60] -> Clusters=[80, 60], Random=60
    if isinstance(config['n_total'], list):
        cluster_targets = config['n_total'][:config['n_clusters']]
        n_random = config['n_total'][config['n_clusters']]
    else:
        # Fallback for integer config
        cluster_targets = config['n_total']
        n_random = 0

    try:
        sel_idx, group_ids = hierarchical_cluster_selection_smart(
            K_pool, cluster_targets, config['n_clusters'], config['min_cluster_size']
        )
    except Exception as e:
        print(f"[{dataset_name}] Clustering Failed: {e}");
        return None

    # --- 2. RANDOM BACKGROUND SAMPLING (Group 2) ---
    # Find indices NOT used in clusters
    all_indices = np.arange(len(X_pool))
    mask_unused = np.isin(all_indices, sel_idx, invert=True)
    available_indices = all_indices[mask_unused]

    if len(available_indices) < n_random:
        rand_idx = available_indices  # Take all if not enough
    else:
        rand_idx = np.random.choice(available_indices, n_random, replace=False)

    # Merge Data
    final_sel_idx = np.concatenate([sel_idx, rand_idx])

    # Merge Group IDs (Append '2's for background)
    rand_group_ids = np.full(len(rand_idx), 2)
    final_group_ids = np.concatenate([group_ids, rand_group_ids])

    # --- 3. ASSIGN TRUTH LABELS ---
    true_labels = np.zeros(len(final_sel_idx), dtype=int)

    # Group 0: Signal Cluster (H1 with corruption)
    mask_0 = (final_group_ids == 0)
    n_0 = mask_0.sum()
    if n_0 > 0:
        lbls = np.ones(n_0, dtype=int)
        n_corr = int(n_0 * config['cluster_corruption'])
        if n_corr > 0: lbls[np.random.choice(n_0, n_corr, replace=False)] = 0
        true_labels[mask_0] = lbls

    # Group 1: Noise Cluster (H0 with corruption)
    mask_1 = (final_group_ids == 1)
    n_1 = mask_1.sum()
    if n_1 > 0:
        lbls = np.zeros(n_1, dtype=int)
        n_corr = int(n_1 * config['cluster_corruption'])
        if n_corr > 0: lbls[np.random.choice(n_1, n_corr, replace=False)] = 1
        true_labels[mask_1] = lbls

    # Group 2: Random Background (Pure H0)
    # (Already zeros, so we leave it)

    return X_pool[final_sel_idx], generate_pvalues(true_labels, config['effect_strength']), true_labels, final_group_ids

# ============================================================================
# 4. OPTIMIZATION
# ============================================================================
def run_optimization(K_matrix, f0_vals, f1_vals, verbose=False):
    n = K_matrix.shape[0]
    if OPT_PARAMS['c_init'] == "Auto": c = np.ones(n) * (1.0 / (K_matrix.sum(axis=1).mean() + 1e-10))
    else: c = np.ones(n) * OPT_PARAMS['c_init']

    grad_sparsity = OPT_PARAMS['lambda_sparse'] * K_matrix.sum(axis=1)

    lr, reg, bnd = OPT_PARAMS['lr'], OPT_PARAMS['lambda_reg'], OPT_PARAMS['lambda_bound']
    losses, grad_norms, alpha_hist, viol_hist = [], [], [], []

    for t in range(OPT_PARAMS['max_iter']):
        alpha = K_matrix @ c
        mix = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)
        grad_nll = -(f0_vals - f1_vals) / mix
        grad_bound = 2 * bnd * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * reg * c) #+ grad_sparsity
        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0: grad_nat = grad_nat * (5.0 / gnorm)
        c -= lr * grad_nat

        loss = -np.sum(np.log(mix)) + reg * (c @ K_matrix @ c)
        losses.append(loss); grad_norms.append(gnorm)
        if t % 50 == 0:
            alpha_hist.append(alpha.copy()); viol_hist.append(np.sum((alpha < 0) | (alpha > 1)))

    return c, {'losses': losses, 'grad_norms': grad_norms, 'alpha_history': alpha_hist, 'violations': viol_hist}

# ============================================================================
# 5. EXECUTION (NO PLOTTING - RESULTS SAVED TO CACHE)
# ============================================================================
def process_dataset(dataset_name):
    print(f"\n[{dataset_name}] Processing...")
    np.random.seed(RANDOM_STATE)
    data = load_and_sample_hierarchical(dataset_name)
    if data is None: return None
    X, p_values, true_labels, group_ids = data
    if np.isnan(X).any(): X = np.nan_to_num(X)

    # Estimate
    def f0_func(p): return np.ones_like(p)
    z = stats.norm.ppf(1 - np.clip(p_values, 1e-10, 1-1e-10))
    mask = p_values < 0.2

    # NEW: Force sigma to 1.0 (or at most 1.0)
    if mask.sum() > 10:
        mu = np.mean(z[mask])
        sig = 1.0
    else:
        mu, sig = 2.5, 1.0

    def f1_func(p): return np.clip(stats.norm.pdf(stats.norm.ppf(1-p), loc=mu, scale=sig)/stats.norm.pdf(stats.norm.ppf(1-p)), 0, 2000.0)
    f0_vals, f1_vals = f0_func(p_values), f1_func(p_values)

    # CV
    print(f"[{dataset_name}] Running CV...")
    base_sigma = estimate_length_scale(X, method='median')
    cv_scores = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fac in SIGMA_GRID:
        curr_sigma = base_sigma * fac
        fold_nlls = []
        for train_idx, test_idx in kf.split(X):
            K_tr = compute_kernel_matrix(X[train_idx], kernel_type='rbf', length_scale=curr_sigma)
            c_tr, _ = run_optimization(K_tr, f0_vals[train_idx], f1_vals[train_idx], verbose=False)
            dist_test = cdist(X[test_idx], X[train_idx], metric='euclidean')
            K_test = np.exp(-(1.0/(2*curr_sigma**2)) * dist_test**2)
            alpha_test = np.clip(K_test @ c_tr, 0, 1)
            mix = np.clip(alpha_test * f0_vals[test_idx] + (1-alpha_test) * f1_vals[test_idx], 1e-12, None)
            fold_nlls.append(-np.sum(np.log(mix)))
        cv_scores.append(np.mean(fold_nlls))
    best_fac = SIGMA_GRID[np.argmin(cv_scores)]
    print(f"[{dataset_name}] Best Sigma: {best_fac}")

    # Final Run
    print(f"[{dataset_name}] Final Optimization...")
    best_sigma = base_sigma * best_fac
    K_final = compute_kernel_matrix(X, kernel_type='rbf', length_scale=best_sigma)
    c_final, history = run_optimization(K_final, f0_vals, f1_vals, verbose=True)
    alpha_final = K_final @ c_final

    lfdr = (alpha_final * f0_vals) / (alpha_final * f0_vals + (1 - alpha_final) * f1_vals)
    q_vals = np.cumsum(np.sort(lfdr)) / np.arange(1, len(lfdr)+1)
    rejections = np.zeros(len(lfdr), dtype=bool)
    if np.sum(q_vals <= 0.1) > 0: rejections[np.argsort(lfdr)[:np.max(np.where(q_vals <= 0.1)[0])+1]] = True
    print(f"[{dataset_name}] Rej: {rejections.sum()}")

    # Isolation
    h1_idx = np.where(true_labels == 0)[0] # Signal
    sim_h1 = np.sort(K_final[:, h1_idx], axis=1)[:, -min(50, len(h1_idx)):].mean(axis=1)
    iso_h1 = 1 - (sim_h1 - sim_h1.min())/(sim_h1.max() - sim_h1.min() + 1e-10)

    h0_idx = np.where(true_labels == 1)[0] # Noise
    sim_h0 = np.sort(K_final[:, h0_idx], axis=1)[:, -min(50, len(h0_idx)):].mean(axis=1)
    iso_h0 = 1 - (sim_h0 - sim_h0.min())/(sim_h0.max() - sim_h0.min() + 1e-10)

    # Return comprehensive results for caching
    return {
        'history': history,
        'iso_h1': iso_h1,
        'iso_h0': iso_h0,
        'p_values': p_values,
        'rejections': rejections,
        'true_labels': true_labels,
        'final_group_ids': group_ids,
        'alpha_final': alpha_final,
        'K_final': K_final,
        'cv_scores': cv_scores,
        'sigma_grid': SIGMA_GRID,
        'best_sigma_factor': best_fac,
        'f0_vals': f0_vals,
        'f1_vals': f1_vals,
        'lfdr': lfdr,
        'X': X
    }

if __name__ == "__main__":
    all_res = {}
    for ds in DATASETS_TO_RUN:
        res = process_dataset(ds)
        if res: all_res[ds] = res

    # Save results to cache
    cache_path = os.path.join(CACHE_DIR, "bench_cv_results.pkl")
    with open(cache_path, "wb") as f:
        pkl.dump(all_res, f)
    print(f"\nResults saved to: {cache_path}")
    print(f"Run 'python plot_bench_cv.py' to generate visualizations.")
