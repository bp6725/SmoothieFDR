"""
End-to-End Pipeline: Global "Entire Space" Inference (Real Data)
File: main_allspace_cv.py

Methodology:
1. Load & Select Data (Exact same logic as main_bench_cv).
2. Split: Hide 20% of points (Test Set).
3. Stage 1: Point-wise Estimation on Training Set.
4. Stage 2: Global KLR on Training Set.
5. Validation: Aggregated Histogram of Alpha on Test Set (All Datasets Combined).

REFACTOR: Results saved to cache for separate visualization (use plot_allspace_cv.py).
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import KFold, train_test_split
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
# 1. CONFIGURATION (STRICTLY FROM main_bench_cv)
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

# Optimization Params (Same as main_bench_cv)
OPTIMIZER = 'vanilla'
OPT_PARAMS = {'c_init': "Auto", 'lr': 0.0005, 'lambda_reg': 10.0, 'lambda_bound': 500.0, 'max_iter': 5000}

# CV Grid
SIGMA_GRID = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_FOLDS = 5

RANDOM_STATE = 42

# ============================================================================
# 2. HIERARCHICAL CLUSTERING LOGIC (UNCHANGED)
# ============================================================================
def hierarchical_cluster_selection_smart(K, n_total, n_clusters_to_pick, min_cluster_size):
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

    if not valid_clusterings: raise ValueError("Selection Failed")
    best_config = max(valid_clusterings, key=lambda x: len(x['valid_clusters']))
    labels = best_config['labels']
    valid_labels = best_config['valid_clusters']

    cluster_centroids = {lbl: K[labels == lbl, :].mean(axis=0) for lbl in valid_labels}
    selected_clusters = []
    first_cluster = max(valid_labels, key=lambda lbl: best_config['cluster_sizes'][lbl])
    selected_clusters.append(first_cluster)
    remaining = [c for c in valid_labels if c != first_cluster]

    while len(selected_clusters) < n_clusters_to_pick and len(remaining) > 0:
        best_dist = -np.inf; best_cluster = None
        for candidate in remaining:
            min_dist = min(np.linalg.norm(cluster_centroids[candidate] - cluster_centroids[sel]) for sel in selected_clusters)
            if min_dist > best_dist: best_dist = min_dist; best_cluster = candidate
        selected_clusters.append(best_cluster)
        remaining.remove(best_cluster)

    points_per_cluster = n_total // len(selected_clusters)
    selected_indices = []; final_cluster_labels = []

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
        if not available: break
        selected_indices.append(np.random.choice(available))
        final_cluster_labels.append(np.argmax(sizes))

    return np.array(selected_indices), np.array(final_cluster_labels)

# ============================================================================
# 3. DATA LOADING (UNCHANGED)
# ============================================================================
def compute_kernel_matrix(X, kernel_type='rbf', length_scale=1.0):
    dist_sq = cdist(X, X, 'sqeuclidean')
    if kernel_type == 'rbf':
        return np.exp(-dist_sq / (2 * length_scale**2))
    return np.eye(len(X))

def estimate_length_scale(X, method='median'):
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        dists = cdist(X[idx], X[idx], 'euclidean')
    else:
        dists = cdist(X, X, 'euclidean')
    return np.median(dists)

def generate_pvalues(labels, effect_strength='medium'):
    p_values = np.zeros(len(labels))
    alphas = {'weak': 0.5, 'medium': 0.05, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)
    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())
    p_values[labels == 0] = np.random.beta(a, 3, size=(labels == 0).sum())
    return np.clip(p_values, 1e-10, 1.0)

def load_and_sample_hierarchical(dataset_name):
    config = DATASET_CONFIG.get(dataset_name)
    if not config: return None
    try:
        data = load_from_ADbench(dataset_name)
        X_full = StandardScaler().fit_transform(data['X_train'])
    except: return None

    if len(X_full) > config['candidate_pool']:
        X_pool = X_full[np.random.choice(len(X_full), config['candidate_pool'], replace=False)]
    else: X_pool = X_full

    sigma = estimate_length_scale(X_pool, method='median') * config['sigma_factor']
    K_pool = compute_kernel_matrix(X_pool, kernel_type='rbf', length_scale=sigma)

    try: sel_idx, group_ids = hierarchical_cluster_selection_smart(K_pool, config['n_total'], config['n_clusters'], config['min_cluster_size'])
    except: return None

    true_labels = np.zeros(len(sel_idx), dtype=int)
    for gid in range(config['n_clusters']):
        mask = (group_ids == gid)
        n_g = mask.sum()
        if n_g == 0: continue
        if gid == 0:
            lbls = np.ones(n_g, dtype=int)
            n_corr = int(n_g * config['cluster_corruption'])
            if n_corr > 0: lbls[np.random.choice(n_g, n_corr, replace=False)] = 0
            true_labels[mask] = lbls
        elif gid == 1:
            lbls = np.zeros(n_g, dtype=int)
            n_corr = int(n_g * config['cluster_corruption'])
            if n_corr > 0: lbls[np.random.choice(n_g, n_corr, replace=False)] = 1
            true_labels[mask] = lbls
        else:
            true_labels[mask] = 0

    return X_pool[sel_idx], generate_pvalues(true_labels, config['effect_strength']), true_labels

# ============================================================================
# 4. OPTIMIZATION - STAGE 1 (UNCHANGED LOGIC)
# ============================================================================
def run_optimization_stage1(K_matrix, f0_vals, f1_vals):
    n = K_matrix.shape[0]
    if OPT_PARAMS['c_init'] == "Auto": c = np.ones(n) * (1.0 / (K_matrix.sum(axis=1).mean() + 1e-10))
    else: c = np.ones(n) * OPT_PARAMS['c_init']

    lr, reg, bnd = OPT_PARAMS['lr'], OPT_PARAMS['lambda_reg'], OPT_PARAMS['lambda_bound']

    for t in range(OPT_PARAMS['max_iter']):
        alpha = K_matrix @ c
        mix = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)
        grad_nll = -(f0_vals - f1_vals) / mix
        grad_bound = 2 * bnd * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * reg * c)
        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0: grad_nat = grad_nat * (5.0 / gnorm)
        c -= lr * grad_nat

    return c

# ============================================================================
# 5. STAGE 2 SOLVER (GLOBAL KLR)
# ============================================================================
class GlobalFDRRegressor:
    """Stage 2: Global Inference using Natural Gradient KLR"""
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
            grad = (sigma - alpha_hat_stage1) + 2 * self.lambda_global * self.c
            self.c -= self.lr * grad
            if np.linalg.norm(grad) < self.tol: break
        return self

    def predict(self, K_test):
        if self.c is None: raise ValueError("Model not fitted")
        return expit(K_test @ self.c)

# ============================================================================
# 6. EXECUTION (NO PLOTTING - RESULTS SAVED TO CACHE)
# ============================================================================
def process_dataset(dataset_name):
    print(f"\n[{dataset_name}] Processing...")
    np.random.seed(RANDOM_STATE)

    # 1. Load Data
    data = load_and_sample_hierarchical(dataset_name)
    if data is None: return None
    X, p_values, true_labels = data
    if np.isnan(X).any(): X = np.nan_to_num(X)

    # 2. Split Data: Hide 20% (Test Set)
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=true_labels)

    X_tr, X_te = X[idx_train], X[idx_test]
    p_tr, p_te = p_values[idx_train], p_values[idx_test]
    y_tr, y_te = true_labels[idx_train], true_labels[idx_test]

    # 3. Density Estimation on TRAIN
    def f0_func(p): return np.ones_like(p)
    z = stats.norm.ppf(1 - np.clip(p_tr, 1e-10, 1-1e-10))
    mask = p_tr < 0.2
    mu, sig = (np.mean(z[mask]), np.std(z[mask])) if mask.sum() > 10 else (2.5, 1.0)
    def f1_func(p): return np.clip(stats.norm.pdf(stats.norm.ppf(1-p), loc=mu, scale=sig)/stats.norm.pdf(stats.norm.ppf(1-p)), 0, 2000.0)
    f0_tr, f1_tr = f0_func(p_tr), f1_func(p_tr)

    # 4. CV for Sigma on TRAIN (Exact same CV logic)
    print(f"[{dataset_name}] Running CV on Train Set...")
    base_sigma = estimate_length_scale(X_tr, method='median')
    cv_scores = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fac in SIGMA_GRID:
        curr_sigma = base_sigma * fac
        fold_nlls = []
        for t_idx, v_idx in kf.split(X_tr):
            # K_train_fold (use cdist)
            dist_tr = cdist(X_tr[t_idx], X_tr[t_idx], 'sqeuclidean')
            K_tr_fold = np.exp(-dist_tr / (2 * curr_sigma**2))

            c_tr = run_optimization_stage1(K_tr_fold, f0_tr[t_idx], f1_tr[t_idx])

            # K_val_fold
            dist_val = cdist(X_tr[v_idx], X_tr[t_idx], 'sqeuclidean')
            K_val_fold = np.exp(-dist_val / (2 * curr_sigma**2))

            alpha_val = np.clip(K_val_fold @ c_tr, 0, 1)
            mix = np.clip(alpha_val * f0_tr[v_idx] + (1-alpha_val) * f1_tr[v_idx], 1e-12, None)
            fold_nlls.append(-np.sum(np.log(mix)))
        cv_scores.append(np.mean(fold_nlls))

    best_fac = SIGMA_GRID[np.argmin(cv_scores)]
    best_sigma = base_sigma * best_fac
    print(f"[{dataset_name}] Best Sigma: {best_fac}")

    # 5. Final Stage 1 on Full TRAIN
    dist_full_tr = cdist(X_tr, X_tr, 'sqeuclidean')
    K_tr_final = np.exp(-dist_full_tr / (2 * best_sigma**2))
    c_stage1 = run_optimization_stage1(K_tr_final, f0_tr, f1_tr)
    alpha_hat_tr = np.clip(K_tr_final @ c_stage1, 0, 1)

    # 6. Stage 2 (Global Inference) on TRAIN
    klr = GlobalFDRRegressor(lambda_global=0.1, lr=0.005, max_iter=2000)
    klr.fit(K_tr_final, alpha_hat_tr)

    # 7. Inference on TEST (Hidden Points)
    dist_test = cdist(X_te, X_tr, 'sqeuclidean')
    K_te = np.exp(-dist_test / (2 * best_sigma**2))
    alpha_pred_te = klr.predict(K_te)

    return {
        'y_true': y_te,
        'alpha_pred': alpha_pred_te,
        'cv_scores': cv_scores,
        'sigma_grid': SIGMA_GRID,
        'best_sigma_factor': best_fac,
        'p_values_test': p_te,
        'X_test': X_te
    }


if __name__ == "__main__":
    all_res = {}
    for ds in DATASETS_TO_RUN:
        try:
            res = process_dataset(ds)
            if res: all_res[ds] = res
        except Exception as e:
            print(f"[{ds}] Failed: {e}")

    # Save results to cache
    cache_path = os.path.join(CACHE_DIR, "allspace_cv_results.pkl")
    with open(cache_path, "wb") as f:
        pkl.dump(all_res, f)
    print(f"\nResults saved to: {cache_path}")
    print(f"Run 'python plot_allspace_cv.py' to generate visualizations.")
