"""
End-to-End Pipeline Validation: Multi-Dataset Loop (Paper Grade)

Updates:
- FIX: Geometric Failure Analysis now has 2 Subplots (Isolation from H1, Isolation from H0).
- FIX: Combined Summary now generates two separate grids for H1/H0 isolation.
- Adheres to: Vanilla Optimizer, Auto Init, Fixed Hyperparameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import KFold
from scipy.sparse.linalg import eigsh
from scipy import linalg
import sys
import os
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# --- IMPORTS ---
from spatial_fdr_evaluation.data.adbench_loader import load_from_ADbench
from spatial_fdr_evaluation.methods.kernels import compute_kernel_matrix, estimate_length_scale

# ============================================================================
# 1. CONFIGURATION
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

# Dataset Config (Turn 40)
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

# Optimization (Strict Adherence)
OPTIMIZER = 'vanilla'
OPT_PARAMS = {'c_init': "Auto", 'lr': 0.0005, 'lambda_reg': 10.0, 'lambda_bound': 500.0, 'max_iter': 5000}

# CV Grid
SIGMA_GRID = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_FOLDS = 5

RANDOM_STATE = 42

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    'figure.titlesize': 28, 'lines.linewidth': 3, 'grid.alpha': 0.4
})

# ============================================================================
# 2. HIERARCHICAL DATA SELECTION
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
# 4. OPTIMIZATION
# ============================================================================
def run_optimization(K_matrix, f0_vals, f1_vals, verbose=False):
    n = K_matrix.shape[0]
    if OPT_PARAMS['c_init'] == "Auto": c = np.ones(n) * (1.0 / (K_matrix.sum(axis=1).mean() + 1e-10))
    else: c = np.ones(n) * OPT_PARAMS['c_init']

    lr, reg, bnd = OPT_PARAMS['lr'], OPT_PARAMS['lambda_reg'], OPT_PARAMS['lambda_bound']
    losses, grad_norms, alpha_hist, viol_hist = [], [], [], []

    for t in range(OPT_PARAMS['max_iter']):
        alpha = K_matrix @ c
        mix = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)
        grad_nll = -(f0_vals - f1_vals) / mix
        grad_bound = 2 * bnd * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * reg * c)
        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0: grad_nat = grad_nat * (5.0 / gnorm)
        c -= lr * grad_nat

        loss = -np.sum(np.log(mix)) + reg * (c @ K_matrix @ c)
        losses.append(loss); grad_norms.append(gnorm)
        if t % 50 == 0:
            alpha_hist.append(alpha.copy()); viol_hist.append(np.sum((alpha < 0) | (alpha > 1)))

    return c, {'losses': losses, 'grad_norms': grad_norms, 'alpha_history': alpha_hist, 'violations': viol_hist}

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
def plot_cv_results(sigma_grid, cv_scores, best_sigma, dataset_name):
    plt.figure(figsize=(12, 8))
    plt.plot(sigma_grid, cv_scores, 'bo-', markerfacecolor='white', markersize=10)
    plt.plot(best_sigma, np.min(cv_scores), 'ro', markersize=15, label=f'Best: {best_sigma}')
    plt.xscale('log'); plt.xticks(sigma_grid, [str(s) for s in sigma_grid])
    plt.xlabel('Sigma'); plt.ylabel('NLL')
    plt.title(f'CV: {dataset_name}', fontweight='bold'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()

def plot_optimization_history(history, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes[0,0].plot(history['losses'], 'b-'); axes[0,0].set_title('Loss')
    axes[0,1].semilogy(history['grad_norms'], 'r-'); axes[0,1].set_title('Gradient')
    iters = np.linspace(0, len(history['losses'])-1, len(history['alpha_history']), dtype=int)
    alphas = history['alpha_history']
    axes[1,0].plot(iters, [a.mean() for a in alphas], 'g-', label='Mean')
    axes[1,0].plot(iters, [a.min() for a in alphas], 'b--', label='Min')
    axes[1,0].plot(iters, [a.max() for a in alphas], 'r--', label='Max')
    axes[1,0].legend(); axes[1,0].set_title('Alpha')
    axes[1,1].plot(iters, history['violations'], 'purple', marker='o'); axes[1,1].set_title('Violations')
    fig.suptitle(f"Optimization: {dataset_name}", fontsize=30, fontweight='bold', y=0.98); plt.tight_layout(); plt.show()

def plot_confusion_clustermap(K, true_labels, rejections, dataset_name):
    categories = np.zeros(len(true_labels), dtype=int)
    categories[(true_labels==1)&(~rejections)]=0; categories[(true_labels==1)&(rejections)]=1
    categories[(true_labels==0)&(~rejections)]=2; categories[(true_labels==0)&(rejections)]=3
    color_map = {0:'#d3d3d3', 1:'#ff0000', 2:'#ffa500', 3:'#0000ff'}
    row_colors = pd.Series(categories).map(color_map); row_colors.name = "Outcome"

    g = sns.clustermap(pd.DataFrame(K), row_colors=row_colors, col_colors=row_colors, cmap="viridis", figsize=(14,14), cbar_pos=None, dendrogram_ratio=0.01)
    g.ax_row_dendrogram.set_visible(False); g.ax_col_dendrogram.set_visible(False)

    legend = [Patch(facecolor='#0000ff', label='TP'), Patch(facecolor='#ffa500', label='FN'),
              Patch(facecolor='#ff0000', label='FP'), Patch(facecolor='#d3d3d3', label='TN')]
    plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.02, 1)); g.fig.suptitle(f"Clustermap: {dataset_name}", fontsize=30, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9, right=0.8); plt.show()

def plot_geometric_failure_dual(iso_h1, iso_h0, p_values, rejections, true_labels, dataset_name):
    """Dual Subplot: Isolation from H1 (Signal) and H0 (Noise)"""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    log_p = -np.log10(np.clip(p_values, 1e-10, 1.0))
    mask_tp = rejections & (true_labels == 0); mask_fn = ~rejections & (true_labels == 0)
    mask_fp = rejections & (true_labels == 1); mask_tn = ~rejections & (true_labels == 1)

    sorted_p = np.sort(p_values)
    bh_cut = sorted_p[np.max(np.where(sorted_p <= (np.arange(1, len(p_values)+1)/len(p_values)*0.1))[0])] if np.any(sorted_p <= 0.1) else None

    # Helper for scatter
    def draw_scatter(ax, iso_data, title):
        ax.scatter(iso_data[mask_tn], log_p[mask_tn], c='lightgray', alpha=0.7, label='TN', s=60)
        ax.scatter(iso_data[mask_fp], log_p[mask_fp], c='red', alpha=0.8, label='FP', marker='x', s=100, linewidth=3)
        ax.scatter(iso_data[mask_fn], log_p[mask_fn], c='orange', alpha=0.9, label='FN', marker='^', s=100)
        ax.scatter(iso_data[mask_tp], log_p[mask_tp], c='blue', alpha=0.8, label='TP', s=80)
        if bh_cut: ax.axhline(-np.log10(bh_cut), color='red', linestyle='--', linewidth=3, label='BH Cutoff')
        ax.axvline(0.5, color='gray', linestyle='--')
        ax.set_xlabel('Isolation (1=Far)'); ax.set_ylabel('-log10(P)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    draw_scatter(axes[0], iso_h1, "Isolation from SIGNAL (H1)")
    draw_scatter(axes[1], iso_h0, "Isolation from NOISE (H0)")

    fig.suptitle(f"Geometric Failure: {dataset_name}", fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout(); plt.show()

# ============================================================================
# 6. COMBINED PLOTS
# ============================================================================
def plot_combined_alpha_convergence(results_dict):
    plt.figure(figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    for (ds_name, res), color in zip(results_dict.items(), colors):
        hist = res['history']; iters = np.linspace(0, len(hist['losses']), len(hist['alpha_history']))
        plt.plot(iters, [a.mean() for a in hist['alpha_history']], '-', linewidth=3, color=color, label=f"{ds_name} (Mean)")
        plt.plot(iters, [a.min() for a in hist['alpha_history']], '--', linewidth=2, color=color, alpha=0.6)
        plt.plot(iters, [a.max() for a in hist['alpha_history']], ':', linewidth=2, color=color, alpha=0.6)
    plt.axhline(0, color='k'); plt.axhline(1, color='k'); plt.title('Combined Alpha Convergence', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1)); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_combined_geo_grid(results_dict, iso_key, title_suffix):
    """Generic Grid Plotter for H1 or H0 Isolation"""
    n_ds = len(results_dict); cols = 3; rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(24, 7*rows))
    axes = axes.flatten()
    for i, (ds_name, res) in enumerate(results_dict.items()):
        ax = axes[i]; iso = res[iso_key]; log_p = -np.log10(np.clip(res['p_values'], 1e-10, 1.0))
        rej, true_lbl = res['rejections'], res['true_labels']
        mask_tp = rej & (true_lbl==0); mask_fn = ~rej & (true_lbl==0)
        mask_fp = rej & (true_lbl==1); mask_tn = ~rej & (true_lbl==1)
        ax.scatter(iso[mask_tn], log_p[mask_tn], c='lightgray', alpha=0.3, s=20)
        ax.scatter(iso[mask_fp], log_p[mask_fp], c='red', marker='x', s=60, alpha=0.7)
        ax.scatter(iso[mask_fn], log_p[mask_fn], c='orange', marker='^', s=60, alpha=0.8)
        ax.scatter(iso[mask_tp], log_p[mask_tp], c='blue', s=40, alpha=0.7)
        ax.set_title(ds_name, fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--')
    for j in range(i+1, len(axes)): axes[j].axis('off')
    fig.suptitle(f"Geometric Failure ({title_suffix}) - All Datasets", fontsize=30, fontweight='bold', y=0.99)
    plt.tight_layout(); plt.show()

# ============================================================================
# 7. EXECUTION
# ============================================================================
def process_dataset(dataset_name):
    print(f"\n[{dataset_name}] Processing...")
    np.random.seed(RANDOM_STATE)
    data = load_and_sample_hierarchical(dataset_name)
    if data is None: return None
    X, p_values, true_labels = data
    if np.isnan(X).any(): X = np.nan_to_num(X)

    # Estimate
    def f0_func(p): return np.ones_like(p)
    z = stats.norm.ppf(1 - np.clip(p_values, 1e-10, 1-1e-10))
    mask = p_values < 0.2
    mu, sig = (np.mean(z[mask]), np.std(z[mask])) if mask.sum() > 10 else (2.5, 1.0)
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
    best_fac = SIGMA_GRID[np.argmin(cv_scores)]; print(f"[{dataset_name}] Best Sigma: {best_fac}")
    plot_cv_results(SIGMA_GRID, cv_scores, best_fac, dataset_name)

    # Final Run
    print(f"[{dataset_name}] Final Optimization...")
    best_sigma = base_sigma * best_fac
    K_final = compute_kernel_matrix(X, kernel_type='rbf', length_scale=best_sigma)
    c_final, history = run_optimization(K_final, f0_vals, f1_vals, verbose=True)
    alpha_final = K_final @ c_final
    plot_optimization_history(history, dataset_name)

    lfdr = (alpha_final * f0_vals) / (alpha_final * f0_vals + (1 - alpha_final) * f1_vals)
    q_vals = np.cumsum(np.sort(lfdr)) / np.arange(1, len(lfdr)+1)
    rejections = np.zeros(len(lfdr), dtype=bool)
    if np.sum(q_vals <= 0.1) > 0: rejections[np.argsort(lfdr)[:np.max(np.where(q_vals <= 0.1)[0])+1]] = True
    print(f"[{dataset_name}] Rej: {rejections.sum()}")
    plot_confusion_clustermap(K_final, true_labels, rejections, dataset_name)

    # Isolation
    h1_idx = np.where(true_labels == 0)[0] # Signal
    sim_h1 = np.sort(K_final[:, h1_idx], axis=1)[:, -min(50, len(h1_idx)):].mean(axis=1)
    iso_h1 = 1 - (sim_h1 - sim_h1.min())/(sim_h1.max() - sim_h1.min() + 1e-10)

    h0_idx = np.where(true_labels == 1)[0] # Noise
    sim_h0 = np.sort(K_final[:, h0_idx], axis=1)[:, -min(50, len(h0_idx)):].mean(axis=1)
    iso_h0 = 1 - (sim_h0 - sim_h0.min())/(sim_h0.max() - sim_h0.min() + 1e-10)

    plot_geometric_failure_dual(iso_h1, iso_h0, p_values, rejections, true_labels, dataset_name)

    return {'history': history, 'iso_h1': iso_h1, 'iso_h0': iso_h0, 'p_values': p_values, 'rejections': rejections, 'true_labels': true_labels}

if __name__ == "__main__":
    all_res = {}
    for ds in DATASETS_TO_RUN:
        res = process_dataset(ds)
        if res: all_res[ds] = res
    if all_res:
        plot_combined_alpha_convergence(all_res)
        plot_combined_geo_grid(all_res, 'iso_h1', 'From Signal H1')
        plot_combined_geo_grid(all_res, 'iso_h0', 'From Noise H0')