#!/usr/bin/env python3
"""
Dependent FDR with Smoothness Regularization
Script: Optimal Sub-selection via MILP - FIXED VERSION

Just fixes the Intel MKL error in your original MILP code.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from spatial_fdr_evaluation.data.adbench_loader import load_from_ADbench
    from spatial_fdr_evaluation.methods.kernels import compute_kernel_matrix, estimate_length_scale
except ImportError:
    print("WARNING: Could not import 'spatial_fdr_evaluation'. Check your python path.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASETS = [
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

CONFIG = {
    'n_total': 200,
    'n_clusters': 3,
    'candidate_pool': 1000,  # Your original setting - keeps MILP tractable
    'sigma_factor': 1.0,
    'cluster_corruption': 0.1,
    'effect_strength': 'medium',
    'output_dir': 'results_milp_batch',
    'random_state': 42
}


# ============================================================================
# MILP SOLVER (Your original, with minor stability fix)
# ============================================================================
def optimal_selection_milp(K, n, n_clusters):
    """
    Fully vectorized MILP
    """
    N = K.shape[0]
    K = (K + K.T) / 2.0 + 1e-8 * np.eye(N)

    x = cp.Variable(N, boolean=True)
    z = cp.Variable((N, n_clusters), boolean=True)

    # Constraints
    constraints = [z[i, :].sum() == x[i] for i in range(N)]
    constraints.append(cp.sum(x) == n)
    constraints.extend([cp.sum(z[:, c]) >= 1 for c in range(n_clusters)])

    # Within-cluster: sum over all clusters
    within_sim = sum(
        (cp.quad_form(z[:, c], K) - cp.sum(cp.multiply(np.diag(K), z[:, c]))) / 2
        for c in range(n_clusters)
    )

    # Between-cluster: sum over all pairs of clusters
    between_sim = sum(
        z[:, c1].T @ K @ z[:, c2]
        for c1 in range(n_clusters)
        for c2 in range(c1 + 1, n_clusters)
    )

    objective = cp.Maximize(within_sim - 0.5 * between_sim)
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.GUROBI, verbose=False)
    except:
        problem.solve(verbose=False)

    if x.value is None:
        raise RuntimeError("Solver failed")

    selected = np.where(x.value > 0.5)[0]
    labels = np.argmax(z.value[x.value > 0.5], axis=1)

    return selected, labels


# ============================================================================
# DATA PIPELINE
# ============================================================================
def generate_pvalues(labels, effect_strength='medium'):
    p_values = np.zeros(len(labels))
    alphas = {'weak': 0.5, 'medium': 0.05, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)

    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())
    p_values[labels == 0] = np.random.beta(a, 5, size=(labels == 0).sum())
    return np.clip(p_values, 1e-10, 1.0)


def run_single_dataset(dataset_name, config):
    # Load
    data = load_from_ADbench(dataset_name)
    X_full = StandardScaler().fit_transform(data['X_train'])

    # Subsample pool
    N_full = len(X_full)
    if N_full > config['candidate_pool']:
        pool_indices = np.random.choice(N_full, config['candidate_pool'], replace=False)
        X_pool = X_full[pool_indices]
    else:
        pool_indices = np.arange(N_full)
        X_pool = X_full

    # Kernel
    sigma = estimate_length_scale(X_pool, method='median') * config['sigma_factor']
    K_pool = compute_kernel_matrix(X_pool, kernel_type='rbf', length_scale=sigma)

    # MILP
    sel_idx_local, cluster_labels = optimal_selection_milp(
        K_pool, config['n_total'], config['n_clusters']
    )

    # Assign Roles
    true_labels = np.zeros(config['n_total'], dtype=int)
    group_ids = cluster_labels.copy()

    for gid in range(config['n_clusters']):
        mask = (group_ids == gid)
        n_g = mask.sum()

        if gid == 0:
            lbls = np.ones(n_g, dtype=int)
            n_corrupt = int(n_g * config['cluster_corruption'])
            if n_corrupt > 0:
                lbls[np.random.choice(n_g, n_corrupt, replace=False)] = 0
            true_labels[mask] = lbls
        elif gid == 1:
            lbls = np.zeros(n_g, dtype=int)
            n_corrupt = int(n_g * config['cluster_corruption'])
            if n_corrupt > 0:
                lbls[np.random.choice(n_g, n_corrupt, replace=False)] = 1
            true_labels[mask] = lbls
        else:
            true_labels[mask] = 0

    p_values = generate_pvalues(true_labels, config['effect_strength'])
    K_selected = K_pool[np.ix_(sel_idx_local, sel_idx_local)]

    return K_selected, group_ids, true_labels, p_values


# ============================================================================
# PLOTTING
# ============================================================================
def save_structural_clustermap(K, group_ids, true_labels, dataset_name, output_dir):
    # Sort for visualization
    sort_keys = group_ids * 10 + true_labels
    sort_idx = np.argsort(sort_keys)
    K_sorted = K[np.ix_(sort_idx, sort_idx)]

    # Colors
    label_colors = pd.Series(true_labels[sort_idx]).map({0: '#2ecc71', 1: '#e74c3c'})
    group_map = {0: '#f39c12', 1: '#9b59b6', 2: '#95a5a6'}
    group_colors = pd.Series(group_ids[sort_idx]).map(group_map)

    row_colors = pd.DataFrame({
        'Cluster': group_colors.values,
        'Truth': label_colors.values
    }, index=range(len(true_labels)))

    plt.figure(figsize=(10, 10))
    g = sns.clustermap(
        K_sorted,
        row_cluster=False, col_cluster=False,
        row_colors=row_colors,
        cmap='viridis',
        xticklabels=False, yticklabels=False
    )

    # Legend
    legend_elements = [
        Patch(facecolor='#f39c12', label='Cluster 0 (Signal)'),
        Patch(facecolor='#9b59b6', label='Cluster 1 (Noise)'),
        Patch(facecolor='#95a5a6', label='Cluster 2 (Bg)'),
        Line2D([0], [0], color='white', label='-------'),
        Patch(facecolor='#e74c3c', label='True H1'),
        Patch(facecolor='#2ecc71', label='True H0')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.suptitle(f"MILP Structure: {dataset_name}", y=0.95)

    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, f'milp_{dataset_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    np.random.seed(CONFIG['random_state'])

    print("=== MILP Data Selection: Batch Run ===")
    print(f"Output Directory: {CONFIG['output_dir']}")
    print("-" * 50)

    results_log = []

    for i, d_name in enumerate(DATASETS):
        print(f"[{i + 1}/{len(DATASETS)}] Processing: {d_name} ...", end=" ", flush=True)
        start_t = time.time()

        try:
            # Run Pipeline
            K_sel, groups, y_true, p_vals = run_single_dataset(d_name, CONFIG)

            # Save Plot
            save_structural_clustermap(K_sel, groups, y_true, d_name, CONFIG['output_dir'])

            # Log stats
            elapsed = time.time() - start_t
            h1_count = np.sum(y_true)
            results_log.append({
                'Dataset': d_name,
                'Status': 'Success',
                'H1_Count': h1_count,
                'Time(s)': round(elapsed, 2)
            })
            print(f"Done ({elapsed:.1f}s) | H1: {h1_count}")

        except Exception as e:
            elapsed = time.time() - start_t
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"   -> Error: {e}")
            results_log.append({
                'Dataset': d_name,
                'Status': 'Failed',
                'H1_Count': 0,
                'Time(s)': round(elapsed, 2)
            })

    print("-" * 50)
    print("Batch Run Complete.")
    print(pd.DataFrame(results_log))