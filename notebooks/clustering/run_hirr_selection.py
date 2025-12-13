#!/usr/bin/env python3
"""
Dependent FDR with Smoothness Regularization
Script: Hierarchical Clustering - Smart Selection

Strategy:
1. Cluster ALL 10,000 points
2. Find all clusters with ≥30 points
3. Pick 3 most distant clusters
4. Sample 200 points total from those 3 clusters
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
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

CONFIG = {"33_skin" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 30,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"19_landsat" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 30,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"31_satimage-2" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 30,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0/2,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"30_satellite" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 50,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"41_Waveform" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 30,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0/2,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"25_musk" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 30,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 1.0,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"4_breastw" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 20,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 0.5,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"45_wine" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 20,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 0.5,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
"15_Hepatitis" :
    {'n_total': 200,  # Total points to select
     'n_clusters': 3,  # Number of clusters to pick
     'min_cluster_size': 20,  # Minimum points per cluster
     'candidate_pool': 10000,  # Cluster on this many points
     'sigma_factor': 0.5,
     'cluster_corruption': 0.1,
     'effect_strength': 'medium',
     'output_dir': 'results_hierarchical_v3',
     'random_state': 42},
}


# ============================================================================
# HIERARCHICAL CLUSTERING - SMART SELECTION
# ============================================================================
def hierarchical_cluster_selection_smart(K, n_total, n_clusters_to_pick, min_cluster_size):
    """
    1. Find all clusters with ≥ min_cluster_size points
    2. Pick n_clusters_to_pick most distant clusters
    3. Sample n_total points from those clusters

    Args:
        K: N×N kernel matrix (e.g., 10000×10000)
        n_total: Total points to select (e.g., 200)
        n_clusters_to_pick: How many clusters to pick (e.g., 3)
        min_cluster_size: Minimum points per cluster (e.g., 30)
    """
    N = K.shape[0]

    print(f"  Hierarchical clustering on {N} points...")

    # Convert kernel to distance
    K_normalized = K / (np.sqrt(np.outer(np.diag(K), np.diag(K))) + 1e-10)
    distance_matrix = 1 - K_normalized
    distance_matrix = np.clip(distance_matrix, 0, None)

    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed_dist, method='ward')

    # Step 1: Find all valid cluster configurations
    print(f"  Finding clusters with ≥{min_cluster_size} points...")

    valid_clusterings = []

    # Try different numbers of clusters
    for n_try in range(2, N // min_cluster_size + 1):
        labels = fcluster(Z, t=n_try, criterion='maxclust')
        unique_labels = np.unique(labels)

        # Check cluster sizes
        cluster_info = {}
        for lbl in unique_labels:
            size = np.sum(labels == lbl)
            if size >= min_cluster_size:
                cluster_info[lbl] = size

        # If we have enough valid clusters, record this configuration
        if len(cluster_info) >= n_clusters_to_pick:
            valid_clusterings.append({
                'n_clusters': n_try,
                'labels': labels,
                'valid_clusters': list(cluster_info.keys()),
                'cluster_sizes': cluster_info
            })

    if not valid_clusterings:
        raise ValueError(f"Could not find {n_clusters_to_pick} clusters with ≥{min_cluster_size} points")

    # Pick the configuration with the most valid clusters
    # (more options to choose from)
    best_config = max(valid_clusterings, key=lambda x: len(x['valid_clusters']))

    labels = best_config['labels']
    valid_cluster_labels = best_config['valid_clusters']

    print(f"  Found {len(valid_cluster_labels)} valid clusters")
    print(f"  Cluster sizes: {[best_config['cluster_sizes'][lbl] for lbl in valid_cluster_labels]}")

    # Step 2: Compute cluster centroids
    print(f"  Computing cluster centroids...")
    cluster_centroids = {}

    for lbl in valid_cluster_labels:
        cluster_mask = (labels == lbl)
        cluster_indices = np.where(cluster_mask)[0]
        # Centroid = mean kernel row for this cluster
        centroid = K[cluster_indices, :].mean(axis=0)
        cluster_centroids[lbl] = centroid

    # Step 3: Greedily pick n_clusters_to_pick most distant clusters
    print(f"  Selecting {n_clusters_to_pick} most distant clusters...")

    selected_clusters = []

    # Start with the largest cluster
    first_cluster = max(valid_cluster_labels, key=lambda lbl: best_config['cluster_sizes'][lbl])
    selected_clusters.append(first_cluster)
    remaining = [c for c in valid_cluster_labels if c != first_cluster]

    # Greedily add most distant clusters
    while len(selected_clusters) < n_clusters_to_pick and len(remaining) > 0:
        best_dist = -np.inf
        best_cluster = None

        for candidate in remaining:
            # Minimum distance to any already-selected cluster
            min_dist = min(
                np.linalg.norm(cluster_centroids[candidate] - cluster_centroids[sel])
                for sel in selected_clusters
            )

            if min_dist > best_dist:
                best_dist = min_dist
                best_cluster = candidate

        selected_clusters.append(best_cluster)
        remaining.remove(best_cluster)

        print(
            f"    Selected cluster {best_cluster} (size={best_config['cluster_sizes'][best_cluster]}, dist={best_dist:.3f})")

    print(f"  Final selected clusters: {selected_clusters}")

    # Step 4: Sample n_total points from the selected clusters
    print(f"  Sampling {n_total} points from {len(selected_clusters)} clusters...")

    points_per_cluster = n_total // len(selected_clusters)
    selected_indices = []
    final_cluster_labels = []

    for cluster_idx, lbl in enumerate(selected_clusters):
        cluster_mask = (labels == lbl)
        cluster_points = np.where(cluster_mask)[0]

        n_sample = min(points_per_cluster, len(cluster_points))
        sampled = np.random.choice(cluster_points, size=n_sample, replace=False)

        selected_indices.extend(sampled)
        final_cluster_labels.extend([cluster_idx] * n_sample)

    # Fill remaining points if needed
    while len(selected_indices) < n_total:
        # Add from largest selected cluster
        cluster_sizes_current = [
            np.sum(np.array(final_cluster_labels) == i)
            for i in range(len(selected_clusters))
        ]
        largest_cluster_idx = np.argmax(cluster_sizes_current)
        lbl = selected_clusters[largest_cluster_idx]

        cluster_points = np.where(labels == lbl)[0]
        available = [p for p in cluster_points if p not in selected_indices]

        if len(available) > 0:
            new_point = np.random.choice(available)
            selected_indices.append(new_point)
            final_cluster_labels.append(largest_cluster_idx)
        else:
            break

    print(f"  Selected {len(selected_indices)} points")
    final_sizes = [np.sum(np.array(final_cluster_labels) == i) for i in range(len(selected_clusters))]
    print(f"  Final cluster distribution: {final_sizes}")

    # For visualization, we want to know the cut height
    # Use the clustering level that gave us these clusters
    cut_height = Z[-(best_config['n_clusters'] - 1), 2]

    return np.array(selected_indices), np.array(final_cluster_labels), Z, labels, cut_height, selected_clusters


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
    """Run smart hierarchical selection"""
    data = load_from_ADbench(dataset_name)
    X_full = StandardScaler().fit_transform(data['X_train'])

    N_full = len(X_full)
    if N_full > config['candidate_pool']:
        pool_indices = np.random.choice(N_full, config['candidate_pool'], replace=False)
        X_pool = X_full[pool_indices]
        print(f"  Subsampled {config['candidate_pool']} from {N_full} points")
    else:
        X_pool = X_full
        print(f"  Using all {N_full} points")

    sigma = estimate_length_scale(X_pool, method='median') * config['sigma_factor']
    K_pool = compute_kernel_matrix(X_pool, kernel_type='rbf', length_scale=sigma)

    # Smart selection
    sel_idx, cluster_labels, Z, all_labels, cut_height, original_cluster_ids = \
        hierarchical_cluster_selection_smart(
            K_pool,
            config['n_total'],
            config['n_clusters'],
            config['min_cluster_size']
        )

    # Assign truth labels
    true_labels = np.zeros(len(sel_idx), dtype=int)
    group_ids = cluster_labels.copy()

    for gid in range(config['n_clusters']):
        mask = (group_ids == gid)
        n_g = mask.sum()

        if n_g == 0:
            continue

        if gid == 0:  # Signal cluster (H1)
            lbls = np.ones(n_g, dtype=int)
            n_corrupt = int(n_g * config['cluster_corruption'])
            if n_corrupt > 0:
                lbls[np.random.choice(n_g, n_corrupt, replace=False)] = 0
            true_labels[mask] = lbls
        elif gid == 1:  # Noise cluster (H0 with corruption)
            lbls = np.zeros(n_g, dtype=int)
            n_corrupt = int(n_g * config['cluster_corruption'])
            if n_corrupt > 0:
                lbls[np.random.choice(n_g, n_corrupt, replace=False)] = 1
            true_labels[mask] = lbls
        else:  # Background (pure H0)
            true_labels[mask] = 0

    p_values = generate_pvalues(true_labels, config['effect_strength'])
    K_selected = K_pool[np.ix_(sel_idx, sel_idx)]

    return {
        'K_selected': K_selected,
        'K_pool': K_pool,
        'group_ids': group_ids,
        'true_labels': true_labels,
        'p_values': p_values,
        'linkage_matrix': Z,
        'all_cluster_labels': all_labels,
        'selected_indices': sel_idx,
        'cut_height': cut_height,
        'original_cluster_ids': original_cluster_ids
    }


# ============================================================================
# PLOTTING
# ============================================================================
def plot_dendrogram_and_clustermap(result_dict, dataset_name, output_dir, config):
    """Create combined dendrogram + clustermap visualization"""
    Z = result_dict['linkage_matrix']
    K_selected = result_dict['K_selected']
    group_ids = result_dict['group_ids']
    true_labels = result_dict['true_labels']
    cut_height = result_dict['cut_height']

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3)

    # ========================================================================
    # LEFT: DENDROGRAM
    # ========================================================================
    ax_dend = fig.add_subplot(gs[0, 0])

    dend = dendrogram(
        Z,
        ax=ax_dend,
        no_labels=True,
        color_threshold=cut_height,
        above_threshold_color='gray'
    )

    ax_dend.axhline(y=cut_height, color='red', linestyle='--', linewidth=2,
                    label=f'Cut at height {cut_height:.2f}')

    ax_dend.set_title(f'Hierarchical Clustering Dendrogram\n{dataset_name}',
                      fontsize=14, fontweight='bold')
    ax_dend.set_xlabel('Sample Index', fontsize=12)
    ax_dend.set_ylabel('Ward Distance', fontsize=12)
    ax_dend.legend(fontsize=10)
    ax_dend.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # RIGHT: KERNEL MATRIX HEATMAP
    # ========================================================================
    ax_clust = fig.add_subplot(gs[0, 1])

    # Sort by cluster then truth
    sort_keys = group_ids * 10 + true_labels
    sort_idx = np.argsort(sort_keys)
    K_sorted = K_selected[np.ix_(sort_idx, sort_idx)]

    # Plot heatmap
    im = ax_clust.imshow(K_sorted, cmap='viridis', aspect='auto')

    # Color bars
    label_colors = pd.Series(true_labels[sort_idx]).map({0: '#2ecc71', 1: '#e74c3c'})
    group_map = {0: '#f39c12', 1: '#9b59b6', 2: '#95a5a6'}
    group_colors = pd.Series(group_ids[sort_idx]).map(group_map)

    from matplotlib.patches import Rectangle
    bar_width = K_sorted.shape[0] * 0.02

    for i in range(len(sort_idx)):
        color = group_colors.iloc[i]
        rect = Rectangle((-bar_width * 2, i), bar_width, 1,
                         facecolor=color, edgecolor='none')
        ax_clust.add_patch(rect)

        color = label_colors.iloc[i]
        rect = Rectangle((-bar_width, i), bar_width, 1,
                         facecolor=color, edgecolor='none')
        ax_clust.add_patch(rect)

    ax_clust.set_xlim(-bar_width * 2.5, K_sorted.shape[1])
    ax_clust.set_ylim(K_sorted.shape[0], 0)

    ax_clust.set_title(f'Selected Points Kernel Matrix\n{len(true_labels)} points from 3 distant clusters',
                       fontsize=14, fontweight='bold')
    ax_clust.set_xlabel('Point Index (sorted)', fontsize=12)
    ax_clust.set_ylabel('Point Index (sorted)', fontsize=12)

    cbar = plt.colorbar(im, ax=ax_clust, fraction=0.046, pad=0.04)
    cbar.set_label('Kernel Similarity', fontsize=10)

    # Legend
    legend_elements = [
        Patch(facecolor='#f39c12', label='Cluster 0 (Signal)'),
        Patch(facecolor='#9b59b6', label='Cluster 1 (Noise)'),
        Patch(facecolor='#95a5a6', label='Cluster 2 (Background)'),
        Line2D([0], [0], color='white', label='---'),
        Patch(facecolor='#e74c3c', label='True H1'),
        Patch(facecolor='#2ecc71', label='True H0')
    ]
    ax_clust.legend(handles=legend_elements, loc='upper left',
                    bbox_to_anchor=(1.15, 1), fontsize=10)

    # Overall title
    h1_count = np.sum(true_labels)
    cluster_sizes = [np.sum(group_ids == c) for c in range(config['n_clusters'])]
    fig.suptitle(f'{dataset_name} - Smart Hierarchical Selection\n' +
                 f'H1={h1_count}, H0={len(true_labels) - h1_count} | Cluster sizes: {cluster_sizes}',
                 fontsize=16, fontweight='bold', y=0.98)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, f'hierarchical_{dataset_name}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {out_path}")


def plot_sns_clustermap_separate(result_dict, dataset_name, output_dir):
    """Seaborn clustermap version"""
    K_selected = result_dict['K_selected']
    group_ids = result_dict['group_ids']
    true_labels = result_dict['true_labels']

    sort_keys = group_ids * 10 + true_labels
    sort_idx = np.argsort(sort_keys)
    K_sorted = K_selected[np.ix_(sort_idx, sort_idx)]

    label_colors = pd.Series(true_labels[sort_idx]).map({0: '#2ecc71', 1: '#e74c3c'})
    group_map = {0: '#f39c12', 1: '#9b59b6', 2: '#95a5a6'}
    group_colors = pd.Series(group_ids[sort_idx]).map(group_map)

    row_colors = pd.DataFrame({
        'Cluster': group_colors.values,
        'Truth': label_colors.values
    })

    g = sns.clustermap(
        K_sorted,
        row_cluster=False,
        col_cluster=False,
        # row_colors=row_colors,
        cmap='viridis',
        figsize=(12, 12),
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Kernel Similarity'}
    )

    legend_elements = [
        Patch(facecolor='#f39c12', label='Cluster 0 (Signal)'),
        Patch(facecolor='#9b59b6', label='Cluster 1 (Noise)'),
        Patch(facecolor='#95a5a6', label='Cluster 2 (Background)'),
        Line2D([0], [0], color='white', label='---'),
        Patch(facecolor='#e74c3c', label='True H1'),
        Patch(facecolor='#2ecc71', label='True H0')
    ]
    g.ax_heatmap.legend(handles=legend_elements, loc='upper left',
                        bbox_to_anchor=(1.02, 1), fontsize=10)

    cluster_sizes = [np.sum(group_ids == c) for c in range(3)]
    plt.suptitle(f'{dataset_name}\nCluster sizes: {cluster_sizes}',
                 y=0.98, fontsize=14, fontweight='bold')

    out_path = os.path.join(output_dir, f'sns_clustermap_{dataset_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # print("=== Smart Hierarchical Clustering Selection ===")
    # print(f"Output: {CONFIG['output_dir']}")
    # print(f"Strategy: Cluster {CONFIG['candidate_pool']} points, find clusters ≥{CONFIG['min_cluster_size']}")
    # print(f"         Pick {CONFIG['n_clusters']} most distant, sample {CONFIG['n_total']} total points")
    print("-" * 60)

    results_log = []

    for i, d_name in enumerate(DATASETS):
        print(f"\n[{i + 1}/{len(DATASETS)}] {d_name}")
        start_t = time.time()

        try:
            result_dict = run_single_dataset(d_name, CONFIG[d_name])

            plot_dendrogram_and_clustermap(result_dict, d_name,
                                           CONFIG[d_name]['output_dir'], CONFIG[d_name])

            plot_sns_clustermap_separate(result_dict, d_name, CONFIG[d_name]['output_dir'])

            elapsed = time.time() - start_t
            h1 = np.sum(result_dict['true_labels'])
            cluster_sizes = [np.sum(result_dict['group_ids'] == c)
                             for c in range(CONFIG[d_name]['n_clusters'])]

            results_log.append({
                'Dataset': d_name,
                'Status': 'OK',
                'H1': h1,
                'H0': len(result_dict['true_labels']) - h1,
                'C0': cluster_sizes[0],
                'C1': cluster_sizes[1],
                'C2': cluster_sizes[2],
                'Time': round(elapsed, 2)
            })

            print(f"  ✓ {elapsed:.1f}s | H1={h1} | Clusters={cluster_sizes}")

        except Exception as e:
            elapsed = time.time() - start_t
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()

            results_log.append({
                'Dataset': d_name,
                'Status': 'FAILED',
                'H1': 0, 'H0': 0, 'C0': 0, 'C1': 0, 'C2': 0,
                'Time': round(elapsed, 2)
            })

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    df = pd.DataFrame(results_log)
    print("\n" + df.to_string(index=False))

    csv_path = os.path.join(CONFIG[DATASETS[0]]['output_dir'], 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    ok = df[df['Status'] == 'OK']
    if len(ok) > 0:
        print(f"\nStats:")
        print(f"  Total time: {ok['Time'].sum():.1f}s")
        print(f"  Avg time: {ok['Time'].mean():.1f}s")