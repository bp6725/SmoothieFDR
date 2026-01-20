"""
Visualization for Euclidean Benchmark with Cross-Validation (main_bench_cv.py)

This script loads cached results and generates all paper-grade visualizations.
Run main_bench_cv.py first to generate the cache file.

Usage:
    python plot_bench_cv.py
    python plot_bench_cv.py --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import os
import argparse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
CACHE_FILE = "bench_cv_results.pkl"
OUTPUT_DIR = "/home/benny/Repos/SmoothieFDR/results/figures/bench_cv"

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    'figure.titlesize': 28, 'lines.linewidth': 3, 'grid.alpha': 0.4
})

# ============================================================================
# INDIVIDUAL DATASET PLOTS
# ============================================================================

def plot_cv_results(sigma_grid, cv_scores, best_sigma, dataset_name, save_path=None):
    """Plot cross-validation results for sigma selection."""
    plt.figure(figsize=(12, 8))
    plt.plot(sigma_grid, cv_scores, 'bo-', markerfacecolor='white', markersize=10)
    plt.plot(best_sigma, np.min(cv_scores), 'ro', markersize=15, label=f'Best: {best_sigma}')
    plt.xscale('log')
    plt.xticks(sigma_grid, [str(s) for s in sigma_grid])
    plt.xlabel('Sigma')
    plt.ylabel('NLL')
    plt.title(f'CV: {dataset_name}', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_optimization_history(history, dataset_name, save_path=None):
    """Plot optimization convergence history."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    axes[0, 0].plot(history['losses'], 'b-')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(history['grad_norms'], 'r-')
    axes[0, 1].set_title('Gradient Norm')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].grid(True, alpha=0.3)

    iters = np.linspace(0, len(history['losses'])-1, len(history['alpha_history']), dtype=int)
    alphas = history['alpha_history']
    axes[1, 0].plot(iters, [a.mean() for a in alphas], 'g-', label='Mean')
    axes[1, 0].plot(iters, [a.min() for a in alphas], 'b--', label='Min')
    axes[1, 0].plot(iters, [a.max() for a in alphas], 'r--', label='Max')
    axes[1, 0].legend()
    axes[1, 0].set_title('Alpha Statistics')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, history['violations'], 'purple', marker='o')
    axes[1, 1].set_title('Constraint Violations')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Optimization: {dataset_name}", fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_clustermap(K, true_labels, rejections, dataset_name, save_path=None):
    """Plot kernel clustermap with confusion categories."""
    categories = np.zeros(len(true_labels), dtype=int)
    categories[(true_labels == 1) & (~rejections)] = 0  # TN
    categories[(true_labels == 1) & (rejections)] = 1   # FP
    categories[(true_labels == 0) & (~rejections)] = 2  # FN
    categories[(true_labels == 0) & (rejections)] = 3   # TP

    color_map = {0: '#d3d3d3', 1: '#ff0000', 2: '#ffa500', 3: '#0000ff'}
    row_colors = pd.Series(categories).map(color_map)
    row_colors.name = "Outcome"

    g = sns.clustermap(pd.DataFrame(K), row_colors=row_colors, col_colors=row_colors,
                       cmap="viridis", figsize=(14, 14), cbar_pos=None, dendrogram_ratio=0.01)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    legend = [Patch(facecolor='#0000ff', label='TP'),
              Patch(facecolor='#ffa500', label='FN'),
              Patch(facecolor='#ff0000', label='FP'),
              Patch(facecolor='#d3d3d3', label='TN')]
    plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.02, 1))
    g.fig.suptitle(f"Clustermap: {dataset_name}", fontsize=30, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9, right=0.8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_geometric_failure_dual(iso_h1, iso_h0, p_values, rejections, true_labels, dataset_name, save_path=None):
    """Dual Subplot: Isolation from H1 (Signal) and H0 (Noise)."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    log_p = -np.log10(np.clip(p_values, 1e-10, 1.0))

    mask_tp = rejections & (true_labels == 0)
    mask_fn = ~rejections & (true_labels == 0)
    mask_fp = rejections & (true_labels == 1)
    mask_tn = ~rejections & (true_labels == 1)

    sorted_p = np.sort(p_values)
    bh_cut = None
    if np.any(sorted_p <= 0.1):
        idx = np.where(sorted_p <= (np.arange(1, len(p_values)+1)/len(p_values)*0.1))[0]
        if len(idx) > 0:
            bh_cut = sorted_p[np.max(idx)]

    def draw_scatter(ax, iso_data, title):
        ax.scatter(iso_data[mask_tn], log_p[mask_tn], c='lightgray', alpha=0.7, label='TN', s=60)
        ax.scatter(iso_data[mask_fp], log_p[mask_fp], c='red', alpha=0.8, label='FP', marker='x', s=100, linewidth=3)
        ax.scatter(iso_data[mask_fn], log_p[mask_fn], c='orange', alpha=0.9, label='FN', marker='^', s=100)
        ax.scatter(iso_data[mask_tp], log_p[mask_tp], c='blue', alpha=0.8, label='TP', s=80)
        if bh_cut:
            ax.axhline(-np.log10(bh_cut), color='red', linestyle='--', linewidth=3, label='BH Cutoff')
        ax.axvline(0.5, color='gray', linestyle='--')
        ax.set_xlabel('Isolation (1=Far)')
        ax.set_ylabel('-log10(P)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    draw_scatter(axes[0], iso_h1, "Isolation from SIGNAL (H1)")
    draw_scatter(axes[1], iso_h0, "Isolation from NOISE (H0)")

    fig.suptitle(f"Geometric Failure: {dataset_name}", fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spatial_adaptation_per_group(iso_h1, iso_h0, alpha_values, group_ids, dataset_name, save_path=None):
    """Combined Plot: All 3 groups on ONE graph showing spatial adaptation."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    groups = {
        0: {'name': 'Signal Cluster (H1)', 'color': 'blue', 'style': '-', 'iso': iso_h1},
        1: {'name': 'Noise Cluster (H0)', 'color': 'red', 'style': '--', 'iso': iso_h0},
        2: {'name': 'Background (H0)', 'color': 'gray', 'style': ':', 'iso': iso_h1}
    }

    for gid, meta in groups.items():
        mask = (group_ids == gid)
        if np.sum(mask) < 5:
            continue

        iso_g = meta['iso'][mask]
        alpha_g = alpha_values[mask]

        sort_idx = np.argsort(iso_g)
        iso_sorted = iso_g[sort_idx]
        alpha_sorted = alpha_g[sort_idx]

        ax.scatter(iso_sorted, alpha_sorted, color=meta['color'], alpha=0.2, s=30)

        window = max(5, len(iso_sorted) // 5)
        mean = pd.Series(alpha_sorted).rolling(window=window, center=True).mean()
        std = pd.Series(alpha_sorted).rolling(window=window, center=True).std()

        ax.plot(iso_sorted, mean, color=meta['color'], linestyle=meta['style'], linewidth=4, label=meta['name'])
        ax.fill_between(iso_sorted, mean - std, mean + std, color=meta['color'], alpha=0.1)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Geometric Isolation from Local Core')
    ax.set_ylabel('Effective Priority (Alpha)')
    ax.set_title(f"Spatial Adaptation Profile: {dataset_name}", fontsize=20, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# COMBINED / AGGREGATE PLOTS
# ============================================================================

def plot_combined_alpha_convergence(results_dict, save_path=None):
    """Combined alpha convergence across all datasets."""
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (ds_name, res), color in zip(results_dict.items(), colors):
        hist = res['history']
        alpha_hist = hist['alpha_history'][0:1000]
        iters = np.linspace(0, 1000, len(alpha_hist))
        plt.plot(iters, [a.mean() for a in alpha_hist], '-', linewidth=3, color=color, label=f"{ds_name} (Mean)")
        plt.plot(iters, [a.min() for a in alpha_hist], '--', linewidth=2, color=color, alpha=0.6)
        plt.plot(iters, [a.max() for a in alpha_hist], ':', linewidth=2, color=color, alpha=0.6)

    plt.axhline(0, color='k')
    plt.axhline(1, color='k')
    plt.title('Alpha Convergence', fontweight='bold',fontsize = 36)
    plt.xlabel('Iteration',fontsize = 24)
    plt.ylabel('Alpha',fontsize = 24)
    # plt.legend(bbox_to_anchor=(1.02, 1))
    plt.legend(bbox_to_anchor=(0.8, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("done")
    else:
        plt.show()


def plot_combined_geo_grid(results_dict, iso_key, title_suffix, save_path=None):
    """Generic Grid Plotter for H1 or H0 Isolation across all datasets."""
    n_ds = len(results_dict)
    cols = 3
    rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(24, 7*rows))
    axes = axes.flatten()

    for i, (ds_name, res) in enumerate(results_dict.items()):
        ax = axes[i]
        iso = res[iso_key]
        log_p = -np.log10(np.clip(res['p_values'], 1e-10, 1.0))
        rej, true_lbl = res['rejections'], res['true_labels']

        mask_tp = rej & (true_lbl == 0)
        mask_fn = ~rej & (true_lbl == 0)
        mask_fp = rej & (true_lbl == 1)
        mask_tn = ~rej & (true_lbl == 1)

        ax.scatter(iso[mask_tn], log_p[mask_tn], c='lightgray', alpha=0.3, s=20)
        ax.scatter(iso[mask_fp], log_p[mask_fp], c='red', marker='x', s=60, alpha=0.7)
        ax.scatter(iso[mask_fn], log_p[mask_fn], c='orange', marker='^', s=60, alpha=0.8)
        ax.scatter(iso[mask_tp], log_p[mask_tp], c='blue', s=40, alpha=0.7)
        ax.set_title(ds_name, fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Geometric Failure ({title_suffix}) - All Datasets", fontsize=30, fontweight='bold', y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_spatial_adaptation(results_dict, save_path=None):
    """
    Grand Summary Plot: Aggregates data from ALL datasets into one graph.
    Demonstrates the global consistency of the 'Fork Pattern'.
    """
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    global_data = {
        0: {'iso': [], 'alpha': [], 'color': 'blue', 'name': 'Global Signal (H1)', 'style': '-'},
        1: {'iso': [], 'alpha': [], 'color': 'red', 'name': 'Global Noise Cluster (H0)', 'style': '--'},
        2: {'iso': [], 'alpha': [], 'color': 'gray', 'name': 'Global Background (H0)', 'style': ':'}
    }

    # Collect data from all datasets
    for ds_name, res in results_dict.items():
        iso_h1 = res['iso_h1']
        iso_h0 = res['iso_h0']
        alpha = res['history']['alpha_history'][-1]
        groups = res['final_group_ids']

        # Group 0: Signal -> Use Iso_H1
        mask0 = (groups == 0)
        global_data[0]['iso'].extend(iso_h1[mask0])
        global_data[0]['alpha'].extend(alpha[mask0])

        # Group 1: Noise Trap -> Use Iso_H0
        mask1 = (groups == 1)
        global_data[1]['iso'].extend(iso_h0[mask1])
        global_data[1]['alpha'].extend(alpha[mask1])

        # Group 2: Background -> Use Iso_H1
        mask2 = (groups == 2)
        global_data[2]['iso'].extend(iso_h1[mask2])
        global_data[2]['alpha'].extend(alpha[mask2])

    # Plot Global Trends
    for gid, meta in global_data.items():
        if len(meta['iso']) == 0:
            continue

        iso_all = np.array(meta['iso'])
        alpha_all = np.array(meta['alpha'])

        ax.scatter(iso_all, alpha_all, color=meta['color'], alpha=0.05, s=10)

        sort_idx = np.argsort(iso_all)
        iso_sorted = iso_all[sort_idx]
        alpha_sorted = alpha_all[sort_idx]

        window = max(50, len(iso_sorted) // 20)
        mean = pd.Series(alpha_sorted).rolling(window=window, center=True).mean()
        std = pd.Series(alpha_sorted).rolling(window=window, center=True).std()

        ax.plot(iso_sorted, mean, color=meta['color'], linestyle=meta['style'], linewidth=5, label=meta['name'])
        ax.fill_between(iso_sorted, mean - (std / 2), mean + (std / 2), color=meta['color'], alpha=0.15)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Geometric Isolation from Local Core\n(0 = Cluster Center, 1 = Scattered)')
    ax.set_ylabel('Effective Priority (Alpha)')
    ax.set_title(f"Global Spatial Adaptation (All {len(results_dict)} Datasets)\nThe 'Fork' Pattern Consistency",
                 fontsize=22, fontweight='bold', pad=20)

    legend_elements = [
        Line2D([0], [0], color='blue', lw=5, label='Signal (H1)'),
        Line2D([0], [0], color='red', lw=5, linestyle='--', label='Noise Trap (H0)'),
        Line2D([0], [0], color='gray', lw=5, linestyle=':', label='Background (H0)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def load_cache(cache_path):
    """Load cached results."""
    with open(cache_path, 'rb') as f:
        return pkl.load(f)


def generate_all_plots(results_dict, output_dir, show=False):
    """Generate all visualizations from cached results."""
    os.makedirs(output_dir, exist_ok=True)

    if False :
        print("Generating per-dataset plots...")
        for ds_name, res in results_dict.items():
            ds_dir = os.path.join(output_dir, ds_name)
            os.makedirs(ds_dir, exist_ok=True)

            # CV Results
            plot_cv_results(
                res['sigma_grid'], res['cv_scores'], res['best_sigma_factor'], ds_name,
                save_path=os.path.join(ds_dir, 'cv_results.png') if not show else None
            )

            # Optimization History
            plot_optimization_history(
                res['history'], ds_name,
                save_path=os.path.join(ds_dir, 'optimization_history.png') if not show else None
            )

            # Confusion Clustermap
            plot_confusion_clustermap(
                res['K_final'], res['true_labels'], res['rejections'], ds_name,
                save_path=os.path.join(ds_dir, 'confusion_clustermap.png') if not show else None
            )

            # Geometric Failure Dual
            plot_geometric_failure_dual(
                res['iso_h1'], res['iso_h0'], res['p_values'], res['rejections'], res['true_labels'], ds_name,
                save_path=os.path.join(ds_dir, 'geometric_failure.png') if not show else None
            )

            # Spatial Adaptation
            plot_spatial_adaptation_per_group(
                res['iso_h1'], res['iso_h0'], res['alpha_final'], res['final_group_ids'], ds_name,
                save_path=os.path.join(ds_dir, 'spatial_adaptation.png') if not show else None
            )

            print(f"  - {ds_name}: done")

    print("\nGenerating combined plots...")

    # Combined Alpha Convergence
    plot_combined_alpha_convergence(
        results_dict,
        save_path=os.path.join(output_dir, 'combined_alpha_convergence.png') if not show else None
    )

    # Combined Geo Grid - H1
    plot_combined_geo_grid(
        results_dict, 'iso_h1', 'From Signal H1',
        save_path=os.path.join(output_dir, 'combined_geo_h1.png') if not show else None
    )

    # Combined Geo Grid - H0
    plot_combined_geo_grid(
        results_dict, 'iso_h0', 'From Noise H0',
        save_path=os.path.join(output_dir, 'combined_geo_h0.png') if not show else None
    )

    # Combined Spatial Adaptation (Fork Pattern)
    plot_combined_spatial_adaptation(
        results_dict,
        save_path=os.path.join(output_dir, 'combined_spatial_adaptation.png') if not show else None
    )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations from bench_cv results')
    parser.add_argument('--cache', default=os.path.join(CACHE_DIR, CACHE_FILE), help='Path to cache file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--show',default=False, action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache}")
    results = load_cache(args.cache)
    print(f"Loaded {len(results)} datasets")

    generate_all_plots(results, args.output_dir, show=args.show)
