"""
Visualization for All-Space Inference (main_allspace_cv.py)

This script loads cached results and generates visualizations.
Run main_allspace_cv.py first to generate the cache file.

Usage:
    python plot_allspace_cv.py
    python plot_allspace_cv.py --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import argparse

# --- CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
CACHE_FILE = "allspace_cv_results.pkl"
OUTPUT_DIR = "/home/benny/Repos/SmoothieFDR/results/figures/allspace_cv"

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    'figure.titlesize': 28, 'lines.linewidth': 3, 'grid.alpha': 0.4
})


def plot_aggregated_histogram(results_dict, save_path=None):
    """
    Aggregates results from all datasets and plots a single Histogram.
    Visualizes separation between Signal (H1, Blue) and Noise (H0, Red).
    """
    all_h1_alphas = []
    all_h0_alphas = []

    for ds_name, res in results_dict.items():
        y_true = res['y_true']
        alpha_pred = res['alpha_pred']

        # Collect Signal (y=0) and Noise (y=1)
        # Assuming 0 is Signal (H1) and 1 is Noise (H0) as per typical ADbench loading
        all_h1_alphas.extend(alpha_pred[y_true == 0])
        all_h0_alphas.extend(alpha_pred[y_true == 1])

    plt.figure(figsize=(12, 8))

    # Plot Histograms
    if len(all_h0_alphas) > 0:
        plt.hist(all_h0_alphas, bins=30, alpha=0.5, color='red', label='Noise (H0)', density=True, edgecolor='black')

    if len(all_h1_alphas) > 0:
        plt.hist(all_h1_alphas, bins=30, alpha=0.5, color='blue', label='Signal (H1)', density=True, edgecolor='black')

    plt.title('Global Inference: Aggregated Alpha Distribution on Hidden Test Sets', fontweight='bold')
    plt.xlabel('Predicted Alpha')
    plt.ylabel('Density (Normalized Count)')
    plt.xlim(0, 1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_dataset_histogram(results_dict, save_path=None):
    """Plot alpha distributions per dataset."""
    n_ds = len(results_dict)
    cols = 3
    rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten()

    for i, (ds_name, res) in enumerate(results_dict.items()):
        ax = axes[i]
        y_true = res['y_true']
        alpha_pred = res['alpha_pred']

        h1_alphas = alpha_pred[y_true == 0]
        h0_alphas = alpha_pred[y_true == 1]

        if len(h0_alphas) > 0:
            ax.hist(h0_alphas, bins=20, alpha=0.5, color='red', label='H0', density=True)
        if len(h1_alphas) > 0:
            ax.hist(h1_alphas, bins=20, alpha=0.5, color='blue', label='H1', density=True)

        ax.set_title(ds_name, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Per-Dataset Alpha Distribution on Test Sets', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cv_summary(results_dict, save_path=None):
    """Plot CV scores summary across datasets."""
    n_ds = len(results_dict)
    cols = 3
    rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten()

    for i, (ds_name, res) in enumerate(results_dict.items()):
        ax = axes[i]
        sigma_grid = res['sigma_grid']
        cv_scores = res['cv_scores']
        best_fac = res['best_sigma_factor']

        ax.plot(sigma_grid, cv_scores, 'bo-', markerfacecolor='white', markersize=8)
        ax.plot(best_fac, np.min(cv_scores), 'ro', markersize=12, label=f'Best: {best_fac}')
        ax.set_xscale('log')
        ax.set_title(ds_name, fontweight='bold')
        ax.set_xlabel('Sigma Factor')
        ax.set_ylabel('CV NLL')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Cross-Validation Results', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_separation_metrics(results_dict, save_path=None):
    """Plot separation metrics (mean alpha for H0 vs H1) per dataset."""
    datasets = []
    h0_means = []
    h1_means = []
    separations = []

    for ds_name, res in results_dict.items():
        y_true = res['y_true']
        alpha_pred = res['alpha_pred']

        h0_mean = np.mean(alpha_pred[y_true == 1]) if np.any(y_true == 1) else 0
        h1_mean = np.mean(alpha_pred[y_true == 0]) if np.any(y_true == 0) else 0

        datasets.append(ds_name)
        h0_means.append(h0_mean)
        h1_means.append(h1_mean)
        separations.append(h1_mean - h0_mean)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot of means
    x = np.arange(len(datasets))
    width = 0.35
    axes[0].bar(x - width/2, h0_means, width, label='H0 (Noise)', color='red', alpha=0.7)
    axes[0].bar(x + width/2, h1_means, width, label='H1 (Signal)', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Predicted Alpha')
    axes[0].set_title('Mean Alpha by Class', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Separation score
    colors = ['green' if s > 0 else 'red' for s in separations]
    axes[1].bar(x, separations, color=colors, alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].set_ylabel('Separation (H1 - H0)')
    axes[1].set_title('Class Separation Score', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_cache(cache_path):
    """Load cached results."""
    with open(cache_path, 'rb') as f:
        return pkl.load(f)


def generate_all_plots(results_dict, output_dir, show=False):
    """Generate all visualizations from cached results."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")

    # Aggregated histogram
    plot_aggregated_histogram(
        results_dict,
        save_path=os.path.join(output_dir, 'aggregated_histogram.png') if not show else None
    )
    print("  - Aggregated histogram: done")

    # Per-dataset histograms
    plot_per_dataset_histogram(
        results_dict,
        save_path=os.path.join(output_dir, 'per_dataset_histograms.png') if not show else None
    )
    print("  - Per-dataset histograms: done")

    # CV summary
    plot_cv_summary(
        results_dict,
        save_path=os.path.join(output_dir, 'cv_summary.png') if not show else None
    )
    print("  - CV summary: done")

    # Separation metrics
    plot_separation_metrics(
        results_dict,
        save_path=os.path.join(output_dir, 'separation_metrics.png') if not show else None
    )
    print("  - Separation metrics: done")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations from allspace_cv results')
    parser.add_argument('--cache', default=os.path.join(CACHE_DIR, CACHE_FILE), help='Path to cache file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache}")
    results = load_cache(args.cache)
    print(f"Loaded {len(results)} datasets")

    generate_all_plots(results, args.output_dir, show=args.show)
