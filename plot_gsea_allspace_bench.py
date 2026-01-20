"""
Visualization for GSEA All-Space Benchmark (main_gsea_allspace_bench.py)

This script loads cached results and generates visualizations.
Run main_gsea_allspace_bench.py first to generate the cache file.

Usage:
    python plot_gsea_allspace_bench.py
    python plot_gsea_allspace_bench.py --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

# --- CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
CACHE_FILE = "gsea_allspace_results.pkl"
OUTPUT_DIR = "/home/benny/Repos/SmoothieFDR/results/figures/gsea_allspace"

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 20, 'lines.linewidth': 2, 'grid.alpha': 0.4
})


def plot_aggregated_results(results_dict, save_path=None):
    """Plot aggregated alpha distributions for signal vs noise."""
    all_signal = []
    all_noise = []

    for dname, res in results_dict.items():
        p_hidden = res['y_hidden_pvals']
        pred = res['alpha_pred']

        # Proxy: p < 0.05 is "Signal"
        mask_sig = p_hidden < 0.05
        all_signal.extend(pred[mask_sig])
        all_noise.extend(pred[~mask_sig])

    plt.figure(figsize=(10, 6))

    if len(all_noise) > 0:
        plt.hist(all_noise, bins=40, alpha=0.5, color='red', density=True,
                 label='Hidden Noise (p>0.05)', edgecolor='black')
    if len(all_signal) > 0:
        plt.hist(all_signal, bins=40, alpha=0.5, color='blue', density=True,
                 label='Hidden Signal (p<0.05)', edgecolor='black')

    plt.title("Aggregated Inference on Unseen Nodes (TGCA Benchmarks)", fontweight='bold')
    plt.xlabel("Predicted Alpha (Probability of Signal)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_dataset_results(results_dict, save_path=None):
    """Plot per-dataset alpha distributions."""
    n_ds = len(results_dict)
    if n_ds == 0:
        return

    cols = min(3, n_ds)
    rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (dname, res) in enumerate(results_dict.items()):
        ax = axes[i]
        p_hidden = res['y_hidden_pvals']
        pred = res['alpha_pred']

        mask_sig = p_hidden < 0.05
        signal_alpha = pred[mask_sig]
        noise_alpha = pred[~mask_sig]

        if len(noise_alpha) > 0:
            ax.hist(noise_alpha, bins=20, alpha=0.5, color='red', density=True, label='Noise')
        if len(signal_alpha) > 0:
            ax.hist(signal_alpha, bins=20, alpha=0.5, color='blue', density=True, label='Signal')

        ax.set_title(dname, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Predicted Alpha')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Per-Dataset Alpha Distribution on Unseen Nodes', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tuning_results(tuning_data, dataset_name, save_path=None):
    """Plot hyperparameter tuning results."""
    plt.figure(figsize=(10, 6))

    param_grid = tuning_data['param_grid']
    scores_by_beta = tuning_data['scores_by_beta']
    best_lambda = tuning_data['best_lambda']

    for beta, scores in scores_by_beta.items():
        plt.plot(param_grid, scores, 'o-', linewidth=2, label=f'Beta={beta}')

    plt.axvline(best_lambda, color='red', linestyle='--', label=f'Best $\\lambda$: {best_lambda:.2e}')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('CV NLL')
    plt.title(f'HP Tuning: {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_separation_summary(results_dict, save_path=None):
    """Plot summary of signal/noise separation across datasets."""
    datasets = []
    signal_means = []
    noise_means = []
    separations = []

    for dname, res in results_dict.items():
        p_hidden = res['y_hidden_pvals']
        pred = res['alpha_pred']

        mask_sig = p_hidden < 0.05
        sig_mean = np.mean(pred[mask_sig]) if np.sum(mask_sig) > 0 else 0
        noise_mean = np.mean(pred[~mask_sig]) if np.sum(~mask_sig) > 0 else 0

        datasets.append(dname[:15])
        signal_means.append(sig_mean)
        noise_means.append(noise_mean)
        separations.append(sig_mean - noise_mean)

    if len(datasets) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(datasets))
    width = 0.35

    # Mean alpha by class
    axes[0].bar(x - width/2, noise_means, width, label='Noise (p>0.05)', color='red', alpha=0.7)
    axes[0].bar(x + width/2, signal_means, width, label='Signal (p<0.05)', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Predicted Alpha')
    axes[0].set_title('Mean Alpha by Class', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Separation
    colors = ['green' if s > 0 else 'red' for s in separations]
    axes[1].bar(x, separations, color=colors, alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].set_ylabel('Separation (Signal - Noise)')
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
        return pickle.load(f)


def generate_all_plots(cache_data, output_dir, show=False):
    """Generate all visualizations from cached results."""
    os.makedirs(output_dir, exist_ok=True)

    results = cache_data['results']
    tuning_results = cache_data['tuning_results']

    print("Generating plots...")

    if results:
        # Aggregated histogram
        plot_aggregated_results(
            results,
            save_path=os.path.join(output_dir, 'aggregated_histogram.png') if not show else None
        )
        print("  - Aggregated histogram: done")

        # Per-dataset results
        plot_per_dataset_results(
            results,
            save_path=os.path.join(output_dir, 'per_dataset_histograms.png') if not show else None
        )
        print("  - Per-dataset histograms: done")

        # Separation summary
        plot_separation_summary(
            results,
            save_path=os.path.join(output_dir, 'separation_summary.png') if not show else None
        )
        print("  - Separation summary: done")

    # Tuning results per dataset
    for dname, tuning_data in tuning_results.items():
        plot_tuning_results(
            tuning_data, dname,
            save_path=os.path.join(output_dir, f'tuning_{dname.replace("/", "_")}.png') if not show else None
        )
        print(f"  - Tuning {dname}: done")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations from gsea_allspace results')
    parser.add_argument('--cache', default=os.path.join(CACHE_DIR, CACHE_FILE), help='Path to cache file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache}")
    cache_data = load_cache(args.cache)
    print(f"Loaded {len(cache_data['results'])} results")

    generate_all_plots(cache_data, args.output_dir, show=args.show)
