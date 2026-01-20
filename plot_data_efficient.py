"""
Visualization for Data-Efficient Testing Experiment (main_data_efficient.py)

This script loads cached results and generates visualizations comparing:
- Random 70% subsampling
- A-optimal 70% subsampling (smart selection)

Key visualization: Scatter plot of log(alpha/(1-alpha)) vs oracle lfdr

Usage:
    python plot_data_efficient.py
    python plot_data_efficient.py --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

# --- CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
CACHE_FILE = "data_efficient_results.pkl"
OUTPUT_DIR = "/home/benny/Repos/SmoothieFDR/results/figures/data_efficient"

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 20, 'lines.linewidth': 2, 'grid.alpha': 0.4
})


def safe_logit(alpha, eps=1e-6):
    """
    Compute log(alpha/(1-alpha)) safely.

    Clips alpha to [eps, 1-eps] to avoid log(0) or log(inf).
    """
    alpha_clipped = np.clip(alpha, eps, 1 - eps)
    return np.log(alpha_clipped / (1 - alpha_clipped))


def plot_logit_alpha_vs_lfdr_scatter(results_dict, save_path=None):
    """
    Main visualization: Scatter plot of log(alpha/(1-alpha)) vs oracle lfdr.

    Compares random vs A-optimal sampling on unseen test points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Collect data from all datasets
    all_lfdr_oracle_random = []
    all_logit_alpha_random = []
    all_lfdr_oracle_aopt = []
    all_logit_alpha_aopt = []

    for ds_name, res in results_dict.items():
        # Random sampling results
        result_random = res['result_random']
        test_idx_random = result_random['test_indices']
        alpha_pred_random = result_random['alpha_pred']
        lfdr_oracle_random = res['lfdr_oracle'][test_idx_random]

        all_lfdr_oracle_random.extend(lfdr_oracle_random)
        all_logit_alpha_random.extend(safe_logit(alpha_pred_random))

        # A-optimal sampling results
        result_aopt = res['result_aopt']
        test_idx_aopt = result_aopt['test_indices']
        alpha_pred_aopt = result_aopt['alpha_pred']
        lfdr_oracle_aopt = res['lfdr_oracle'][test_idx_aopt]

        all_lfdr_oracle_aopt.extend(lfdr_oracle_aopt)
        all_logit_alpha_aopt.extend(safe_logit(alpha_pred_aopt))

    all_lfdr_oracle_random = np.array(all_lfdr_oracle_random)
    all_logit_alpha_random = np.array(all_logit_alpha_random)
    all_lfdr_oracle_aopt = np.array(all_lfdr_oracle_aopt)
    all_logit_alpha_aopt = np.array(all_logit_alpha_aopt)

    # Left plot: Random sampling
    ax1 = axes[0]
    ax1.scatter(all_lfdr_oracle_random, all_logit_alpha_random,
                alpha=0.4, s=30, c='red', edgecolors='none')

    # Add reference line (perfect prediction)
    x_range = np.linspace(0.01, 0.99, 100)
    ax1.plot(x_range, safe_logit(x_range), 'k--', linewidth=2, label='Perfect prediction')

    ax1.set_xlabel('Oracle lfdr')
    ax1.set_ylabel('log(alpha/(1-alpha)) predicted')
    ax1.set_title('Random Subsampling (70%)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Right plot: A-optimal sampling
    ax2 = axes[1]
    ax2.scatter(all_lfdr_oracle_aopt, all_logit_alpha_aopt,
                alpha=0.4, s=30, c='blue', edgecolors='none')

    ax2.plot(x_range, safe_logit(x_range), 'k--', linewidth=2, label='Perfect prediction')

    ax2.set_xlabel('Oracle lfdr')
    ax2.set_ylabel('log(alpha/(1-alpha)) predicted')
    ax2.set_title('A-Optimal Subsampling (70%)', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    fig.suptitle('Data-Efficient Testing: Prediction Quality on Unseen Points',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_logit_alpha_vs_lfdr_combined(results_dict, save_path=None):
    """
    Combined scatter plot comparing both methods on same axes.
    """
    plt.figure(figsize=(12, 10))

    # Collect data from all datasets
    all_lfdr_random = []
    all_logit_random = []
    all_lfdr_aopt = []
    all_logit_aopt = []

    for ds_name, res in results_dict.items():
        # Random
        result_random = res['result_random']
        test_idx_random = result_random['test_indices']
        alpha_pred_random = result_random['alpha_pred']
        lfdr_oracle_random = res['lfdr_oracle'][test_idx_random]

        all_lfdr_random.extend(lfdr_oracle_random)
        all_logit_random.extend(safe_logit(alpha_pred_random))

        # A-optimal
        result_aopt = res['result_aopt']
        test_idx_aopt = result_aopt['test_indices']
        alpha_pred_aopt = result_aopt['alpha_pred']
        lfdr_oracle_aopt = res['lfdr_oracle'][test_idx_aopt]

        all_lfdr_aopt.extend(lfdr_oracle_aopt)
        all_logit_aopt.extend(safe_logit(alpha_pred_aopt))

    # Plot both
    plt.scatter(all_lfdr_random, all_logit_random,
                alpha=0.3, s=25, c='red', edgecolors='none', label='Random (70%)')
    plt.scatter(all_lfdr_aopt, all_logit_aopt,
                alpha=0.3, s=25, c='blue', edgecolors='none', label='A-Optimal (70%)')

    # Reference line
    x_range = np.linspace(0.01, 0.99, 100)
    plt.plot(x_range, safe_logit(x_range), 'k--', linewidth=2.5, label='Perfect prediction')

    plt.xlabel('Oracle lfdr')
    plt.ylabel('log(alpha/(1-alpha)) predicted')
    plt.title('Data-Efficient Testing: Random vs A-Optimal Sampling',
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_error_comparison(results_dict, save_path=None):
    """
    Compare prediction error (MSE) between methods across datasets.
    """
    datasets = []
    mse_random = []
    mse_aopt = []

    for ds_name, res in results_dict.items():
        # Random
        result_random = res['result_random']
        test_idx_random = result_random['test_indices']
        alpha_pred_random = result_random['alpha_pred']
        alpha_oracle_random = res['alpha_oracle'][test_idx_random]
        mse_r = np.mean((alpha_pred_random - alpha_oracle_random)**2)

        # A-optimal
        result_aopt = res['result_aopt']
        test_idx_aopt = result_aopt['test_indices']
        alpha_pred_aopt = result_aopt['alpha_pred']
        alpha_oracle_aopt = res['alpha_oracle'][test_idx_aopt]
        mse_a = np.mean((alpha_pred_aopt - alpha_oracle_aopt)**2)

        datasets.append(ds_name[:12])
        mse_random.append(mse_r)
        mse_aopt.append(mse_a)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_random, width, label='Random', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, mse_aopt, width, label='A-Optimal', color='blue', alpha=0.7)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('MSE (alpha predicted vs oracle)')
    ax.set_title('Prediction Error Comparison: Random vs A-Optimal Sampling',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement percentage
    for i, (r, a) in enumerate(zip(mse_random, mse_aopt)):
        if r > 0:
            improvement = (r - a) / r * 100
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.0f}%',
                       xy=(x[i], max(r, a) + 0.001),
                       ha='center', fontsize=9, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_dataset_scatter(results_dict, output_dir, show=False):
    """
    Generate per-dataset scatter plots.
    """
    for ds_name, res in results_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Random
        result_random = res['result_random']
        test_idx_random = result_random['test_indices']
        alpha_pred_random = result_random['alpha_pred']
        lfdr_oracle_random = res['lfdr_oracle'][test_idx_random]

        ax1 = axes[0]
        ax1.scatter(lfdr_oracle_random, safe_logit(alpha_pred_random),
                    alpha=0.6, s=40, c='red', edgecolors='darkred', linewidth=0.5)
        x_range = np.linspace(0.01, 0.99, 100)
        ax1.plot(x_range, safe_logit(x_range), 'k--', linewidth=2)
        ax1.set_xlabel('Oracle lfdr')
        ax1.set_ylabel('log(alpha/(1-alpha))')
        ax1.set_title('Random Sampling', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)

        # A-optimal
        result_aopt = res['result_aopt']
        test_idx_aopt = result_aopt['test_indices']
        alpha_pred_aopt = result_aopt['alpha_pred']
        lfdr_oracle_aopt = res['lfdr_oracle'][test_idx_aopt]

        ax2 = axes[1]
        ax2.scatter(lfdr_oracle_aopt, safe_logit(alpha_pred_aopt),
                    alpha=0.6, s=40, c='blue', edgecolors='darkblue', linewidth=0.5)
        ax2.plot(x_range, safe_logit(x_range), 'k--', linewidth=2)
        ax2.set_xlabel('Oracle lfdr')
        ax2.set_ylabel('log(alpha/(1-alpha))')
        ax2.set_title('A-Optimal Sampling', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)

        fig.suptitle(f'{ds_name}: Prediction on Unseen Points',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if not show:
            ds_dir = os.path.join(output_dir, ds_name.replace('/', '_'))
            os.makedirs(ds_dir, exist_ok=True)
            plt.savefig(os.path.join(ds_dir, 'logit_alpha_vs_lfdr.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_summary_stats(results_dict, save_path=None):
    """
    Summary statistics table.
    """
    datasets = []
    n_test = []
    corr_random = []
    corr_aopt = []
    mse_random = []
    mse_aopt = []

    for ds_name, res in results_dict.items():
        # Random
        result_random = res['result_random']
        test_idx_random = result_random['test_indices']
        alpha_pred_random = result_random['alpha_pred']
        alpha_oracle_random = res['alpha_oracle'][test_idx_random]

        # A-optimal
        result_aopt = res['result_aopt']
        test_idx_aopt = result_aopt['test_indices']
        alpha_pred_aopt = result_aopt['alpha_pred']
        alpha_oracle_aopt = res['alpha_oracle'][test_idx_aopt]

        datasets.append(ds_name[:15])
        n_test.append(len(test_idx_random))

        # Correlation
        corr_r = np.corrcoef(alpha_pred_random, alpha_oracle_random)[0, 1]
        corr_a = np.corrcoef(alpha_pred_aopt, alpha_oracle_aopt)[0, 1]
        corr_random.append(corr_r)
        corr_aopt.append(corr_a)

        # MSE
        mse_r = np.mean((alpha_pred_random - alpha_oracle_random)**2)
        mse_a = np.mean((alpha_pred_aopt - alpha_oracle_aopt)**2)
        mse_random.append(mse_r)
        mse_aopt.append(mse_a)

    # Create table
    fig, ax = plt.subplots(figsize=(14, len(datasets) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for i in range(len(datasets)):
        improvement = (mse_random[i] - mse_aopt[i]) / mse_random[i] * 100 if mse_random[i] > 0 else 0
        table_data.append([
            datasets[i],
            n_test[i],
            f'{corr_random[i]:.3f}',
            f'{corr_aopt[i]:.3f}',
            f'{mse_random[i]:.4f}',
            f'{mse_aopt[i]:.4f}',
            f'{improvement:+.1f}%'
        ])

    # Add mean row
    table_data.append([
        'MEAN',
        int(np.mean(n_test)),
        f'{np.mean(corr_random):.3f}',
        f'{np.mean(corr_aopt):.3f}',
        f'{np.mean(mse_random):.4f}',
        f'{np.mean(mse_aopt):.4f}',
        f'{np.mean([(mse_random[i] - mse_aopt[i]) / mse_random[i] * 100 for i in range(len(datasets)) if mse_random[i] > 0]):+.1f}%'
    ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Dataset', 'N test', 'Corr (R)', 'Corr (A)', 'MSE (R)', 'MSE (A)', 'Improvement'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the improvement column
    for i in range(len(table_data)):
        improvement_str = table_data[i][-1]
        improvement_val = float(improvement_str.replace('%', '').replace('+', ''))
        if improvement_val > 0:
            table[(i+1, 6)].set_facecolor('#90EE90')
        elif improvement_val < 0:
            table[(i+1, 6)].set_facecolor('#FFB6C1')

    plt.title('Data-Efficient Testing: Summary Statistics\n(R=Random, A=A-Optimal)',
              fontweight='bold', fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_cache(cache_path):
    """Load cached results."""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def generate_all_plots(results_dict, output_dir, show=False):
    """Generate all visualizations from cached results."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")

    # Main scatter plot (side by side)
    plot_logit_alpha_vs_lfdr_scatter(
        results_dict,
        save_path=os.path.join(output_dir, 'logit_alpha_vs_lfdr_sidebyside.png') if not show else None
    )
    print("  - Logit alpha vs lfdr (side by side): done")

    # Combined scatter plot
    plot_logit_alpha_vs_lfdr_combined(
        results_dict,
        save_path=os.path.join(output_dir, 'logit_alpha_vs_lfdr_combined.png') if not show else None
    )
    print("  - Logit alpha vs lfdr (combined): done")

    # MSE comparison
    plot_prediction_error_comparison(
        results_dict,
        save_path=os.path.join(output_dir, 'mse_comparison.png') if not show else None
    )
    print("  - MSE comparison: done")

    # Summary table
    plot_summary_stats(
        results_dict,
        save_path=os.path.join(output_dir, 'summary_stats.png') if not show else None
    )
    print("  - Summary statistics: done")

    # Per-dataset plots
    print("  - Generating per-dataset plots...")
    plot_per_dataset_scatter(results_dict, output_dir, show=show)
    print("  - Per-dataset plots: done")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations from data_efficient results')
    parser.add_argument('--cache', default=os.path.join(CACHE_DIR, CACHE_FILE), help='Path to cache file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache}")
    results = load_cache(args.cache)
    print(f"Loaded {len(results)} datasets")

    generate_all_plots(results, args.output_dir, show=args.show)
