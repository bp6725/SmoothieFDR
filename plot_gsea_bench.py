"""
Visualization for GSEA Benchmark (main_gsea_bench.py)

This script loads cached results and generates visualizations.
Run main_gsea_bench.py first to generate the cache file.

Usage:
    python plot_gsea_bench.py
    python plot_gsea_bench.py --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import argparse

# --- CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
CACHE_FILE = "gsea_bench_results.pkl"
OUTPUT_DIR = "/home/benny/Repos/SmoothieFDR/results/figures/gsea_bench"

# --- PAPER-GRADE PLOT STYLING ---
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 20, 'lines.linewidth': 2, 'grid.alpha': 0.4
})


def plot_optimization_history(history, dataset_name, save_path=None):
    """Plot optimization convergence history."""
    if not history['losses']:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(history['losses'], 'b-')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)

    axes[0, 1].semilogy(history['grad_norms'], 'r-')
    axes[0, 1].set_title('Gradient Norm')
    axes[0, 1].grid(True)

    alphas = history['alpha_history']
    iters = range(len(alphas))
    axes[1, 0].plot(iters, [a.mean() for a in alphas], 'g-', label='Mean')
    axes[1, 0].plot(iters, [a.min() for a in alphas], 'b--', label='Min')
    axes[1, 0].plot(iters, [a.max() for a in alphas], 'r--', label='Max')
    axes[1, 0].legend()
    axes[1, 0].set_title('Alpha Stats')
    axes[1, 0].set_ylim(-0.2, 1.2)
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['violations'], 'purple', marker='o')
    axes[1, 1].set_title('Violations')
    axes[1, 1].grid(True)

    fig.suptitle(f"Optimization: {dataset_name}", fontsize=20, y=0.98)
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


def plot_auc_comparison(results_df, save_path=None):
    """Plot AUC comparison between BH and Graph FDR."""
    if results_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot
    x = np.arange(len(results_df))
    width = 0.35

    axes[0].bar(x - width/2, results_df['auc_bh'], width, label='BH', color='red', alpha=0.7)
    axes[0].bar(x + width/2, results_df['auc_graph'], width, label='Graph FDR', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    labels = [f"{row['benchmark'][:4]}_{row['dataset'][:8]}" for _, row in results_df.iterrows()]
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('AUC Comparison by Dataset', fontweight='bold')
    axes[0].legend()
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Scatter plot: BH vs Graph
    axes[1].scatter(results_df['auc_bh'], results_df['auc_graph'], s=100, alpha=0.7)
    axes[1].plot([0.4, 1], [0.4, 1], 'k--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('AUC (BH)')
    axes[1].set_ylabel('AUC (Graph FDR)')
    axes[1].set_title('AUC: Graph FDR vs BH', fontweight='bold')
    axes[1].set_xlim(0.4, 1)
    axes[1].set_ylim(0.4, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add labels to scatter points
    for i, row in results_df.iterrows():
        axes[1].annotate(row['dataset'][:6],
                         (row['auc_bh'], row['auc_graph']),
                         textcoords="offset points",
                         xytext=(5, 5),
                         fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rejection_comparison(results_df, save_path=None):
    """Plot number of rejections comparison."""
    if results_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width/2, results_df['n_rej_bh'], width, label='BH', color='red', alpha=0.7)
    ax.bar(x + width/2, results_df['n_rej_graph'], width, label='Graph FDR', color='blue', alpha=0.7)
    ax.set_xticks(x)
    labels = [f"{row['benchmark'][:4]}_{row['dataset'][:8]}" for _, row in results_df.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of Rejections')
    ax.set_title('Rejections Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_loss_convergence(histories, save_path=None):
    """
    Global loss convergence across all datasets.

    Shows optimization loss curves for all datasets on one figure.
    """
    # Filter out empty histories
    valid_histories = {k: v for k, v in histories.items() if v and v.get('losses')}
    if not valid_histories:
        print("No valid optimization histories to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(valid_histories))))

    # Left: Raw loss values (log scale)
    ax1 = axes[0]
    for (ds_name, history), color in zip(valid_histories.items(), colors):
        losses = history['losses']
        iterations = np.arange(len(losses)) * 50  # log_interval assumed 50
        short_name = ds_name.split('/')[-1][:12] if '/' in ds_name else ds_name[:12]
        ax1.plot(iterations, losses, '-', linewidth=2, color=color, label=short_name)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Convergence (Raw)', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Right: Normalized loss (relative to initial)
    ax2 = axes[1]
    for (ds_name, history), color in zip(valid_histories.items(), colors):
        losses = np.array(history['losses'])
        iterations = np.arange(len(losses)) * 50
        # Normalize: (loss - min) / (initial - min)
        loss_norm = (losses - losses.min()) / (losses[0] - losses.min() + 1e-10)
        short_name = ds_name.split('/')[-1][:12] if '/' in ds_name else ds_name[:12]
        ax2.plot(iterations, loss_norm, '-', linewidth=2, color=color, label=short_name)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Normalized Loss')
    ax2.set_title('Loss Convergence (Normalized)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle('Global Loss Convergence - GSEA Benchmark', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_summary_table(results_df, save_path=None):
    """Create a summary table figure."""
    if results_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    table_data = results_df[['benchmark', 'dataset', 'best_lambda', 'auc_bh', 'auc_graph']].copy()
    table_data['best_lambda'] = table_data['best_lambda'].apply(lambda x: f'{x:.2e}')
    table_data['auc_bh'] = table_data['auc_bh'].apply(lambda x: f'{x:.3f}')
    table_data['auc_graph'] = table_data['auc_graph'].apply(lambda x: f'{x:.3f}')

    table = ax.table(cellText=table_data.values,
                     colLabels=['Benchmark', 'Dataset', 'Lambda', 'AUC BH', 'AUC Graph'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color cells based on AUC improvement
    for i in range(len(results_df)):
        auc_bh = float(results_df.iloc[i]['auc_bh'])
        auc_graph = float(results_df.iloc[i]['auc_graph'])
        if auc_graph > auc_bh:
            table[(i+1, 4)].set_facecolor('#90EE90')  # Light green
        elif auc_graph < auc_bh:
            table[(i+1, 4)].set_facecolor('#FFB6C1')  # Light red

    plt.title('GSEA Benchmark Results Summary', fontweight='bold', fontsize=14, pad=20)

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

    results_df = cache_data['results_df']
    histories = cache_data['histories']
    tuning_results = cache_data['tuning_results']

    print("Generating plots...")

    # Per-dataset plots
    for ds_key in histories.keys():
        ds_dir = os.path.join(output_dir, ds_key.replace('/', '_'))
        os.makedirs(ds_dir, exist_ok=True)

        # Optimization history
        if ds_key in histories and histories[ds_key]:
            plot_optimization_history(
                histories[ds_key], ds_key,
                save_path=os.path.join(ds_dir, 'optimization_history.png') if not show else None
            )

        # Tuning results
        if ds_key in tuning_results:
            plot_tuning_results(
                tuning_results[ds_key], ds_key,
                save_path=os.path.join(ds_dir, 'tuning_results.png') if not show else None
            )

        print(f"  - {ds_key}: done")

    # Summary plots
    print("\nGenerating summary plots...")

    # Global Loss Convergence
    plot_combined_loss_convergence(
        histories,
        save_path=os.path.join(output_dir, 'combined_loss_convergence.png') if not show else None
    )
    print("  - Global loss convergence: done")

    # AUC comparison
    plot_auc_comparison(
        results_df,
        save_path=os.path.join(output_dir, 'auc_comparison.png') if not show else None
    )
    print("  - AUC comparison: done")

    # Rejection comparison
    plot_rejection_comparison(
        results_df,
        save_path=os.path.join(output_dir, 'rejection_comparison.png') if not show else None
    )
    print("  - Rejection comparison: done")

    # Summary table
    plot_summary_table(
        results_df,
        save_path=os.path.join(output_dir, 'summary_table.png') if not show else None
    )
    print("  - Summary table: done")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations from gsea_bench results')
    parser.add_argument('--cache', default=os.path.join(CACHE_DIR, CACHE_FILE), help='Path to cache file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache}")
    cache_data = load_cache(args.cache)
    print(f"Loaded {len(cache_data['results_df'])} results")

    generate_all_plots(cache_data, args.output_dir, show=args.show)
