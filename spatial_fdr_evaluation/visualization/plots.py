"""
Visualization functions for spatial FDR evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
from scipy.spatial.distance import cdist

def visualize_structural_groups(K: np.ndarray, group_ids: np.ndarray, save_path: Optional[str] = None):
    """
    Visualizes the structural groups (H1, Compact H0, Close H0, Far H0)
    using a sorted kernel heatmap and similarity boxplots.
    """
    labels = ["H1 Signal", "H0 Compact", "H0 Close", "H0 Far"]
    colors = ['red', 'blue', 'orange', 'gray']

    # 1. Clustermap (Sorted by Group)
    sort_idx = np.argsort(group_ids)
    K_sorted = K[np.ix_(sort_idx, sort_idx)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap
    sns.heatmap(K_sorted, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False, ax=ax1)
    ax1.set_title("Kernel Matrix Sorted by Group\n(Top-Left=H1, then Compact, Close, Far)")

    # 2. Similarity Boxplot (Similarity to H1)
    h1_indices = np.where(group_ids == 0)[0]
    sims = []
    names = []

    # Compare H1-H1 self-sim vs H0_groups-H1 cross-sim
    if len(h1_indices) > 0:
        sims.append(K[np.ix_(h1_indices, h1_indices)].mean(axis=1))
        names.append("H1 Self")

        for gid in [1, 2, 3]:
            idx = np.where(group_ids == gid)[0]
            if len(idx) > 0:
                sims.append(K[np.ix_(idx, h1_indices)].mean(axis=1))
                names.append(labels[gid])

    ax2.boxplot(sims, labels=names)
    ax2.set_title("Similarity to Signal Block (H1)")
    ax2.set_ylabel("Avg Kernel Similarity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dataset_overview(locations, true_labels, p_values, save_path=None):
    """Plots spatial labels and p-value distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. True Labels
    ax = axes[0]
    scatter = ax.scatter(locations[:, 0], locations[:, 1], c=true_labels,
                         cmap='RdYlGn_r', s=30, vmin=0, vmax=1,
                         edgecolors='black', linewidth=0.3)
    ax.set_title('True Labels (0=H1, 1=H0)')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    plt.colorbar(scatter, ax=ax, label='Label')

    # 2. P-values
    ax = axes[1]
    ax.hist(p_values[true_labels == 1], bins=30, alpha=0.5, color='green', density=True, label='H0')
    ax.hist(p_values[true_labels == 0], bins=30, alpha=0.5, color='red', density=True, label='H1')
    ax.set_title('P-value Distribution')
    ax.legend()

    # 3. Spatial P-values
    ax = axes[2]
    scatter = ax.scatter(locations[:, 0], locations[:, 1], c=p_values,
                         cmap='RdYlGn', s=30, vmin=0, vmax=1,
                         edgecolors='black', linewidth=0.3)
    ax.set_title('Observed P-values')
    plt.colorbar(scatter, ax=ax, label='p-value')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_kernel_diagnostics(locations, K, true_labels, save_path=None):
    """Detailed histograms of kernel values vs distances for different pairs."""
    distances = cdist(locations, locations, metric='euclidean')

    is_null = (true_labels == 1)
    is_alt = (true_labels == 0)

    # Sampling for speed if N is large
    if len(locations) > 1000:
        idx = np.random.choice(len(locations), 1000, replace=False)
        sub_dist = distances[np.ix_(idx, idx)]
        sub_K = K[np.ix_(idx, idx)]
        sub_labels = true_labels[idx]
        is_null = (sub_labels == 1)
        is_alt = (sub_labels == 0)
    else:
        sub_dist = distances
        sub_K = K

    # Extract pairs (Upper triangle)
    mask_tri = np.triu(np.ones_like(sub_K, dtype=bool), k=1)

    # H0-H0
    mask_h0 = np.outer(is_null, is_null) & mask_tri
    d_h0 = sub_dist[mask_h0]
    k_h0 = sub_K[mask_h0]

    # H1-H1
    mask_h1 = np.outer(is_alt, is_alt) & mask_tri
    d_h1 = sub_dist[mask_h1]
    k_h1 = sub_K[mask_h1]

    # Cross
    mask_cross = np.outer(is_null, is_alt)
    d_cross = sub_dist[mask_cross]
    k_cross = sub_K[mask_cross]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Distances
    bins = 40
    axes[0, 0].hist(d_h0, bins=bins, color='blue', alpha=0.7);
    axes[0, 0].set_title("H0-H0 Distances")
    axes[0, 1].hist(d_h1, bins=bins, color='red', alpha=0.7);
    axes[0, 1].set_title("H1-H1 Distances")
    axes[0, 2].hist(d_cross, bins=bins, color='purple', alpha=0.7);
    axes[0, 2].set_title("H0-H1 Distances")

    # Kernel
    axes[1, 0].hist(k_h0, bins=bins, color='blue', alpha=0.7);
    axes[1, 0].set_title("H0-H0 Kernel Sim")
    axes[1, 1].hist(k_h1, bins=bins, color='red', alpha=0.7);
    axes[1, 1].set_title("H1-H1 Kernel Sim")
    axes[1, 2].hist(k_cross, bins=bins, color='purple', alpha=0.7);
    axes[1, 2].set_title("H0-H1 Kernel Sim")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_method_comparison_detail(locations, true_labels, p_values,
                                  lfdr, rejections_spatial, rejections_bh,
                                  stats_spatial, stats_bh, save_path=None):
    """
    Plots the 2x3 comparison grid (Spatial Results, LFDR, Stats vs BH Results, P-val, Stats).
    """
    n = len(true_labels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Spatial FDR ---
    ax = axes[0, 0]
    # Map TN/FP/FN/TP
    colors = ['green', 'red', 'orange', 'blue']  # TN, FP, FN, TP
    labels = ['TN', 'FP', 'FN', 'TP']

    class_map = np.zeros(n, dtype=int)
    class_map[~rejections_spatial & (true_labels == 1)] = 0  # TN
    class_map[rejections_spatial & (true_labels == 1)] = 1  # FP
    class_map[~rejections_spatial & (true_labels == 0)] = 2  # FN
    class_map[rejections_spatial & (true_labels == 0)] = 3  # TP

    for i in range(4):
        mask = class_map == i
        if mask.any():
            ax.scatter(locations[mask, 0], locations[mask, 1], c=colors[i], label=labels[i], s=30, edgecolors='k',
                       lw=0.2)
    ax.set_title(f"Spatial FDR (FDR={stats_spatial.get('observed_fdr', 0):.2f})")
    ax.legend()

    # LFDR Map
    ax = axes[0, 1]
    sc = ax.scatter(locations[:, 0], locations[:, 1], c=lfdr, cmap='RdYlGn_r', s=30)
    plt.colorbar(sc, ax=ax, label='lfdr')
    ax.set_title("Estimated Local FDR")

    # Stats Text
    ax = axes[0, 2]
    ax.axis('off')
    txt = f"SPATIAL FDR\n{'-' * 20}\n"
    for k, v in stats_spatial.items():
        if isinstance(v, float):
            txt += f"{k}: {v:.4f}\n"
        else:
            txt += f"{k}: {v}\n"
    ax.text(0.1, 0.5, txt, fontsize=12, fontfamily='monospace', bbox=dict(facecolor='lightblue', alpha=0.3))

    # --- Row 2: BH ---
    ax = axes[1, 0]
    class_map_bh = np.zeros(n, dtype=int)
    class_map_bh[~rejections_bh & (true_labels == 1)] = 0  # TN
    class_map_bh[rejections_bh & (true_labels == 1)] = 1  # FP
    class_map_bh[~rejections_bh & (true_labels == 0)] = 2  # FN
    class_map_bh[rejections_bh & (true_labels == 0)] = 3  # TP

    for i in range(4):
        mask = class_map_bh == i
        if mask.any():
            ax.scatter(locations[mask, 0], locations[mask, 1], c=colors[i], label=labels[i], s=30, edgecolors='k',
                       lw=0.2)
    ax.set_title(f"BH (FDR={stats_bh.get('observed_fdr', 0):.2f})")
    ax.legend()

    # P-value Map
    ax = axes[1, 1]
    sc = ax.scatter(locations[:, 0], locations[:, 1], c=p_values, cmap='RdYlGn', s=30)
    plt.colorbar(sc, ax=ax, label='p-value')
    ax.set_title("Raw P-values")

    # Stats Text
    ax = axes[1, 2]
    ax.axis('off')
    txt = f"BENJAMINI-HOCHBERG\n{'-' * 20}\n"
    for k, v in stats_bh.items():
        if isinstance(v, float):
            txt += f"{k}: {v:.4f}\n"
        else:
            txt += f"{k}: {v}\n"
    ax.text(0.1, 0.5, txt, fontsize=12, fontfamily='monospace', bbox=dict(facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_power_comparison(
    results: Dict[str, Dict[str, list]],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 4)
):
    """
    Plot power comparison across spatial structure conditions.

    Parameters
    ----------
    results : dict
        Nested dict: {condition: {method: [metrics_list]}}
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(16, 4)
        Figure size
    """
    conditions = list(results.keys())
    methods = list(next(iter(results.values())).keys())
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize)
    if len(conditions) == 1: axes = [axes]

    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        data_dict = {}
        for method in methods:
            if method in results[condition]:
                data_dict[method] = [m['power'] for m in results[condition][method]]
        df = pd.DataFrame(data_dict)
        df.boxplot(ax=ax)
        ax.set_title(f'{condition.capitalize()}')
        ax.set_ylim([0, 1])

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fdr_calibration(
        results: Dict[str, Dict[str, list]],
        nominal_fdr: float = 0.1,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 8)
):
    """
    Plot FDR calibration check.

    Parameters
    ----------
    results : dict
        Nested dict: {condition: {method: [metrics_list]}}
    nominal_fdr : float, default=0.1
        Target FDR level
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(8, 8)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    conditions = ['none', 'weak', 'medium', 'strong']
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D']

    for method_name in results['none'].keys():
        for condition, color, marker in zip(conditions, colors, markers):
            if method_name in results[condition]:
                # Empirical FDR across replications
                fdr_values = [m['FDR'] for m in results[condition][method_name]]

                mean_fdr = np.mean(fdr_values)
                std_fdr = np.std(fdr_values)

                ax.errorbar(
                    nominal_fdr, mean_fdr,
                    yerr=std_fdr,
                    marker=marker, markersize=10,
                    color=color,
                    label=f'{method_name} - {condition}',
                    capsize=5, capthick=2, linewidth=2
                )

    # Reference line: perfect calibration
    ax.plot([0, 0.15], [0, 0.15], 'k--', linewidth=2, label='Perfect calibration')
    ax.axhline(nominal_fdr, color='gray', linestyle=':', linewidth=2, label='Nominal FDR')
    ax.axvline(nominal_fdr, color='gray', linestyle=':', linewidth=2)

    ax.set_xlabel('Nominal FDR', fontsize=14)
    ax.set_ylabel('Empirical FDR', fontsize=14)
    ax.set_xlim([0, 0.15])
    ax.set_ylim([0, 0.15])
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('FDR Calibration Check', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_lambda_sensitivity(
    lambda_values: List[float],
    power_values: List[float],
    fdr_values: List[float],
    nominal_fdr: float = 0.1,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot sensitivity to regularization parameter λ.

    Parameters
    ----------
    lambda_values : list
        List of λ values tested
    power_values : list
        Corresponding power values
    fdr_values : list
        Corresponding FDR values
    nominal_fdr : float, default=0.1
        Target FDR level
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(12, 5)
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Power vs lambda
    ax1.semilogx(lambda_values, power_values, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Regularization λ', fontsize=13)
    ax1.set_ylabel('Power (TPR)', fontsize=13)
    ax1.set_title('Power vs. Regularization', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # FDR vs lambda
    ax2.semilogx(lambda_values, fdr_values, 'o-', linewidth=2, markersize=8, color='red')
    ax2.axhline(nominal_fdr, color='black', linestyle='--', linewidth=2, label=f'Nominal FDR = {nominal_fdr}')
    ax2.set_xlabel('Regularization λ', fontsize=13)
    ax2.set_ylabel('Empirical FDR', fontsize=13)
    ax2.set_title('FDR Control vs. Regularization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(fdr_values) * 1.2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_power_gain_bars(
    results: Dict[str, Dict[str, list]],
    baseline_method: str = 'BH',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot power gain relative to baseline as bar chart.

    Parameters
    ----------
    results : dict
        Nested dict: {condition: {method: [metrics_list]}}
    baseline_method : str, default='BH'
        Name of baseline method
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(10, 6)
        Figure size
    """
    conditions = ['none', 'weak', 'medium', 'strong']
    methods = [m for m in results['none'].keys() if m != baseline_method]

    # Compute mean power for each condition and method
    power_gains = {method: [] for method in methods}

    for condition in conditions:
        baseline_power = np.mean([m['power'] for m in results[condition][baseline_method]])

        for method in methods:
            if method in results[condition]:
                method_power = np.mean([m['power'] for m in results[condition][method]])
                gain = (method_power - baseline_power) / baseline_power if baseline_power > 0 else 0
                power_gains[method].append(gain * 100)  # Convert to percentage
            else:
                power_gains[method].append(0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(conditions))
    width = 0.8 / len(methods)

    for idx, method in enumerate(methods):
        offset = (idx - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, power_gains[method], width,
               label=method, alpha=0.8)

    ax.set_xlabel('Spatial Structure', fontsize=13)
    ax.set_ylabel('Power Gain over BH (%)', fontsize=13)
    ax.set_title('Relative Power Gain Across Spatial Structures',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conditions])
    ax.legend(fontsize=11)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_spatial_alpha_map(
    locations: np.ndarray,
    alpha_values: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot spatial map of estimated α(loc) values.

    Works for 2D locations only.

    Parameters
    ----------
    locations : np.ndarray, shape (n_samples, 2)
        2D spatial coordinates
    alpha_values : np.ndarray, shape (n_samples,)
        Estimated α(loc) values
    true_labels : np.ndarray, optional
        True labels (1 = H0, 0 = H1)
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(12, 5)
        Figure size
    """
    if locations.shape[1] != 2:
        print("Spatial map plotting only works for 2D locations")
        return None

    if true_labels is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    # Plot estimated α
    scatter1 = ax1.scatter(
        locations[:, 0], locations[:, 1],
        c=alpha_values, cmap='RdYlBu_r',
        s=50, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax1.set_xlabel('Spatial Dimension 1', fontsize=12)
    ax1.set_ylabel('Spatial Dimension 2', fontsize=12)
    ax1.set_title('Estimated α(loc) - Prior Null Probability',
                  fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='α(loc)')

    if true_labels is not None:
        # Plot ground truth
        scatter2 = ax2.scatter(
            locations[:, 0], locations[:, 1],
            c=true_labels, cmap='RdBu',
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        ax2.set_xlabel('Spatial Dimension 1', fontsize=12)
        ax2.set_ylabel('Spatial Dimension 2', fontsize=12)
        ax2.set_title('Ground Truth Labels', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter2, ax=ax2, ticks=[0, 1])
        cbar.set_ticklabels(['H1 (Signal)', 'H0 (Null)'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_optimization_diagnostics(
        losses: List[float],
        grad_norms: List[float],
        alpha_history: List[np.ndarray],
        violations_history: List[int],
        locations: np.ndarray,
        alpha_final: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        in_top_right: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (18, 10)
):
    """Plot 6-panel optimization diagnostics."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Evolution')
    ax1.grid(True, alpha=0.3)

    # Gradient norm
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(grad_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm (log scale)')
    ax2.grid(True, alpha=0.3)

    # Alpha evolution
    ax3 = fig.add_subplot(gs[0, 2])
    checkpoint_iters = np.linspace(0, len(losses) - 1, len(alpha_history), dtype=int)
    ax3.plot(checkpoint_iters, [a.min() for a in alpha_history], 'b-', label='min', linewidth=2)
    ax3.plot(checkpoint_iters, [a.max() for a in alpha_history], 'r-', label='max', linewidth=2)
    ax3.plot(checkpoint_iters, [a.mean() for a in alpha_history], 'g-', label='mean', linewidth=2)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax3.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('α')
    ax3.set_title('α Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Violations
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(checkpoint_iters, violations_history, 'purple', marker='o', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('# Violations')
    ax4.set_title('Constraint Violations')
    ax4.grid(True, alpha=0.3)

    # Final alpha map
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(locations[:, 0], locations[:, 1], c=alpha_final,
                          cmap='RdYlGn', s=50, vmin=0, vmax=1,
                          edgecolors='black', linewidth=0.5)
    if in_top_right is not None:
        ax5.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
        ax5.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Final α Map')
    plt.colorbar(scatter, ax=ax5, label='α(x)')

    # True labels
    ax6 = fig.add_subplot(gs[1, 2])
    if true_labels is not None:
        scatter = ax6.scatter(locations[:, 0], locations[:, 1], c=true_labels,
                              cmap='RdYlGn_r', s=50, vmin=0, vmax=1,
                              edgecolors='black', linewidth=0.5)
        if in_top_right is not None:
            ax6.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
            ax6.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_title('True Labels')
        plt.colorbar(scatter, ax=ax6)
    else:
        ax6.text(0.5, 0.5, 'No labels', ha='center', va='center', transform=ax6.transAxes)
        ax6.axis('off')

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()