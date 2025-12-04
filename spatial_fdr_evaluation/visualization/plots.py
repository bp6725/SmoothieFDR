"""
Visualization functions for spatial FDR evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd


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
    conditions = ['none', 'weak', 'medium', 'strong']
    methods = list(next(iter(results.values())).keys())
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        
        # Prepare data for boxplot
        data_dict = {}
        for method in methods:
            if method in results[condition]:
                power_values = [m['power'] for m in results[condition][method]]
                data_dict[method] = power_values
        
        df = pd.DataFrame(data_dict)
        
        # Boxplot
        df.boxplot(ax=ax)
        
        # Add mean line for baseline
        if 'BH' in data_dict:
            mean_bh = np.mean(data_dict['BH'])
            ax.axhline(mean_bh, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'BH mean: {mean_bh:.3f}')
        
        ax.set_ylabel('Power (TPR)', fontsize=12)
        ax.set_title(f'Spatial: {condition.capitalize()}', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


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