"""
Diagnostic Script: Visualize f0 and f1 Estimation

This script generates synthetic spatial data and visualizes how f0 and f1
are estimated using the current method. Helps diagnose why α → 0.

Usage:
    python f0_f1_diagnostic.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

# ============================================================================
# SYNTHETIC DATA GENERATION (Same as evaluation framework)
# ============================================================================

def generate_spatial_clusters(n_samples, d, n_clusters, cluster_strength):
    """
    Generate synthetic spatial data with clustered alternatives.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    d : int
        Spatial dimension
    n_clusters : int
        Number of alternative clusters
    cluster_strength : float
        Strength of spatial clustering (sigma for cluster centers)
        
    Returns
    -------
    locations : (n_samples, d)
        Spatial locations
    labels : (n_samples,)
        True labels (0=null, 1=alternative)
    """
    # Random spatial locations
    locations = np.random.randn(n_samples, d)
    
    # Generate cluster centers for alternatives
    cluster_centers = np.random.randn(n_clusters, d) * 2
    
    # Determine proportion of alternatives (20%)
    n_alternatives = int(0.2 * n_samples)
    n_nulls = n_samples - n_alternatives
    
    labels = np.zeros(n_samples, dtype=int)
    
    # Place alternatives near cluster centers
    for i in range(n_alternatives):
        # Choose random cluster
        cluster_idx = np.random.randint(n_clusters)
        center = cluster_centers[cluster_idx]
        
        # Add noise around cluster center
        locations[i] = center + np.random.randn(d) * cluster_strength
        labels[i] = 1
    
    # Nulls are already randomly placed
    # (the last n_nulls samples remain as originally sampled)
    
    return locations, labels


def generate_pvalues(labels, effect_size=2.0):
    """
    Generate p-values from z-scores.
    
    Parameters
    ----------
    labels : array
        True labels (0=null, 1=alternative)
    effect_size : float
        Mean shift for alternatives
        
    Returns
    -------
    p_values : array
        P-values
    """
    n = len(labels)
    z_scores = np.zeros(n)
    
    # Nulls: N(0, 1)
    z_scores[labels == 0] = np.random.randn(np.sum(labels == 0))
    
    # Alternatives: N(effect_size, 1)
    z_scores[labels == 1] = np.random.randn(np.sum(labels == 1)) + effect_size
    
    # Convert to p-values (one-sided)
    p_values = 1 - stats.norm.cdf(z_scores)
    
    return p_values


# ============================================================================
# F0 AND F1 ESTIMATION (Current method)
# ============================================================================

def estimate_marginal_mixture_robust(p_values, verbose=False):
    """
    Estimates marginal f0 and f1 distributions robustly.
    
    This is the CURRENT method being used.
    Strategy: Right-Tail Anchoring with dilated null detection.
    """
    # 1. Clean and Transform to Z-space
    p_clean = p_values[(~np.isnan(p_values)) & (p_values > 0) & (p_values < 1)]
    p_clean = np.clip(p_clean, 1e-15, 1 - 1e-15)
    z_scores = stats.norm.ppf(p_clean)

    if len(z_scores) < 100:
        # Fallback for tiny data
        return 0.9, lambda p: np.ones_like(np.atleast_1d(p)), lambda p: 2 * (1 - np.atleast_1d(p))

    # 2. Fit Global Density f(z) using KDE
    n = len(z_scores)
    std_dev = np.std(z_scores, ddof=1)
    bw_scott = 1.06 * std_dev * (n ** (-0.2))

    kde = KernelDensity(bandwidth=bw_scott, kernel='gaussian')
    kde.fit(z_scores.reshape(-1, 1))

    def f_z_global(z_in):
        return np.exp(kde.score_samples(np.atleast_1d(z_in).reshape(-1, 1)))

    # 3. Estimate Empirical Null Parameters (Right-Tail Anchoring)
    median_z = np.median(z_scores)
    q75_z = np.percentile(z_scores, 75)

    # Robust sigma estimate
    sigma_est = (q75_z - median_z) / 0.6745
    sigma_est = max(1.0, sigma_est)

    delta_est = median_z

    if verbose:
        print(f"Marginal Estimation:")
        print(f"  Empirical Null: N({delta_est:.3f}, {sigma_est:.3f}²)")
        if sigma_est > 1.1:
            print(f"  -> Detected spatial correlation/overdispersion")

    # 4. Define P-space Densities

    def get_jacobian(z):
        return 1.0 / np.maximum(stats.norm.pdf(z), 1e-10)

    def f0(p):
        """Null density in p-space (potentially U-shaped if dilated)."""
        p_arr = np.atleast_1d(p)
        p_safe = np.clip(p_arr, 1e-15, 1 - 1e-15)
        z = stats.norm.ppf(p_safe)
        
        # Null density in Z
        pdf_z = stats.norm.pdf(z, loc=delta_est, scale=sigma_est)
        
        # Transform to P
        return pdf_z * get_jacobian(z)

    def f1(p):
        """Alternative density via deconvolution."""
        p_arr = np.atleast_1d(p)
        p_safe = np.clip(p_arr, 1e-15, 1 - 1e-15)
        z = stats.norm.ppf(p_safe)
        
        # Global density
        pdf_global = f_z_global(z)
        
        # Null density
        pdf_null = stats.norm.pdf(z, loc=delta_est, scale=sigma_est)
        
        # Estimate pi0 from right tail
        ratio = pdf_global / (pdf_null + 1e-10)
        pi0_temp = np.min(ratio[z > median_z]) if np.any(z > median_z) else 1.0
        pi0_temp = np.clip(pi0_temp, 0.5, 1.0)
        
        # Residual
        excess = pdf_global - (pi0_temp * pdf_null)
        excess[z > delta_est] = 0  # Signal only on left
        excess = np.maximum(0, excess)
        
        # Transform to P
        return (excess * get_jacobian(z)) / (1 - pi0_temp + 1e-6)

    # 5. Final Pi0 Estimate
    lambda_val = 0.5
    num_null_region = np.sum(p_clean > lambda_val)
    mass_f0_in_region = 0.5
    
    pi0_estimate = (num_null_region / len(p_clean)) / mass_f0_in_region
    pi0_estimate = np.clip(pi0_estimate, 0.1, 1.0)

    if verbose:
        print(f"  π₀ = {pi0_estimate:.3f}")

    return pi0_estimate, f0, f1


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_f0_f1_comparison(datasets_list, save_path=None):
    """
    Plot f0 and f1 for multiple datasets.
    
    Parameters
    ----------
    datasets_list : list of dict
        Each dict has 'p_values', 'labels', 'f0', 'f1', 'pi0'
    save_path : str, optional
        Path to save figure
    """
    n_datasets = len(datasets_list)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid: 3 rows
    # Row 1: Aggregate plot
    # Row 2-3: Individual examples
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== AGGREGATE PLOT ==========
    ax_agg = fig.add_subplot(gs[0, :])
    
    # Fine grid for plotting
    p_grid = np.linspace(0.001, 0.999, 1000)
    
    # Average f0 and f1 across all datasets
    f0_all = np.zeros((n_datasets, len(p_grid)))
    f1_all = np.zeros((n_datasets, len(p_grid)))
    
    for i, data in enumerate(datasets_list):
        f0_all[i] = data['f0'](p_grid)
        f1_all[i] = data['f1'](p_grid)
    
    f0_mean = np.mean(f0_all, axis=0)
    f0_std = np.std(f0_all, axis=0)
    f1_mean = np.mean(f1_all, axis=0)
    f1_std = np.std(f1_all, axis=0)
    
    # Plot means with confidence bands
    ax_agg.plot(p_grid, f0_mean, 'b-', linewidth=3, label='f₀(p) - Null', alpha=0.8)
    ax_agg.fill_between(p_grid, f0_mean - f0_std, f0_mean + f0_std, 
                        color='blue', alpha=0.2)
    
    ax_agg.plot(p_grid, f1_mean, 'r-', linewidth=3, label='f₁(p) - Alternative', alpha=0.8)
    ax_agg.fill_between(p_grid, f1_mean - f1_std, f1_mean + f1_std,
                        color='red', alpha=0.2)
    
    # Reference line
    ax_agg.axhline(1.0, color='gray', linestyle='--', alpha=0.5, 
                   label='Theoretical Null (Uniform)')
    
    # Ratio analysis
    ratio_mean = f1_mean / (f0_mean + 1e-10)
    frac_f1_wins = np.mean(ratio_mean > 1)
    
    ax_agg.set_xlabel('p-value', fontsize=14, fontweight='bold')
    ax_agg.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax_agg.set_title(f'AGGREGATE: f₀ and f₁ across {n_datasets} datasets', 
                     fontsize=16, fontweight='bold')
    ax_agg.legend(fontsize=12, loc='upper right')
    ax_agg.grid(True, alpha=0.3)
    ax_agg.set_ylim(bottom=0, top=min(10, np.max([f0_mean, f1_mean]) * 1.2))
    
    # Add diagnostic text
    pi0_mean = np.mean([d['pi0'] for d in datasets_list])
    text_str = f"Mean π₀ = {pi0_mean:.3f}\n"
    text_str += f"f₁>f₀ for {frac_f1_wins:.1%} of range\n"
    
    if frac_f1_wins > 0.6:
        text_str += "🚨 PROBLEM: f₁ dominates!"
        bbox_color = 'red'
    else:
        text_str += "✓ Looks reasonable"
        bbox_color = 'green'
    
    ax_agg.text(0.98, 0.97, text_str, transform=ax_agg.transAxes,
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.6))
    
    # ========== INDIVIDUAL EXAMPLES ==========
    # Show 6 individual datasets
    n_examples = min(6, n_datasets)
    example_indices = np.linspace(0, n_datasets - 1, n_examples, dtype=int)
    
    for plot_idx, data_idx in enumerate(example_indices):
        row = 1 + plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        data = datasets_list[data_idx]
        
        # Plot f0 and f1
        f0_vals = data['f0'](p_grid)
        f1_vals = data['f1'](p_grid)
        
        ax.plot(p_grid, f0_vals, 'b-', linewidth=2, label='f₀', alpha=0.7)
        ax.plot(p_grid, f1_vals, 'r-', linewidth=2, label='f₁', alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        
        # Histogram of p-values
        ax.hist(data['p_values'], bins=30, density=True, alpha=0.3, 
                color='black', label='p-values')
        
        ax.set_xlabel('p-value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Dataset {data_idx + 1} (π₀={data["pi0"]:.2f})', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=min(10, np.max([f0_vals, f1_vals]) * 1.2))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_ratio_analysis(datasets_list, save_path=None):
    """
    Plot the f1/f0 ratio - this is what the optimizer sees.
    """
    n_datasets = len(datasets_list)
    p_grid = np.linspace(0.001, 0.999, 1000)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== Plot 1: Ratio curves ==========
    ax = axes[0]
    
    ratios_all = []
    for data in datasets_list:
        f0_vals = data['f0'](p_grid)
        f1_vals = data['f1'](p_grid)
        ratio = f1_vals / (f0_vals + 1e-10)
        ratios_all.append(ratio)
        
        # Plot individual (faint)
        ax.plot(p_grid, ratio, 'gray', alpha=0.1, linewidth=0.5)
    
    # Plot mean
    ratio_mean = np.mean(ratios_all, axis=0)
    ax.plot(p_grid, ratio_mean, 'purple', linewidth=3, label='Mean f₁/f₀')
    
    # Reference line
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Ratio = 1')
    
    # Shade regions
    ax.fill_between(p_grid, 0, ratio_mean, where=(ratio_mean > 1), 
                    alpha=0.3, color='red', label='f₁ > f₀ (prefers α=0)')
    ax.fill_between(p_grid, 0, ratio_mean, where=(ratio_mean < 1),
                    alpha=0.3, color='blue', label='f₀ > f₁ (prefers α=1)')
    
    ax.set_xlabel('p-value', fontsize=14, fontweight='bold')
    ax.set_ylabel('f₁(p) / f₀(p)', fontsize=14, fontweight='bold')
    ax.set_title('Likelihood Ratio: What the Optimizer Sees', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=min(20, np.percentile(ratio_mean, 99)))
    
    # ========== Plot 2: Cumulative fraction ==========
    ax = axes[1]
    
    # For each p-value, what fraction of datasets have f1 > f0?
    frac_f1_wins = np.zeros(len(p_grid))
    for i in range(len(p_grid)):
        wins = sum(ratios[i] > 1 for ratios in ratios_all)
        frac_f1_wins[i] = wins / n_datasets
    
    ax.plot(p_grid, frac_f1_wins, 'purple', linewidth=3)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.fill_between(p_grid, 0, frac_f1_wins, where=(frac_f1_wins > 0.5),
                    alpha=0.3, color='red')
    ax.fill_between(p_grid, 0, frac_f1_wins, where=(frac_f1_wins < 0.5),
                    alpha=0.3, color='blue')
    
    ax.set_xlabel('p-value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fraction of datasets where f₁ > f₀', fontsize=14, fontweight='bold')
    ax.set_title('Consistency Check Across Datasets', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add text
    overall_frac = np.mean(frac_f1_wins)
    text = f"Overall: f₁>f₀ for {overall_frac:.1%} of (p-value, dataset) pairs"
    if overall_frac > 0.6:
        text += "\n🚨 CRITICAL: This will push α → 0!"
        bbox_color = 'red'
    else:
        text += "\n✓ Reasonable"
        bbox_color = 'green'
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.6))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run diagnostic analysis."""
    
    print("=" * 70)
    print("F0 AND F1 DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    # Parameters
    n_datasets = 50
    n_samples = 500
    d = 2
    n_clusters = 3
    cluster_strength = 0.3
    effect_size = 2.0
    
    print(f"\nGenerating {n_datasets} synthetic datasets...")
    print(f"  N = {n_samples}, d = {d}")
    print(f"  True π₀ = 0.8 (80% nulls)")
    print(f"  Spatial clustering: {n_clusters} clusters, strength={cluster_strength}")
    print(f"  Effect size: {effect_size}")
    
    # Generate datasets
    datasets = []
    
    for i in range(n_datasets):
        if (i + 1) % 10 == 0:
            print(f"  Processing dataset {i + 1}/{n_datasets}...")
        
        # Generate data
        locations, labels = generate_spatial_clusters(
            n_samples, d, n_clusters, cluster_strength
        )
        p_values = generate_pvalues(labels, effect_size)
        
        # Estimate f0 and f1
        pi0, f0, f1 = estimate_marginal_mixture_robust(p_values, verbose=(i == 0))
        
        datasets.append({
            'p_values': p_values,
            'labels': labels,
            'f0': f0,
            'f1': f1,
            'pi0': pi0
        })
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Plot 1: f0 and f1 comparison
    print("\n1. Creating f0/f1 comparison plot...")
    fig1 = plot_f0_f1_comparison(datasets, save_path='f0_f1_comparison.png')
    
    # Plot 2: Ratio analysis
    print("2. Creating ratio analysis plot...")
    fig2 = plot_ratio_analysis(datasets, save_path='f0_f1_ratio_analysis.png')
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    pi0_values = [d['pi0'] for d in datasets]
    print(f"\nπ₀ estimates:")
    print(f"  Mean: {np.mean(pi0_values):.3f} (True: 0.800)")
    print(f"  Std:  {np.std(pi0_values):.3f}")
    print(f"  Range: [{np.min(pi0_values):.3f}, {np.max(pi0_values):.3f}]")
    
    # Check f1/f0 at key points
    p_test = np.array([0.01, 0.05, 0.1, 0.5])
    print(f"\nRatio f₁/f₀ at key p-values (averaged over datasets):")
    
    for p in p_test:
        ratios = [d['f1'](p) / d['f0'](p) for d in datasets]
        mean_ratio = np.mean(ratios)
        
        status = "✓" if (p < 0.1 and mean_ratio > 1) or (p >= 0.5 and mean_ratio < 1) else "⚠️"
        print(f"  p={p:.2f}: {mean_ratio:.2f} {status}")
    
    print("\n" + "=" * 70)
    print("DONE! Check the generated figures:")
    print("  - f0_f1_comparison.png")
    print("  - f0_f1_ratio_analysis.png")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
