"""
Simple example demonstrating the spatial FDR evaluation framework.

This script shows a minimal working example without requiring ADbench data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import framework components
from spatial_fdr_evaluation.data import (
    generate_evaluation_data,
    extract_spatial_structure
)
from spatial_fdr_evaluation.methods import (
    benjamini_hochberg,
    SpatialFDR,
    estimate_length_scale
)
from spatial_fdr_evaluation.evaluation import compute_metrics
from spatial_fdr_evaluation.visualization import plot_spatial_alpha_map


def main():
    print("=" * 70)
    print("Spatial FDR Evaluation - Simple Example")
    print("=" * 70)
    
    # 1. Generate synthetic spatial data
    print("\n1. Generating synthetic spatial data...")
    np.random.seed(42)
    n_samples = 500
    n_features = 2  # Use 2D for easy visualization
    
    # Create spatial locations (2D points)
    locations = np.random.randn(n_samples, n_features)
    print(f"   Generated {n_samples} locations in {n_features}D space")
    
    # 2. Generate evaluation scenario with spatial clustering
    print("\n2. Generating evaluation data with medium spatial clustering...")
    true_labels, p_values = generate_evaluation_data(
        locations,
        spatial_strength='medium',  # Medium spatial clustering
        effect_strength='medium',   # Medium signal strength
        n_clusters=5,
        random_state=42
    )
    
    n_true_signals = np.sum(true_labels == 0)
    print(f"   Generated p-values for {n_samples} tests")
    print(f"   True signals (H1): {n_true_signals} ({n_true_signals/n_samples*100:.1f}%)")
    print(f"   True nulls (H0): {n_samples - n_true_signals} ({(1-n_true_signals/n_samples)*100:.1f}%)")
    
    # 3. Run Benjamini-Hochberg baseline
    print("\n3. Running Benjamini-Hochberg (BH) baseline...")
    discoveries_bh = benjamini_hochberg(p_values, alpha=0.1)
    metrics_bh = compute_metrics(discoveries_bh, true_labels)
    
    print(f"   BH Results:")
    print(f"     Power (TPR):  {metrics_bh['power']:.3f}")
    print(f"     FDR:          {metrics_bh['FDR']:.3f}")
    print(f"     Precision:    {metrics_bh['precision']:.3f}")
    print(f"     Discoveries:  {metrics_bh['n_discoveries']}")
    
    # 4. Run Spatial FDR method
    print("\n4. Running Spatial FDR with RKHS regularization...")
    
    # Estimate length scale from data
    length_scale = estimate_length_scale(locations, method='median')
    print(f"   Estimated length scale: {length_scale:.3f}")
    
    # Initialize and fit model
    model = SpatialFDR(
        kernel_type='matern',
        lambda_bound=100.0,
        lambda_reg=0.1,
        kernel_params={'nu': 1.5, 'length_scale': length_scale},
        optimizer='natural_gradient',
        verbose=True
    )
    
    model.fit_pointwise(locations, p_values)
    
    # Make discoveries
    discoveries_spatial = model.reject(p_values, alpha=0.1)
    metrics_spatial = compute_metrics(discoveries_spatial, true_labels)
    
    print(f"\n   Spatial FDR Results:")
    print(f"     Power (TPR):  {metrics_spatial['power']:.3f}")
    print(f"     FDR:          {metrics_spatial['FDR']:.3f}")
    print(f"     Precision:    {metrics_spatial['precision']:.3f}")
    print(f"     Discoveries:  {metrics_spatial['n_discoveries']}")
    
    # 5. Compare results
    print("\n5. Comparison:")
    power_gain = (metrics_spatial['power'] - metrics_bh['power']) / metrics_bh['power'] * 100
    print(f"   Power gain over BH: {power_gain:+.1f}%")
    print(f"   FDR control: BH={metrics_bh['FDR']:.3f}, Spatial={metrics_spatial['FDR']:.3f}")
    
    # 6. Visualize results
    print("\n6. Creating visualizations...")
    
    # Get estimated α(loc) values
    alpha_estimates = model.predict_alpha()
    
    # Create output directory
    output_dir = Path('./example_results')
    output_dir.mkdir(exist_ok=True)
    
    # Plot spatial map
    fig = plot_spatial_alpha_map(
        locations,
        alpha_estimates,
        true_labels,
        save_path=output_dir / 'spatial_alpha_map.png'
    )
    
    # Additional plot: discoveries overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BH discoveries
    ax1.scatter(
        locations[:, 0], locations[:, 1],
        c=discoveries_bh.astype(int),
        cmap='RdYlGn', s=50, alpha=0.7,
        edgecolors='black', linewidth=0.5
    )
    ax1.set_title(f'BH Discoveries (n={metrics_bh["n_discoveries"]})', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    
    # Spatial FDR discoveries
    ax2.scatter(
        locations[:, 0], locations[:, 1],
        c=discoveries_spatial.astype(int),
        cmap='RdYlGn', s=50, alpha=0.7,
        edgecolors='black', linewidth=0.5
    )
    ax2.set_title(f'Spatial FDR Discoveries (n={metrics_spatial["n_discoveries"]})',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'discoveries_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"   Saved visualizations to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey finding:")
    if power_gain > 10:
        print(f"  ✓ Spatial FDR achieved {power_gain:.1f}% power gain while maintaining FDR control")
    else:
        print(f"  - Modest power gain ({power_gain:.1f}%), spatial structure may be weak")


if __name__ == '__main__':
    main()
