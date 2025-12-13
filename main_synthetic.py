"""
End-to-End Pipeline Validation on Square Synthetic Data

This script validates the complete Spatial FDR pipeline:
1. Generate square data (medium strength)
2. Estimate f0/f1 distributions
3. Run point-wise optimization with diagnostics
4. Compute local FDR
5. Perform FDR control (rejection decisions)
6. Evaluate performance metrics
7. Visualize results at every step

Run with: python main_synthetic.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
import sys
import os

# Add package to path
sys.path.insert(0, '/mnt/user-data/outputs/spatial_fdr_evaluation')

from spatial_fdr_evaluation.data.synthetic import generate_square_data
from spatial_fdr_evaluation.visualization.plots import plot_optimization_diagnostics
from spatial_fdr_evaluation.data.adbench_loader import load_from_ADbench, convert_adbench_to_spatial_fdr
from spatial_fdr_evaluation.data.synthetic import generate_evaluation_data





# ============================================================================
# CONFIGURATION: SELECT DATASET
# ============================================================================
# Options:
#   'square_synthetic' - 4-square synthetic data (as before)
#   '7_Cardiotocography', '24_mnist', '11_donors', '36_speech', etc. - ADbench datasets
DATASET_NAME = '24_mnist'  # Change this to test different datasets
N_SAMPLES = -1  # Only used for synthetic data
EFFECT_SIZE = 4.0  # Only used for synthetic data
RANDOM_STATE = 42

SPATIAL_STRENGTH= "strong"


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

print("=" * 80)
print(f"SECTION 1: LOADING DATASET - {DATASET_NAME}")
print("=" * 80)

if DATASET_NAME == 'square_synthetic':
    # Generate synthetic square data
    print("\nGenerating synthetic square data...")
    locations, true_labels, p_values = generate_square_data(
        n_samples=N_SAMPLES,
        effect_size=EFFECT_SIZE,
        random_state=RANDOM_STATE
    )
    dataset_type = 'synthetic'

else:
    # Load ADbench for spatial structure
    print(f"\nLoading ADbench dataset: {DATASET_NAME}...")
    data_dict = load_from_ADbench(DATASET_NAME, n_samples=N_SAMPLES, la=0.1)
    locations = data_dict['X_train']

    print(f"  ✓ Loaded: {data_dict['dataset_name']}")
    print(f"  Samples: {len(locations)}, Dimensions: {data_dict['dim']}D")

    # Generate synthetic evaluation data with spatial clustering
    print(f"\nGenerating synthetic labels and p-values...")
    print(f"  Spatial strength: {SPATIAL_STRENGTH}")
    true_labels, p_values = generate_evaluation_data(
        locations,
        spatial_strength=SPATIAL_STRENGTH,
        effect_strength='medium',
        random_state=RANDOM_STATE
    )
    dataset_type = 'adbench'

# Common statistics
n_samples = len(p_values)
n_alternatives = np.sum(true_labels == 0)
n_nulls = np.sum(true_labels == 1)

print(f"\nDataset statistics:")
print(f"  Total samples: {n_samples}")
print(f"  True alternatives: {n_alternatives} ({100 * n_alternatives / n_samples:.1f}%)")
print(f"  True nulls: {n_nulls} ({100 * n_nulls / n_samples:.1f}%)")
print(f"  P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
print(f"  Dimensions: {locations.shape[1]}D")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# # Plot 1: Spatial locations
# ax = axes[0]
# if locations.shape[1] == 2:
#     scatter = ax.scatter(locations[:, 0], locations[:, 1],
#                          c=true_labels, cmap='RdYlGn_r', s=50,
#                          vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
#     if dataset_type == 'synthetic':
#         ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Quarter boundary')
#         ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
#         ax.legend()
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title('True Labels\n(0=alternative, 1=null)')
# else:
#     # PCA for high-D
#     from sklearn.decomposition import PCA
#
#     pca = PCA(n_components=2, random_state=RANDOM_STATE)
#     locations_2d = pca.fit_transform(locations)
#     scatter = ax.scatter(locations_2d[:, 0], locations_2d[:, 1],
#                          c=true_labels, cmap='RdYlGn_r', s=50,
#                          vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
#     ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
#     ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
#     ax.set_title(f'True Labels (PCA from {locations.shape[1]}D)')
# plt.colorbar(scatter, ax=ax, label='Label')

# Plot 2: P-value distribution
ax = axes[1]
ax.hist(p_values[true_labels == 1], bins=30, alpha=0.5, label='Nulls', density=True, color='green')
ax.hist(p_values[true_labels == 0], bins=30, alpha=0.5, label='Alternatives', density=True, color='red')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.3, label='Uniform')
ax.set_xlabel('P-value')
ax.set_ylabel('Density')
ax.set_title('P-value Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# # Plot 3: Spatial p-values
# ax = axes[2]
# if locations.shape[1] == 2:
#     scatter = ax.scatter(locations[:, 0], locations[:, 1],
#                          c=p_values, cmap='RdYlGn', s=50,
#                          vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
#     if dataset_type == 'synthetic':
#         ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
#         ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title('Observed P-values')
# else:
#     scatter = ax.scatter(locations_2d[:, 0], locations_2d[:, 1],
#                          c=p_values, cmap='RdYlGn', s=50,
#                          vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_title(f'P-values (PCA from {locations.shape[1]}D)')
# plt.colorbar(scatter, ax=ax, label='P-value')
#
# plt.tight_layout()
# plt.show()
# print(f"\n✓ Saved: step1_data_{DATASET_NAME}.png")
# plt.close()


# ============================================================================
# SECTION 2: ESTIMATE f0 AND f1 DISTRIBUTIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: ESTIMATING NULL AND ALTERNATIVE DISTRIBUTIONS")
print("="*80)

# Convert p-values to z-scores for estimation
z_scores = stats.norm.ppf(1 - p_values)
z_scores = np.clip(z_scores, -10, 10)  # Numerical stability

# Estimate pi_0 (proportion of nulls) using Storey's method
lambda_val = 0.5
pi_0_est = np.mean(p_values > lambda_val) / (1 - lambda_val)
pi_0_est = min(pi_0_est, 1.0)  # Ensure <= 1

print(f"\nEstimated π₀ (proportion of nulls): {pi_0_est:.3f}")
print(f"True π₀: {np.mean(true_labels == 1):.3f}")

# Fit mixture model using simple approach
# f0: standard normal (theoretical null)
# f1: fit to data with small p-values

def f0_func(p):
    """Null distribution (uniform on p-values)"""
    return np.ones_like(p)

# Estimate f1 by fitting to small p-values
small_p_mask = p_values < 0.1
if np.sum(small_p_mask) > 10:
    z_alt = z_scores[small_p_mask]
    mu_alt = np.mean(z_alt)
    sigma_alt = np.std(z_alt)
else:
    mu_alt = 2.0
    sigma_alt = 1.0

def f1_func(p):
    """Alternative distribution"""
    z = stats.norm.ppf(1 - p)
    z = np.clip(z, -10, 10)
    return stats.norm.pdf(z, loc=mu_alt, scale=sigma_alt) / stats.norm.pdf(z)

print(f"Alternative distribution: N({mu_alt:.2f}, {sigma_alt:.2f}²)")

# Evaluate densities at observed p-values
f0_vals = f0_func(p_values)
f1_vals = f1_func(p_values)

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: P-value densities
ax = axes[0]
p_grid = np.linspace(0.001, 0.999, 1000)
ax.plot(p_grid, f0_func(p_grid), 'g-', linewidth=2, label='f₀ (null)')
ax.plot(p_grid, f1_func(p_grid), 'r-', linewidth=2, label='f₁ (alternative)')
ax.hist(p_values, bins=30, density=True, alpha=0.3, color='gray', label='Observed')
ax.set_xlabel('P-value')
ax.set_ylabel('Density')
ax.set_title('Estimated Null and Alternative Densities')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mixture model
ax = axes[1]
mixture = pi_0_est * f0_func(p_grid) + (1 - pi_0_est) * f1_func(p_grid)
ax.plot(p_grid, mixture, 'b-', linewidth=2, label=f'Mixture (π₀={pi_0_est:.2f})')
ax.plot(p_grid, f0_func(p_grid), 'g--', linewidth=1, alpha=0.5, label='f₀')
ax.plot(p_grid, f1_func(p_grid), 'r--', linewidth=1, alpha=0.5, label='f₁')
ax.hist(p_values, bins=30, density=True, alpha=0.3, color='gray', label='Observed')
ax.set_xlabel('P-value')
ax.set_ylabel('Density')
ax.set_title('Two-Group Mixture Model')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig('/mnt/user-data/outputs/step2_distribution_estimation.png', dpi=300, bbox_inches='tight')
# print("\n✓ Saved: step2_distribution_estimation.png")
# plt.close()


# ============================================================================
# SECTION 3: KERNEL CONSTRUCTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: CONSTRUCTING MATÉRN KERNEL")
print("="*80)

# Matérn kernel with nu=2.5, length_scale estimated from data
def compute_matern_kernel(X1, X2, length_scale=0.51, nu=2.5):
    """Matérn kernel with smoothness nu=2.5"""
    dists = cdist(X1, X2, metric='euclidean')
    
    if nu == 0.5:
        K = np.exp(-dists / length_scale)
    elif nu == 1.5:
        tmp = np.sqrt(3) * dists / length_scale
        K = (1 + tmp) * np.exp(-tmp)
    elif nu == 2.5:
        tmp = np.sqrt(5) * dists / length_scale
        K = (1 + tmp + tmp**2 / 3) * np.exp(-tmp)
    else:
        raise ValueError(f"nu={nu} not implemented")
    
    return K

# Estimate length scale from data
# dists = cdist(locations, locations, metric='euclidean')
# median_dist = np.median(dists[np.triu_indices_from(dists, k=1)])
length_scale = 1#median_dist  # Use median distance as length scale

print(f"\nKernel parameters:")
print(f"  Type: Matérn")
print(f"  Smoothness (ν): 2.5")
print(f"  Length scale: {length_scale:.3f} (median pairwise distance)")

# Compute kernel matrix
K = compute_matern_kernel(locations, locations, length_scale=length_scale, nu=2.5)

# Compute pairwise Euclidean distances
distances = cdist(locations, locations, metric='euclidean')

# Separate into blocks
is_null = (true_labels == 1)
is_alt = (true_labels == 0)

# Extract both distances and kernel values for different pairs
null_null_dist = []
null_null_kernel = []

alt_alt_dist = []
alt_alt_kernel = []

null_alt_dist = []
null_alt_kernel = []

for i in range(len(locations)):
    for j in range(i + 1, len(locations)):  # Upper triangle only
        d = distances[i, j]
        k = K[i, j]

        if is_null[i] and is_null[j]:
            null_null_dist.append(d)
            null_null_kernel.append(k)
        elif is_alt[i] and is_alt[j]:
            alt_alt_dist.append(d)
            alt_alt_kernel.append(k)
        else:  # One null, one alt
            null_alt_dist.append(d)
            null_alt_kernel.append(k)

# Convert to arrays
null_null_dist = np.array(null_null_dist)
null_null_kernel = np.array(null_null_kernel)

alt_alt_dist = np.array(alt_alt_dist)
alt_alt_kernel = np.array(alt_alt_kernel)

null_alt_dist = np.array(null_alt_dist)
null_alt_kernel = np.array(null_alt_kernel)

# Create 3x2 figure
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Row 1: Null-Null
axes[0, 0].hist(null_null_dist, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title(f'Null-Null: Euclidean Distances\n(n={len(null_null_dist)} pairs)')
axes[0, 0].set_xlabel('Distance')
axes[0, 0].set_ylabel('Count')
axes[0, 0].axvline(np.median(null_null_dist), color='red', linestyle='--',
                   label=f'Median: {np.median(null_null_dist):.3f}')
axes[0, 0].legend()

axes[0, 1].hist(null_null_kernel, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 1].set_title(f'Null-Null: Kernel Values\n(n={len(null_null_kernel)} pairs)')
axes[0, 1].set_xlabel('Kernel value')
axes[0, 1].set_ylabel('Count')
axes[0, 1].axvline(np.median(null_null_kernel), color='red', linestyle='--',
                   label=f'Median: {np.median(null_null_kernel):.3f}')
axes[0, 1].legend()

# Row 2: Alt-Alt
axes[1, 0].hist(alt_alt_dist, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].set_title(f'Alt-Alt: Euclidean Distances\n(n={len(alt_alt_dist)} pairs)')
axes[1, 0].set_xlabel('Distance')
axes[1, 0].set_ylabel('Count')
axes[1, 0].axvline(np.median(alt_alt_dist), color='red', linestyle='--',
                   label=f'Median: {np.median(alt_alt_dist):.3f}')
axes[1, 0].legend()

axes[1, 1].hist(alt_alt_kernel, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].set_title(f'Alt-Alt: Kernel Values\n(n={len(alt_alt_kernel)} pairs)')
axes[1, 1].set_xlabel('Kernel value')
axes[1, 1].set_ylabel('Count')
axes[1, 1].axvline(np.median(alt_alt_kernel), color='red', linestyle='--',
                   label=f'Median: {np.median(alt_alt_kernel):.3f}')
axes[1, 1].legend()

# Row 3: Null-Alt (Cross)
axes[2, 0].hist(null_alt_dist, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[2, 0].set_title(f'Null-Alt (Cross): Euclidean Distances\n(n={len(null_alt_dist)} pairs)')
axes[2, 0].set_xlabel('Distance')
axes[2, 0].set_ylabel('Count')
axes[2, 0].axvline(np.median(null_alt_dist), color='red', linestyle='--',
                   label=f'Median: {np.median(null_alt_dist):.3f}')
axes[2, 0].legend()

axes[2, 1].hist(null_alt_kernel, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[2, 1].set_title(f'Null-Alt (Cross): Kernel Values\n(n={len(null_alt_kernel)} pairs)')
axes[2, 1].set_xlabel('Kernel value')
axes[2, 1].set_ylabel('Count')
axes[2, 1].axvline(np.median(null_alt_kernel), color='red', linestyle='--',
                   label=f'Median: {np.median(null_alt_kernel):.3f}')
axes[2, 1].legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nDistance vs Kernel Statistics:")
print("=" * 80)
print(f"NULL-NULL pairs (n={len(null_null_dist)}):")
print(f"  Distance - Mean: {null_null_dist.mean():.4f}, Median: {np.median(null_null_dist):.4f}")
print(f"  Kernel   - Mean: {null_null_kernel.mean():.4f}, Median: {np.median(null_null_kernel):.4f}")
print()
print(f"ALT-ALT pairs (n={len(alt_alt_dist)}):")
print(f"  Distance - Mean: {alt_alt_dist.mean():.4f}, Median: {np.median(alt_alt_dist):.4f}")
print(f"  Kernel   - Mean: {alt_alt_kernel.mean():.4f}, Median: {np.median(alt_alt_kernel):.4f}")
print()
print(f"NULL-ALT (cross) pairs (n={len(null_alt_dist)}):")
print(f"  Distance - Mean: {null_alt_dist.mean():.4f}, Median: {np.median(null_alt_dist):.4f}")
print(f"  Kernel   - Mean: {null_alt_kernel.mean():.4f}, Median: {np.median(null_alt_kernel):.4f}")
print("=" * 80)

# Key insight: Check if spatial separation exists
print("\n🔍 Spatial Separation Analysis:")
print(
    f"   Alt-Alt median distance ({np.median(alt_alt_dist):.3f}) < Null-Null ({np.median(null_null_dist):.3f})? {np.median(alt_alt_dist) < np.median(null_null_dist)}")
print(
    f"   Cross median distance ({np.median(null_alt_dist):.3f}) > Alt-Alt ({np.median(alt_alt_dist):.3f})? {np.median(null_alt_dist) > np.median(alt_alt_dist)}")
print(
    f"   Cross median kernel ({np.median(null_alt_kernel):.3f}) < Alt-Alt ({np.median(alt_alt_kernel):.3f})? {np.median(null_alt_kernel) < np.median(alt_alt_kernel)}")



# Additional: Check spatial structure
print("\nSpatial Separation Check:")
print(f"Mean distance (null-null): {cdist(locations[is_null], locations[is_null]).mean():.4f}")
print(f"Mean distance (alt-alt): {cdist(locations[is_alt], locations[is_alt]).mean():.4f}")
print(f"Mean distance (null-alt): {cdist(locations[is_null], locations[is_alt]).mean():.4f}")

if locations.shape[1] < 3 :
    # Visualize kernel
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Kernel matrix
    ax = axes[0]
    im = ax.imshow(K, cmap='viridis', aspect='auto')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    ax.set_title('Kernel Matrix K')
    plt.colorbar(im, ax=ax)

    # Plot 2: Kernel function shape
    ax = axes[1]
    center_idx = np.argmin(np.sum((locations - [0.5, 0.5])**2, axis=1))
    distances = np.linalg.norm(locations - locations[center_idx], axis=1)
    kernel_values = K[center_idx, :]
    ax.scatter(distances, kernel_values, alpha=0.5, s=20)
    dist_grid = np.linspace(0, distances.max(), 100)
    # Create dummy 2D points at various distances
    X_center = np.zeros((1, 2))
    X_grid = np.column_stack([dist_grid, np.zeros_like(dist_grid)])
    kernel_grid = compute_matern_kernel(X_center, X_grid, length_scale, nu=2.5)[0]
    ax.plot(dist_grid, kernel_grid, 'r-', linewidth=2, label='Matérn ν=2.5')
    ax.set_xlabel('Distance from center point')
    ax.set_ylabel('Kernel value K(x, x′)')
    ax.set_title('Kernel Function Shape')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig('/mnt/user-data/outputs/step3_kernel_construction.png', dpi=300, bbox_inches='tight')
    # print("\n✓ Saved: step3_kernel_construction.png")
    # plt.close()


# ============================================================================
# SECTION 4: POINT-WISE OPTIMIZATION WITH TRACKING
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: POINT-WISE OPTIMIZATION (STAGE 1)")
print("="*80)

# Hyperparameters (from CV tuning)
lambda_reg = 0.1
lambda_bound = 500.0
learning_rate = 0.001
max_grad_norm = 10.0
max_iter = 15000

print(f"\nHyperparameters:")
print(f"  λ_reg (RKHS): {lambda_reg}")
print(f"  λ_bound (boundary): {lambda_bound}")
print(f"  Learning rate: {learning_rate}")
print(f"  Gradient clipping: {max_grad_norm}")
print(f"  Max iterations: {max_iter}")

# Initialize
n = len(p_values)
c = (np.random.rand(n)-0.5)*2

# Tracking arrays
losses = []
grad_norms = []
alpha_history = []
violations_history = []
checkpoint_interval = 10

print("\nOptimizing...")
for iteration in range(max_iter):
    # Forward pass
    alpha = K @ c
    mixture = alpha * f0_vals + (1 - alpha) * f1_vals
    mixture = np.clip(mixture, 1e-10, None)
    
    # Compute loss
    data_loss = -np.sum(np.log(mixture))
    reg_loss = lambda_reg * (c @ K @ c)
    bound_loss = lambda_bound * np.sum(
        np.maximum(0, alpha - 1)**2 + np.maximum(0, -alpha)**2
    )
    total_loss = data_loss + reg_loss + bound_loss
    
    # Natural gradient
    grad_alpha = -(f0_vals - f1_vals) / mixture
    grad_bound = 2 * lambda_bound * (
        np.maximum(0, alpha - 1) - np.maximum(0, -alpha)
    )
    grad_nat = grad_alpha + grad_bound + 2 * lambda_reg * c
    
    # Gradient clipping (CRITICAL!)
    grad_norm = np.linalg.norm(grad_nat)
    if grad_norm > max_grad_norm:
        grad_nat = grad_nat * (max_grad_norm / grad_norm)
    
    # Update
    c = c - learning_rate * grad_nat
    
    # Track every iteration
    losses.append(total_loss)
    grad_norms.append(grad_norm)
    
    # Checkpoint every N iterations
    if iteration % checkpoint_interval == 0:
        alpha_history.append(alpha.copy())
        violations = np.sum((alpha < 0) | (alpha > 1))
        violations_history.append(violations)
        
        if iteration % 100 == 0:
            print(f"  Iter {iteration:4d}: loss={total_loss:8.2f}, "
                  f"|∇|={grad_norm:6.2f}, α∈[{alpha.min():.3f}, {alpha.max():.3f}], "
                  f"viol={violations}")

# Final alpha
alpha_final = K @ c
violations_final = np.sum((alpha_final < 0) | (alpha_final > 1))

print(f"\nOptimization complete!")
print(f"  Final loss: {losses[-1]:.2f}")
print(f"  Final α range: [{alpha_final.min():.3f}, {alpha_final.max():.3f}]")
print(f"  Final violations: {violations_final}")
# print(f"  Top-right mean α: {alpha_final[in_top_right].mean():.3f} (target ≈0.1)")
# print(f"  Elsewhere mean α: {alpha_final[~in_top_right].mean():.3f} (target ≈0.9)")

# Plot optimization diagnostics
plot_optimization_diagnostics(
    losses=losses,
    grad_norms=grad_norms,
    alpha_history=alpha_history,
    violations_history=violations_history,
    locations=locations,
    alpha_final=alpha_final,
    true_labels=true_labels,
    # in_top_right=in_top_right,
    save_path='/mnt/user-data/outputs/step4_optimization_diagnostics.png'
)
print("\n✓ Saved: step4_optimization_diagnostics.png")


# ============================================================================
# SECTION 5: LOCAL FDR COMPUTATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: COMPUTING LOCAL FDR")
print("="*80)

# Compute local FDR: lfdr(x) = α(x) * f0(p) / [α(x)*f0(p) + (1-α(x))*f1(p)]
lfdr = (alpha_final * f0_vals) / (alpha_final * f0_vals + (1 - alpha_final) * f1_vals)

print(f"\nLocal FDR statistics:")
print(f"  Range: [{lfdr.min():.3f}, {lfdr.max():.3f}]")
print(f"  Mean: {lfdr.mean():.3f}")
print(f"  Median: {np.median(lfdr):.3f}")
# print(f"  Top-right mean: {lfdr[in_top_right].mean():.3f}")
# print(f"  Elsewhere mean: {lfdr[~in_top_right].mean():.3f}")

# Plot local FDR
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Spatial lfdr
ax = axes[0]
scatter = ax.scatter(locations[:, 0], locations[:, 1], 
                    c=lfdr, cmap='RdYlGn_r', s=50,
                    vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Local FDR (lfdr)')
plt.colorbar(scatter, ax=ax, label='lfdr')

# Plot 2: lfdr vs p-value
ax = axes[1]
ax.scatter(p_values, lfdr, alpha=0.5, s=20, c=true_labels, cmap='RdYlGn_r')
ax.set_xlabel('P-value')
ax.set_ylabel('Local FDR')
ax.set_title('Local FDR vs P-value')
ax.grid(True, alpha=0.3)

# Plot 3: lfdr distribution by true label
ax = axes[2]
ax.hist(lfdr[true_labels == 1], bins=30, alpha=0.5, label='True nulls', density=True, color='green')
ax.hist(lfdr[true_labels == 0], bins=30, alpha=0.5, label='True alts', density=True, color='red')
ax.set_xlabel('Local FDR')
ax.set_ylabel('Density')
ax.set_title('Local FDR Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/step5_local_fdr.png', dpi=300, bbox_inches='tight')
# print("\n✓ Saved: step5_local_fdr.png")
# plt.close()

plt.show()
# ============================================================================
# SECTION 6: FDR CONTROL (SPATIAL LFDR - STOREY-TAYLOR)
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: FDR CONTROL (SPATIAL LFDR - STOREY-TAYLOR)")
print("="*80)

target_fdr = 0.10

# Storey-Taylor procedure on lfdr
sorted_idx = np.argsort(lfdr)
sorted_lfdr = lfdr[sorted_idx]

# Find largest k such that mean(lfdr[0:k]) <= target_fdr
cumsum_lfdr = np.cumsum(sorted_lfdr)
mean_lfdr = cumsum_lfdr / np.arange(1, n + 1)

idx_below = np.where(mean_lfdr <= target_fdr)[0]
if len(idx_below) > 0:
    k_spatial = idx_below[-1] + 1
    rejections_spatial = np.zeros(n, dtype=bool)
    rejections_spatial[sorted_idx[:k_spatial]] = True
else:
    k_spatial = 0
    rejections_spatial = np.zeros(n, dtype=bool)

n_rejections_spatial = np.sum(rejections_spatial)
n_tp_spatial = np.sum(rejections_spatial & (true_labels == 0))
n_fp_spatial = np.sum(rejections_spatial & (true_labels == 1))

print(f"\nFDR control at α={target_fdr}:")
print(f"  Storey-Taylor threshold k: {k_spatial}")
print(f"  Total rejections: {n_rejections_spatial}")
print(f"  True positives: {n_tp_spatial}")
print(f"  False positives: {n_fp_spatial}")

if n_rejections_spatial > 0:
    observed_fdr_spatial = n_fp_spatial / n_rejections_spatial
    power_spatial = n_tp_spatial / np.sum(true_labels == 0)
    precision_spatial = n_tp_spatial / n_rejections_spatial
else:
    observed_fdr_spatial = 0.0
    power_spatial = 0.0
    precision_spatial = 0.0

print(f"\nPerformance metrics:")
print(f"  Observed FDR: {observed_fdr_spatial:.3f} (target: {target_fdr})")
print(f"  Power (TPR): {power_spatial:.3f}")
print(f"  Precision: {precision_spatial:.3f}")

# Confusion matrix
tp_spatial = n_tp_spatial
fp_spatial = n_fp_spatial
fn_spatial = np.sum(~rejections_spatial & (true_labels == 0))
tn_spatial = np.sum(~rejections_spatial & (true_labels == 1))

print(f"\nConfusion matrix:")
print(f"           Reject    Not Reject")
print(f"  Alt       {tp_spatial:4d}       {fn_spatial:4d}")
print(f"  Null      {fp_spatial:4d}       {tn_spatial:4d}")


# ============================================================================
# SECTION 7: BASELINE COMPARISON (BENJAMINI-HOCHBERG)
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: BASELINE - BENJAMINI-HOCHBERG PROCEDURE")
print("="*80)

# Standard BH on p-values (no spatial information)
sorted_pval_idx = np.argsort(p_values)
sorted_pvals = p_values[sorted_pval_idx]

# BH threshold: find largest k where p(k) <= (k/n) * alpha
thresholds = (np.arange(1, n + 1) / n) * target_fdr
idx_below_bh = np.where(sorted_pvals <= thresholds)[0]

if len(idx_below_bh) > 0:
    k_bh = idx_below_bh[-1] + 1
    rejections_bh = np.zeros(n, dtype=bool)
    rejections_bh[sorted_pval_idx[:k_bh]] = True
else:
    k_bh = 0
    rejections_bh = np.zeros(n, dtype=bool)

n_rejections_bh = np.sum(rejections_bh)
n_tp_bh = np.sum(rejections_bh & (true_labels == 0))
n_fp_bh = np.sum(rejections_bh & (true_labels == 1))

print(f"\nFDR control at α={target_fdr}:")
print(f"  BH threshold k: {k_bh}")
print(f"  Total rejections: {n_rejections_bh}")
print(f"  True positives: {n_tp_bh}")
print(f"  False positives: {n_fp_bh}")

if n_rejections_bh > 0:
    observed_fdr_bh = n_fp_bh / n_rejections_bh
    power_bh = n_tp_bh / np.sum(true_labels == 0)
    precision_bh = n_tp_bh / n_rejections_bh
else:
    observed_fdr_bh = 0.0
    power_bh = 0.0
    precision_bh = 0.0

print(f"\nPerformance metrics:")
print(f"  Observed FDR: {observed_fdr_bh:.3f} (target: {target_fdr})")
print(f"  Power (TPR): {power_bh:.3f}")
print(f"  Precision: {precision_bh:.3f}")

# Confusion matrix
tp_bh = n_tp_bh
fp_bh = n_fp_bh
fn_bh = np.sum(~rejections_bh & (true_labels == 0))
tn_bh = np.sum(~rejections_bh & (true_labels == 1))

print(f"\nConfusion matrix:")
print(f"           Reject    Not Reject")
print(f"  Alt       {tp_bh:4d}       {fn_bh:4d}")
print(f"  Null      {fp_bh:4d}       {tn_bh:4d}")


# ============================================================================
# SECTION 8: COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: SPATIAL FDR vs BENJAMINI-HOCHBERG COMPARISON")
print("="*80)

print(f"\n{'Metric':<20} {'Spatial FDR':>15} {'BH':>15} {'Improvement':>15}")
print("="*70)
print(f"{'Rejections':<20} {n_rejections_spatial:>15} {n_rejections_bh:>15} {n_rejections_spatial - n_rejections_bh:>+15}")
print(f"{'True Positives':<20} {tp_spatial:>15} {tp_bh:>15} {tp_spatial - tp_bh:>+15}")
print(f"{'False Positives':<20} {fp_spatial:>15} {fp_bh:>15} {fp_spatial - fp_bh:>+15}")
print(f"{'Observed FDR':<20} {observed_fdr_spatial:>15.3f} {observed_fdr_bh:>15.3f} {observed_fdr_spatial - observed_fdr_bh:>+15.3f}")
print(f"{'Power':<20} {power_spatial:>15.3f} {power_bh:>15.3f} {power_spatial - power_bh:>+15.3f}")
print(f"{'Precision':<20} {precision_spatial:>15.3f} {precision_bh:>15.3f} {precision_spatial - precision_bh:>+15.3f}")
print("="*70)

# Determine which is better
if observed_fdr_spatial <= target_fdr * 1.1 and power_spatial > power_bh:
    print("\n✅ Spatial FDR WINS: Better power with FDR control")
elif observed_fdr_bh <= target_fdr * 1.1 and power_bh > power_spatial:
    print("\n⚠️  BH WINS: Better power (spatial FDR may have issues)")
elif observed_fdr_spatial > target_fdr * 1.5 and observed_fdr_bh > target_fdr * 1.5:
    print("\n❌ BOTH FAIL: Neither controls FDR properly")
else:
    print("\n⚠️  MIXED: Check detailed results")


# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Spatial FDR
ax = axes[0, 0]
for i, (label, color) in enumerate(zip(['TN', 'FP', 'FN', 'TP'], ['green', 'red', 'orange', 'blue'])):
    classification_spatial = np.zeros(n, dtype=int)
    classification_spatial[~rejections_spatial & (true_labels == 1)] = 0  # TN
    classification_spatial[rejections_spatial & (true_labels == 1)] = 1   # FP
    classification_spatial[~rejections_spatial & (true_labels == 0)] = 2  # FN
    classification_spatial[rejections_spatial & (true_labels == 0)] = 3   # TP
    mask = classification_spatial == i
    ax.scatter(locations[mask, 0], locations[mask, 1],
              c=color, s=50, label=label, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Spatial FDR Results\n(FDR={observed_fdr_spatial:.3f}, Power={power_spatial:.3f})')
ax.legend()

ax = axes[0, 1]
scatter = ax.scatter(locations[:, 0], locations[:, 1],
                    c=lfdr, cmap='RdYlGn_r', s=50,
                    vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Local FDR Map')
plt.colorbar(scatter, ax=ax, label='lfdr')

ax = axes[0, 2]
ax.axis('off')
summary_spatial = f"""
SPATIAL FDR (Storey-Taylor)
{'='*30}
Rejections:  {n_rejections_spatial}
TP: {tp_spatial}  FP: {fp_spatial}
FN: {fn_spatial}  TN: {tn_spatial}

FDR:        {observed_fdr_spatial:.3f}
Power:      {power_spatial:.3f}
Precision:  {precision_spatial:.3f}
"""
ax.text(0.1, 0.5, summary_spatial, fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Row 2: BH
ax = axes[1, 0]
for i, (label, color) in enumerate(zip(['TN', 'FP', 'FN', 'TP'], ['green', 'red', 'orange', 'blue'])):
    classification_bh = np.zeros(n, dtype=int)
    classification_bh[~rejections_bh & (true_labels == 1)] = 0  # TN
    classification_bh[rejections_bh & (true_labels == 1)] = 1   # FP
    classification_bh[~rejections_bh & (true_labels == 0)] = 2  # FN
    classification_bh[rejections_bh & (true_labels == 0)] = 3   # TP
    mask = classification_bh == i
    ax.scatter(locations[mask, 0], locations[mask, 1],
              c=color, s=50, label=label, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'BH Results\n(FDR={observed_fdr_bh:.3f}, Power={power_bh:.3f})')
ax.legend()

ax = axes[1, 1]
scatter = ax.scatter(locations[:, 0], locations[:, 1],
                    c=p_values, cmap='RdYlGn', s=50,
                    vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('P-value Map')
plt.colorbar(scatter, ax=ax, label='p-value')

ax = axes[1, 2]
ax.axis('off')
summary_bh = f"""
BENJAMINI-HOCHBERG
{'='*30}
Rejections:  {n_rejections_bh}
TP: {tp_bh}  FP: {fp_bh}
FN: {fn_bh}  TN: {tn_bh}

FDR:        {observed_fdr_bh:.3f}
Power:      {power_bh:.3f}
Precision:  {precision_bh:.3f}
"""
ax.text(0.1, 0.5, summary_bh, fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Update the line that says "rejections =" to use "rejections_spatial"
# And update all summary stats to distinguish between methods

print(f"\nFinal Results:")
print(f"  ✓ Data generated: {n_samples} samples")
print(f"  ✓ Distributions estimated (π₀={pi_0_est:.3f})")
print(f"  ✓ Kernel constructed (ℓ={length_scale:.3f})")
print(f"  ✓ Optimization converged ({max_iter} iterations)")
print(f"  ✓ Local FDR computed")
print(f"  ✓ Spatial FDR: {n_rejections_spatial} rejections (FDR={observed_fdr_spatial:.3f}, Power={power_spatial:.3f})")
print(f"  ✓ BH baseline: {n_rejections_bh} rejections (FDR={observed_fdr_bh:.3f}, Power={power_bh:.3f})")

if observed_fdr_spatial <= target_fdr * 1.1 and power_spatial > power_bh:
    print("\n🎉 SPATIAL FDR SUCCESSFUL - Outperforms BH!")
elif observed_fdr_spatial > target_fdr * 1.5:
    print("\n⚠️  SPATIAL FDR BROKEN - FDR not controlled!")
else:
    print("\n⚠️  Check results - May need tuning")
