"""
End-to-End Synthetic Validation: Global "Entire Space" Inference
File: main_allspace_synth_cv.py

Methodology:
1. Synthetic Data: "Square" Dataset with P-values.
   - Top-Right: Signal (High -log10(p))
   - Elsewhere: Noise (Uniform p-values)
2. Stage 1 Simulation: Derive Alpha estimates from these P-values.
3. Stage 2 Global Inference: Run Kernel Logistic Regression (KLR) to recover the surface.
4. Visualization: Input P-values vs. Output Alpha Surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
import sys
import os
from typing import Tuple

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
N_TRAIN = 400
GRID_RES = 100
SIGMA = 0.15          # Kernel width
NOISE_LEVEL = 0.05    # Noise for Stage 1 simulation

# Optimization Params for Stage 2 (Global KLR)
STAGE2_PARAMS = {
    'lambda_global': 0.01,
    'lr': 0.01,
    'max_iter': 3000,
    'tol': 1e-5
}

# Plot Styling
plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 18, 'figure.titlesize': 22,
    'lines.linewidth': 2, 'grid.alpha': 0.3
})

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================
def compute_kernel_matrix(X, kernel_type='rbf', length_scale=1.0):
    """Computes RBF kernel matrix (N x N)."""
    dist_sq = cdist(X, X, 'sqeuclidean')
    if kernel_type == 'rbf':
        return np.exp(-dist_sq / (2 * length_scale**2))
    return np.eye(len(X))

def generate_square_data(
        n_samples: int = 200,
        effect_size: float = 2.0,
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate toy data with 4-square spatial structure."""
    np.random.seed(random_state)
    locations = np.random.rand(n_samples, 2)
    in_top_right = (locations[:, 0] > 0.5) & (locations[:, 1] > 0.5)

    labels = np.zeros(n_samples, dtype=int)

    # Top-right: 90% alternatives (Signal)
    top_right_indices = np.where(in_top_right)[0]
    n_alt_topright = int(0.9 * len(top_right_indices))
    null_topright = top_right_indices[n_alt_topright:]
    labels[null_topright] = 1 # Null

    # Other three quarters: 10% alternatives (Signal)
    other_indices = np.where(~in_top_right)[0]
    n_alt_other = int(0.1 * len(other_indices))
    labels[other_indices] = 1 # Null
    labels[other_indices[:n_alt_other]] = 0 # Signal

    # Generate p-values
    z_scores = np.random.randn(n_samples)
    # Add effect to signal (labels==0)
    z_scores[labels == 0] += effect_size
    p_values = 1 - stats.norm.cdf(z_scores)

    return locations, labels, p_values

def get_true_alpha_square(X):
    """Returns the Ground Truth Alpha (0.9 in top-right, 0.1 elsewhere)."""
    alpha = np.ones(len(X)) * 0.1
    mask = (X[:, 0] > 0.5) & (X[:, 1] > 0.5)
    alpha[mask] = 0.9
    return alpha

# ============================================================================
# 3. STAGE 2 SOLVER: GLOBAL KLR
# ============================================================================
class GlobalFDRRegressor:
    """Stage 2: Global Surface Inference via Kernel Logistic Regression."""
    def __init__(self, lambda_global=1.0, lr=0.01, max_iter=2000, tol=1e-5):
        self.lambda_global = lambda_global
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.c = None

    def fit(self, K_train, alpha_hat_stage1):
        n = K_train.shape[0]
        # Warm Start
        epsilon = 0.01
        alpha_clipped = np.clip(alpha_hat_stage1, epsilon, 1 - epsilon)
        target_logits = logit(alpha_clipped)
        self.c = np.linalg.solve(K_train + 1e-6 * np.eye(n), target_logits)

        # Optimization
        for i in range(self.max_iter):
            g = K_train @ self.c
            sigma = expit(g)
            grad = (sigma - alpha_hat_stage1) + 2 * self.lambda_global * self.c
            self.c -= self.lr * grad
            if np.linalg.norm(grad) < self.tol: break
        return self

    def predict(self, K_test):
        if self.c is None: raise ValueError("Model not fitted")
        return expit(K_test @ self.c)

# ============================================================================
# 4. EXECUTION PIPELINE
# ============================================================================
def run_square_experiment():
    print("Initializing Square Experiment...")

    # --- A. Generate Data ---
    print(f"Generating Data ({N_TRAIN} points)...")
    X_train, labels, p_values = generate_square_data(n_samples=N_TRAIN, random_state=RANDOM_STATE)

    # Calculate -log10(p) for visualization
    log_p = -np.log10(np.clip(p_values, 1e-10, 1.0))

    # --- B. Simulate Stage 1 Output ---
    # In a real run, we would optimize alpha_hat based on p_values.
    # For this proof-of-concept, we simulate alpha_hat based on the true distribution + noise.
    alpha_true_train = get_true_alpha_square(X_train)
    alpha_hat_stage1 = np.clip(
        alpha_true_train + np.random.normal(0, NOISE_LEVEL, N_TRAIN),
        0.01, 0.99
    )

    # --- C. Run Stage 2 (Global Inference) ---
    print("Running Stage 2: Global KLR...")
    K_train = compute_kernel_matrix(X_train, kernel_type='rbf', length_scale=SIGMA)
    klr = GlobalFDRRegressor(**STAGE2_PARAMS)
    klr.fit(K_train, alpha_hat_stage1)

    # --- D. Infer on Grid ---
    print("Inferring Entire Space...")
    x_range = np.linspace(0, 1, GRID_RES)
    y_range = np.linspace(0, 1, GRID_RES)
    XX, YY = np.meshgrid(x_range, y_range)
    X_grid = np.column_stack([XX.ravel(), YY.ravel()])

    dist_grid = cdist(X_grid, X_train, metric='euclidean')
    K_grid = np.exp(-(dist_grid**2) / (2 * SIGMA**2))
    alpha_pred_grid = klr.predict(K_grid).reshape(GRID_RES, GRID_RES)

    # --- E. Visualization (P-values vs Alpha) ---
    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # PLOT 1: Ground Truth P-values
    # We use a threshold to emphasize the signal, but color by magnitude
    sc1 = axes[0].scatter(X_train[:,0], X_train[:,1], c=log_p, cmap='Reds', s=60, edgecolors='k', linewidth=0.5)
    axes[0].set_title("Input: Observed P-values (-log10)", fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("X1"); axes[0].set_ylabel("X2")

    # Add ground truth box for reference (dashed line)
    rect = plt.Rectangle((0.5, 0.5), 0.5, 0.5, linewidth=3, edgecolor='blue', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)

    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('-log10(p)')

    # PLOT 2: Inferred Alpha Surface
    im2 = axes[1].contourf(XX, YY, alpha_pred_grid, levels=20, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Output: Inferred Alpha Surface $\\alpha(x)$", fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].set_xlabel("X1"); axes[1].set_ylabel("X2")

    # Overlay the training points faintly to show coverage
    axes[1].scatter(X_train[:,0], X_train[:,1], c='k', s=10, alpha=0.15)

    # Add ground truth box for reference
    rect2 = plt.Rectangle((0.5, 0.5), 0.5, 0.5, linewidth=3, edgecolor='white', facecolor='none', linestyle='--')
    axes[1].add_patch(rect2)

    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Estimated Probability P(H1)')

    plt.suptitle("From P-values to Global Function: Square Discontinuity", fontsize=24, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    run_square_experiment()