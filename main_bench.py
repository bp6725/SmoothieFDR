"""
End-to-End Pipeline Validation on Benchmark Structural Data

Updates:
- FIX: Added missing import 'compute_kernel_matrix'.
- CONFIG: Vanilla NGD Optimizer (Constant LR).
- SAMPLER: Mixed Membership (Cluster Corruption).
- DIAGNOSTICS: Full density and failure analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy import linalg
import sys
import os
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# --- IMPORTS ---
from spatial_fdr_evaluation.data.adbench_loader import load_from_ADbench
# FIX: Added compute_kernel_matrix to imports
from spatial_fdr_evaluation.methods.kernels import compute_kernel_matrix, estimate_length_scale

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAME = '33_skin'

# 'adam'    = Robust PyTorch Adam
# 'vanilla' = Simple NGD (Constant LR, No Momentum)
OPTIMIZER = 'vanilla'

SAMPLING_CONFIG = {
    'n_total': 600,
    'background_ratio': 0.5,
    'cluster_corruption': 0.2,
    'sigma_factor': 0.5,
    'effect_strength': 'medium'
}

RANDOM_STATE = 42

# ============================================================================
# NEW SAMPLER: MIXED MEMBERSHIP CLUSTERS
# ============================================================================
def generate_pvalues(labels, effect_strength='medium'):
    p_values = np.zeros(len(labels))
    alphas = {'weak': 0.5, 'medium': 0.05, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)

    # H0 -> Uniform
    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())
    # H1 -> Beta
    p_values[labels == 0] = np.random.beta(a, 3, size=(labels == 0).sum())
    return np.clip(p_values, 1e-10, 1.0)

def load_and_sample_mixed_clusters(dataset_name, n_total, background_ratio, cluster_corruption, sigma_factor, effect_strength):
    try:
        data = load_from_ADbench(dataset_name)
        X_full = StandardScaler().fit_transform(data['X_train'])
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    # Spectral Clustering
    sigma = estimate_length_scale(X_full, method='median') * sigma_factor
    K_full = compute_kernel_matrix(X_full, kernel_type='rbf', length_scale=sigma)

    D = np.array(K_full.sum(axis=1)).flatten() + 1e-10
    A_norm = (np.diag(1/np.sqrt(D)) @ K_full @ np.diag(1/np.sqrt(D)))

    try:
        _, evecs = eigsh(A_norm, k=10, which='LA')
    except:
        _, evecs = linalg.eigh(A_norm)
        evecs = evecs[:, -10:]

    kmeans = KMeans(n_clusters=3, random_state=42).fit(evecs[:, ::-1][:, :3])
    labels_struct = kmeans.labels_

    counts = np.bincount(labels_struct)
    sorted_clusters = np.argsort(counts)[::-1]
    cid_signal = sorted_clusters[0]
    cid_noise  = sorted_clusters[1]

    n_background = int(n_total * background_ratio)
    n_clusters = n_total - n_background
    n_per_cluster = n_clusters // 2

    pool_sig = np.where(labels_struct == cid_signal)[0]
    pool_noi = np.where(labels_struct == cid_noise)[0]
    pool_bg  = np.where((labels_struct != cid_signal) & (labels_struct != cid_noise))[0]

    if len(pool_bg) < n_background:
        remaining = np.setdiff1d(np.arange(len(X_full)), np.concatenate([pool_sig[:n_per_cluster], pool_noi[:n_per_cluster]]))
        pool_bg = remaining

    idx_sig = np.random.choice(pool_sig, n_per_cluster, replace=False)
    idx_noi = np.random.choice(pool_noi, n_per_cluster, replace=False)
    idx_bg  = np.random.choice(pool_bg, n_background, replace=False)

    # Corruption
    lbl_sig = np.zeros(n_per_cluster, dtype=int)
    n_flip_sig = int(n_per_cluster * cluster_corruption)
    if n_flip_sig > 0:
        lbl_sig[np.random.choice(n_per_cluster, n_flip_sig, replace=False)] = 1

    lbl_noi = np.ones(n_per_cluster, dtype=int)
    n_flip_noi = int(n_per_cluster * cluster_corruption)
    if n_flip_noi > 0:
        lbl_noi[np.random.choice(n_per_cluster, n_flip_noi, replace=False)] = 0

    lbl_bg = np.ones(n_background, dtype=int)

    indices = np.concatenate([idx_sig, idx_noi, idx_bg])
    true_labels = np.concatenate([lbl_sig, lbl_noi, lbl_bg])

    perm = np.random.permutation(len(indices))
    indices = indices[perm]
    true_labels = true_labels[perm]

    p_values = generate_pvalues(true_labels, effect_strength)

    return X_full[indices], p_values, true_labels

# ============================================================================
# HELPER: OPTIMIZATION PLOT
# ============================================================================
def plot_optimization_history_only(losses, grad_norms, alpha_history, violations_history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]; ax.plot(losses, 'b-', linewidth=2); ax.set_title('Loss Evolution'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]; ax.semilogy(grad_norms, 'r-', linewidth=2); ax.set_title('Gradient Norm'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    iters = np.linspace(0, len(losses)-1, len(alpha_history), dtype=int)
    ax.plot(iters, [a.mean() for a in alpha_history], 'g-', label='mean')
    ax.plot(iters, [a.min() for a in alpha_history], 'b--', label='min')
    ax.plot(iters, [a.max() for a in alpha_history], 'r--', label='max')
    ax.legend(); ax.set_title('Alpha Evolution'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]; ax.plot(iters, violations_history, 'purple', marker='o'); ax.set_title('Constraint Violations'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
print(f"Loading {DATASET_NAME} with MIXED CLUSTERS...")
np.random.seed(RANDOM_STATE)
data_tuple = load_and_sample_mixed_clusters(DATASET_NAME, **SAMPLING_CONFIG)

if data_tuple is None: raise ValueError("Failed to load data.")
locations, p_values, true_labels = data_tuple
if np.isnan(locations).any(): locations = np.nan_to_num(locations)

n_samples = len(p_values)
print(f"Stats: N={n_samples}, H1={np.sum(true_labels==0)}, H0={np.sum(true_labels==1)}")

# ============================================================================
# SECTION 2: ESTIMATION
# ============================================================================
print("\n" + "="*80 + "\nSECTION 2: ESTIMATING DENSITIES\n" + "="*80)
def f0_func(p): return np.ones_like(p)

z_scores = stats.norm.ppf(1 - np.clip(p_values, 1e-10, 1-1e-10))
small_p_mask = p_values < 0.2
if np.sum(small_p_mask) > 10:
    mu_alt, sigma_alt = np.mean(z_scores[small_p_mask]), np.std(z_scores[small_p_mask])
else:
    mu_alt, sigma_alt = 2.5, 1.0
print(f"Estimated Alternative: N({mu_alt:.2f}, {sigma_alt:.2f})")

def f1_func(p):
    z = stats.norm.ppf(1 - np.clip(p, 1e-10, 1-1e-10))
    raw_pdf = stats.norm.pdf(z, loc=mu_alt, scale=sigma_alt) / stats.norm.pdf(z)
    # Clip to prevent 10^7
    return np.clip(raw_pdf, 0, 2000.0)

f0_vals, f1_vals = f0_func(p_values), f1_func(p_values)

# Diagnostic Histograms
ratio = f1_vals / f0_vals
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
log_bins = np.logspace(np.log10(1e-2), np.log10(ratio.max()), 30)

axes[0].hist(f0_vals, bins=20, color='blue', alpha=0.7); axes[0].set_title("f0(p)")
axes[1].hist(f1_vals, bins=log_bins, color='red', alpha=0.7); axes[1].set_title("f1(p) (Log Bins)"); axes[1].set_xscale('log')
axes[2].hist(ratio, bins=log_bins, color='purple', alpha=0.7); axes[2].set_title("Ratio f1/f0 (Log Bins)"); axes[2].set_xscale('log')
plt.show()

# ============================================================================
# SECTION 3: KERNEL
# ============================================================================
print("\n" + "="*80 + "\nSECTION 3: KERNEL\n" + "="*80)
length_scale = estimate_length_scale(locations, method='median') * SAMPLING_CONFIG['sigma_factor']
print(f"Kernel Length Scale: {length_scale:.4f}")

def compute_matern_kernel(X1, X2, length_scale):
    dists = cdist(X1, X2, metric='euclidean')
    tmp = np.sqrt(3) * dists / length_scale
    K = (1 + tmp) * np.exp(-tmp)
    return K

K = compute_matern_kernel(locations, locations, length_scale)

# Calculate Isolation
h1_indices = np.where(true_labels == 0)[0]
K_to_h1 = K[:, h1_indices]
top_k = min(10, len(h1_indices))
sim_to_signal_group = np.sort(K_to_h1, axis=1)[:, -top_k:].mean(axis=1)
isolation_score = 1 - sim_to_signal_group

# ============================================================================
# SECTION 4: OPTIMIZATION
# ============================================================================
print("\n" + "="*80 + f"\nSECTION 4: OPTIMIZATION ({OPTIMIZER.upper()})\n" + "="*80)

# Init c=0.05
c = np.ones(n_samples) * 0.005
alpha_final = None
losses, grad_norms, alpha_history, violations_history = [], [], [], []
lambda_reg = 10.0; lambda_bound = 500.0; max_iter = 5000

if OPTIMIZER == 'adam':
    lr = 0.02
    m, v = np.zeros_like(c), np.zeros_like(c)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    print(f"Running NGD-Adam (lr={lr})...")

    for t in range(1, max_iter + 1):
        alpha = K @ c
        alpha_safe = np.clip(alpha, -0.1, 1.1)
        mixture = np.clip(alpha_safe * f0_vals + (1 - alpha_safe) * f1_vals, 1e-12, None)

        loss = -np.sum(np.log(mixture)) + lambda_reg * (c @ K @ c) + \
               lambda_bound * np.sum(np.maximum(0, alpha - 1)**2 + np.maximum(0, -alpha)**2)

        if np.isnan(loss): break

        grad_nll = -(f0_vals - f1_vals) / mixture
        grad_bound = 2 * lambda_bound * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * lambda_reg * c)

        m = beta1 * m + (1 - beta1) * grad_nat
        v = beta2 * v + (1 - beta2) * (grad_nat**2)
        m_hat, v_hat = m / (1 - beta1**t), v / (1 - beta2**t)
        c -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        losses.append(loss); grad_norms.append(np.linalg.norm(grad_nat))
        if t % 50 == 0:
            alpha_history.append(alpha.copy())
            violations_history.append(np.sum((alpha < 0) | (alpha > 1)))
        if t % 500 == 0: print(f"  Iter {t}: Loss={loss:.1f}")
    alpha_final = K @ c

elif OPTIMIZER == 'vanilla':
    lr = 0.0005
    print(f"Running Simple NGD (Constant LR={lr})...")

    for t in range(1, max_iter + 1):
        alpha = K @ c
        mixture = np.clip(alpha * f0_vals + (1 - alpha) * f1_vals, 1e-12, None)

        loss = -np.sum(np.log(mixture)) + lambda_reg * (c @ K @ c) + \
               lambda_bound * np.sum(np.maximum(0, alpha - 1)**2 + np.maximum(0, -alpha)**2)

        if np.isnan(loss): print("Loss is NaN!"); break

        grad_nll = -(f0_vals - f1_vals) / mixture
        grad_bound = 2 * lambda_bound * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_nat = grad_nll + grad_bound + (2 * lambda_reg * c)

        gnorm = np.linalg.norm(grad_nat)
        if gnorm > 5.0: grad_nat = grad_nat * (5.0 / gnorm)

        c -= lr * grad_nat
        losses.append(loss); grad_norms.append(gnorm)

        if t % 50 == 0:
            alpha_history.append(alpha.copy())
            violations_history.append(np.sum((alpha < 0) | (alpha > 1)))
        if t % 500 == 0: print(f"  Iter {t}: Loss={loss:.1f}")
    alpha_final = K @ c

print("Optimization Complete.")
plot_optimization_history_only(losses, grad_norms, alpha_history, violations_history)

# ============================================================================
# SECTION 5: RESULTS & PLOTS
# ============================================================================
print("\n" + "="*80 + "\nSECTION 5: RESULTS\n" + "="*80)

lfdr = (alpha_final * f0_vals) / (alpha_final * f0_vals + (1 - alpha_final) * f1_vals)
sorted_idx = np.argsort(lfdr)
q_vals = np.cumsum(lfdr[sorted_idx]) / np.arange(1, n_samples + 1)
rejections = np.zeros(n_samples, dtype=bool)
if np.sum(q_vals <= 0.1) > 0:
    rejections[sorted_idx[:np.max(np.where(q_vals <= 0.1)[0])+1]] = True

n_rej = np.sum(rejections)
n_tp = np.sum(rejections & (true_labels == 0))
fdr = np.sum(rejections & (true_labels == 1)) / n_rej if n_rej > 0 else 0.0
power = n_tp / np.sum(true_labels == 0)
print(f"Rejections: {n_rej}, FDR: {fdr:.3f}, Power: {power:.3f}")

# Final Plot
fig, ax = plt.subplots(figsize=(12, 8))
log_p = -np.log10(np.clip(p_values, 1e-10, 1.0))

mask_tp, mask_fn = rejections & (true_labels == 0), ~rejections & (true_labels == 0)
mask_fp, mask_tn = rejections & (true_labels == 1), ~rejections & (true_labels == 1)

ax.scatter(isolation_score[mask_tn], log_p[mask_tn], c='lightgray', alpha=0.3, label='TN')
ax.scatter(isolation_score[mask_fp], log_p[mask_fp], c='red', alpha=0.8, label='FP', marker='x', s=60)
ax.scatter(isolation_score[mask_fn], log_p[mask_fn], c='orange', alpha=0.9, label='FN', marker='^', s=70)
ax.scatter(isolation_score[mask_tp], log_p[mask_tp], c='blue', alpha=0.8, label='TP', s=60)

sorted_p = np.sort(p_values)
bh_crit = (np.arange(1, len(p_values)+1) / len(p_values)) * 0.1
bh_rej = sorted_p <= bh_crit
if np.any(bh_rej):
    bh_cut = sorted_p[np.max(np.where(bh_rej)[0])]
    ax.axhline(-np.log10(bh_cut), color='red', linestyle='--', linewidth=2, label='BH Cutoff')

ax.axvline(0.5, color='gray', linestyle='--'); ax.set_xlabel('Geometric Isolation'); ax.set_ylabel('-log10(P-value)')
ax.set_title(f'Geometric Failure Analysis ({DATASET_NAME}) - {OPTIMIZER.upper()}')
custom_lines = [Line2D([0], [0], color='white', marker='o', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], color='white', marker='^', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], color='white', marker='x', markeredgecolor='red', markersize=10),
                Line2D([0], [0], color='white', marker='o', markerfacecolor='lightgray', markersize=10),
                Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
ax.legend(custom_lines, ['H1 as H1 (TP)', 'H1 as H0 (FN)', 'H0 as H1 (FP)', 'H0 as H0 (TN)', 'BH Cutoff'], loc='upper right')
ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()