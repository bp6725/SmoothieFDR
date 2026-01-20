"""
GSEA All-Space Benchmark: Global Inference on Gene Set Enrichment Analysis

This script runs Stage 2 global inference on GSEA benchmarks using graph diffusion kernels.
Results are saved to cache for separate visualization.

REFACTOR: Results saved to cache for separate visualization (use plot_gsea_allspace_bench.py).
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from scipy.special import expit, logit
import networkx as nx
import warnings
import sys
import os
from sklearn.model_selection import KFold, train_test_split

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# --- CACHE CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# 0. STAGE 2 SOLVER (GLOBAL KLR)
# ============================================================================
class GlobalFDRRegressor:
    """Stage 2: Global Inference using Natural Gradient KLR"""

    def __init__(self, lambda_global=1.0, lr=0.005, max_iter=2000, tol=1e-5):
        self.lambda_global = lambda_global
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.c = None

    def fit(self, K_train, alpha_hat_stage1):
        n = K_train.shape[0]
        alpha_clipped = np.clip(alpha_hat_stage1, 0.01, 0.99)
        target_logits = logit(alpha_clipped)

        # Warm Start (Ridge)
        self.c = np.linalg.solve(K_train + 1e-4 * np.eye(n), target_logits)

        # Optimize
        for i in range(self.max_iter):
            g = K_train @ self.c
            sigma = expit(g)
            grad = K_train @ (sigma - alpha_clipped) + 2 * self.lambda_global * (K_train @ self.c)

            gnorm = np.linalg.norm(grad)
            self.c -= self.lr * grad

            if gnorm < self.tol:
                break
        return self

    def predict(self, K_test_cross):
        if self.c is None:
            raise ValueError("Model not fitted")
        return expit(K_test_cross @ self.c)


# ============================================================================
# 1. KERNEL & VALIDATORS
# ============================================================================

class KernelValidator:
    @staticmethod
    def validate_structure(K):
        if np.isnan(K).any():
            K = np.nan_to_num(K)

        all_vals = np.linalg.eigvalsh(K)
        vals = all_vals[-5:]
        vals = np.sort(vals)[::-1]

        lambda_1 = vals[0] if vals[0] > 0 else 1e-10
        lambda_2 = vals[1]
        ratio = lambda_2 / lambda_1

        print(f"    [Structure] Ratio (lam2/lam1): {ratio:.4f}")

        if ratio > 0.99:
            num_components = np.sum(np.isclose(all_vals, 1.0))
            if num_components > (len(all_vals) * 0.9):
                print("    -> WARNING: Graph is shattered (Identity Matrix).")
                return False
            return True
        return True

    @staticmethod
    def validate_signal_alignment(adj_matrix, p_values, n_perms=100):
        """Checks if the P-values actually cluster on the graph (Moran's I / permutation)."""
        p_clipped = np.clip(p_values, 1e-10, 1 - 1e-10)
        z_scores = norm.ppf(1 - p_clipped)
        z_scores = np.nan_to_num(z_scores)

        z_centered = z_scores - np.mean(z_scores)
        obs_score = z_centered.T @ adj_matrix @ z_centered

        perm_scores = []
        for _ in range(n_perms):
            z_perm = np.random.permutation(z_centered)
            perm_scores.append(z_perm.T @ adj_matrix @ z_perm)

        perm_scores = np.array(perm_scores)
        mean_perm = np.mean(perm_scores)
        std_perm = np.std(perm_scores) + 1e-10

        p_val_spatial = np.mean(perm_scores >= obs_score)
        z_score_spatial = (obs_score - mean_perm) / std_perm

        print(f"    [Alignment] Spatial Z-Score: {z_score_spatial:.2f} (p={p_val_spatial:.4f})")

        if z_score_spatial > 2.0:
            print("    -> SIGNIFICANT: Signal aligns with Graph.")
            return True
        else:
            print("    -> WARNING: Signal looks random on this graph.")
            return False


class SpatialFDRGraphKernel:
    def __init__(self, adjacency_matrix, kernel_type='diffusion', normalized=True, **kwargs):
        self.W = self._validate_adjacency(adjacency_matrix)
        self.n = self.W.shape[0]
        self.kernel_type = kernel_type
        self.normalized = normalized
        self.params = kwargs
        self.L = self._compute_laplacian()
        self.K = self._compute_kernel()

    def _validate_adjacency(self, W):
        if hasattr(W, 'nodes'):
            W = nx.to_numpy_array(W)
        if sp.issparse(W):
            W = W.toarray()
        if not np.allclose(W, W.T):
            W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        return W

    def _compute_laplacian(self):
        degrees = np.array(self.W.sum(axis=1)).flatten()
        if self.normalized:
            with np.errstate(divide='ignore'):
                d_inv_sqrt = np.power(degrees, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = np.diag(d_inv_sqrt)
            I = np.eye(self.n)
            return I - D_inv_sqrt @ self.W @ D_inv_sqrt
        else:
            return np.diag(degrees) - self.W

    def _compute_kernel(self):
        beta = self.params.get('beta', 2.0)
        eigvals, eigvecs = np.linalg.eigh(self.L)
        exp_eigvals = np.exp(-beta * eigvals)
        K = (eigvecs * exp_eigvals) @ eigvecs.T
        return (K + K.T) / 2


# ============================================================================
# 2. OPTIMIZER & TUNER
# ============================================================================

def optimize_pointwise(K, p_values, f0, f1, lambda_reg=10.0, lambda_bound=500.0,
                       learning_rate=0.005, max_iter=10000, tol=1e-6):
    n = K.shape[0]
    c = np.ones(n) * (1.0 / (K.sum(axis=1).mean() + 1e-10))

    for t in range(max_iter):
        alpha = K @ c
        mix = alpha * f0 + (1 - alpha) * f1
        mix = np.clip(mix, 1e-12, None)

        grad_nll = -(f0 - f1) / mix
        grad_bound = 2 * lambda_bound * (np.maximum(0, alpha - 1) - np.maximum(0, -alpha))
        grad_reg = 2 * lambda_reg * c
        grad_total = grad_nll + grad_bound + grad_reg

        gnorm = np.linalg.norm(grad_total)
        if gnorm > 5.0:
            grad_total = grad_total * (5.0 / gnorm)
        c -= learning_rate * grad_total

        if gnorm < tol and t > 100:
            break

    return c, None


class HyperparameterTuner:
    @staticmethod
    def tune_lambda(p_values, ppi, f0, f1, param_grid=None, n_folds=3, dataset_name="Dataset"):
        """Performs K-Fold CV to find the best lambda_reg on the NLL."""
        if param_grid is None:
            param_grid = list(np.logspace(-15, 20, 15))

        beta_grid = [2]

        print(f"    Starting HP Tuning (Grid: {param_grid})...")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        results_by_beta = {b: [] for b in beta_grid}
        flat_scores = []
        flat_params = []

        for bam in beta_grid:
            K = SpatialFDRGraphKernel(ppi, kernel_type='diffusion', normalized=True, beta=bam).K

            for lam in param_grid:
                fold_nlls = []
                for train_idx, test_idx in kf.split(p_values):
                    K_train = K[np.ix_(train_idx, train_idx)]
                    K_test_cross = K[np.ix_(test_idx, train_idx)]

                    p_train = p_values[train_idx]
                    f0_train, f1_train = f0[train_idx], f1[train_idx]

                    c_train, _ = optimize_pointwise(K_train, p_train, f0_train, f1_train, lambda_reg=lam)

                    alpha_test = K_test_cross @ c_train
                    alpha_test = np.clip(alpha_test, 0.00001, 0.99999)

                    mix_test = alpha_test * f0[test_idx] + (1 - alpha_test) * f1[test_idx]
                    mix_test = np.clip(mix_test, 1e-5, None)
                    nll = -np.sum(np.log(mix_test))
                    fold_nlls.append(nll)

                mean_score = np.mean(fold_nlls)
                results_by_beta[bam].append(mean_score)
                flat_scores.append(mean_score)
                flat_params.append((lam, bam))

        best_idx = np.argmin(flat_scores)
        best_lambda, best_beta = flat_params[best_idx]
        print(f"    [Tuning] Best Lambda: {best_lambda} (with Beta: {best_beta})")

        return best_lambda, best_beta, param_grid, results_by_beta


# ============================================================================
# 3. BENCHMARK RUNNER
# ============================================================================

class FDRMethods:
    @staticmethod
    def estimate_densities(p_values):
        f0_vals = np.ones_like(p_values)
        p_clipped = np.clip(p_values, 1e-10, 1 - 1e-10)
        z = norm.ppf(1 - p_clipped)
        mask = p_values < 0.2
        mu, sig = (np.mean(z[mask]), np.std(z[mask])) if mask.sum() > 10 else (2.5, 1.0)
        f1_vals = np.clip(norm.pdf(z, loc=mu, scale=sig) / norm.pdf(z), 0, 5000.0)
        return f0_vals, f1_vals


class BenchmarkRunner:
    def __init__(self, pickle_path):
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

    def align_data(self, ppi, deg_df):
        if 'ensmb' in deg_df.columns:
            deg_df = deg_df.set_index('ensmb')
        if deg_df.index.duplicated().any():
            deg_df = deg_df[~deg_df.index.duplicated(keep='first')]

        if hasattr(ppi, 'nodes'):
            graph_nodes = list(ppi.nodes())
            is_nx = True
        else:
            graph_nodes = ppi.index.tolist()
            is_nx = False

        data_genes = deg_df.index.tolist()
        common = sorted(list(set(graph_nodes).intersection(set(data_genes))))
        if len(common) < 100:
            return None, None, None

        if is_nx:
            adj = nx.to_numpy_array(ppi.subgraph(common), nodelist=common)
        else:
            adj = ppi.loc[common, common].values

        return adj, deg_df.loc[common], common

    def stratified_subsample(self, p_values, genes, adj_matrix, n_target=5000, n_bins=100):
        N = len(p_values)
        if N <= n_target:
            return p_values, genes, adj_matrix

        print(f"    Subsampling {N} -> {n_target} genes...")
        df = pd.DataFrame({'p': p_values, 'idx': range(N)})
        try:
            df['bin'] = pd.qcut(df['p'], q=n_bins, labels=False, duplicates='drop')
        except:
            df['bin'] = pd.qcut(df['p'].rank(method='first'), q=n_bins, labels=False)

        samples_per_bin = n_target // n_bins
        s_idx = df.groupby('bin', group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_bin))
        )['idx'].values
        s_idx = np.sort(s_idx)
        return p_values[s_idx], [genes[i] for i in s_idx], adj_matrix[np.ix_(s_idx, s_idx)]

    def run_inference_split(self):
        results_log = {}
        tuning_results = {}

        for bname, dsets in self.data.items():
            print(f"\nBenchmark: {bname}")

            if bname != 'TGCA':
                continue

            d_count = 0
            for dname, content in dsets.items():
                print(f"  Dataset: {dname}")

                # 1. Align
                ppi_raw, deg_df = content['ppi_network'], content['DEG']
                ppi_mat, deg_aligned, genes = self.align_data(ppi_raw, deg_df)
                if ppi_mat is None:
                    continue

                try:
                    pvals = deg_aligned.rename(columns={"P.Value": 'pvals'})['pvals'].values
                except:
                    continue

                # 2a. Tuning Step (1000 genes) - STRATIFIED
                print("    Step A: Tuning Lambda on 1000 genes...")
                p_tune, genes_tune, ppi_tune = self.stratified_subsample(
                    pvals, genes, ppi_mat, n_target=1000, n_bins=50
                )
                f0_tune, f1_tune = FDRMethods.estimate_densities(p_tune)
                best_lambda, best_beta, param_grid, tuning_scores = HyperparameterTuner.tune_lambda(
                    p_tune, ppi_tune, f0_tune, f1_tune, dataset_name=dname
                )

                # Store tuning results
                tuning_results[dname] = {
                    'param_grid': param_grid,
                    'scores_by_beta': tuning_scores,
                    'best_lambda': best_lambda,
                    'best_beta': best_beta
                }

                # 2b. Main Execution (5000 genes) - STRATIFIED
                print(f"    Step B: Main Execution on 5000 genes (Lambda={best_lambda})...")
                p_samp, genes_samp, ppi_samp = self.stratified_subsample(
                    pvals, genes, ppi_mat, n_target=5000, n_bins=100
                )

                # 3. Build Full Kernel with Best Params
                print(f"    Building kernel (beta={best_beta})...")
                gk = SpatialFDRGraphKernel(
                    ppi_samp, kernel_type='diffusion', normalized=True, beta=best_beta
                )

                # 4. Validation (BOTH checks)
                print(f"  Validating Graph-Signal Relationship...")
                is_graph_valid = KernelValidator.validate_structure(gk.K)
                is_signal_aligned = KernelValidator.validate_signal_alignment(ppi_samp, p_samp)

                if not is_graph_valid or not is_signal_aligned:
                    print("    Skipping (Validation failed)")
                    continue

                # 5. Split into Train/Test (80/20)
                print("    Splitting into train/test for inference evaluation...")
                indices = np.arange(len(p_samp))
                idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

                # Slice data and kernel
                p_train = p_samp[idx_train]
                f0_train, f1_train = FDRMethods.estimate_densities(p_train)

                K_train = gk.K[np.ix_(idx_train, idx_train)]
                K_test_cross = gk.K[np.ix_(idx_test, idx_train)]

                # 6. Stage 1: Optimize Alpha on Training Set ONLY
                print(f"    Stage 1: Learning Alpha on Train (Lambda={best_lambda})...")
                c_opt, _ = optimize_pointwise(
                    K_train, p_train, f0_train, f1_train,
                    lambda_reg=best_lambda
                )
                alpha_train = np.clip(K_train @ c_opt, 0.001, 0.999)

                # 7. Stage 2: Global KLR
                print("    Stage 2: Fitting Global KLR...")
                klr = GlobalFDRRegressor(lambda_global=0.5)
                klr.fit(K_train, alpha_train)

                # 8. Inference on Unseen Nodes
                print("    Inference: Predicting Unseen Points...")
                alpha_pred = klr.predict(K_test_cross)

                results_log[dname] = {
                    'y_hidden_pvals': p_samp[idx_test],
                    'alpha_pred': alpha_pred,
                    'best_lambda': best_lambda,
                    'best_beta': best_beta
                }

                d_count += 1

        return results_log, tuning_results


if __name__ == "__main__":
    runner = BenchmarkRunner("/home/benny/Repos/SmoothieFDR/Data/benchmarks.pkl")
    results, tuning_results = runner.run_inference_split()

    # Save results to cache
    cache_data = {
        'results': results,
        'tuning_results': tuning_results
    }
    cache_path = os.path.join(CACHE_DIR, "gsea_allspace_results.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"\nResults saved to: {cache_path}")
    print(f"Run 'python plot_gsea_allspace_bench.py' to generate visualizations.")
