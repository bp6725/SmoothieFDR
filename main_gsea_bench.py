"""
GSEA Benchmark: Point-wise FDR on Gene Set Enrichment Analysis

This script runs the Spatial FDR method on GSEA benchmarks using graph diffusion kernels.
Results are saved to cache for separate visualization.

REFACTOR: Results saved to cache for separate visualization (use plot_gsea_bench.py).
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom, norm
from statsmodels.stats.multitest import multipletests
import networkx as nx
import warnings
import sys
import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.linalg import eigvalsh

# --- PATH SETUP ---
sys.path.insert(0, os.path.abspath('.'))

# --- CACHE CONFIGURATION ---
CACHE_DIR = "/home/benny/Repos/SmoothieFDR/results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================================
# 0. VALIDATORS
# ============================================================================

class KernelValidator:
    @staticmethod
    def validate_structure(K):
        """
        Checks if the Graph has good cluster structure (Spectral Gap).
        """
        print(f"DEBUG: K statistics - Min: {K.min()}, Max: {K.max()}, Has NaNs: {np.isnan(K).any()}")

        if np.isnan(K).any() or np.isinf(K).any():
            print("CRITICAL: Kernel matrix contains NaNs or Infs! Fixing with 0 replacement...")
            K = np.nan_to_num(K)

        all_vals = np.linalg.eigvalsh(K)
        vals = all_vals[-5:]
        vals = np.sort(vals)[::-1]

        lambda_1 = vals[0] if vals[0] > 0 else 1e-10
        lambda_2 = vals[1]

        ratio = lambda_2 / lambda_1
        print(f"    [Structure] Top Eigs: {vals[:3]}")
        print(f"    [Structure] Spectral Ratio (lam2/lam1): {ratio:.4f}")

        if ratio > 0.99:
            print("    -> Graph has very strong, nearly disconnected clusters.")
            num_components = np.sum(np.isclose(all_vals, 1.0))
            print(f"Number of disconnected clusters: {num_components}")
            print(f"Total nodes: {len(all_vals)}")

            if num_components > (len(all_vals) * 0.9):
                print("WARNING: Graph is shattered (too many tiny clusters).")
                return False
            else:
                print("SUCCESS: Meaningful cluster structure detected.")
                return True
        elif ratio < 0.5:
            print("    -> Graph is very connected (Hairball).")
            return False
        else:
            print("    -> Graph has moderate structure.")
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


# ============================================================================
# 1. OPTIMIZER (CPU Vanilla)
# ============================================================================

def optimize_pointwise(K, p_values, f0, f1, lambda_reg=10.0, lambda_bound=500.0,
                       learning_rate=0.05, max_iter=15000, tol=1e-6):
    """Vanilla Natural Gradient Descent (CPU/NumPy) with history tracking."""
    n = K.shape[0]
    c = np.ones(n) * (1.0 / (K.sum(axis=1).mean() + 1e-10))

    history = {
        'losses': [],
        'grad_norms': [],
        'alpha_history': [],
        'violations': []
    }

    for t in range(max_iter):
        alpha = K @ c
        mix = alpha * f0 + (1 - alpha) * f1
        mix = np.clip(mix, 1e-12, None)

        grad_nll = -(f0 - f1) / mix
        grad_upper = np.maximum(0, alpha - 1)
        grad_lower = np.maximum(0, -alpha)
        grad_bound = 2 * lambda_bound * (grad_upper - grad_lower)
        grad_reg = 2 * lambda_reg * c

        grad_total = grad_nll + grad_bound + grad_reg

        gnorm = np.linalg.norm(grad_total)
        if gnorm > 5.0:
            grad_total = grad_total * (5.0 / gnorm)

        c -= learning_rate * grad_total

        if t % 50 == 0:
            loss = -np.sum(np.log(mix)) + lambda_reg * (c @ K @ c)
            history['losses'].append(loss)
            history['grad_norms'].append(gnorm)
            history['alpha_history'].append(alpha.copy())
            history['violations'].append(np.sum((alpha < 0) | (alpha > 1)))

            if gnorm < tol:
                break

    return c, history


# ============================================================================
# 2. HYPERPARAMETER TUNER
# ============================================================================

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
# 3. KERNEL & METHODS
# ============================================================================

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
        if hasattr(W, 'nodes'): W = nx.to_numpy_array(W)
        if sp.issparse(W): W = W.toarray()
        if not np.allclose(W, W.T): W = (W + W.T) / 2
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
        if self.kernel_type == 'diffusion':
            beta = self.params.get('beta', 1.0)
            eigvals, eigvecs = np.linalg.eigh(self.L)
            exp_eigvals = np.exp(-beta * eigvals)
            K = (eigvecs * exp_eigvals) @ eigvecs.T
            return (K + K.T) / 2
        else:
            raise ValueError("Only diffusion supported in this block")


class FDRMethods:
    @staticmethod
    def standard_bh(p_values, alpha=0.05):
        rejected, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return rejected, q_values

    @staticmethod
    def estimate_densities(p_values):
        f0_vals = np.ones_like(p_values)

        p_clipped = np.clip(p_values, 1e-10, 1 - 1e-10)
        z = norm.ppf(1 - p_clipped)
        mask = p_values < 0.2
        if mask.sum() > 10:
            mu, sig = np.mean(z[mask]), np.std(z[mask])
        else:
            mu, sig = 2.5, 1.0

        numer = norm.pdf(z, loc=mu, scale=sig)
        denom = norm.pdf(z)
        f1_vals = np.clip(numer / denom, 0, 5000.0)
        return f0_vals, f1_vals

    @staticmethod
    def smooth_graph_fdr(p_values, adj_matrix, dataset_name, alpha=0.05, lambda_reg=10.0, beta=2):
        n = len(p_values)
        gk = SpatialFDRGraphKernel(adj_matrix, kernel_type='diffusion', normalized=True, beta=beta)

        print(f"  Validating Graph-Signal Relationship...")
        is_graph_valid = KernelValidator.validate_structure(gk.K)
        KernelValidator.validate_signal_alignment(adj_matrix, p_values)

        print(f" Finish Validating")

        if not is_graph_valid:
            return None, None, None, None, None

        f0, f1 = FDRMethods.estimate_densities(p_values)
        c_opt, history = optimize_pointwise(gk.K, p_values, f0, f1, lambda_reg=lambda_reg)

        alpha_final = gk.K @ c_opt
        alpha_final = np.clip(alpha_final, 0.00001, 0.99999)

        f0_vals = f0
        f1_vals = f1

        lfdr = (alpha_final * f0_vals) / (alpha_final * f0_vals + (1 - alpha_final) * f1_vals)

        idx = np.argsort(lfdr)
        sorted_lfdr = lfdr[idx]
        q_vals_sorted = np.cumsum(sorted_lfdr) / np.arange(1, len(lfdr) + 1)

        rejections = np.zeros(len(lfdr), dtype=bool)
        threshold = alpha if alpha is not None else 0.1

        if np.sum(q_vals_sorted <= threshold) > 0:
            max_idx = np.max(np.where(q_vals_sorted <= threshold)[0])
            rejections[idx[:max_idx + 1]] = True

        print(f"[{dataset_name}] Rej: {rejections.sum()}")

        q_values_aligned = np.zeros_like(lfdr)
        q_values_aligned[idx] = q_vals_sorted

        return rejections, q_values_aligned, alpha_final, gk.K, history


class EnrichmentEngine:
    def __init__(self, pathway_dict, all_genes_universe):
        self.pathway_dict = pathway_dict
        self.universe = set(all_genes_universe)
        self.M = len(self.universe)

    def test_enrichment(self, selected_genes):
        selected_set = set(selected_genes).intersection(self.universe)
        N = len(selected_set)
        results = []

        for pname, pgenes in self.pathway_dict.items():
            pset = set(pgenes).intersection(self.universe)
            n = len(pset)
            if n < 3:
                results.append({'pathway': pname, 'pval': 1.0, 'overlap': 0, 'size': n})
                continue

            k = len(selected_set.intersection(pset))
            pval = hypergeom.sf(k - 1, self.M, n, N) if k > 0 else 1.0
            results.append({'pathway': pname, 'pval': pval, 'overlap': k, 'size': n})

        res = pd.DataFrame(results)
        return res.sort_values('pval') if not res.empty else res


# ============================================================================
# 4. BENCHMARK RUNNER (NO PLOTTING - RESULTS SAVED TO CACHE)
# ============================================================================

class BenchmarkRunner:
    def __init__(self, pickle_path):
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

    def align_data(self, ppi, deg_df):
        """Robust Alignment & Deduplication"""
        if 'ensmb' in deg_df.columns:
            deg_df = deg_df.set_index('ensmb')
        if deg_df.index.duplicated().any():
            deg_df = deg_df[~deg_df.index.duplicated(keep='first')]

        if hasattr(ppi, 'nodes'):
            graph_nodes = list(ppi.nodes())
            is_nx = True
        elif isinstance(ppi, pd.DataFrame):
            graph_nodes = ppi.index.tolist()
            is_nx = False
        else:
            return None, None, None

        data_genes = deg_df.index.tolist()
        if len(graph_nodes) > 0 and len(data_genes) > 0 and type(graph_nodes[0]) != type(data_genes[0]):
            try:
                if isinstance(graph_nodes[0], int):
                    deg_df.index = deg_df.index.astype(int)
                elif isinstance(graph_nodes[0], str):
                    deg_df.index = deg_df.index.astype(str)
                data_genes = deg_df.index.tolist()
            except:
                pass

        common = sorted(list(set(graph_nodes).intersection(set(data_genes))))
        if len(common) == 0:
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

    def run(self):
        results_log = []
        all_histories = {}
        all_tuning_results = {}

        for bname, dsets in self.data.items():
            print(f"\nBenchmark: {bname}")
            d_count = 0

            for dname, content in dsets.items():
                if d_count >= 10:
                    break
                print(f"  Dataset ({d_count + 1}/10): {dname}")

                # 1. Align
                ppi_raw, deg_df = content['ppi_network'], content['DEG']
                ppi_mat, deg_aligned, genes = self.align_data(ppi_raw, deg_df)
                if ppi_mat is None:
                    continue

                try:
                    pvals = deg_aligned.rename(columns={"P.Value": 'pvals'})['pvals'].values
                except:
                    continue

                # 2a. Tuning Step (1000 genes)
                print("    Step A: Tuning Lambda on 1000 genes...")
                p_tune, genes_tune, ppi_tune = self.stratified_subsample(pvals, genes, ppi_mat, n_target=1000, n_bins=50)
                f0_tune, f1_tune = FDRMethods.estimate_densities(p_tune)
                best_lambda, best_beta, param_grid, tuning_scores = HyperparameterTuner.tune_lambda(
                    p_tune, ppi_tune, f0_tune, f1_tune, dataset_name=dname
                )

                # Store tuning results
                all_tuning_results[f"{bname}_{dname}"] = {
                    'param_grid': param_grid,
                    'scores_by_beta': tuning_scores,
                    'best_lambda': best_lambda,
                    'best_beta': best_beta
                }

                # 2b. Main Execution (5000 genes)
                print(f"    Step B: Main Execution on 5000 genes (Lambda={best_lambda})...")
                p_samp, genes_samp, ppi_samp = self.stratified_subsample(pvals, genes, ppi_mat, n_target=5000, n_bins=100)

                # Run BH
                rej_bh, _ = FDRMethods.standard_bh(p_samp)
                genes_bh = [genes_samp[i] for i in range(len(genes_samp)) if rej_bh[i]]

                # Run Graph FDR
                genes_graph = []
                rej_graph, _, pi_0, K_final, history = FDRMethods.smooth_graph_fdr(
                    p_samp, ppi_samp,
                    dataset_name=dname,
                    lambda_reg=best_lambda,
                    beta=best_beta,
                    alpha=0.1
                )
                if rej_graph is None:
                    continue
                genes_graph = [genes_samp[i] for i in range(len(genes_samp)) if rej_graph[i]]

                # Store history
                all_histories[f"{bname}_{dname}"] = history

                # Enrichment & AUC Calculation
                eng = EnrichmentEngine(content['pathways'], genes_samp)
                true_labels = set(content['positive_labels'])

                res_bh = eng.test_enrichment(genes_bh)
                res_graph = eng.test_enrichment(genes_graph)

                auc_bh = self._evaluate_auc(res_bh, true_labels)
                auc_graph = self._evaluate_auc(res_graph, true_labels)

                metrics = {
                    'benchmark': bname,
                    'dataset': dname,
                    'best_lambda': best_lambda,
                    'best_beta': best_beta,
                    'auc_bh': auc_bh,
                    'auc_graph': auc_graph,
                    'n_rej_bh': int(rej_bh.sum()),
                    'n_rej_graph': int(rej_graph.sum())
                }
                results_log.append(metrics)
                print(f"    AUC: BH={metrics['auc_bh']:.3f}, Graph={metrics['auc_graph']:.3f}")
                d_count += 1

        return pd.DataFrame(results_log), all_histories, all_tuning_results

    def _evaluate_auc(self, df_res, true_label_set):
        """Calculates AUC of the ROC curve."""
        if df_res.empty:
            return 0.5

        y_true = df_res['pathway'].apply(lambda x: 1 if x in true_label_set else 0).values

        if len(np.unique(y_true)) < 2:
            return 0.5

        y_score = -np.log10(df_res['pval'].values + 1e-100)

        try:
            return roc_auc_score(y_true, y_score)
        except:
            return 0.5


if __name__ == "__main__":
    runner = BenchmarkRunner("/home/benny/Repos/SmoothieFDR/Data/benchmarks.pkl")
    results_df, histories, tuning_results = runner.run()

    # Save results to cache
    cache_data = {
        'results_df': results_df,
        'histories': histories,
        'tuning_results': tuning_results
    }
    cache_path = os.path.join(CACHE_DIR, "gsea_bench_results.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"\nResults saved to: {cache_path}")
    print(f"Run 'python plot_gsea_bench.py' to generate visualizations.")

    if not results_df.empty:
        print("\nFinal Summary:")
        print(results_df[['benchmark', 'dataset', 'best_lambda', 'auc_bh', 'auc_graph']])
