import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .adbench_loader import load_from_ADbench
from ..methods.kernels import compute_kernel_matrix, estimate_length_scale

# The "Top 9" Datasets with verified structure
Valid_Datasets = [
    '33_skin',  # Gap: 0.3682
    '19_landsat',  # Gap: 0.3463
    '31_satimage-2',  # Gap: 0.3326
    '30_satellite',  # Gap: 0.3295
    '41_Waveform',  # Gap: 0.2452
    '25_musk',  # Gap: 0.2432
    '4_breastw',  # Gap: 0.1997
    '45_wine',  # Gap: 0.1761
    '15_Hepatitis'  # Gap: 0.0902 (Robustness check)
]


def generate_pvalues(labels, effect_strength='medium'):
    """
    Generates p-values based on labels.
    H0 (1) -> Uniform(0,1)
    H1 (0) -> Beta(a, 1)
    """
    p_values = np.zeros(len(labels))

    # Parameters for H1 distribution (Lower alpha = Stronger signal)
    alphas = {'weak': 0.5, 'medium': 0.1, 'strong': 0.02}
    a = alphas.get(effect_strength, 0.1)

    # H0: Uniform
    p_values[labels == 1] = np.random.uniform(0, 1, size=(labels == 1).sum())

    # H1: Beta
    p_values[labels == 0] = np.random.beta(a, 1, size=(labels == 0).sum())

    # Clip to avoid numerical instability
    return np.clip(p_values, 1e-10, 1.0)


def load_and_sample_mixed_clusters(dataset_name,
                                   n_total=500,
                                   cluster_corruption=0.2,  # 20% of points in a cluster get the "wrong" label
                                   background_ratio=0.5,  # 50% of data is background noise
                                   sigma_factor=0.5,
                                   effect_strength='medium'):
    """
    Creates a Mixed-Membership Spatial Dataset:
    1. 'Signal Cluster': Dense block, mostly H1, but with some H0s inside.
    2. 'Noise Cluster': Dense block, mostly H0, but with some H1s inside.
    3. 'Background': Distant H0 points.
    """
    # 1. Load & Kernel
    try:
        data = load_from_ADbench(dataset_name)
        X = StandardScaler().fit_transform(data['X_train'])
    except:
        return None

    # Spectral Clustering to find blocks
    sigma = estimate_length_scale(X, method='median') * sigma_factor
    K = compute_kernel_matrix(X, kernel_type='rbf', length_scale=sigma)

    # Fast Eigen decomposition
    D = np.array(K.sum(axis=1)).flatten() + 1e-10
    A_norm = (np.diag(1 / np.sqrt(D)) @ K @ np.diag(1 / np.sqrt(D)))
    try:
        _, evecs = eigsh(A_norm, k=10, which='LA')
    except:
        _, evecs = linalg.eigh(A_norm)
        evecs = evecs[:, -10:]

    # Heuristic k=3 (Signal Block, Noise Block, Background)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(evecs[:, ::-1][:, :optimal_k])
    labels_full = kmeans.labels_

    # 2. Assign Roles to Clusters
    # We need 2 distinct dense clusters. We pick the two largest.
    counts = np.bincount(labels_full)
    largest_clusters = np.argsort(counts)[::-1]

    cid_signal = largest_clusters[0]  # The "Signal Source"
    cid_noise = largest_clusters[1]  # The "Noise Source"

    # 3. Sampling Quotas
    n_background = int(n_total * background_ratio)
    n_clusters = n_total - n_background
    n_per_cluster = n_clusters // 2

    # 4. Extract Pools
    pool_signal_cluster = np.where(labels_full == cid_signal)[0]
    pool_noise_cluster = np.where(labels_full == cid_noise)[0]
    pool_background = np.where((labels_full != cid_signal) & (labels_full != cid_noise))[0]

    # Fallback if background pool is empty (e.g. k=2 dataset)
    if len(pool_background) < n_background:
        # Steal from the largest cluster
        pool_background = np.concatenate([pool_background, pool_signal_cluster[n_per_cluster:]])
        pool_signal_cluster = pool_signal_cluster[:n_per_cluster]

    # 5. Sample Indices
    idx_sig_cluster = np.random.choice(pool_signal_cluster, n_per_cluster, replace=False)
    idx_noi_cluster = np.random.choice(pool_noise_cluster, n_per_cluster, replace=False)
    idx_background = np.random.choice(pool_background, n_background, replace=False)

    # 6. Assign True Labels (0=H1, 1=H0)
    # Start with base assumption
    labels_sig_cluster = np.zeros(n_per_cluster, dtype=int)  # All H1
    labels_noi_cluster = np.ones(n_per_cluster, dtype=int)  # All H0
    labels_background = np.ones(n_background, dtype=int)  # All H0

    # --- CORRUPTION STEP ---
    # Flip H1 -> H0 in Signal Cluster (Holes in cheese)
    n_flip_sig = int(n_per_cluster * cluster_corruption)
    flip_idx = np.random.choice(n_per_cluster, n_flip_sig, replace=False)
    labels_sig_cluster[flip_idx] = 1  # Become H0

    # Flip H0 -> H1 in Noise Cluster (Needles in haystack)
    n_flip_noi = int(n_per_cluster * cluster_corruption)
    flip_idx = np.random.choice(n_per_cluster, n_flip_noi, replace=False)
    labels_noi_cluster[flip_idx] = 0  # Become H1

    # 7. Combine
    indices = np.concatenate([idx_sig_cluster, idx_noi_cluster, idx_background])
    true_labels = np.concatenate([labels_sig_cluster, labels_noi_cluster, labels_background])

    # Shuffle together
    perm = np.random.permutation(len(indices))
    indices = indices[perm]
    true_labels = true_labels[perm]

    # 8. Generate P-values
    p_values = generate_pvalues(true_labels, effect_strength)

    # Return (X, p, y)
    return X[indices], p_values, true_labels