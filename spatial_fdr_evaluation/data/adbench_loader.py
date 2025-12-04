"""
Data loading utilities for ADbench anomaly detection datasets.
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Dict, Tuple
from scipy import stats
from scipy.stats import rankdata


class AdBench_Simulator:
    """Wrapper for ADbench DataGenerator"""

    def __init__(self):
        pass

    @staticmethod
    def load_dataset(data_set='2_annthyroid', la=0.1, resample=True, return_all_data=False):
        """Load single dataset from ADbench"""
        from third_party.ADbench.data_generator import DataGenerator

        data_generator = DataGenerator(dataset=data_set, generate_duplicates=resample)
        data = data_generator.generator(la=la, return_all_data=return_all_data)
        return data


def load_from_ADbench(dataset_name, n_dims_to_take=-1, n_samples=-1, la=0.1):
    """
    Load dataset from ADbench and return in standard format

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., '2_annthyroid', '3_cardio')
    n_dims_to_take : int
        Number of dimensions to use (-1 = all)
    n_samples : int
        Number of samples to use (-1 = all)
    la : float
        Label amount

    Returns
    -------
    dict
        Dictionary with X_train, X_test, Y_train, Y_test, metadata
    """
    dataset = AdBench_Simulator.load_dataset(dataset_name, la, True, False)
    X_train, X_test = dataset['X_train'], dataset['X_test']
    Y_train, Y_test = dataset['y_train'], dataset['y_test']

    if n_samples != -1:
        X_train = X_train[0:n_samples]
        Y_train = Y_train[0:n_samples]  # FIX: Also subsample Y_train!

    if n_dims_to_take != -1:
        raise NotImplementedError('No other dims for you.')

    # Note: Normalization is commented out in original code
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    running_results = {}
    running_results['X_train'] = X_train
    running_results['X_test'] = X_test
    running_results['Y_train'] = Y_train
    running_results['Y_test'] = Y_test
    running_results['dataset_name'] = dataset_name
    running_results["is_supervised"] = sum(Y_train) > 0
    running_results["dim"] = X_train.shape[1]

    return running_results


def convert_adbench_to_spatial_fdr(data_dict, use_train=True, anomaly_method='isolation_forest', random_state=42):
    """
    Convert ADbench dataset to spatial FDR format: (locations, true_labels, p_values)

    Parameters
    ----------
    data_dict : dict
        Output from load_from_ADbench
    use_train : bool
        Use train (True) or test (False) split
    anomaly_method : str
        Method to compute p-values: 'isolation_forest', 'lof', 'knn'
    random_state : int
        Random seed

    Returns
    -------
    locations : np.ndarray, shape (n_samples, n_features)
        Feature vectors as spatial locations
    true_labels : np.ndarray, shape (n_samples,)
        True labels: 0 = anomaly/alternative, 1 = normal/null
    p_values : np.ndarray, shape (n_samples,)
        P-values from anomaly scores
    """
    # Select split
    if use_train:
        X = data_dict['X_train']
        Y = data_dict['Y_train']
    else:
        X = data_dict['X_test']
        Y = data_dict['Y_test']

    # Compute anomaly scores
    if anomaly_method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(random_state=random_state, contamination='auto')
        clf.fit(X)
        scores = -clf.score_samples(X)  # Higher = more anomalous

    elif anomaly_method == 'lof':
        from sklearn.neighbors import LocalOutlierFactor
        clf = LocalOutlierFactor(novelty=False, contamination='auto')
        scores = -clf.fit_predict(X)

    elif anomaly_method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        k = min(10, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        scores = np.mean(distances, axis=1)
    else:
        raise ValueError(f"Unknown method: {anomaly_method}")

    # Convert scores to p-values via ranking
    ranks = rankdata(scores, method='average')
    p_values = 1 - (ranks / len(scores))
    p_values = np.clip(p_values, 1e-10, 1 - 1e-10)

    # Convert labels: ADbench uses 1=anomaly, we use 0=anomaly
    locations = X
    true_labels = 1 - Y  # Flip: 0→1 (null), 1→0 (alternative)

    return locations, true_labels, p_values


def extract_spatial_structure(X: np.ndarray,
                              bandwidth: float = 'scott',
                              kernel: str = 'gaussian',
                              random_state: int = 42) -> Tuple[np.ndarray, KernelDensity]:
    """
    Extract spatial structure from data using Kernel Density Estimation.

    This provides the spatial locations and their density for realistic
    spatial FDR evaluation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data
    bandwidth : float or str, default='scott'
        Bandwidth for KDE. If 'scott', uses Scott's rule.
    kernel : str, default='gaussian'
        Kernel type for KDE
    random_state : int, default=42
        Random seed

    Returns
    -------
    locations : np.ndarray, shape (n_samples, n_features)
        Spatial locations (same as X)
    kde : KernelDensity
        Fitted KDE model for sampling distribution p(loc)
    """
    n_samples, n_features = X.shape

    # Compute bandwidth using Scott's rule if needed
    if bandwidth == 'scott':
        bandwidth = n_samples ** (-1.0 / (n_features + 4))

    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(X)

    return X, kde


def subsample_locations(X: np.ndarray,
                       n_samples: int,
                       random_state: int = 42) -> np.ndarray:
    """
    Subsample actual locations from dataset.

    Selects a random subset of actual data points from X.

    Parameters
    ----------
    X : np.ndarray
        Full dataset
    n_samples : int
        Number of samples to select
    random_state : int
        Random seed

    Returns
    -------
    locations : np.ndarray
        Subsampled locations (actual data points from X)
    """
    np.random.seed(random_state)

    # Return all if asking for more than available
    if n_samples >= len(X):
        return X.copy()

    # Uniform random subsample from actual data points
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    return X[indices]
