"""
Benchmark Configuration System for SmoothieFDR.

This module provides a unified configuration system for running
experiments across different datasets and benchmark types.

Usage:
    # Load from YAML
    config = load_config('configs/euclidean_bench.yaml')

    # Use presets
    config = EUCLIDEAN_PRESETS['default']

    # Programmatic
    config = BenchmarkConfig(
        datasets=['33_skin', '19_landsat'],
        benchmark_filter=['TGCA']  # Only run on TGCA
    )
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Any
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    # Sampling
    n_total: Union[int, List[int]] = 450  # Total samples or [cluster1, cluster2, background]
    n_clusters: int = 2
    min_cluster_size: int = 30
    candidate_pool: int = 10000

    # Kernel
    sigma_factor: float = 1.0

    # Signal generation
    cluster_corruption: float = 0.2
    effect_strength: str = 'medium'  # 'weak', 'medium', 'strong'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetConfig':
        return cls(**d)


@dataclass
class BenchmarkConfig:
    """
    Configuration for Euclidean (ADbench) benchmark experiments.

    Parameters
    ----------
    datasets : list of str
        Dataset names to run (e.g., ['33_skin', '19_landsat'])
    dataset_configs : dict
        Per-dataset configuration overrides
    benchmark_filter : list of str, optional
        For GSEA: filter to specific benchmarks (e.g., ['TGCA'])
        Set to None to run all benchmarks

    Optimization
    ------------
    optimizer : str
        'vanilla' (default) or 'adam'
    learning_rate : float
    lambda_reg : float
    lambda_bound : float
    max_iter : int
    lambda_sparse : float
        Sparsity penalty (optional)

    Cross-Validation
    ----------------
    sigma_grid : list of float
        Sigma multipliers for CV
    n_folds : int
        Number of CV folds

    Experiment
    ----------
    random_state : int
    n_seeds : int
        Number of random seeds per dataset
    """
    # Dataset selection
    datasets: List[str] = field(default_factory=lambda: [
        '33_skin',
        '19_landsat',
        '31_satimage-2',
        '30_satellite',
        '41_Waveform',
        '25_musk',
        '4_breastw',
        '45_wine',
        '15_Hepatitis'
    ])

    # Per-dataset configs (optional overrides)
    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=dict)

    # Benchmark filtering (for GSEA)
    benchmark_filter: Optional[List[str]] = None

    # Optimization parameters
    optimizer: str = 'vanilla'
    learning_rate: float = 0.0005
    lambda_reg: float = 10.0
    lambda_bound: float = 500.0
    max_iter: int = 5000
    lambda_sparse: float = 0.001

    # Cross-validation
    sigma_grid: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
    ])
    n_folds: int = 5

    # Experiment settings
    random_state: int = 42
    n_seeds: int = 1

    # Output
    results_dir: str = 'results'

    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset."""
        if dataset_name in self.dataset_configs:
            return self.dataset_configs[dataset_name]
        return DatasetConfig()  # Return defaults

    def should_run_benchmark(self, benchmark_name: str) -> bool:
        """Check if benchmark should be run (for GSEA filtering)."""
        if self.benchmark_filter is None:
            return True
        return benchmark_name in self.benchmark_filter

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert DatasetConfig objects
        d['dataset_configs'] = {
            k: v.to_dict() if isinstance(v, DatasetConfig) else v
            for k, v in d['dataset_configs'].items()
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'BenchmarkConfig':
        """Create from dictionary."""
        # Convert dataset_configs back to DatasetConfig objects
        if 'dataset_configs' in d:
            d['dataset_configs'] = {
                k: DatasetConfig.from_dict(v) if isinstance(v, dict) else v
                for k, v in d['dataset_configs'].items()
            }
        return cls(**d)


@dataclass
class GSEABenchmarkConfig:
    """
    Configuration for GSEA (graph-based) benchmark experiments.

    Parameters
    ----------
    data_path : str
        Path to benchmarks.pkl file
    benchmarks : list of str, optional
        Specific benchmarks to run. None = run all.
    max_datasets_per_benchmark : int
        Maximum datasets to process per benchmark
    subsample_tuning : int
        Number of genes for hyperparameter tuning
    subsample_main : int
        Number of genes for main execution
    """
    # Data
    data_path: str = "Data/benchmarks.pkl"

    # Benchmark selection
    benchmarks: Optional[List[str]] = None  # None = all, or ['TGCA', 'GEO', ...]
    max_datasets_per_benchmark: int = 10

    # Subsampling
    subsample_tuning: int = 1000
    subsample_main: int = 5000
    n_bins_tuning: int = 50
    n_bins_main: int = 100

    # Kernel parameters
    beta_grid: List[float] = field(default_factory=lambda: [2])
    lambda_grid: List[float] = field(default_factory=lambda: list())

    # Optimization (uses same defaults as BenchmarkConfig)
    learning_rate: float = 0.05
    lambda_reg: float = 10.0
    lambda_bound: float = 500.0
    max_iter: int = 15000

    # CV
    n_folds: int = 3

    # Experiment
    random_state: int = 42

    # Validation flags
    validate_structure: bool = True
    validate_alignment: bool = True

    def __post_init__(self):
        """Set default lambda grid if not provided."""
        if not self.lambda_grid:
            self.lambda_grid = list(__import__('numpy').logspace(-15, 20, 15))

    def should_run_benchmark(self, benchmark_name: str) -> bool:
        """Check if benchmark should be run."""
        if self.benchmarks is None:
            return True
        return benchmark_name in self.benchmarks

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GSEABenchmarkConfig':
        return cls(**d)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Default dataset configurations for Euclidean benchmarks
_DEFAULT_DATASET_CONFIGS = {
    "33_skin": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=30,
        sigma_factor=1.0
    ),
    "19_landsat": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=30,
        sigma_factor=1.0
    ),
    "31_satimage-2": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=30,
        sigma_factor=0.5
    ),
    "30_satellite": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=50,
        sigma_factor=1.0
    ),
    "41_Waveform": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=30,
        sigma_factor=0.5
    ),
    "25_musk": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=30,
        sigma_factor=1.0
    ),
    "4_breastw": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=20,
        sigma_factor=0.5
    ),
    "45_wine": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=20,
        sigma_factor=0.5
    ),
    "15_Hepatitis": DatasetConfig(
        n_total=[50, 200, 200],
        n_clusters=2,
        min_cluster_size=20,
        sigma_factor=0.5
    ),
}

# Euclidean benchmark presets
EUCLIDEAN_PRESETS: Dict[str, BenchmarkConfig] = {
    'default': BenchmarkConfig(
        dataset_configs=_DEFAULT_DATASET_CONFIGS
    ),

    'quick_test': BenchmarkConfig(
        datasets=['33_skin', '4_breastw'],
        dataset_configs=_DEFAULT_DATASET_CONFIGS,
        max_iter=1000,
        n_seeds=1
    ),

    'full_evaluation': BenchmarkConfig(
        dataset_configs=_DEFAULT_DATASET_CONFIGS,
        n_seeds=5,
        max_iter=10000
    ),
}

# GSEA benchmark presets
GSEA_PRESETS: Dict[str, GSEABenchmarkConfig] = {
    'default': GSEABenchmarkConfig(),

    'tgca_only': GSEABenchmarkConfig(
        benchmarks=['TGCA'],
        max_datasets_per_benchmark=10
    ),

    'quick_test': GSEABenchmarkConfig(
        benchmarks=['TGCA'],
        max_datasets_per_benchmark=2,
        subsample_tuning=500,
        subsample_main=2000,
        max_iter=5000
    ),

    'full_evaluation': GSEABenchmarkConfig(
        benchmarks=None,  # All benchmarks
        max_datasets_per_benchmark=20,
        max_iter=20000
    ),
}


# =============================================================================
# I/O FUNCTIONS
# =============================================================================

def load_config(path: Union[str, Path]) -> Union[BenchmarkConfig, GSEABenchmarkConfig]:
    """
    Load configuration from YAML or JSON file.

    Parameters
    ----------
    path : str or Path
        Path to configuration file (.yaml, .yml, or .json)

    Returns
    -------
    config : BenchmarkConfig or GSEABenchmarkConfig
        Loaded configuration
    """
    path = Path(path)

    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown config format: {path.suffix}")

    # Determine config type
    config_type = data.pop('type', 'euclidean')

    if config_type == 'gsea':
        return GSEABenchmarkConfig.from_dict(data)
    else:
        return BenchmarkConfig.from_dict(data)


def save_config(
    config: Union[BenchmarkConfig, GSEABenchmarkConfig],
    path: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """
    Save configuration to file.

    Parameters
    ----------
    config : BenchmarkConfig or GSEABenchmarkConfig
        Configuration to save
    path : str or Path
        Output path
    format : str
        'yaml' or 'json'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    # Add type marker
    if isinstance(config, GSEABenchmarkConfig):
        data['type'] = 'gsea'
    else:
        data['type'] = 'euclidean'

    if format == 'yaml':
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_config_from_preset(
    preset_name: str,
    config_type: str = 'euclidean',
    **overrides
) -> Union[BenchmarkConfig, GSEABenchmarkConfig]:
    """
    Create configuration from preset with optional overrides.

    Parameters
    ----------
    preset_name : str
        Preset name ('default', 'quick_test', etc.)
    config_type : str
        'euclidean' or 'gsea'
    **overrides
        Override specific fields

    Returns
    -------
    config : BenchmarkConfig or GSEABenchmarkConfig
    """
    if config_type == 'gsea':
        presets = GSEA_PRESETS
        config_class = GSEABenchmarkConfig
    else:
        presets = EUCLIDEAN_PRESETS
        config_class = BenchmarkConfig

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")

    # Get preset and convert to dict
    base_config = presets[preset_name]
    config_dict = base_config.to_dict()

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value

    return config_class.from_dict(config_dict)
