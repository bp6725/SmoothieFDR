"""FDR control methods."""

from .baseline import benjamini_hochberg
from .spatial_fdr import SpatialFDR
from .kernels import (
    matern_kernel,
    rbf_kernel,
    compute_kernel_matrix,
    estimate_length_scale
)
from .optimization import (
    PointwiseOptimizer,
    OptimizerConfig,
    optimize_pointwise,
    create_optimizer,
    OPTIMIZER_PRESETS
)
from .global_inference import (
    GlobalFDRRegressor,
    GlobalFDRConfig,
    create_global_regressor
)

__all__ = [
    # Baseline
    'benjamini_hochberg',
    # Main class
    'SpatialFDR',
    # Kernels
    'matern_kernel',
    'rbf_kernel',
    'compute_kernel_matrix',
    'estimate_length_scale',
    # Optimization (Stage 1)
    'PointwiseOptimizer',
    'OptimizerConfig',
    'optimize_pointwise',
    'create_optimizer',
    'OPTIMIZER_PRESETS',
    # Global inference (Stage 2)
    'GlobalFDRRegressor',
    'GlobalFDRConfig',
    'create_global_regressor',
]
