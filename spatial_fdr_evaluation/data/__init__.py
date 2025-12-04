"""Data loading and generation utilities."""

from .loader import (
    load_from_ADbench,
    extract_spatial_structure,
    subsample_locations
)

from .synthetic import (
    generate_spatial_clusters,
    generate_clustered_labels,
    generate_pvalues,
    generate_evaluation_data
)

__all__ = [
    'load_from_ADbench',
    'extract_spatial_structure',
    'subsample_locations',
    'generate_spatial_clusters',
    'generate_clustered_labels',
    'generate_pvalues',
    'generate_evaluation_data'
]
