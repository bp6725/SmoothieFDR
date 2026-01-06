"""
Configuration module for SmoothieFDR experiments.

This module provides configuration classes and presets for running
experiments across different datasets and benchmark settings.
"""

from .benchmark_config import (
    BenchmarkConfig,
    DatasetConfig,
    GSEABenchmarkConfig,
    load_config,
    save_config,
    EUCLIDEAN_PRESETS,
    GSEA_PRESETS,
)

__all__ = [
    'BenchmarkConfig',
    'DatasetConfig',
    'GSEABenchmarkConfig',
    'load_config',
    'save_config',
    'EUCLIDEAN_PRESETS',
    'GSEA_PRESETS',
]
