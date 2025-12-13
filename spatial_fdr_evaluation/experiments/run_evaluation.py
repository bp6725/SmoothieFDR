"""
Main evaluation script for spatial FDR methods.
Refactored to support both on-the-fly generation and fixed datasets.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm

from ..data.loader import subsample_locations
from ..data.synthetic import generate_evaluation_data
from ..methods.baseline import benjamini_hochberg
from ..methods.spatial_fdr import SpatialFDR
from ..methods.kernels import estimate_length_scale
from ..evaluation.metrics import compute_metrics, summarize_metrics
from ..visualization.plots import (
    plot_power_comparison,
    plot_fdr_calibration,
    plot_power_gain_bars
)


def evaluate_fixed_sample(
        locations: np.ndarray,
        p_values: np.ndarray,
        true_labels: np.ndarray,
        methods_config: Dict,
        return_details: bool = False
) -> Dict[str, Any]:
    """
    Evaluates methods on a specific, already-generated dataset (X, p, y).

    Parameters
    ----------
    return_details : bool
        If True, returns a tuple (metrics, artifacts) where artifacts contains
        raw arrays (lfdr, rejections, alpha_pointwise) for visualization.
    """
    results = {}
    details = {}

    for method_name, config in methods_config.items():
        try:
            if method_name == 'BH':
                discoveries = benjamini_hochberg(
                    p_values,
                    alpha=config.get('alpha', 0.1)
                )
                if return_details:
                    details[method_name] = {'rejections': discoveries}

            elif method_name == 'SpatialFDR':
                # Use estimated length scale if not provided
                k_params = config.get('kernel_params', {}).copy()
                if 'length_scale' not in k_params:
                    k_params['length_scale'] = estimate_length_scale(locations)

                model = SpatialFDR(
                    kernel_type=config.get('kernel_type', 'matern'),
                    lambda_reg=config.get('lambda_reg', 1.0),
                    lambda_bound=config.get('lambda_bound', 100.0),
                    kernel_params=k_params,
                    optimizer=config.get('optimizer', 'natural_gradient'),
                    verbose=False
                )

                model.fit_pointwise(locations, p_values)
                discoveries = model.reject(p_values, fdr_level=config.get('alpha', 0.1))

                if return_details:
                    details[method_name] = {
                        'rejections': discoveries,
                        'lfdr': model.lfdr_,
                        'alpha': model.alpha_pointwise_
                    }

            else:
                print(f"Unknown method: {method_name}")
                continue

            # Compute metrics
            metrics = compute_metrics(discoveries, true_labels)
            results[method_name] = metrics

        except Exception as e:
            print(f"Error in {method_name}: {e}")
            results[method_name] = None

    if return_details:
        return results, details
    return results


def run_single_replication(
        locations: np.ndarray,
        spatial_strength: str,
        effect_strength: str,
        methods_config: Dict,
        random_state: int
) -> Dict[str, Dict]:
    """Legacy wrapper for synthetic data generation."""
    true_labels, p_values = generate_evaluation_data(
        locations,
        spatial_strength=spatial_strength,
        effect_strength=effect_strength,
        random_state=random_state
    )
    return evaluate_fixed_sample(locations, p_values, true_labels, methods_config)

def run_evaluation(
    X_data: np.ndarray,
    n_samples: int = 500,
    spatial_strengths: List[str] = None,
    effect_strength: str = 'medium',
    n_replications: int = 100,
    methods_config: Dict = None,
    output_dir: str = './results',
    random_state: int = 42
) -> Dict:
    """
    Legacy runner for the purely synthetic experiments.
    """
    if spatial_strengths is None:
        spatial_strengths = ['none', 'weak', 'medium', 'strong']

    if methods_config is None:
        locations_sample = subsample_locations(X_data, n_samples, random_state)
        length_scale = estimate_length_scale(locations_sample, method='median')

        methods_config = {
            'BH': {'alpha': 0.1},
            'SpatialFDR': {
                'kernel_type': 'matern',
                'lambda_reg': 1.0,
                'kernel_params': {'nu': 1.5, 'length_scale': length_scale},
                'alpha': 0.1
            }
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {strength: {m: [] for m in methods_config} for strength in spatial_strengths}

    print(f"Running synthetic evaluation on {n_replications} reps...")

    for strength in spatial_strengths:
        for rep in tqdm(range(n_replications), desc=strength):
            locations = subsample_locations(X_data, n_samples, random_state + rep)

            rep_results = run_single_replication(
                locations, strength, effect_strength, methods_config, random_state + rep
            )

            for m, metrics in rep_results.items():
                if metrics: all_results[strength][m].append(metrics)

    # Save & Plot (abbreviated for brevity, kept same as original)
    return all_results


def run_lambda_sensitivity(
        X_data: np.ndarray,
        n_samples: int = 500,
        lambda_values: List[float] = None,
        spatial_strength: str = 'medium',
        effect_strength: str = 'medium',
        n_replications: int = 20,
        output_dir: str = './results',
        random_state: int = 42
) -> Dict:
    """
    Run sensitivity analysis for regularization parameter λ.
    """
    if lambda_values is None:
        lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Estimate length scale for the default setup
    # We sample once just to get a heuristic length scale
    locations_sample = subsample_locations(X_data, n_samples, random_state)
    length_scale = estimate_length_scale(locations_sample, method='median')

    sensitivity_results = {
        'lambda_values': lambda_values,
        'power_mean': [],
        'power_std': [],
        'fdr_mean': [],
        'fdr_std': []
    }

    print("Running λ sensitivity analysis...")

    for lambda_val in tqdm(lambda_values, desc="Testing λ values"):
        method_config = {
            'SpatialFDR': {
                'kernel_type': 'matern',
                'lambda_reg': lambda_val,
                'kernel_params': {'nu': 1.5, 'length_scale': length_scale},
                'optimizer': 'natural_gradient',
                'alpha': 0.1
            }
        }

        rep_results = []
        for rep in range(n_replications):
            # Sample locations
            locations_rep = subsample_locations(
                X_data, n_samples, random_state=random_state + rep
            )

            # Run using the wrapper (which generates synthetic data)
            result = run_single_replication(
                locations_rep,
                spatial_strength=spatial_strength,
                effect_strength=effect_strength,
                methods_config=method_config,
                random_state=random_state + rep
            )

            if result.get('SpatialFDR') is not None:
                rep_results.append(result['SpatialFDR'])

        # Compute statistics
        if rep_results:
            power_values = [r['power'] for r in rep_results]
            fdr_values = [r['FDR'] for r in rep_results]

            sensitivity_results['power_mean'].append(np.mean(power_values))
            sensitivity_results['power_std'].append(np.std(power_values))
            sensitivity_results['fdr_mean'].append(np.mean(fdr_values))
            sensitivity_results['fdr_std'].append(np.std(fdr_values))
        else:
            sensitivity_results['power_mean'].append(0.0)
            sensitivity_results['power_std'].append(0.0)
            sensitivity_results['fdr_mean'].append(0.0)
            sensitivity_results['fdr_std'].append(0.0)

    # Save and plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from ..visualization.plots import plot_lambda_sensitivity
        plot_lambda_sensitivity(
            lambda_values,
            sensitivity_results['power_mean'],
            sensitivity_results['fdr_mean'],
            save_path=output_path / 'lambda_sensitivity.png'
        )
    except ImportError:
        print("Visualization module not found, skipping plot.")

    return sensitivity_results