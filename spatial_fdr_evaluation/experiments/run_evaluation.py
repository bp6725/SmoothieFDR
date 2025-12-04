"""
Main evaluation script for spatial FDR methods.

Runs complete evaluation across multiple conditions and replications.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm

from ..data.loader import  subsample_locations
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


def run_single_replication(
    locations: np.ndarray,
    spatial_strength: str,
    effect_strength: str,
    methods_config: Dict,
    random_state: int
) -> Dict[str, Dict]:
    """
    Run a single replication of the evaluation.
    
    Parameters
    ----------
    locations : np.ndarray
        Spatial coordinates
    spatial_strength : str
        Spatial clustering strength
    effect_strength : str
        Signal effect strength
    methods_config : dict
        Configuration for methods to evaluate
    random_state : int
        Random seed for this replication
        
    Returns
    -------
    results : dict
        Dictionary mapping method names to their metrics
    """
    # Generate evaluation data
    true_labels, p_values = generate_evaluation_data(
        locations,
        spatial_strength=spatial_strength,
        effect_strength=effect_strength,
        random_state=random_state
    )
    
    results = {}
    
    # Run each method
    for method_name, method_config in methods_config.items():
        # try:
        if True :
            if method_name == 'BH':
                # Benjamini-Hochberg baseline
                discoveries = benjamini_hochberg(
                    p_values,
                    alpha=method_config.get('alpha', 0.1)
                )
                
            elif method_name == 'SpatialFDR':
                # Spatial FDR via RKHS
                model = SpatialFDR(
                    kernel_type=method_config.get('kernel_type', 'matern'),
                    lambda_reg=method_config.get('lambda_reg', 10),
                    lambda_bound=100.0,  # ADD THIS LINE
                    kernel_params=method_config.get('kernel_params', {}),
                    optimizer=method_config.get('optimizer', 'natural_gradient'),
                    verbose=False
                )

                model.fit_pointwise(locations, p_values)  # CHANGED: fit → fit_pointwise
                discoveries = model.reject(p_values,
                                           fdr_level=method_config.get('alpha', 0.1))  # CHANGED: alpha → fdr_level

                print(f"α range: [{model.alpha_pointwise_.min():.3f}, {model.alpha_pointwise_.max():.3f}]")
                print(f"α mean: {model.alpha_pointwise_.mean():.3f}")
                print(f"α values near 0: {np.sum(model.alpha_pointwise_ < 0.1)}")

            else:
                print(f"Unknown method: {method_name}")
                continue
            
            # Compute metrics
            metrics = compute_metrics(discoveries, true_labels)
            results[method_name] = metrics
            
        # except Exception as e:
        #     print(f"Error in {method_name}: {e}")
        #     results[method_name] = None
        #
    return results


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
    Run complete evaluation across all conditions.
    
    Parameters
    ----------
    X_data : np.ndarray
        Real data for extracting spatial structure
    n_samples : int, default=500
        Number of locations to sample
    spatial_strengths : list, optional
        List of spatial clustering strengths to test
    effect_strength : str, default='medium'
        Signal effect strength
    n_replications : int, default=100
        Number of Monte Carlo replications
    methods_config : dict, optional
        Configuration for methods to evaluate
    output_dir : str, default='./results'
        Directory to save results
    random_state : int, default=42
        Base random seed
        
    Returns
    -------
    all_results : dict
        Nested dictionary with all results
    """
    # Default configurations
    if spatial_strengths is None:
        spatial_strengths = ['none', 'weak', 'medium', 'strong']
    
    if methods_config is None:
        # Estimate length scale from data
        # _, kde = extract_spatial_structure(X_data)
        locations_sample = subsample_locations(X_data, n_samples, random_state)
        length_scale = estimate_length_scale(locations_sample, method='median')
        
        methods_config = {
            'BH': {'alpha': 0.1},
            'SpatialFDR': {
                'kernel_type': 'matern',
                'lambda_reg': 0.1,
                'kernel_params': {'nu': 1.5, 'length_scale': length_scale},
                'optimizer': 'natural_gradient',
                'alpha': 0.1
            }
        }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract spatial structure once
    # _, kde = extract_spatial_structure(X_data, random_state=random_state)
    
    # Storage for results
    all_results = {
        strength: {method: [] for method in methods_config.keys()}
        for strength in spatial_strengths
    }
    
    # Run evaluation
    print("Running spatial FDR evaluation...")
    print(f"Spatial strengths: {spatial_strengths}")
    print(f"Methods: {list(methods_config.keys())}")
    print(f"Replications: {n_replications}")
    print(f"Samples per replication: {n_samples}")
    print("-" * 60)
    
    for strength in spatial_strengths:
        print(f"\nEvaluating spatial strength: {strength}")
        
        for rep in tqdm(range(n_replications), desc=f"  {strength}"):
            # Sample locations for this replication
            locations = subsample_locations(
                X_data,
                n_samples,
                random_state=random_state + rep
            )
            
            # Run single replication
            rep_results = run_single_replication(
                locations,
                spatial_strength=strength,
                effect_strength=effect_strength,
                methods_config=methods_config,
                random_state=random_state + rep
            )
            
            # Store results
            for method_name, metrics in rep_results.items():
                if metrics is not None:
                    all_results[strength][method_name].append(metrics)
    
    # Summarize results
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for strength in spatial_strengths:
        print(f"\nSpatial Strength: {strength.upper()}")
        print("-" * 40)
        
        for method_name in methods_config.keys():
            if all_results[strength][method_name]:
                summary = summarize_metrics(all_results[strength][method_name])
                
                print(f"\n  {method_name}:")
                print(f"    Power: {summary['power']['mean']:.3f} ± {summary['power']['std']:.3f}")
                print(f"    FDR:   {summary['FDR']['mean']:.3f} ± {summary['FDR']['std']:.3f}")
                print(f"    Discoveries: {summary['n_discoveries']['mean']:.1f}")
    
    # Save results
    results_file = output_path / 'evaluation_results.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for strength in all_results:
        json_results[strength] = {}
        for method in all_results[strength]:
            json_results[strength][method] = all_results[strength][method]
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_power_comparison(
        all_results,
        save_path=output_path / 'power_comparison.png'
    )
    
    plot_fdr_calibration(
        all_results,
        save_path=output_path / 'fdr_calibration.png'
    )
    
    plot_power_gain_bars(
        all_results,
        save_path=output_path / 'power_gain_bars.png'
    )
    
    print(f"Plots saved to: {output_path}")
    
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
    
    Parameters
    ----------
    X_data : np.ndarray
        Real data for extracting spatial structure
    n_samples : int, default=500
        Number of locations to sample
    lambda_values : list, optional
        List of λ values to test
    spatial_strength : str, default='medium'
        Spatial clustering strength
    effect_strength : str, default='medium'
        Signal effect strength
    n_replications : int, default=20
        Number of replications per λ
    output_dir : str, default='./results'
        Directory to save results
    random_state : int, default=42
        Base random seed
        
    Returns
    -------
    sensitivity_results : dict
        Results for each λ value
    """
    if lambda_values is None:
        lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Extract spatial structure
    _, kde = extract_spatial_structure(X_data, random_state=random_state)
    locations = subsample_locations(X_data, n_samples, random_state)
    length_scale = estimate_length_scale(locations, method='median')
    
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
            locations_rep = subsample_locations(
                X_data, n_samples, random_state=random_state + rep
            )
            
            result = run_single_replication(
                locations_rep,
                spatial_strength=spatial_strength,
                effect_strength=effect_strength,
                methods_config=method_config,
                random_state=random_state + rep
            )
            
            if result['SpatialFDR'] is not None:
                rep_results.append(result['SpatialFDR'])
        
        # Compute statistics
        power_values = [r['power'] for r in rep_results]
        fdr_values = [r['FDR'] for r in rep_results]
        
        sensitivity_results['power_mean'].append(np.mean(power_values))
        sensitivity_results['power_std'].append(np.std(power_values))
        sensitivity_results['fdr_mean'].append(np.mean(fdr_values))
        sensitivity_results['fdr_std'].append(np.std(fdr_values))
    
    # Save and plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    from ..visualization.plots import plot_lambda_sensitivity
    plot_lambda_sensitivity(
        lambda_values,
        sensitivity_results['power_mean'],
        sensitivity_results['fdr_mean'],
        save_path=output_path / 'lambda_sensitivity.png'
    )
    
    return sensitivity_results
