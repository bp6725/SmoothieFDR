# Spatial FDR Evaluation Framework

A comprehensive framework for evaluating False Discovery Rate (FDR) control methods that leverage spatial structure through Reproducing Kernel Hilbert Spaces (RKHS).

## Overview

This framework implements and evaluates spatial FDR control methods, comparing them against standard baselines like Benjamini-Hochberg. The key innovation is using kernel-based regularization to borrow strength from spatially nearby hypotheses, improving power while maintaining FDR control.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from spatial_fdr_evaluation.experiments import run_evaluation
import numpy as np

# Load your spatial data
X_data = np.random.randn(1000, 50)  # Replace with your data

# Run evaluation
results = run_evaluation(
    X_data=X_data,
    n_samples=500,
    n_replications=100,
    output_dir='./results'
)
```

### Command Line Usage

```bash
# Run on single dataset
python main.py --datasets 25_musk --n_samples 500 --n_reps 100

# Run on multiple datasets
python main.py --datasets 25_musk 11_donors 24_mnist --n_reps 50

# With lambda sensitivity analysis
python main.py --datasets 25_musk --lambda_sensitivity
```

## Framework Structure

```
spatial_fdr_evaluation/
├── data/               # Data loading and synthetic generation
│   ├── loader.py       # ADbench data loading
│   └── synthetic.py    # Synthetic evaluation data generation
│
├── methods/            # FDR control methods
│   ├── baseline.py     # Benjamini-Hochberg
│   ├── spatial_fdr.py  # Spatial RKHS-based FDR
│   └── kernels.py      # Kernel functions (Matérn, RBF)
│
├── evaluation/         # Metrics and analysis
│   └── metrics.py      # Power, FDR, TPR calculations
│
├── visualization/      # Plotting functions
│   └── plots.py        # All visualization utilities
│
└── experiments/        # Experiment runners
    └── run_evaluation.py  # Main evaluation script
```

## Key Features

### 1. Spatial FDR Control via RKHS

The `SpatialFDR` class implements kernel logistic regression for estimating spatially-varying prior null probabilities α(loc):

```python
from spatial_fdr_evaluation.methods import SpatialFDR

# Initialize model
model = SpatialFDR(
    kernel_type='matern',
    lambda_reg=0.1,
    kernel_params={'nu': 1.5, 'length_scale': 1.0}
)

# Fit to data
model.fit(locations, p_values)

# Make predictions
alpha_estimates = model.predict_alpha(new_locations)
discoveries = model.reject(p_values, alpha=0.1)
```

### 2. Flexible Kernel Choice

Support for multiple kernel types with explicit smoothness control:

```python
# Matérn kernel with different smoothness levels
# nu = 0.5: C^0 (continuous)
# nu = 1.5: C^1 (once differentiable)
# nu = 2.5: C^2 (twice differentiable)

from spatial_fdr_evaluation.methods import compute_kernel_matrix

K = compute_kernel_matrix(
    locations,
    kernel_type='matern',
    nu=1.5,  # Smoothness parameter
    length_scale=1.0
)
```

### 3. Synthetic Data Generation

Create realistic evaluation scenarios with spatial clustering:

```python
from spatial_fdr_evaluation.data import generate_evaluation_data

true_labels, p_values = generate_evaluation_data(
    locations,
    spatial_strength='medium',  # 'none', 'weak', 'medium', 'strong'
    effect_strength='medium',   # 'weak', 'medium', 'strong'
)
```

### 4. Comprehensive Evaluation

Automatic evaluation across multiple conditions:

- **Spatial strengths**: None (random), Weak, Medium, Strong
- **Effect strengths**: Weak, Medium, Strong signals
- **Metrics**: Power, FDR, TPR, Precision, F1
- **Multiple replications**: Monte Carlo evaluation

### 5. Visualization

Built-in plotting functions:

```python
from spatial_fdr_evaluation.visualization import (
    plot_power_comparison,
    plot_fdr_calibration,
    plot_lambda_sensitivity,
    plot_spatial_alpha_map
)

# Power comparison across conditions
plot_power_comparison(results, save_path='power_comparison.png')

# FDR calibration check
plot_fdr_calibration(results, save_path='fdr_calibration.png')

# Spatial map of estimated α(loc)
plot_spatial_alpha_map(locations, alpha_values, true_labels)
```

## Evaluation Protocol

The framework implements a rigorous evaluation protocol:

1. **Extract spatial structure** from real data (via KDE)
2. **Generate synthetic labels** with controlled spatial clustering
3. **Generate p-values** with realistic signal strength
4. **Run multiple methods** (BH baseline + Spatial FDR)
5. **Compute metrics** (Power, FDR, etc.)
6. **Repeat** across many replications
7. **Visualize** and summarize results

## Example Workflow

```python
# 1. Load real spatial data
from spatial_fdr_evaluation.data import extract_spatial_structure

X_data = load_your_data()  # Your real data
locations, kde = extract_spatial_structure(X_data)

# 2. Generate evaluation scenario
from spatial_fdr_evaluation.data import generate_evaluation_data

true_labels, p_values = generate_evaluation_data(
    locations,
    spatial_strength='medium',
    effect_strength='medium'
)

# 3. Run methods
from spatial_fdr_evaluation.methods import benjamini_hochberg, SpatialFDR

# Baseline
discoveries_bh = benjamini_hochberg(p_values, alpha=0.1)

# Spatial FDR
model = SpatialFDR(lambda_reg=0.1)
model.fit(locations, p_values)
discoveries_spatial = model.reject(p_values, alpha=0.1)

# 4. Evaluate
from spatial_fdr_evaluation.evaluation import compute_metrics

metrics_bh = compute_metrics(discoveries_bh, true_labels)
metrics_spatial = compute_metrics(discoveries_spatial, true_labels)

print(f"BH Power: {metrics_bh['power']:.3f}, FDR: {metrics_bh['FDR']:.3f}")
print(f"Spatial Power: {metrics_spatial['power']:.3f}, FDR: {metrics_spatial['FDR']:.3f}")
```

## Configuration

### Methods Configuration

```python
methods_config = {
    'BH': {
        'alpha': 0.1
    },
    'SpatialFDR': {
        'kernel_type': 'matern',
        'lambda_reg': 0.1,
        'kernel_params': {'nu': 1.5, 'length_scale': 1.0},
        'optimizer': 'natural_gradient',
        'alpha': 0.1
    }
}
```

### Evaluation Parameters

```python
evaluation_config = {
    'n_samples': 500,
    'spatial_strengths': ['none', 'weak', 'medium', 'strong'],
    'effect_strength': 'medium',
    'n_replications': 100,
    'random_state': 42
}
```

## Output

Results are saved to the specified output directory:

```
results/
├── dataset_name/
│   ├── evaluation_results.json      # Raw results
│   ├── power_comparison.png         # Power across conditions
│   ├── fdr_calibration.png          # FDR control verification
│   ├── power_gain_bars.png          # Relative power gains
│   └── lambda_sensitivity.png       # Regularization analysis
```

## Expected Results

### No Spatial Structure (Random Assignment)
- Spatial FDR ≈ BH in power
- Both control FDR at nominal level
- No spurious power gain

### Weak Spatial Clustering
- Spatial FDR: 10-20% power gain
- FDR controlled

### Medium Spatial Clustering
- Spatial FDR: 30-50% power gain
- FDR controlled

### Strong Spatial Clustering
- Spatial FDR: 50-100% power gain
- FDR controlled

## Advanced Usage

### Custom Kernels

```python
from spatial_fdr_evaluation.methods.kernels import matern_kernel

# Define custom kernel
def my_kernel(X, Y, **params):
    # Your custom kernel implementation
    pass

# Use in SpatialFDR
# (requires modifying spatial_fdr.py to accept custom kernels)
```

### Hyperparameter Tuning

```python
from spatial_fdr_evaluation.experiments import run_lambda_sensitivity

sensitivity_results = run_lambda_sensitivity(
    X_data,
    lambda_values=[0.001, 0.01, 0.1, 1.0, 10.0],
    n_replications=20
)
```

## Integration with Your Data

Replace the `load_from_ADbench` function in `main.py` with your data loading:

```python
def load_from_ADbench(dataset_name: str):
    # Your custom data loading
    X_train, X_test, y_train, y_test = load_my_data(dataset_name)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
```

## Citation

If you use this framework in your research, please cite:

```
@article{perets2024spatial,
  title={Spatially Smooth Bayesian FDR via Reproducing Kernels},
  author={Perets, Binyamin and Mannor, Shie},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
