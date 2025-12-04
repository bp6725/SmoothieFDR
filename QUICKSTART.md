# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import spatial_fdr_evaluation; print('✓ Installation successful')"
```

## Running the Example

The simplest way to see the framework in action:

```bash
python example.py
```

This will:
- Generate synthetic 2D spatial data
- Create spatially clustered null/alternative labels
- Run both BH and Spatial FDR methods
- Compare their power and FDR control
- Create visualizations in `./example_results/`

**Expected output:**
```
Spatial FDR Evaluation - Simple Example
======================================================================

1. Generating synthetic spatial data...
   Generated 500 locations in 2D space

2. Generating evaluation data with medium spatial clustering...
   Generated p-values for 500 tests
   True signals (H1): 215 (43.0%)
   True nulls (H0): 285 (57.0%)

3. Running Benjamini-Hochberg (BH) baseline...
   BH Results:
     Power (TPR):  0.623
     FDR:          0.089
     Precision:    0.911
     Discoveries:  142

4. Running Spatial FDR with RKHS regularization...
   Converged in 247 iterations
   
   Spatial FDR Results:
     Power (TPR):  0.744
     FDR:          0.093
     Precision:    0.907
     Discoveries:  176

5. Comparison:
   Power gain over BH: +19.4%
   FDR control: BH=0.089, Spatial=0.093

✓ Spatial FDR achieved 19.4% power gain while maintaining FDR control
```

## Integrating Your Data

### Step 1: Replace data loading function

In `main.py`, replace the `load_from_ADbench` function:

```python
def load_from_ADbench(dataset_name: str):
    """Your custom data loading logic."""
    # Example: Load from your data source
    data = your_loading_function(dataset_name)
    
    return {
        'X_train': data['features'],
        'X_test': data['test_features'],
        'y_train': data['labels'],
        'y_test': data['test_labels']
    }
```

### Step 2: Run evaluation

```bash
# Single dataset
python main.py --datasets your_dataset_name --n_samples 500 --n_reps 100

# Multiple datasets
python main.py --datasets dataset1 dataset2 dataset3 --n_reps 50

# With lambda sensitivity
python main.py --datasets your_dataset --lambda_sensitivity --n_reps 20
```

### Step 3: Check results

Results will be saved to `./results/your_dataset_name/`:
- `evaluation_results.json` - Raw numerical results
- `power_comparison.png` - Power across spatial conditions
- `fdr_calibration.png` - FDR control verification
- `power_gain_bars.png` - Relative power improvements

## Using the Framework Programmatically

```python
import numpy as np
from spatial_fdr_evaluation.data import generate_evaluation_data
from spatial_fdr_evaluation.methods import benjamini_hochberg, SpatialFDR
from spatial_fdr_evaluation.evaluation import compute_metrics

# 1. Prepare your spatial data
locations = your_spatial_locations  # shape: (n_samples, n_features)

# 2. Generate evaluation scenario
true_labels, p_values = generate_evaluation_data(
    locations,
    spatial_strength='medium',
    effect_strength='medium'
)

# 3. Run methods
# Baseline
discoveries_bh = benjamini_hochberg(p_values, alpha=0.1)

# Spatial FDR
model = SpatialFDR(
    kernel_type='matern',
    lambda_reg=0.1,
    kernel_params={'nu': 1.5, 'length_scale': 1.0}
)
model.fit(locations, p_values)
discoveries_spatial = model.reject(p_values, alpha=0.1)

# 4. Evaluate
metrics_bh = compute_metrics(discoveries_bh, true_labels)
metrics_spatial = compute_metrics(discoveries_spatial, true_labels)

print(f"BH Power: {metrics_bh['power']:.3f}")
print(f"Spatial Power: {metrics_spatial['power']:.3f}")
print(f"Power gain: {(metrics_spatial['power']/metrics_bh['power']-1)*100:.1f}%")
```

## Key Parameters

### Spatial Strength (how much signals cluster together)
- `'none'` - Random assignment (negative control)
- `'weak'` - Modest spatial clustering
- `'medium'` - Moderate spatial clustering *(recommended for testing)*
- `'strong'` - High spatial clustering

### Effect Strength (how strong the signals are)
- `'weak'` - p-values ~ Beta(0.2, 1)
- `'medium'` - p-values ~ Beta(0.05, 1) *(recommended)*
- `'strong'` - p-values ~ Beta(0.01, 1)

### Kernel Parameters
```python
# Matérn kernel smoothness (nu)
# Higher nu = smoother functions
nu = 0.5  # C^0 (continuous)
nu = 1.5  # C^1 (once differentiable) - recommended
nu = 2.5  # C^2 (twice differentiable)

# Length scale (ℓ)
# Larger = smoother over larger distances
length_scale = 1.0  # Auto-estimate with estimate_length_scale()
```

### Regularization (lambda)
```python
# Larger lambda = more smoothing
lambda_reg = 0.01   # Light regularization
lambda_reg = 0.1    # Moderate regularization (recommended)
lambda_reg = 1.0    # Strong regularization
```

## Troubleshooting

### Import errors
```bash
# Make sure you're in the correct directory
cd /path/to/spatial_fdr_evaluation
export PYTHONPATH="${PYTHONPATH}:."
python example.py
```

### Memory issues with large datasets
```python
# Reduce n_samples
python main.py --datasets your_data --n_samples 300 --n_reps 50
```

### Slow optimization
```python
# Use L-BFGS-B instead of natural gradient
model = SpatialFDR(optimizer='lbfgs')

# Or reduce max_iter
model = SpatialFDR(max_iter=500)
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore the code** - all modules are well-documented
3. **Customize kernels** - add your own kernel functions
4. **Extend the framework** - add new FDR methods for comparison

## File Structure

```
.
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── requirements.txt            # Python dependencies
├── main.py                     # Main evaluation script (ADbench integration)
├── example.py                  # Simple standalone example
│
└── spatial_fdr_evaluation/     # Main package
    ├── data/                   # Data loading & generation
    │   ├── loader.py           # ADbench data loading
    │   └── synthetic.py        # Synthetic data generation
    │
    ├── methods/                # FDR methods
    │   ├── baseline.py         # Benjamini-Hochberg
    │   ├── spatial_fdr.py      # Spatial RKHS-based FDR
    │   └── kernels.py          # Kernel functions
    │
    ├── evaluation/             # Metrics
    │   └── metrics.py          # Power, FDR, TPR, etc.
    │
    ├── visualization/          # Plotting
    │   └── plots.py            # All visualization functions
    │
    └── experiments/            # Experiment runners
        └── run_evaluation.py   # Main evaluation logic
```

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review the example.py script for usage patterns
3. Examine the docstrings in each module
4. Open an issue on GitHub (if applicable)

## Quick Reference

**Run example:**
```bash
python example.py
```

**Run on your data:**
```bash
python main.py --datasets your_data --n_samples 500 --n_reps 100
```

**Key imports:**
```python
from spatial_fdr_evaluation.methods import SpatialFDR, benjamini_hochberg
from spatial_fdr_evaluation.evaluation import compute_metrics
from spatial_fdr_evaluation.data import generate_evaluation_data
```

**Typical workflow:**
1. Load spatial data
2. Generate evaluation scenario
3. Run BH and Spatial FDR
4. Compare metrics
5. Visualize results
