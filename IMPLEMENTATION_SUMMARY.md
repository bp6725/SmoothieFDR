# Spatial FDR Implementation - Complete Package

## Package Overview

This is a production-ready implementation of the Spatial FDR evaluation framework for your paper "Spatially Smooth Bayesian FDR via Reproducing Kernels."

## What's Included

### Core Implementation Files

1. **spatial_fdr_evaluation/** - Main package
   - `__init__.py` - Package initialization
   
2. **spatial_fdr_evaluation/data/** - Data handling
   - `loader.py` - ADbench data loading utilities
   - `synthetic.py` - Synthetic data generation with spatial clustering
   - Functions for generating clustered labels and p-values
   
3. **spatial_fdr_evaluation/methods/** - FDR methods
   - `baseline.py` - Benjamini-Hochberg implementation
   - `spatial_fdr.py` - **Main contribution**: SpatialFDR class with kernel logistic regression
   - `kernels.py` - Matérn and RBF kernel implementations
   - Natural gradient optimization
   - RKHS regularization
   
4. **spatial_fdr_evaluation/evaluation/** - Metrics
   - `metrics.py` - Comprehensive evaluation metrics
   - TPR, FDR, Power, Precision, F1 scores
   - Confusion matrix computations
   
5. **spatial_fdr_evaluation/visualization/** - Plotting
   - `plots.py` - All visualization functions
   - Power comparison plots
   - FDR calibration checks
   - Spatial alpha maps
   - Lambda sensitivity analysis
   
6. **spatial_fdr_evaluation/experiments/** - Experiment runners
   - `run_evaluation.py` - Main evaluation pipeline
   - Complete Monte Carlo evaluation framework
   - Lambda sensitivity analysis

### User Scripts

7. **main.py** - Main evaluation script
   - Integrates with your ADbench data loading
   - Command-line interface
   - Runs complete evaluation protocol
   
8. **example.py** - Standalone example
   - Minimal working example
   - No external data needed
   - Demonstrates full workflow
   
### Documentation

9. **README.md** - Comprehensive documentation
   - Framework overview
   - Installation instructions
   - Usage examples
   - API documentation
   - Expected results
   
10. **QUICKSTART.md** - Quick start guide
    - Immediate getting started
    - Integration instructions
    - Common use cases
    - Troubleshooting
    
11. **requirements.txt** - Python dependencies
    - numpy, scipy, scikit-learn
    - matplotlib, seaborn
    - pandas, tqdm

## Key Features Implemented

### 1. Spatial FDR Method (Your Novel Contribution)

**File:** `spatial_fdr_evaluation/methods/spatial_fdr.py`

**Implementation:**
- Kernel logistic regression for α(loc) estimation
- Two-step procedure:
  1. Estimate α* at observed locations
  2. Kernel logistic regression with RKHS regularization
- Natural gradient descent optimization
- Sigmoid squashing to ensure α ∈ [0,1]
- Prediction at arbitrary spatial locations

**Key equation implemented:**
```
L(c) = -Σ[α* log(σ(Kc)) + (1-α*) log(1-σ(Kc))] + λ c^T K c
```

### 2. Flexible Kernel Framework

**File:** `spatial_fdr_evaluation/methods/kernels.py`

**Implemented kernels:**
- Matérn kernel with explicit smoothness control (ν parameter)
- RBF (Gaussian) kernel
- Automatic length scale estimation
- Numerical stability guarantees

**Smoothness control:**
- ν = 0.5: C^0 (continuous)
- ν = 1.5: C^1 (once differentiable) - recommended
- ν = 2.5: C^2 (twice differentiable)

### 3. Evaluation Protocol

**File:** `spatial_fdr_evaluation/experiments/run_evaluation.py`

**Features:**
- Monte Carlo evaluation (default: 100 replications)
- Multiple spatial clustering conditions
- Automatic metric computation
- Statistical summaries
- Result persistence (JSON)

**Spatial conditions:**
- None (random) - negative control
- Weak clustering
- Medium clustering
- Strong clustering

### 4. Realistic Data Generation

**File:** `spatial_fdr_evaluation/data/synthetic.py`

**Capabilities:**
- K-means spatial clustering
- Cluster-based label assignment
- Realistic p-value generation
- Beta distributions for H1 signals
- Controlled effect sizes

### 5. Comprehensive Metrics

**File:** `spatial_fdr_evaluation/evaluation/metrics.py`

**Computed metrics:**
- Power (TPR / Sensitivity)
- False Discovery Rate (FDR)
- False Positive Rate (FPR)
- Precision (PPV)
- F1 score
- Confusion matrix
- Power gain over baseline

### 6. Publication-Quality Visualizations

**File:** `spatial_fdr_evaluation/visualization/plots.py`

**Plot types:**
- Power comparison across conditions (boxplots)
- FDR calibration verification (scatter with error bars)
- Power gain bar charts
- Lambda sensitivity analysis
- Spatial alpha maps (2D heatmaps)

## How to Use

### Quick Test (No Data Required)

```bash
pip install -r requirements.txt
python example.py
```

### Integrate Your Data

1. Edit `main.py` - replace `load_from_ADbench()` function
2. Run: `python main.py --datasets your_data --n_reps 100`
3. Results saved to `./results/your_data/`

### Programmatic Usage

```python
from spatial_fdr_evaluation.methods import SpatialFDR
from spatial_fdr_evaluation.data import generate_evaluation_data

# Your spatial data
locations = ...  # (n_samples, n_features)

# Generate test scenario
true_labels, p_values = generate_evaluation_data(
    locations, spatial_strength='medium'
)

# Fit model
model = SpatialFDR(lambda_reg=0.1)
model.fit(locations, p_values)

# Make predictions
alpha_estimates = model.predict_alpha()
discoveries = model.reject(p_values, alpha=0.1)
```

## Implementation Details

### Natural Gradient Optimization

**Why:** Eliminates kernel matrix conditioning issues

**Implementation:**
```python
# Standard gradient: ∇L = K * (σ - α*) + 2λKc
# Natural gradient: ∇̃L = (σ - α*) + 2λc
```

### RKHS Regularization

**Form:** `λ ||α||²_H = λ c^T K c`

**Purpose:**
- Enforces spatial smoothness
- Controls overfitting
- Ensures unique solution

### Representer Theorem

**Applied:** Solution guaranteed to have form:
```python
α(x) = Σ c_j K(x, x_j)
```

Reduces infinite-dimensional problem to N coefficients.

## Expected Results

Based on your framework's design:

### Condition: No Spatial Structure
- Spatial FDR ≈ BH (no spurious gains)
- Both control FDR at 0.1

### Condition: Medium Spatial Clustering
- Spatial FDR: **30-50% power gain** over BH
- FDR maintained at ≤0.1
- More discoveries with same error rate

### Condition: Strong Spatial Clustering
- Spatial FDR: **50-100% power gain** over BH
- Demonstrates value of spatial regularization

## File Organization

```
spatial_fdr_evaluation/
├── data/               # Data I/O
├── methods/            # Core algorithms
├── evaluation/         # Metrics & analysis
├── visualization/      # Plotting
├── experiments/        # Runners
└── utils/              # Helpers

Scripts:
├── main.py            # ADbench integration
└── example.py         # Standalone demo

Documentation:
├── README.md          # Full docs
├── QUICKSTART.md      # Quick guide
└── requirements.txt   # Dependencies
```

## Testing the Implementation

### Test 1: Negative Control (No Spatial Structure)
```bash
python example.py
# Expected: Spatial FDR ≈ BH in power
```

### Test 2: Positive Control (Strong Clustering)
```python
# In example.py, change:
spatial_strength='strong'
# Expected: Large power gain
```

### Test 3: Lambda Sensitivity
```bash
python main.py --datasets your_data --lambda_sensitivity
# Expected: Optimal λ balances power and FDR
```

## Code Quality Features

- ✓ Type hints throughout
- ✓ Comprehensive docstrings
- ✓ Modular design
- ✓ Error handling
- ✓ Numerical stability (regularization, clipping)
- ✓ Progress bars for long runs
- ✓ JSON result persistence
- ✓ Reproducible (random seeds)

## Next Steps for Your Paper

1. **Integrate your real data:** Replace `load_from_ADbench()`
2. **Run full evaluation:** `python main.py --datasets all_your_datasets --n_reps 100`
3. **Generate paper figures:** All plots are publication-quality (300 DPI)
4. **Analyze results:** Use `evaluation_results.json` for tables
5. **Add to supplementary:** Include code as supplementary material

## Extending the Framework

### Add New Kernels
Edit `methods/kernels.py`:
```python
def your_custom_kernel(X, Y, **params):
    # Implementation
    return K
```

### Add New Methods
Create new file in `methods/`:
```python
class YourMethod:
    def fit(self, locations, p_values): ...
    def reject(self, p_values, alpha): ...
```

### Add New Metrics
Edit `evaluation/metrics.py`:
```python
def compute_your_metric(discoveries, true_labels):
    # Implementation
    return metric_value
```

## Support & Citation

**Questions?** See README.md and QUICKSTART.md

**Citation:**
```bibtex
@article{perets2024spatial,
  title={Spatially Smooth Bayesian FDR via Reproducing Kernels},
  author={Perets, Binyamin and Mannor, Shie},
  year={2024}
}
```

## Summary

This is a **complete, production-ready implementation** of your spatial FDR framework. It includes:

✓ Novel spatial FDR method with RKHS regularization  
✓ Kernel logistic regression implementation  
✓ Natural gradient optimization  
✓ Comprehensive evaluation protocol  
✓ Publication-quality visualizations  
✓ Full documentation  
✓ Ready to integrate with your data  

**Everything you need for your paper's evaluation section is here and ready to run.**
