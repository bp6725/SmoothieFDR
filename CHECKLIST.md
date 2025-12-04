# Installation & Usage Checklist

## ☐ Step 1: Installation

```bash
# Navigate to the downloaded package
cd spatial_fdr_evaluation/

# Install dependencies
pip install -r requirements.txt

# Optional: Install as editable package
pip install -e .
```

**Verify installation:**
```bash
python -c "import spatial_fdr_evaluation; print('✓ Package installed successfully')"
```

---

## ☐ Step 2: Run Basic Example

Test that everything works:

```bash
python example.py
```

**Expected:** 
- Script completes without errors
- Shows power gain (typically 15-30%)
- Creates `./example_results/` with plots

**If it works:** ✓ Core functionality is operational

---

## ☐ Step 3: Integrate Your Data

**Option A: Modify main.py (Recommended)**

1. Open `main.py`
2. Find the `load_from_ADbench()` function (around line 20)
3. Replace with your data loading:

```python
def load_from_ADbench(dataset_name: str):
    """Your custom data loading."""
    # YOUR CODE HERE
    X_train, X_test, y_train, y_test = your_loading_function(dataset_name)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
```

4. Run:
```bash
python main.py --datasets your_dataset_name --n_samples 500 --n_reps 100
```

**Option B: Use Programmatically**

Create your own script:

```python
import numpy as np
from spatial_fdr_evaluation.experiments import run_evaluation

# Load your data
X_data = your_data_loading_function()

# Run evaluation
results = run_evaluation(
    X_data=X_data,
    n_samples=500,
    n_replications=100,
    output_dir='./my_results'
)
```

---

## ☐ Step 4: Check Results

After running evaluation, check the output directory:

```
./results/your_dataset_name/
├── evaluation_results.json      # ← Raw numbers for tables
├── power_comparison.png         # ← Figure: Power across conditions
├── fdr_calibration.png          # ← Figure: FDR control verification
└── power_gain_bars.png          # ← Figure: Relative improvements
```

**Key things to verify:**
- [ ] FDR is controlled (empirical ≤ 0.1 for nominal 0.1)
- [ ] Power gain over BH in spatially clustered conditions
- [ ] No spurious gain in 'none' (random) condition

---

## ☐ Step 5: Generate Paper Figures

All plots are already publication-quality (300 DPI, proper fonts).

**To customize:**

```python
from spatial_fdr_evaluation.visualization import (
    plot_power_comparison,
    plot_fdr_calibration,
    plot_power_gain_bars
)

# Load your results
with open('results/your_data/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Customize and save
plot_power_comparison(
    results,
    save_path='paper_figures/fig_power.png',
    figsize=(16, 4)  # Adjust size
)
```

---

## ☐ Step 6: Run Full Evaluation (For Paper)

For your paper, run comprehensive evaluation:

```bash
# All datasets, many replications
python main.py \
    --datasets dataset1 dataset2 dataset3 \
    --n_samples 500 \
    --n_reps 100 \
    --spatial_strengths none weak medium strong \
    --output_dir ./paper_results

# With lambda sensitivity
python main.py \
    --datasets your_main_dataset \
    --lambda_sensitivity \
    --n_reps 50
```

**This will take time!** For 3 datasets × 4 conditions × 100 reps:
- Estimated time: 30-120 minutes (depends on n_samples)
- Use `--n_reps 20` for quick testing

---

## ☐ Step 7: Create Results Table

From `evaluation_results.json`, extract numbers for paper table:

```python
import json
import pandas as pd

with open('results/dataset/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Create summary table
table_data = []
for condition in ['none', 'weak', 'medium', 'strong']:
    for method in ['BH', 'SpatialFDR']:
        metrics = results[condition][method]
        # Compute mean across replications
        power_mean = np.mean([m['power'] for m in metrics])
        fdr_mean = np.mean([m['FDR'] for m in metrics])
        
        table_data.append({
            'Condition': condition,
            'Method': method,
            'Power': f"{power_mean:.3f}",
            'FDR': f"{fdr_mean:.3f}"
        })

df = pd.DataFrame(table_data)
print(df.to_latex(index=False))  # LaTeX table for paper
```

---

## ☐ Common Issues & Solutions

### ImportError: No module named 'spatial_fdr_evaluation'

**Solution:**
```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 2: Install as package
pip install -e .
```

### Memory Error (Large Datasets)

**Solution:** Reduce n_samples
```bash
python main.py --datasets large_dataset --n_samples 300
```

### Slow Optimization

**Solution:** Use L-BFGS-B optimizer
```python
model = SpatialFDR(optimizer='lbfgs')
```

Or reduce max iterations:
```python
model = SpatialFDR(max_iter=500)
```

### Results Don't Match Expected

**Check:**
1. Spatial structure in your data (run example.py to verify framework works)
2. Length scale estimation (try manual values: 0.1, 1.0, 10.0)
3. Lambda parameter (run sensitivity analysis)
4. Number of samples (need ~500+ for good kernel estimation)

---

## ☐ Final Checklist for Paper

Before submission:

- [ ] Ran full evaluation on all datasets (n_reps ≥ 100)
- [ ] Generated all figures (power, FDR calibration, sensitivity)
- [ ] Created results table from JSON
- [ ] Verified FDR control (empirical ≤ nominal)
- [ ] Documented power gains in each condition
- [ ] Included code as supplementary material
- [ ] Cited framework in methods section

---

## Quick Command Reference

```bash
# Test installation
python example.py

# Single dataset
python main.py --datasets my_data --n_reps 100

# Multiple datasets
python main.py --datasets data1 data2 data3 --n_reps 50

# With lambda sensitivity
python main.py --datasets my_data --lambda_sensitivity

# Custom parameters
python main.py \
    --datasets my_data \
    --n_samples 500 \
    --n_reps 100 \
    --spatial_strengths none medium strong \
    --effect_strength medium \
    --output_dir ./my_results
```

---

## Getting Help

1. **QUICKSTART.md** - Quick reference guide
2. **README.md** - Full documentation
3. **IMPLEMENTATION_SUMMARY.md** - Technical details
4. Check docstrings: `help(SpatialFDR)`
5. Example code in `example.py`

---

## Success Criteria

Your implementation is working correctly if:

✓ Example runs without errors  
✓ FDR controlled at nominal level  
✓ Power gain in spatially clustered conditions  
✓ No gain in random (none) condition  
✓ Figures are generated  
✓ Results are reproducible (same random seed → same results)

---

## Next: Paper Writing

Once evaluation is complete:

1. **Methods Section:** Describe the SpatialFDR algorithm
   - Kernel logistic regression
   - RKHS regularization  
   - Natural gradient optimization

2. **Results Section:** Present evaluation findings
   - Power gains across conditions
   - FDR control verification
   - Comparison to BH

3. **Figures:** Include the generated plots
   - Power comparison (main result)
   - FDR calibration (validation)
   - Lambda sensitivity (supplementary)

4. **Supplementary:** Attach the code
   - Entire `spatial_fdr_evaluation/` package
   - `example.py` as demonstration
   - `requirements.txt` for reproducibility

---

**Ready to start? Begin with Step 1! ✓**
