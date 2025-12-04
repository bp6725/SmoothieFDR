# f₀ and f₁ Diagnostic Tools

## 📊 **Purpose**

These tools help diagnose why α → 0 in your spatial FDR implementation by visualizing how f₀(p) and f₁(p) are estimated.

---

## 🚀 **Quick Start**

### **Option 1: Python Script (Recommended)**

```bash
python f0_f1_diagnostic.py
```

**Output:**
- `f0_f1_comparison.png` - f₀ and f₁ curves across datasets
- `f0_f1_ratio_analysis.png` - Likelihood ratio analysis
- Terminal output with summary statistics

### **Option 2: Jupyter Notebook (Interactive)**

```bash
jupyter notebook f0_f1_diagnostic.ipynb
```

**Benefits:**
- Interactive exploration
- Modify parameters on the fly
- See results immediately

---

## 📈 **What the Plots Show**

### **Plot 1: f₀ and f₁ Comparison**

**Top row (Aggregate):**
- Blue line: f₀(p) - Null density
- Red line: f₁(p) - Alternative density
- Gray dashed: Theoretical null (should be flat at 1.0)
- Shaded regions: ±1 std across datasets

**Bottom rows: Individual examples**
- 6 sample datasets
- Shows variability across runs

**What to look for:**
- ✅ **Good:** f₀ flat at 1.0, f₁ spiked at p≈0
- ❌ **Bad:** f₀ U-shaped, f₁ > f₀ everywhere

### **Plot 2: Ratio Analysis**

**Top panel:**
- Purple line: f₁(p) / f₀(p) ratio
- Red shaded: Where f₁ > f₀ (optimizer prefers α=0)
- Blue shaded: Where f₀ > f₁ (optimizer prefers α=1)

**Bottom panel:**
- Fraction of datasets where f₁ > f₀
- Should be high at p≈0 (signals), low elsewhere

**Critical diagnostic:**
- If f₁ > f₀ for >60% of range → 🚨 **α will go to 0!**

---

## 🔍 **Interpreting Results**

### **Scenario 1: Working Correctly**

```
π₀ estimates: Mean 0.800 (True: 0.800) ✓
Ratio f₁/f₀ at key p-values:
  p=0.01: 15.2 ✓  (alternatives dominate)
  p=0.05: 8.3  ✓
  p=0.10: 3.1  ✓
  p=0.50: 0.3  ✓  (nulls dominate)

f₁ > f₀ for 18% of p-value range ✓
```

**Plots show:**
- f₀ is flat (uniform)
- f₁ is concentrated near p=0
- Ratio > 1 only for small p

### **Scenario 2: BROKEN (Your Current State)**

```
π₀ estimates: Mean 0.804 (True: 0.800) ✓
Ratio f₁/f₀ at key p-values:
  p=0.01: 405019.81 ⚠️  (EXTREME!)
  p=0.05: 11259.58  ⚠️
  p=0.10: 0.00      ⚠️  (Zero!)
  p=0.50: 0.00      ✓

f₁ > f₀ for 65% of p-value range 🚨
```

**Plots show:**
- f₀ is U-shaped (dilated null)
- f₁ has extreme spikes
- Ratio is unstable (huge at some p, zero elsewhere)

---

## 🎯 **What the Numbers Mean**

### **The Ratio f₁/f₀**

This is what the likelihood optimizer sees:

```
L(α) = -Σ log[α·f₀(pᵢ) + (1-α)·f₁(pᵢ)]
```

**If f₁/f₀ is large at pᵢ:**
- Setting α=0 gives: log[f₁(pᵢ)] (large)
- Setting α=1 gives: log[f₀(pᵢ)] (small)
- **Optimizer chooses α=0**

**Your current state:**
- f₁/f₀ = 405,020 at p=0.01
- f₁/f₀ = 0 at p=0.10

This creates a pathological loss landscape!

---

## 🔧 **Parameters You Can Modify**

In the script, edit these at the top:

```python
# Line ~372 in f0_f1_diagnostic.py
n_datasets = 50        # Number of datasets to generate
n_samples = 500        # Samples per dataset
d = 2                  # Spatial dimension
n_clusters = 3         # Number of alternative clusters
cluster_strength = 0.3 # Spatial clustering strength
effect_size = 2.0      # Shift for alternatives
```

**Try different settings:**
- Increase `effect_size` to 3.0 → stronger signals
- Decrease `cluster_strength` to 0.1 → weaker spatial structure
- Change `n_samples` to 1000 → more data

---

## 📊 **Expected Output**

### **Terminal Summary**

```
======================================================================
SUMMARY STATISTICS
======================================================================

π₀ estimates:
  Mean: 0.804 (True: 0.800)
  Std:  0.033
  Range: [0.740, 0.896]

Ratio f₁/f₀ at key p-values (averaged over datasets):
  p=0.01: 15.23 ✓
  p=0.05: 8.14  ✓
  p=0.10: 2.87  ✓
  p=0.50: 0.31  ✓

======================================================================
```

### **Files Created**

1. **f0_f1_comparison.png**
   - Multi-panel figure showing f₀ and f₁
   - Aggregate + 6 examples
   - ~300 KB

2. **f0_f1_ratio_analysis.png**
   - Ratio curves
   - Consistency check
   - ~200 KB

---

## 🚨 **Red Flags**

### **Sign 1: U-Shaped f₀**

If you see f₀(p) high at p=0 and p=1, low in middle:
- This is the "dilated null"
- Theoretically interesting, but WRONG for p-values
- P-values are ALWAYS uniform under null

### **Sign 2: Extreme Ratios**

If f₁/f₀ > 1000 or = 0:
- Numerical instability
- Division by near-zero values
- Loss landscape is pathological

### **Sign 3: f₁ > f₀ Everywhere**

If >60% of p-range has f₁ > f₀:
- Optimizer will set α=0 everywhere
- Results in 70% FDR
- Total failure

---

## ✅ **Next Steps**

1. **Run the diagnostic:**
   ```bash
   python f0_f1_diagnostic.py
   ```

2. **Check the plots** - Look for U-shaped f₀

3. **If broken, switch to simple estimation:**
   - Use theoretical null: f₀(p) = 1
   - Fit Beta to small p-values for f₁
   - Code provided in earlier discussions

4. **Re-run evaluation** to verify FDR control

---

## 📚 **Files Included**

- `f0_f1_diagnostic.py` - Standalone script
- `f0_f1_diagnostic.ipynb` - Jupyter notebook
- `README_DIAGNOSTIC.md` - This file

---

## 💡 **Key Insight**

**The fundamental issue:**

Your "Efron-style" estimation creates a U-shaped f₀ in p-space. While theoretically sophisticated, this is **incompatible** with the definition of p-values (which are uniform under null).

**Result:** The optimizer sees f₁ > f₀ almost everywhere → sets α=0 → rejects everything → 70% FDR.

**Solution:** Use theoretical null (f₀=1) and fit f₁ to alternatives only.

---

**Good luck! Run the diagnostic and let me know what you see!** 🎯
