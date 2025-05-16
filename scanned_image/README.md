
# Bayesian Optimization for Scanned Image Processing

This folder contains scripts related to applying Bayesian Optimization (BO) on scanned images and generating synthetic scanned versions using optimized parameters.

## Files Overview

### 1. `Bayesian_search.py`
Applies Bayesian Optimization on scanned images.

- Defines search spaces such as:
  - Brightness
  - Contrast
  - Sharpness
  - Shadow offset
  - And more (you can add your own)

**How to Run:**
```bash
python Bayesian_search.py <num_samples> <use_model_guided> <model1> <model2> ...
```

- `<num_samples>`: Number of samples from the target domain to use in the BO search.
- `<use_model_guided>`: Whether to use model-guided optimization (`1` for `True` or `0` for `False`).
- `<model1>, <model2>, ...`: Names of models to be used for BO search.

---

### 2. `evaluate_parameters.py`
Contains the evaluation functions used during the BO process.

- You must configure:
  - Target domain data path
  - Trained model paths
  - Output paths
  - Paper texture path
- Sample files can be found in: `../data/scanned_data`

---

### 3. `scanned_image_generation.py`
Generates scanned-style images using the parameters found via BO.

**How to Run:**
```bash
python scanned_image_generation.py <optional_parameters>
```

---

### 4. `utils.py`
Provides auxiliary functions used in both the optimization and image generation processes.
