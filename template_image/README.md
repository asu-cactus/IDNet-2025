# Bayesian Optimization for Template Image Processing

This repository contains scripts for applying **Bayesian Optimization (BO)** to template images and generating synthetic images using optimized parameters.

## File Overview
### `Bayesian_search.py`
Performs Bayesian Optimization on template images.

**Search space includes:**
- Segment position
- Font style
- Font size
- Font color
- And more (custom parameters can be added)

#### 🚀 How to Run
From the root directory of the project:

```bash
python template_image/Bayesian_search.py \
  --area LOCATION_NAME \
  --segment TEMPLATE_SEGMENT \
  --target_samples NUM_SAMPLES \
  --with_model IS_GUIDED \
  --lambda0 FACTOR_OF_SIMILARITY \
  --lambda1 FACTOR_OF_CONSISTENCY \
  --candidate_models MODEL_NAMES \
  --config_info CONFIG_PATH \
  --fonts_path PATH_TO_FONTS \
  --output_file CONFIG_FILE
```

**Example:**
```bash
python template_image/Bayesian_search.py \
  --area ALB \
  --segment surname \
  --target_samples 10 \
  --with_model 1 \
  --lambda0 1 \
  --lambda1 1 \
  --candidate_models resnet50 vit-large \
  --config_info data/configures/ALB_parameters.json \
  --fonts_path small_fonts \
  --output_file ALB_parameters.json
```

---

### `template_image_generation.py`
Generates synthetic template images using parameters found via BO.  
You must configure the template layout, portrait image, and PII fields.

#### 🚀 How to Run
```bash
python template_image_generation.py \
  --parameters_path PARAMETERS_JSON \
  --JSONPATH PATH_TO_PII_FILES ...
```

---

### `utils.py`
Provides shared utility functions for both the optimization and image generation workflows.

---

To prevent abuse, we only provide a single generation file for ALB template generation with some parameters hard coded. For other regions or templates, the hyperparameters may differ slightly. Please contact us if you're interested in accessing additional scripts for research purposes.

