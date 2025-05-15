# IDNet-2025
IDNet-2025 is a synthetic data generation framework to generate a large number of identity documents using only a few documents from a target domain without including any private information.

## Environment Setup and Installations
Python>=10.0 is required to run the project. To install all the dependencies, either run the following command or manually install the dependencies in the [requirements](/requirements.txt) file.
```bash
pip install -r requirements.txt
```


## Setting Up Datasets and Models
#### Setting Up Models
Download the pretrained models from [here](https://huggingface.co/datasets/cactuslab/IDNet-2025/blob/main/models.tar.gz) and place the unzipped models folder inside the data/ directory.

#### Setting Up Target Domain Images
Download the target domain images file from [here](https://drive.google.com/file/d/1iqZ0rDuO0GSkc3Osrr7V_--JbvPOky5X/view?usp=sharing) and place the unzipped 'target_images' folder inside the data/ directory.

#### Setting Up Synthetic Dataset
Download the datasets from [here](https://huggingface.co/datasets/cactuslab/IDNet-2025) and place the unzipped folders inside the data/ directory.


## Running the Experiments Scripts

### Running Bayesian Optimization
To run the Bayesian Optimization baseline, run the following command under the root directory of the project.
```bash
python experiments/Bayesian_search.py target_samples with_model lambda0 lambda1 candidate_models
```
In the above command, the parameter target_samples denotes the number of samples to be used (int), with_model denotes whether the optimization will be model-guided or not (0 or 1), lambda0 and lambda1 are Bayesian optimization hyperparameters, and candidate_models stand for the names of models (space separated) that will guide the optimization. Example commands are given below:
##### Example of Search W/ Model-Guided Optimization
```bash
python experiments/Bayesian_search.py 20 1 1 1 resnet50
```
##### Example of Search W/ SSIM Only Objective
```bash
python experiments/Bayesian_search.py 20 0 1 1 ssim
```
You may also execute experiments/run.py which provides sample commands to execute Bayesian_search.py.


### Running Hyperband Search Baseline
To run the Hyperband search baseline, run the following command under the root directory of the project.
```bash
python experiments/Hyperband_search.py param_r param_eta target_samples with_model candidate_models
```
Here, param_r and param_eta represent maximum resources and successive halving parameters of the Hyperband search method. Other parameters target_samples, with_model, and candidate_models are similar to those in Bayesian search. Example commands:
##### Example of Search W/ Model-Guided Optimization
```bash
python experiments/Hyperband_search.py 700 3 20 1 resnet50
```
##### Example of Search W/ SSIM Only Objective
```bash
python experiments/Hyperband_search.py 700 3 20 0 ssim
```
Also, experiments/run_hyperband.py can be executed, which runs Hyperband_search.py with example commands.


### Running CycleGAN Baseline
To run the CycleGAN baseline, run the following command under the root directory of the project.
```bash
python experiments/test_cyclegan.py dataset
```
Example command:
To run the CycleGAN baseline, run the following command under the root directory of the project.
```bash
python experiments/test_cyclegan.py idnet2sidtd1
```
Additionally, experiments/cycle_run.sh runs test_cyclegan.py with example commands.


### Running Bayesian Optimization on Scanned Images
To run the Bayesian Optimization on scanned images, run the following command under the root directory of the project.
```bash
python scanned_image/Bayesian_search.py target_samples with_model candidate_models
```
To generate scanned images, run scanned_image/scan_image_generation.py


### Running Bayesian Optimization on Template Images
To run the Bayesian Optimization on template images, run the following command under the root directory of the project.
```bash
python template_image/Bayesian_search.py --area LOCATION_NAME --segment TEMPLATE_SEGMENT --target_samples NUM_SAMPLES --with_model IS_GUIDED --candidate_models MODEL_NAMES --config_info CONFIG_PATH --fonts_path PATH_TO_FONTS --output_file CONFIG_FILE
```
Example command:
```bash
python template_image/Bayesian_search.py --area ALB --segment surname --target_samples 10 --with_model 1 --candidate_models resnet50 vit-large --config_info data/configures/ALB_parameters.json --fonts_path small_fonts --output_file ALB_parameters.json
```
