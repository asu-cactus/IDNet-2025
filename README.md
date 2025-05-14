# IDNet-2025
IDNet-2025 is a synthetic data generation framework to generate a large number of identity documents using only a few documents from a target domain without including any private information.

### Environment Setup and Installations
Python>=10.0 is required to run the project. To install all the dependencies, either run the following command or manually install the dependencies in the [requirements](/requirements.txt) file.
```bash
pip install -r requirements.txt
```

### Setting Up Datasets and Models
#### Setting Up Models
Download the pretrained models from [here](https://huggingface.co/datasets/cactuslab/IDNet-2025) and place the unzipped models folder inside the data/ directory.

#### Setting Up Target Domain Images
Download the target domain images file from [here](https://drive.google.com/file/d/1iqZ0rDuO0GSkc3Osrr7V_--JbvPOky5X/view?usp=sharing) and place the unzipped 'target_images' folder inside the data/ directory.

#### Setting Up Synthetic Dataset
Download the datasets from [here](https://huggingface.co/datasets/cactuslab/IDNet-2025) and place the unzipped folders inside the data/ directory.

## Running the Scripts
### Running the Experiments
#### Running Hyperband Search baseline
To run the Hyperband search baseline, run the following command under the root directory of the project.
```bash
python Hyperband_search.py param_r param_eta num_samples is_model_guided guiding_models
```


