# AI-based Precision Prognostics and Therapy Personalization for Childhood Brain Tumors

## Project Overview

This repository contains the code for the study "AI-based Precision Prognostics and Therapy Personalization for Childhood Brain Tumors". The project focuses on predicting patient survival in medulloblastoma using DNA methylation beta values and copy number variation (CNV) data, along with metastasis status.

The core of the project is a deep learning model implemented in PyTorch, featuring:
* A sparse input layer (`sparselinear`) to handle high-dimensional methylation and CNV data guided by a gene connectivity prior.
* Subsequent dense layers with ELU activations and LayerNorm.
* Concatenation with metastasis stage.
* A Multi-Task Logistic Regression (MTLR) head for discrete-time survival prediction.

The codebase allows for model training, running inference on test data using a pre-trained model, evaluating prediction performance (C-index, time-dependent AUC, Brier score), and plotting results.

## Requirements

* Python (>=3.9 recommended, tested with environment specified below)
* Conda package manager
* Git (for cloning the repository)

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/sstefan01/methylation-survival.git
    cd methylation-survival
    ```

2.  **Create Conda Environment:** Create the environment using the provided file. This installs all necessary dependencies with specific versions.
    ```bash
    conda env create -f environment.yml
    ```
    *(This might take up to 10 minutes)*

3.  **Activate Environment:** Activate the newly created environment (the name is specified inside `environment.yml`, usually defaults to the name in the file, e.g., `submission_env`).
    ```bash
    conda activate <environment_name_from_yaml>
    ```

## Data Acquisition and Placement

Several data files are required to run the different scripts. Data and model weights can be found at Zenodo using the following DOI/link: **[Zenodo DOI/Link Here]**.

1.  **Connectivity Matrix:**
    * The file `conn_mat_beta.csv` is required.
    * **Action:** Place this file inside the `data/` directory.

2.  **Pre-trained Model:**
    * To run inference (`inference.py`) and subsequent evaluation/plotting without re-training, you need the pre-trained model weights.
    * **Action:** Download the model state dictionary file (e.g., `mantismb_model_statedict.pt`) from Zenodo and place it in the `models/` directory. Ensure the path in `config.yaml` under `inference.input_model_path` matches this file.

3.  **Cohort Data (Features & Survival):**
    * The code requires separate files for different data types (beta, cnv) and survival/clinical information for multiple patient cohorts (e.g., ICGC, Cavalli, Sturm, Northcott).
    * **Training Data:** The data files for the cohorts used in training (`cavalli`, `northcott`, `sturm` as per default `config.yaml`) need to be obtained.
        * **Action:** Download the training datasets from Zenodo.
        * Place the downloaded files into the `data/` directory. Ensure the filenames match those specified in the `data` section of `config.yaml` (e.g., `beta_cavalli.csv`, `cnv_cavalli.csv`, `surv_cavalli.csv`, etc.).
    * **Test Data (ICGC):** The default test cohort is ICGC.
        * **Action:** Obtain the ICGC data files (`beta_icgc.csv`, `cnv_icgc.csv`, `surv_icgc.csv`) and place them in the `data/` directory. These files are also available through Zenodo Ensure filenames match the config.

4.  **Data Directory Structure (Expected):**
    ```
    <project_root>/
    ├── data/
    │   ├── conn_mat_beta.csv       
    │   ├── beta_cavalli.csv       
    │   ├── cnv_cavalli.csv
    │   ├── surv_cavalli.csv
    │   ├── beta_northcott.csv
    │   ├── cnv_northcott.csv
    │   ├── surv_northcott.csv
    │   ├── beta_sturm.csv
    │   ├── cnv_sturm.csv
    │   ├── surv_sturm.csv
    │   ├── beta_icgc.csv           
    │   ├── cnv_icgc.csv
    │   ├── surv_icgc.csv
    │   └── ... (other cohort files if using) ...
    ├── models/
    │   └── mantismb_model_statedict.pt # Pre-trained model weights
    ├── results/
    │   ├── plots/                  # Output directory for plots
    │   └── ... (output predictions/metrics saved here) ...
    ├── src/
    │   ├── __init__.py
    │   ├── model.py
    │   └── utils.py
    ├── config.yaml                 # Configuration file
    ├── train.py                    # Training script
    ├── inference.py                # Inference script
    ├── evaluate.py                 # Evaluation script
    ├── plot_results.py             # Plotting script
    ├── environment.yml             # Conda environment file
    └── README.md                   # This file
    ```

## Configuration (`config.yaml`)

The `config.yaml` file controls various aspects of the scripts, including file paths, model hyperparameters, cohort selection for training/testing, and column names.

* **Paths:** Ensure all paths in the `data:` section correctly point to your downloaded/placed data files relative to the project root. Update `inference.input_model_path` to your pre-trained model file.
* **Run Setup:** The `run_setup:` section defines which cohorts are used for training (`training_cohorts`) and which single cohort is used for testing (`test_cohort`) when running `train.py`/`inference.py`/`evaluate.py`. It also lists the `feature_types` to load and concatenate.
* **Hyperparameters:** Model structure and training parameters are defined here.

## Usage

Ensure your Conda environment is activated (`conda activate <environment_name_from_yaml>`) before running any scripts. All commands should be run from the project root directory.

**1. Training (Optional - if not using pre-trained model)**

* Modify `config.yaml` if necessary (e.g., change training cohorts, hyperparameters).
* Run the training script:
    ```bash
    python train.py --config config.yaml
    ```
* This will train a new model and save the `state_dict` to the location specified in `config['training']['output_model_dir'] / config['training']['output_model_name']`. You would then update `config['inference']['input_model_path']` to use this newly trained model for subsequent steps.

**2. Inference (Generating Predictions)**
* **Run the script:**
    * To use the **default pre-trained model** specified in `config.yaml`:
        ```bash
        python inference.py --config config.yaml
        ```
    * To use a **different model file** (e.g., one you trained yourself), use the `--model_path` argument:
        ```bash
        # Replace path/to/your/model.pt with the actual path
        python inference.py --config config.yaml --model_path path/to/your/model_statedict.pt
        ```
    * You can also override the output file path:
        ```bash
        python inference.py --config config.yaml --model_path path/to/your/model_statedict.pt --output results/my_custom_predictions.csv
        ```
* **Output:** This generates a prediction CSV file (e.g., `results/survival_predictions.csv` or the path specified by `--output`) containing patient IDs and predicted survival probabilities at different time points.

**3. Evaluation (Calculating Metrics)**

* Requires the prediction CSV generated by `inference.py`.
* Ensure `config.yaml` points to the correct survival data files for the training cohorts (for AUC) and the test cohort (using correct OS time/event columns if specified).
* Run the evaluation script, providing the path to the predictions:
    ```bash
    # Example: Evaluate predictions saved in results/survival_predictions.csv
    python evaluate.py --config config.yaml --pred_path results/survival_predictions.csv --output_metrics results/evaluation_metrics.json
    ```
* This will print evaluation metrics (C-index, AUC, Brier Score) to the console and optionally save them to the specified JSON file (`--output_metrics`).

**4. Plotting Results**

* Requires the metrics JSON file (from `evaluate.py`) and potentially the predictions CSV file (if plotting survival curves).
* Run the plotting script:
    ```bash
 
    # Plot metrics and survival curves
    python plot_results.py --config config.yaml --metrics_file results/evaluation_metrics.json --plot_survival --predictions_file results/survival_predictions.csv --output_dir results/plots
    ```
* This will save PNG plot files in the specified output directory (`--output_dir`).

## System requirements

### Minimum requirements
- **Operating system**: macOS 10.15+, Ubuntu 18.04+, or Windows 10/11 (with WSL2)  
- **CPU**: any 64‑bit Intel or Apple‑Silicon processor  
- **Memory**: ≥ 8 GB RAM  
- **Python**: ≥ 3.8  
- **Conda**: ≥ 4.10  
- **Disk space**: ≥ 2 GB for code, dependencies, and example data  
- **GPU**: none required (CPU only)  

### Tested environment
- **Operating system**: macOS 14.0 (Ventura) (Build 23A344)  
- **Hardware**: Apple M1 Pro (arm64), 16 GB RAM  
- **Python**: 3.9.21  
- **Conda**: 23.3.1

### Dependencies
All Python packages are pinned in `environment.yml`.  

## License

This project is licensed under MIT license.

## Contact

For questions, please contact sabina_stefan@dfci.harvard.edu or volker_hovestadt@dfci.harvard.edu
