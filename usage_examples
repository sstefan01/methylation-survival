# Usage Examples

This file provides common examples for training, inference, evaluation, and applying MANTIS models to custom datasets.

All commands should be run from the root of the repository after activating the Python environment.

```bash
source .minimal_req/bin/activate
```

## 1. Run inference with the released pretrained MANTIS-MB model

The default pretrained model path is specified in `config.yaml`:

```yaml
inference:
  input_model_path: "models/mantismb_model_weights.pt"
```

To run inference using the default model and the test cohort defined in `config.yaml`:

```bash
python inference.py --config config.yaml
```

Predictions will be written to the path specified in `config.yaml`:

```yaml
inference:
  output_predictions_path: "results/survival_predictions.csv"
```

You can also override the model path and output path from the command line:

```bash
python inference.py \
  --config config.yaml \
  --model_path models/my_trained_model.pt \
  --output results/my_predictions.csv
```

## 2. Train a new model

Training cohorts and the test cohort are defined in the `run_setup` section of `config.yaml`:

```yaml
run_setup:
  training_cohorts: ["cav", "northcott", "sturm"]
  test_cohort: "jones"
```

To train a model:

```bash
python train.py --config config.yaml
```

The trained model will be saved according to:

```yaml
training:
  output_model_dir: "models/"
  output_model_name: "trained_survival_model.pt"
```

## 3. Evaluate predictions

After running inference, evaluate predictions using:

```bash
python evaluate.py \
  --config config.yaml \
  --pred_path results/survival_predictions.csv \
  --output_metrics results/evaluation_metrics.json
```

This computes survival prediction metrics such as the c-index, time-dependent AUC, and Brier score.

## 4. Plot results

To generate plots from predictions and evaluation outputs:

```bash
python plot_results.py \
  --config config.yaml \
  --metrics_file results/evaluation_metrics.json \
  --plot_survival \
  --predictions_file results/survival_predictions.csv \
  --output_dir results/plots
```

## 5. Change input feature combinations

Input features are controlled using two fields in `config.yaml`:

```yaml
run_setup:
  feature_types: ["beta", "cnv"]
  include_metastasis: True
```

For beta values only:

```yaml
run_setup:
  feature_types: ["beta"]
  include_metastasis: False
```

For CNVs only:

```yaml
run_setup:
  feature_types: ["cnv"]
  include_metastasis: False
```

For beta values and CNVs:

```yaml
run_setup:
  feature_types: ["beta", "cnv"]
  include_metastasis: False
```

For beta values, CNVs, and metastasis status:

```yaml
run_setup:
  feature_types: ["beta", "cnv"]
  include_metastasis: True
```

Input dimensions are inferred automatically from the selected feature types, so users do not need to manually edit parameters such as `input_dim` or `c2_input_offset`.

## 6. Use alternative survival heads

The survival head is controlled by:

```yaml
model:
  survival_head_type: "mtlr"
```

Available options are:

```yaml
model:
  survival_head_type: "mtlr"
```

```yaml
model:
  survival_head_type: "deepsurv"
```

```yaml
model:
  survival_head_type: "deephit"
```

After changing the survival head, train as usual:

```bash
python train.py --config config.yaml
```

## 7. Run inference on a custom external cohort

To apply a trained model to a new patient cohort, prepare input files that match the released feature annotations.

For example:

```text
data/
├── beta_custom.csv
├── cnv_custom.csv
└── surv_custom.csv
```

The beta value and CNV files should be aligned to the released CpG/probe and CNV feature annotations. Sample identifiers should match across molecular and survival/clinical files.

Because `config.yaml` uses cohort-specific file paths, add paths for the custom cohort in the `data` section:

```yaml
data:
  custom_beta_path: "data/beta_custom.csv"
  custom_cnv_path: "data/cnv_custom.csv"
  custom_surv_path: "data/surv_custom.csv"
```

Then set the custom cohort as the test cohort:

```yaml
run_setup:
  training_cohorts: ["cav", "northcott", "sturm"]
  test_cohort: "custom"
  feature_types: ["beta", "cnv"]
  include_metastasis: True
```

If the model includes metastasis status, the survival/clinical file should contain the column specified by:

```yaml
data:
  clinical_feature_col: "m"
```

Then run inference:

```bash
python inference.py \
  --config config.yaml \
  --model_path models/mantismb_model_weights.pt \
  --output results/custom_predictions.csv
```

## 8. Required columns in survival/clinical files

The relevant column names are specified in the `data` section of `config.yaml`:

```yaml
data:
  time_column: "pfs_time"
  event_column: "pfs_event"
  clinical_feature_col: "m"
  id_column: "Patient_ID"
  os_time_column: "time"
  os_event_column: "event"
```

For inference, `Patient_ID` should match the sample identifiers used in the beta and CNV matrices. If `include_metastasis: True`, the clinical file should also include the metastasis column specified by `clinical_feature_col`.

## 9. Recommended workflow for new datasets

For a new patient cohort:

1. Align beta values to the released CpG/probe annotations.
2. Align CNV features to the released CNV feature annotations, if using a CNV-containing model.
3. Ensure sample identifiers match across beta, CNV, and clinical files.
4. Add the custom file paths to `config.yaml`.
5. Set `run_setup.test_cohort` to the custom cohort name.
6. Select the appropriate input features using `feature_types` and `include_metastasis`.
7. Run inference using the pretrained or custom model.
8. If observed outcomes are available, run `evaluate.py`.

Example:

```bash
python inference.py \
  --config config.yaml \
  --model_path models/mantismb_model_weights.pt \
  --output results/custom_predictions.csv

python evaluate.py \
  --config config.yaml \
  --pred_path results/custom_predictions.csv \
  --output_metrics results/custom_metrics.json
```

## Notes

- Feature order must match the released feature annotation files.
- Models trained with CNV inputs require CNV features at inference.
- Models trained with metastasis status require metastasis status at inference.
- For external cohorts, predictions should be interpreted in the context of available clinical annotation.
