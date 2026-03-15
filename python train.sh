python train.py
python inference.py   --custom_beta data/beta_acns331_20240104.csv   --custom_cnv data/CNV_gene_acns331_20240115.csv   --custom_surv data/acns331_surv.csv   --output results/acns331_predictions.csv
python evaluate.py --config config.yaml --pred_path results/acns331_predictions.csv --custom_surv data/acns331_surv.csv --output_metrics results/evaluation_metrics_ensmbled_acns331.json

python inference.py  --model_path models/trained_survival_model_statedict.pt --custom_beta data/beta_val_uniform_preprocess_rCM.csv   --custom_cnv data/CNV_g34_test.csv  --custom_surv data/group34_rCM_samples.csv   --output results/rCM_predictions.csv
python evaluate.py --config config.yaml --pred_path results/rCM_predictions.csv --custom_surv data/group34_rCM_samples.csv --output_metrics results/evaluation_metrics_rCM.json




python inference.py   --custom_beta data/beta_sjmb03_20240104.csv   --custom_cnv data/CNV_gene_sjmb03_20240111.csv   --custom_surv data/sjmb03_surv.csv   --output results/sjmb03_predictions.csv
python evaluate.py --config config.yaml --pred_path results/sjmb03_predictions.csv --custom_surv data/sjmb03_surv.csv --output_metrics results/evaluation_metrics_ensmbled_sjmb03.json


python inference.py  --model_path models/trained_survival_model_statedict.pt --custom_beta data/beta_sj_high.csv   --custom_cnv data/cnv_sj_high.csv   --custom_surv data/surv_sj_high.csv   --output results/sjmb03_predictions.csv
python evaluate.py --config config.yaml --pred_path results/sjmb03_predictions.csv --custom_surv data/surv_sj_high.csv --output_metrics results/evaluation_metrics_sjmb03_high.json

python inference.py  --model_path models/trained_survival_model_statedict.pt --custom_beta data/beta_sj_average.csv   --custom_cnv data/cnv_sj_average.csv   --custom_surv data/surv_sj_average.csv   --output results/sjmb03_average_predictions.csv
python evaluate.py --config config.yaml --pred_path results/sjmb03_average_predictions.csv --custom_surv data/surv_sj_average.csv --output_metrics results/evaluation_metrics_sjmb03_average.json