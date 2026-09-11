# configs/

Layered, version-controlled configuration. Nothing under `configs/` ever contains a
credential — secrets are environment-variable-only (see `.env.example`).

| Path | Status | Purpose |
|---|---|---|
| `logging.yaml` | **active** | stdlib logging dictConfig, loaded by `ad_mri.core.logging`. |
| `data/` | **active** | Per-dataset ingestion/validation/label-taxonomy configs, validated by `ad_mri.data_pipeline.dataset_config.DatasetConfig`. Copy `dataset_example.yaml` per real dataset. |
| `preprocessing/` | **active** | Spatial/intensity/QC/augmentation configs, validated by `ad_mri.datasets.preprocessing_config.PreprocessingConfig`. `default.yaml` is used by both training and inference (see README "Preprocessing pipeline"). |
| `model/` | **active** | CNN architecture configs, validated by `ad_mri.modeling.model_config.ModelConfig`. `cnn3d_default.yaml` — see `docs/model_architecture.md`. |
| `training/` | **active** | Training run configs, validated by `ad_mri.training.training_config.TrainingConfig`. `default.yaml` — see README "Training pipeline". |
| `hpo/` | **active** | Hyperparameter-search space + budget, validated by `ad_mri.training.hpo_config.HPOConfig`. `default.yaml` — see `docs/model_selection.md`. |
| `explainability/` | **active** | LIME/SHAP parameter configs, validated by `ad_mri.explainability.config.LimeConfig`/`ShapConfig`. `lime_default.yaml`/`shap_default.yaml` — see `docs/explainability.md`. |
| `inference/` | **active** | Probability-calibration config, validated by `ad_mri.inference.calibration.CalibrationConfig`. `calibration_default.yaml` (`method: none` — uncalibrated) — see `docs/inference.md`. Applied only when explicitly passed via `--calibration-config`; `ad-mri predict` is uncalibrated by default. |
| `env/` | reserved | Per-environment overrides layered on top of the above — added once the data/model/training config schemas exist. |

Evaluation (`ad-mri evaluate`) has no dedicated config directory: it reuses the same
`preprocessing/*.yaml` a model was trained with (hash-checked against the checkpoint's own
provenance — see `docs/evaluation.md`), and its remaining parameters (confidence level, bootstrap
resample count, threshold, seed) are CLI flags, not persisted config — there is nothing about an
evaluation run that needs to be versioned independently of the checkpoint + dataset it evaluated.
Inference (`ad-mri predict`) is the same, plus which checkpoint/preprocessing/calibration config to
serve is deployment-level configuration (`INFERENCE_MODEL_PATH` etc., environment variables — see
`.env.example`), not a versioned config file.

For this phase, application-level configuration (server, database, security, logging
level/format, MLflow/Redis endpoints) is sourced entirely from environment variables via
`ad_mri.core.config.Settings` — see `.env.example` for the full list.

Full rationale: `docs/TECHNICAL_SPECIFICATION.md` §14 (Configuration Management).
