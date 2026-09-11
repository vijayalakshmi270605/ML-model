# ad-mri — Alzheimer's Disease MRI Classification System

A production-grade, modular ML system for Alzheimer's Disease classification from brain MRI,
built around a CNN/deep-learning classifier with LIME/SHAP explainability, MLflow experiment
tracking, DVC data versioning, and a FastAPI serving layer.

> **Not a certified medical device.** This is a research / decision-support engineering system.
> It makes no clinical diagnostic claims, and no performance number appears anywhere in this
> codebase unless it was actually produced by a logged MLflow run. See
> [`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md) §0.

## Status

**Phases 1-14 are done — this is the final phase.** repository scaffolding, dataset ingestion &
management, MRI preprocessing, the CNN model architecture, the training pipeline, robust model
selection/hyperparameter optimization, a comprehensive evaluation framework, LIME/SHAP
explainability, a production inference engine, a production-grade FastAPI backend, a
PostgreSQL-backed audit trail, containerization/CI-CD/deployment, ML/system monitoring, and a
security/privacy/responsible-AI review. A
full tested dataset pipeline, a full tested MONAI preprocessing
pipeline (the *same* pipeline used by both training and inference), a configurable 2D/3D CNN
classifier (`ConvNetClassifier` — see [`docs/model_architecture.md`](docs/model_architecture.md)),
a full `Trainer` (checkpointing, early stopping, LR scheduling, mixed precision, class-imbalance
handling, MLflow tracking, resume-from-checkpoint), subject-level stratified K-fold
cross-validation + Optuna-based hyperparameter search (each trial scored by CV, never the test set
— see [`docs/model_selection.md`](docs/model_selection.md)), a standalone evaluation CLI (accuracy,
precision/recall/specificity/F1, ROC-AUC, PR-AUC, confusion matrix, bootstrap confidence
intervals, and JSON/CSV/markdown/PNG reports — see [`docs/evaluation.md`](docs/evaluation.md)), an
explainability service producing real LIME + SHAP explanations for any checkpoint (see
[`docs/explainability.md`](docs/explainability.md)), a cached, timeout-bounded inference engine
(see [`docs/inference.md`](docs/inference.md)), the REST API serving it — `POST
/api/v1/predictions`, `POST /api/v1/explanations`, `GET /api/v1/models`, versioned health/ready/
version endpoints, centralized exception handling, scoped API-key authentication+authorization, an
in-memory rate limiter, secure response headers, and hardened CORS/upload/logging handling (see
[`docs/api.md`](docs/api.md)) — and a
five-table PostgreSQL audit trail (`model_versions`, `prediction_requests`, `prediction_results`,
`explanation_records`, `system_events`), Alembic migrations, connection pooling, and a
repository/service layer, wired into the API best-effort (an unreachable audit database never fails
a request — see [`docs/database.md`](docs/database.md)) — are implemented and exercised by tests.
Four containerized services (`api`, `worker`, `training`, `mlflow` — see
[`docs/deployment.md`](docs/deployment.md)), a 7-step GitHub Actions CI pipeline (lint, type
check, security scan, unit tests, integration tests, Docker build + Trivy image scan), a
`development → staging → production` CD pipeline (GitHub Environments-gated promotion, with a
real, automatic health-check-triggered rollback), and a documented secrets-management
architecture (nothing is ever committed to this repository) round out the deployment story.
Real-time system/model metrics (CPU/memory/GPU, request latency/throughput/error rate,
prediction/class/confidence distributions, input-validation/preprocessing failure rates, model
latency), a Prometheus + Grafana stack (8 alert rules, a 17-panel dashboard, both real,
`promtool`-validated), offline data-drift reports (PSI + KS test against a training-distribution
baseline), and delayed, ground-truth-backed model-drift reports (never assuming ground truth is
immediately available — see [`docs/monitoring.md`](docs/monitoring.md)) round out the
monitoring story. A scoped, multi-key authorization architecture (`ApiScope`: `predict`/`label`,
layered on the existing single-key auth, 401-vs-403 tested), a decompression-bomb guard on
compressed NIfTI uploads, secure response headers on every response (including one from a
genuinely unhandled exception — a real Starlette architectural gap found and closed, not
assumed away), inline ML-safety disclaimers on every prediction/explanation response, and a
requirement-by-requirement security review (`bandit`: 0 findings; `pip-audit`: 0 known
vulnerabilities) round out the security/responsible-AI story — see
[`docs/security.md`](docs/security.md) and [`docs/responsible_ai.md`](docs/responsible_ai.md).

The MLflow Model Registry (multiple model *versions*, promotion history — distinct from both the
one currently-served model `GET /api/v1/models` reports and the narrower `model_versions` audit
table), S3/MinIO object storage (a `LocalFilesystemObjectStorage` backend is real today; see
`storage.object_storage`), Kubernetes manifests (`deploy/k8s/` is a placeholder — Docker
Compose is the real, implemented deployment target), Alertmanager routing (the Prometheus alert
*rules* are real; email/Slack/PagerDuty delivery is not wired up), and Evidently-based drift
reports (a declared, `train`-extra dependency this phase's own PSI/KS implementation was used
instead of, for stability — see [`docs/monitoring.md`](docs/monitoring.md) §9), TLS
termination/encryption-at-rest, and OAuth2/JWT/per-user identity (a shared, scoped API key
remains the authentication/authorization model — see [`docs/security.md`](docs/security.md) §7 for
the complete, honest residual-risk list) are **not implemented yet**; every touchpoint there is a
typed, documented stub that fails loudly (`NotImplementedError` internally) rather than silently
faking a result. See the roadmap in
[`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md).

> **This model has not been clinically validated.** Nothing in this deployment tooling — CI/CD,
> environment promotion, rollback — changes that; see the opening disclaimer in
> [`docs/deployment.md`](docs/deployment.md) and at the top of this README.

## Architecture at a glance

```
Data Sources → Data Pipeline (DVC) → Training (PyTorch/MONAI → MLflow) → Model Registry
                                                                              │
                                    ┌─────────────────────────────────────────┘
                                    ▼
                    FastAPI Inference API ◄──► Explainability (LIME/SHAP) via async worker
                                    │
                                    ▼
                    PostgreSQL (audit trail) + object storage (artifacts)
```

Full design: [`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md).

## Repository layout

```
src/ad_mri/
├── core/            # config (env-driven), structured logging, exceptions, constants, seeding
├── data_pipeline/    # ingestion, DICOM→NIfTI, validation, manifest, holdout + CV splitting
├── datasets/         # MRIDataset, preprocessing config + MONAI transform pipeline, visualization
├── modeling/         # ModelConfig, ConvNetClassifier, factory, summary/FLOPs, checkpointing
├── training/         # Trainer, cross-validation, HPO (Optuna), reporting, CLI entry point
├── evaluation/       # metrics, confusion matrix, ROC/PR curves, statistics (CI), report, evaluate CLI
├── explainability/   # LIME, SHAP, service, slicing, sampling, overlays, CLI entry point
├── inference/        # InferenceEngine, ModelLoader (cache), calibration, CLI entry point
├── model_registry/   # MLflow Model Registry client
├── api/              # FastAPI app: routers, schemas, middleware, exception handlers, security, DI
├── worker/           # Celery async jobs (explainability, batch prediction)
├── db/               # Audit-trail models, repositories, service layer, session, Alembic migrations
├── storage/          # Object-storage seam for large artifacts (checkpoints, explanation manifests)
├── monitoring/       # System/model/drift metrics, alert thresholds, health checks
└── cli/              # `ad-mri` command-line interface

configs/    # logging.yaml, data/, preprocessing/, model/, training/, hpo/, explainability/*.yaml (active); reserved env configs
data/       # DVC-tracked raw/interim/processed/splits (gitignored; empty until you run the pipeline)
docker/     # api / worker / training / mlflow Dockerfiles + local-dev and production docker-compose files
deploy/     # Per-environment config templates; prometheus/grafana configs; k8s/terraform placeholders
.github/    # CI (lint/type-check/security/test/build) and CD (dev->staging->prod) workflows
tests/      # unit, integration, api, data_validation, explainability
docs/       # technical specification, ADRs, model cards, API/database/deployment docs
```

See "Directory guide" below for what each package is responsible for.

## Prerequisites

- Python 3.11+
- (Optional, for the full stack) Docker + Docker Compose
- (Optional, for training-phase work later) a CUDA-capable GPU

## Install

```bash
# from the repository root
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"          # runtime + dev/test/lint dependencies
cp .env.example .env             # then edit values as needed for your machine
```

Training/MLOps dependencies (torch, monai, mlflow, dvc, lime, shap, celery, ...) are a separate,
heavier extra — install only when you reach the training phase:

```bash
pip install -e ".[train]"
```

## Run the API

```bash
ad-mri serve
# equivalent: uvicorn ad_mri.api.main:app --reload
```

Then check:

- http://localhost:8000/healthz — liveness
- http://localhost:8000/readyz — readiness (503 if PostgreSQL isn't running — expected without `docker compose up db`)
- http://localhost:8000/docs — interactive OpenAPI docs
- http://localhost:8000/metrics — Prometheus metrics

Every `/api/v1/*` route is fully implemented — see [`docs/api.md`](docs/api.md) for the full
reference (endpoints, schemas, auth/rate-limiting/CORS architecture, error contract):
`POST /api/v1/predictions`, `POST /api/v1/explanations`, `GET /api/v1/models`,
`GET /api/v1/health`/`/ready`/`/version`. Set `INFERENCE_MODEL_PATH` (see `.env.example`) or
`/predictions`/`/explanations` respond `503` rather than crashing; `/models` returns an empty list
either way. Only the full MLflow Model Registry (multiple versions/promotion — distinct from the
one currently-served model `/models` already reports) and the DB-backed prediction audit trail
remain unimplemented.

## Dataset ingestion & management

Requires the `train` extra (`pip install -e ".[train]"`) for `pandas`/`nibabel`/`scikit-learn`
(and `pydicom` if you have DICOM data). Each dataset gets its own config file under
`configs/data/*.yaml` — copy [`configs/data/dataset_example.yaml`](configs/data/dataset_example.yaml),
point `root_dir`/`labels_csv` at your data, and adjust `label_map` to your taxonomy. **Raw medical
imaging data is never committed to git** — `data/{raw,interim,processed,splits}/*` is gitignored;
version it with DVC instead (see below).

Expected layout per dataset (one folder per subject):

```
<root_dir>/<subject_id>/*.nii.gz      # NIfTI, or
<root_dir>/<subject_id>/*.dcm         # a DICOM series (auto-converted if convert_dicom: true)
<labels_csv>                          # subject_id -> raw diagnosis label, shipped separately
```

Four-stage pipeline, each stage runnable independently or via `dvc repro`:

```bash
# 1. Ingest: discover per-subject scans (+ convert any DICOM series) -> raw inventory CSV
ad-mri data ingest --dataset-config configs/data/dataset_example.yaml

# 2. Validate: file existence/format/corruption/dimensions/metadata/label checks
ad-mri data validate --dataset-config configs/data/dataset_example.yaml

# 3. Manifest: QC-passed rows -> the final versioned manifest (+ class-distribution report)
ad-mri data manifest --dataset-config configs/data/dataset_example.yaml

# 4. Split: reproducible, subject-level, leakage-safe train/val/test split
ad-mri data split --dataset-config configs/data/dataset_example.yaml
```

Outputs (all under the gitignored `data/` tree, `<name>` = the dataset config's `name`):
`data/interim/<name>/inventory.csv` → `data/interim/<name>/validated.csv` →
`data/processed/<name>/manifest.csv` → `data/splits/<name>/{train,val,test}.csv`.

**Guarantees**: splitting groups by `subject_id` — a subject with multiple scans/sessions always
lands entirely in one split (see `assert_no_subject_leakage` in
[`data_pipeline/splitting.py`](src/ad_mri/data_pipeline/splitting.py)); the same `seed` always
reproduces the same split; class imbalance is logged (`data_pipeline/manifest.py`) and
`data_pipeline/class_balance.py` computes inverse-frequency class weights /
a `WeightedRandomSampler` for the training phase to consume.

### DVC workflow

`dvc.yaml` declares the same four stages, parameterized by [`params.yaml`](params.yaml)
(`dataset_config`, `dataset_name`):

```bash
pip install -e ".[train]"        # brings in dvc[s3]
dvc init                         # once per repo
dvc remote add -d storage <your-remote-url>   # e.g. s3://bucket/path, or a local path for testing

# edit params.yaml to point at your dataset config, then:
dvc repro                        # runs ingest -> validate -> manifest -> split, skipping unchanged stages
dvc add data/raw/<name>          # start tracking the raw data itself (never via git)
git add dvc.yaml params.yaml data/raw/<name>.dvc .gitignore
git commit -m "Track <name> dataset via DVC"
dvc push                         # upload data to the configured remote
```

On another machine / in CI: `dvc pull` fetches the tracked data, `dvc repro` regenerates the
manifest/splits deterministically from it.

## Preprocessing pipeline

One function — [`datasets.transforms.build_transforms(config, train=...)`](src/ad_mri/datasets/transforms.py)
— is the single preprocessing implementation used by **both** training and production inference;
there is no second copy. `train=True` adds MONAI-based augmentation on top of the exact same
deterministic steps `train=False` uses. Configuration lives in
[`configs/preprocessing/default.yaml`](configs/preprocessing/default.yaml), validated by
`ad_mri.datasets.preprocessing_config.PreprocessingConfig`.

**Deterministic pipeline** (identical for train/val/test/inference): safe NIfTI load
(`SafeLoadImaged` — wraps any load failure in a clear `PreprocessingError`) → channel-first →
orient to a target axis convention (default `RAS`) → resample to a target voxel spacing → per-volume
intensity normalization (`zscore_nonzero` | `zscore` | `percentile_clip` | `min_max` — all computed
from that volume's own values, never dataset-wide statistics, so there's no channel for val/test
statistics to leak into training) → resize/pad-or-crop to a fixed spatial shape → optional quality
control (`QualityControld`: rejects non-finite output or a suspiciously all-background volume;
`qc_mode: raise` fails loud, `warn` logs and continues) → cast to `float32` tensor.

**Training-only augmentation** (never applied at val/test/inference): small-angle rotation +
translation + scale jitter (`RandAffined`; degrees, not 90° turns), Gaussian intensity noise,
intensity shift/scale jitter. Left-right flip is **off by default** — Alzheimer's-related atrophy
can be asymmetric, so mirroring isn't guaranteed to produce a medically valid sample for this task;
enable `augmentation.allow_flip` explicitly only if you've verified it's appropriate for your data.

**Reproducibility & versioning**: `build_transforms(..., seed=...)` pins MONAI's augmentation RNG
for exact repeatability; `PreprocessingConfig.version` (human semver) + `.compute_hash()`
(content-addressed fingerprint) let any preprocessed sample/model be traced back to the exact
recipe that produced it — `save_preprocessing_config`/`load_preprocessing_config` persist both.

```bash
# Sanity-check preprocessing visually on one scan (original vs. preprocessed
# mid-slices + intensity histograms) before trusting it on a whole dataset:
ad-mri data preview-preprocessing --image path/to/scan.nii.gz
# -> writes path/to/scan.preprocessing_preview.png
```

```python
# The same call, from either training or inference code:
from ad_mri.datasets.preprocessing_config import load_preprocessing_config
from ad_mri.datasets.transforms import apply_transforms, build_transforms

config = load_preprocessing_config("configs/preprocessing/default.yaml")

train_transform = build_transforms(config, train=True)    # MRIDataset(..., transform=train_transform)
infer_transform = build_transforms(config, train=False)   # InferenceEngine._run_pipeline uses this internally

sample = apply_transforms(infer_transform, {"image": "path/to/scan.nii.gz"})
tensor = sample["image"]  # torch.float32, shape (1, *config.target_shape)
```

`InferenceEngine._run_pipeline` (see [`inference/predictor.py`](src/ad_mri/inference/predictor.py))
wires this in for real — it validates the image (`datasets.validation.validate_image_for_pipeline`:
existence, readability, dimensionality, a non-degenerate affine), runs the identical `train=False`
pipeline, then a real cached-model forward pass — see "Inference" below.

## Model architecture

`ConvNetClassifier` ([`modeling/architectures.py`](src/ad_mri/modeling/architectures.py)) is a
configurable, progressively-deeper 2D/3D CNN: `[Conv -> Norm -> Activation -> Dropout -> Pool]
x num_blocks` -> global average pool -> FC head. Every structural choice (depth, channel widths,
normalization, activation, dropout, weight init) comes from
[`ModelConfig`](src/ad_mri/modeling/model_config.py)
([`configs/model/cnn3d_default.yaml`](configs/model/cnn3d_default.yaml)) — `num_classes` has no
default; it must match your dataset's label taxonomy. Full architecture diagram, layer-by-layer
explanation, tensor shape flow, and real (not hand-computed) parameter/FLOPs numbers:
[`docs/model_architecture.md`](docs/model_architecture.md).

```bash
# Layer-by-layer summary: shapes, parameter count, estimated MACs/FLOPs
ad-mri model summary --model-config configs/model/cnn3d_default.yaml
```

```python
from ad_mri.modeling.model_config import load_model_config
from ad_mri.modeling.factory import build_model
from ad_mri.modeling.checkpoint import save_checkpoint, load_checkpoint

config = load_model_config("configs/model/cnn3d_default.yaml")
model = build_model(config)                          # nn.Module, weights already initialized
save_checkpoint(model, config, "model.pt")            # bundles weights + architecture + a content hash
model, config, metadata = load_checkpoint("model.pt")  # self-describing: no need to already know the architecture
```

## Training pipeline

`Trainer` ([`training/trainer.py`](src/ad_mri/training/trainer.py)) owns the full training loop —
data loading, optimization, checkpointing, MLflow logging — and is the **only** place training
happens: nothing in `api/`/`inference/` imports from `training/`, and `training/` never imports
from `api/`/`inference/`. Runnable from a real CLI, not just a notebook:

```bash
# Equivalent invocations — both call the same run_training() entry point:
python -m ad_mri.training.train \
    --model-config configs/model/cnn3d_default.yaml \
    --training-config configs/training/default.yaml \
    --preprocessing-config configs/preprocessing/default.yaml \
    --train-manifest data/splits/<name>/train.csv \
    --val-manifest data/splits/<name>/val.csv

ad-mri train --model-config ... --training-config ... --train-manifest ... --val-manifest ...

# Resume an interrupted or completed run:
ad-mri train ... --resume-from artifacts/checkpoints/<run-name>/last.pt
```

What it does, per epoch: trains, validates, computes accuracy/precision/recall/F1 (macro +
per-class) and ROC-AUC where applicable (`evaluation/metrics.py`), steps the configured LR
scheduler, checks early stopping, and saves `last.pt` (every epoch, per
`save_every_n_epochs`) and `best.pt` (whenever the monitored metric improves) — each checkpoint
bundles model weights **and** optimizer/scheduler/scaler state **and** the full
model/training/preprocessing config **and** a SHA-256 of the exact train/val manifest files used,
so a checkpoint is fully self-describing and traceable (requirement: save dataset version +
preprocessing version alongside the model). At the end of a run it renders training-curve and
confusion-matrix PNGs and (if enabled) logs params/metrics/artifacts to MLflow — MLflow is
supplementary tracking: an unreachable tracking server logs a warning and training proceeds,
it never aborts a run.

Configurable via [`configs/training/default.yaml`](configs/training/default.yaml)
(`TrainingConfig`): batch size, epochs, optimizer (`adam`/`adamw`/`sgd`), LR scheduler
(`none`/`step`/`cosine`/`plateau`), weight decay, gradient clipping, mixed precision (CUDA only —
harmless no-op on CPU), early-stopping patience/metric/mode, and class-imbalance strategy
(`none` / `sampler` — `WeightedRandomSampler` / `loss_weights` — weighted `CrossEntropyLoss`, both
backed by `data_pipeline.class_balance` from Phase 2). A bad/corrupted sample encountered mid-epoch
is logged and skipped, not fatal; a non-finite (NaN/Inf) loss *is* fatal (`TrainingError`) rather
than silently corrupting the model. `core.seeding.set_global_seed` plus a `deterministic` flag
(stricter cudnn/algorithm determinism, off by default — costs throughput) cover reproducibility.

## Model selection & hyperparameter optimization

Subject-level, stratified K-fold cross-validation and Optuna-based hyperparameter search — see
[`docs/model_selection.md`](docs/model_selection.md) for the full train/val/CV/test-set
distinction this is built around. **The test set is never an input to any command in this
section** — `data cross-validate`/`cv run`/`cv search` have no `--test-manifest` flag; there is no
code path here that can read it.

```bash
# Build K stratified, subject-level, leakage-safe CV folds from the development
# pool (train.csv + val.csv) — reuses the same leakage check as the Phase 2 split.
ad-mri data cross-validate \
    --train-manifest data/splits/<name>/train.csv --val-manifest data/splits/<name>/val.csv \
    --output-dir data/splits/<name>/cv --n-folds 5 --seed 42

# Cross-validate one configuration: trains one model per fold (reusing Trainer
# unchanged), reports mean/std/95% CI per metric, writes a comparison report.
ad-mri cv run --model-config configs/model/cnn3d_default.yaml \
    --training-config configs/training/default.yaml --cv-dir data/splits/<name>/cv

# Hyperparameter search: every trial is scored by K-fold CV (not one split) over
# learning rate, batch size, optimizer, weight decay, dropout, and architecture
# parameters (base_channels, num_blocks) — configs/hpo/default.yaml. Selects the
# best trial by a predefined metric (never test-set accuracy) and writes
# best_model.yaml / best_training.yaml, ready to feed straight into `ad-mri train`.
ad-mri cv search --model-config configs/model/cnn3d_default.yaml \
    --training-config configs/training/default.yaml --hpo-config configs/hpo/default.yaml \
    --cv-dir data/splits/<name>/cv --output-dir artifacts/hpo_reports/ad-mri-hpo
```

An invalid sampled hyperparameter combination (e.g. `num_blocks` too deep for `input_shape`) is
caught per-trial (`ConfigurationError`) and marked failed rather than aborting the whole search.
MLflow nests three levels deep — study → trial → fold — so every fold's full Phase 5 tracking
(checkpoints, curves, confusion matrix) is preserved alongside each trial's CV summary and the
study's best-trial selection.

## Evaluation

Standalone evaluation on a held-out dataset — see [`docs/evaluation.md`](docs/evaluation.md) for
the full metric/report reference. **This is the only command meant to ever read `test.csv`**: run
it exactly once, after every modeling decision above is frozen.

```bash
# Equivalent invocations — both call the same run_evaluation() entry point:
ad-mri evaluate --model artifacts/checkpoints/final-model/best.pt \
    --dataset data/splits/<name>/test.csv \
    --preprocessing-config configs/preprocessing/default.yaml \
    --output-dir artifacts/evaluation/final-model

python -m ad_mri.evaluation.evaluate --model artifacts/checkpoints/final-model/best.pt \
    --dataset data/splits/<name>/test.csv
```

Computes accuracy, precision/recall/specificity/F1 (macro, weighted, and per-class), ROC-AUC,
PR-AUC, and a full confusion matrix (`evaluation.metrics` — the same function the training loop
uses, so no second implementation), plus a non-parametric bootstrap 95% confidence interval on
each headline metric (`evaluation.statistics.bootstrap_metric_ci`, since a single test set has no
folds to summarize across a t-distribution). Class labels are read back from the checkpoint's own
training provenance, never re-derived from the test manifest, and every output is labeled by class
**name**. `--threshold` overrides the default argmax decision rule for binary models only. Writes
a confusion-matrix PNG, ROC-curve PNG, PR-curve PNG, text classification report, JSON report, CSV
report, per-sample predictions CSV, and a markdown summary — every one of them stamped with the
model version, config hashes, and dataset SHA-256 (requirement: "store evaluation results with
model version"). A preprocessing config that doesn't hash-match what the checkpoint was trained
with is rejected outright rather than silently producing invalidated numbers.

## Explainability

LIME and SHAP explanations for one checkpoint's prediction on one scan — see
[`docs/explainability.md`](docs/explainability.md) for the full method reference and, importantly,
its **Limitations** section (read before trusting an explanation) and its **Model explanation vs.
medical interpretation** section (this system makes no clinical claims, explanations included).

```bash
# Equivalent invocations — both call the same run_explanation() entry point:
ad-mri explain run --model artifacts/checkpoints/final-model/best.pt \
    --image data/raw/subject-042/scan.nii.gz \
    --background-manifest data/splits/<name>/train.csv \
    --methods lime,shap

python -m ad_mri.explainability.explain --model artifacts/checkpoints/final-model/best.pt \
    --image data/raw/subject-042/scan.nii.gz

# Aggregate |SHAP value| across several samples into one downsampled region-importance
# grid for one class — "global" only in this multi-sample, downsampled form:
ad-mri explain global-shap --model artifacts/checkpoints/final-model/best.pt \
    --samples-manifest data/splits/<name>/val.csv \
    --background-manifest data/splits/<name>/train.csv \
    --target-class AD
```

`POST /api/v1/explanations` can also attach an optional, off-by-default `narrative_summary` field
— a plain-language rendering of the numeric output above via the OpenAI API
(`explainability.llm_summary`, requires the `llm` extra + `OPENAI_API_KEY`). Only aggregate
region-importance numbers already computed locally are ever sent — never image data, a file path,
or a patient identifier — and `disclaimer` (not this field) remains the response's authoritative
safety statement regardless. See [`docs/explainability.md`](docs/explainability.md) §7.5.

**LIME** (`explainability.lime_explainer`) perturbs superpixels on one representative 2D slice
(quickshift/SLIC-segmented) and re-inserts each perturbed slice into a *copy* of the real 3D
volume — every candidate is scored by a genuine forward pass through the real 3D model, never a
2D proxy — then fits a local linear surrogate whose coefficients rank the superpixels by
contribution. **SHAP** (`explainability.shap_explainer`, `shap.GradientExplainer` by default)
computes per-voxel expected-gradient attributions directly on the full input tensor — no
flattening, no slicing needed for the computation itself — against a versioned background sample
set drawn from a manifest. Both keep preprocessing identical to inference
(`datasets.transforms.build_transforms(..., train=False)`) and are seed-reproducible. Every call
writes a `manifest.json` recording model version, input ID, preprocessing version, the
method(s)/parameters used, and every generated artifact (overlay PNGs, the raw `.npy` SHAP values,
`prediction.json`) — see `docs/explainability.md` §8.

## Inference

The production inference engine — see [`docs/inference.md`](docs/inference.md) for the full
pipeline/requirement reference and [`docs/api.md`](docs/api.md) for the REST API it's served
behind. One implementation (`inference.predictor.InferenceEngine.predict()`) serves the CLI,
`python -m ad_mri.inference.predict`, and `POST /api/v1/predictions` alike.

```bash
# Equivalent invocations — both call the same run_prediction() entry point:
ad-mri predict --model artifacts/checkpoints/final-model/best.pt \
    --image data/raw/subject-042/scan.nii.gz

python -m ad_mri.inference.predict --model .../best.pt --image .../scan.nii.gz

# The API (needs INFERENCE_MODEL_PATH set — see .env.example):
curl -X POST http://localhost:8000/api/v1/predictions -F "file=@scan.nii.gz"
curl http://localhost:8000/api/v1/models
curl http://localhost:8000/api/v1/ready
```

MRI file → validation → preprocessing → **cached** model loading
(`inference.model_loader.ModelLoader`, keyed by checkpoint+preprocessing+device — a checkpoint is
read from disk at most once) → inference under `torch.inference_mode()` (no gradients, ever) →
optional calibration (off by default — plain softmax; `--calibration-config` enables temperature
scaling) → prediction + confidence → optional LIME/SHAP explanation (`--explain`, reusing the
already-loaded checkpoint via `ExplainabilityService.from_loaded_model` — no second disk load). The
full pipeline runs with a configurable `--timeout-seconds` (enforced via a worker-thread
`Future.result(timeout=...)`, raising `InferenceError` rather than hanging), returns the clean,
deterministic `PredictionResult`/`PredictionResponse` structure shown in `docs/inference.md` §3,
and never logs an uploaded file's original filename, a raw file path, or a validation exception's
message — only a one-way-hashed image reference and the exception's type name. A model/preprocessing
mismatch is rejected at load time (`ConfigurationError`), the same guarantee `ad-mri evaluate` and
`ad-mri explain` give.

## REST API

The production FastAPI backend — see [`docs/api.md`](docs/api.md) for the full reference. Every
route calls `InferenceEngine`/`ExplainabilityService` directly; no business/model logic lives in a
route handler.

```bash
curl -X POST http://localhost:8000/api/v1/predictions -F "file=@scan.nii.gz"
curl -X POST "http://localhost:8000/api/v1/explanations?methods=lime,shap" -F "file=@scan.nii.gz"
curl http://localhost:8000/api/v1/models
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/version
```

- **Async where it matters**: upload I/O is natively `async`; the CPU-bound
  predict/explain call runs via `asyncio.to_thread` so one in-flight request never blocks another.
- **Centralized exception handling**: routes never catch the inference/explainability layer's
  exceptions themselves — `api.exception_handlers` maps every domain exception to a status code and
  a consistent `{"detail", "error_code"}` body, once, application-wide.
- **Auth architecture**: an API-key scheme (`API_KEY_REQUIRED`), off by default, required in
  `APP_ENV=prod`, plus a Phase 14 scoped multi-key authorization layer (`API_KEYS`, `ApiScope`:
  `predict`/`label`) layered on top — a key can be limited to just `predict` without `label`, with
  a real 401-vs-403 distinction; a documented, narrow seam (`api.security` + `api.deps
  .require_scope`) for a stronger scheme (OAuth2/JWT) later. See [`docs/security.md`](docs/security.md).
- **Rate limiting architecture**: a real, enforced-by-default in-memory sliding-window limiter
  (`RATE_LIMIT_PER_MINUTE`), with a documented seam for a Redis-backed one in a multi-instance
  deployment.
- **Secure file handling**: uploads are streamed to a generically-named temp file (never the
  client's original filename, and path-traversal-safe by construction — adversarially tested),
  capped at `MAX_UPLOAD_SIZE_MB` (413 if exceeded), rejected if the volume's own header declares a
  voxel count beyond a configurable ceiling (a decompression-bomb guard, Phase 14), and always
  cleaned up (`finally: shutil.rmtree(...)`) — success or failure.
- **Secure response headers**: `SecurityHeadersMiddleware` (Phase 14) sets `X-Content-Type-Options`,
  `X-Frame-Options`, a route-appropriate `Content-Security-Policy`, and more on every response —
  including one produced by a genuinely unhandled exception, a real Starlette gap found and closed
  during Phase 14's review (see [`docs/security.md`](docs/security.md) §5).
- **CORS**: the outermost middleware, so even a rate-limited/error response carries correct
  headers for a browser client; explicit origin/method/header allow-lists, no wildcards.
- **OpenAPI**: a real `securityScheme` for the API key (the `/docs` "Authorize" button targets it),
  tag descriptions, and every schema documented.
- **ML safety**: every prediction/explanation response carries an inline `disclaimer` field
  distinguishing a model prediction from a clinical diagnosis (and, for explanations, association
  from causality) — see [`docs/responsible_ai.md`](docs/responsible_ai.md).

## Test

```bash
pytest                                    # full suite
pytest --cov=ad_mri --cov-report=term-missing   # with coverage
pytest tests/unit tests/api tests/data_validation tests/explainability  # unit only (CI's "unit-tests" job)
pytest tests/integration                  # integration only (CI's "integration-tests" job)
```

Or via `make`: `make test` / `make test-unit` / `make test-integration` / `make test-cov`
(Windows: run the `pytest` commands directly, or use Git Bash / WSL for `make`).

## Lint, format, type-check

```bash
ruff check src tests
black src tests
mypy src
bandit -c pyproject.toml -r src   # SAST — CI's "security" job
pip-audit --skip-editable         # dependency vulnerabilities — CI's "security" job
```

Install git hooks once so these run automatically pre-commit:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml --profile tools run --rm migrate
```

Brings up four services — `api`, an async `worker`, an `mlflow` tracking server, plus `db`
(PostgreSQL) and `redis` — each with its own health check, resource limits, and restart policy;
see [`docs/deployment.md`](docs/deployment.md) §1-2. `migrate` is a one-off Alembic job
(`profiles: [tools]`, so it never starts with a plain `up`) — run it once against a fresh
database. `docker-compose.prod.yml` is the production counterpart: never builds (pulls a pinned,
already-CI-scanned image tag instead), publishes only the API's port, and runs every container
hardened (read-only root filesystem, dropped capabilities, no new privileges) — see
[`docs/deployment.md`](docs/deployment.md) §2.

## Database migrations

Schema is defined in `src/ad_mri/db/models.py` and versioned via Alembic
(`src/ad_mri/db/migrations/`) — five tables (`model_versions`, `prediction_requests`,
`prediction_results`, `explanation_records`, `system_events`) backing the audit trail described in
[`docs/database.md`](docs/database.md). Once PostgreSQL is running and `.env` points at it:

```bash
alembic upgrade head
```

Never stores raw MRI data or secrets — only redacted references, hashes, and metadata; see
[`docs/database.md`](docs/database.md) §6 for exactly what is enforced. The API writes to this
audit trail best-effort (`api.audit`): an unreachable database never fails a `/predictions` or
`/explanations` request, it only degrades observability. Set `AUDIT_ENABLED=false` to disable
writes entirely.

## CI/CD & production deployment

Full details: [`docs/deployment.md`](docs/deployment.md). In brief:

- **CI** (`.github/workflows/ci.yml`, on every push/PR): lint (`ruff`/`black`) → type check
  (`mypy`) → security checks (`bandit` + `pip-audit`) → unit tests → integration tests → build +
  Trivy-scan every image (`api`/`worker`/`training`/`mlflow`) → push to GHCR (default branch
  only, using the workflow's own auto-issued token — never a stored secret).
- **CD** (`.github/workflows/cd.yml`): `development` deploys automatically once CI passes;
  `staging`/`production` are always a deliberate, reviewer-gated `workflow_dispatch` promoting
  one already-built image tag — never a fresh rebuild. A failed post-deploy health check
  **automatically rolls back** to the previously-deployed tag.
- **Secrets**: never committed, anywhere — CI uses only its auto-issued token; CD reads
  per-environment GitHub Environment secrets; the running application reads a secrets file that
  lives outside this repository on the target host. See
  [`docs/deployment.md`](docs/deployment.md) §6 and `deploy/environments/` for the non-secret
  per-environment config templates.

> This model has not been clinically validated. Deploying it to any environment does not change
> that — see the disclaimer at the top of this README.

## Monitoring

Full details: [`docs/monitoring.md`](docs/monitoring.md). In brief:

```bash
docker compose -f docker/docker-compose.yml --profile observability up -d   # Grafana :3000, Prometheus :9090
ad-mri monitor build-drift-reference --manifest data/splits/<name>/train.csv --output artifacts/monitoring/drift_reference.json
ad-mri monitor drift-report --reference artifacts/monitoring/drift_reference.json
ad-mri monitor register-baseline --model-version-id <id> --from-evaluation-report artifacts/evaluation/.../report.json
ad-mri monitor model-drift-report --model-version-id <id> --class-names AD,CN
```

- **System/model metrics**, real-time, via the existing `GET /metrics` (Prometheus): CPU/memory/
  GPU, request count/latency (throughput/error rate are PromQL over those, not extra series),
  prediction/class/confidence distributions, input-validation/preprocessing failure rates, model
  latency, and real-time intensity/image-quality signals — see `docs/monitoring.md` §2-3.
- **Data drift**: `ad-mri monitor build-drift-reference` (once, from the training manifest) then
  `ad-mri monitor drift-report` (offline, PSI + Kolmogorov-Smirnov against that baseline,
  including predicted-class distribution drift) — `docs/monitoring.md` §4.
- **Delayed model-performance monitoring**: `POST /api/v1/monitoring/ground-truth` /
  `ad-mri monitor record-ground-truth` attaches a true label to a *past* prediction whenever it
  actually becomes available (never assumed immediate); `ad-mri monitor model-drift-report`
  computes real accuracy/F1 against a registered baseline once ≥30 labeled predictions exist —
  `docs/monitoring.md` §5.
- **Retraining triggers**: every drift/degradation signal is a trigger for a human decision, never
  an automated retrain — `docs/monitoring.md` §8.

## Security, privacy & responsible AI

Full details: [`docs/security.md`](docs/security.md) (application security — the final,
requirement-by-requirement review) and [`docs/responsible_ai.md`](docs/responsible_ai.md) (ML
safety). In brief:

- **Verified clean, for real, as of this review**: 536/536 tests passing; `ruff`/`black --check`/
  `mypy` clean; `bandit -c pyproject.toml -r src` — 0 findings; `pip-audit --skip-editable` — 0
  known vulnerabilities.
- **Input hardening**: file-type/size validation, streamed upload size caps, a decompression-bomb
  guard on compressed NIfTI headers (`max_voxel_count`), and adversarially-tested path-traversal
  safety on every uploaded filename.
- **AuthN/AuthZ**: API-key authentication plus a scoped, multi-key authorization layer
  (`ApiScope`: `predict`/`label`) — a real 401 (unauthenticated) vs. 403 (authenticated, wrong
  scope) distinction, fully backward compatible with the legacy single-key mode.
- **Secure response headers** on every response, including one from a genuinely unhandled
  exception (a real Starlette architectural gap — found, verified, and closed, not assumed away).
- **Secrets, audit logging, rate limiting**: no secret ever committed; every prediction/explanation/
  system event persisted to a best-effort audit trail; per-key/IP rate limiting honoring a
  reconfigured API-key header name.
- **ML safety**: an inline `disclaimer` field on every prediction/explanation response
  distinguishing a model prediction from a clinical diagnosis and an explanation from a causal
  claim; exposed model version + content hashes; a calibrated-or-disclosed confidence signal (full
  probability distribution, never a bare label); documented limitations, dataset limitations, and
  bias/generalization risks.
- **Residual risks, documented rather than hidden**: no TLS termination in-process (expected — a
  deployment concern), no encryption at rest, no OAuth2/JWT/per-user identity, per-process (not
  distributed) rate limiting — see `docs/security.md` §7 for the complete list.

## Directory guide

| Package | Responsibility |
|---|---|
| `core` | Environment-driven `Settings` (pydantic-settings), structlog configuration, the exception hierarchy, shared enums/constants (including `ApiScope`), `disclaimers.py` (the ML-safety prediction/explanation disclaimer text, Phase 14), and reproducibility seeding. Nothing elsewhere hardcodes a path, credential, or secret — it all flows through here. |
| `data_pipeline` | **Implemented.** Per-dataset config (`dataset_config.py`), scan discovery + optional DICOM→NIfTI conversion (`ingestion.py`, `conversion.py`), ingestion-time dataset validation (`validation.py`), manifest generation + class-distribution reporting (`manifest.py`), class-weight/sampler utilities (`class_balance.py`), leakage-safe subject-level holdout splitting (`splitting.py`), and leakage-safe subject-level stratified K-fold cross-validation folding (`cross_validation.py`). Classical spatial preprocessing (skull-strip / atlas registration in `preprocessing.py`) remains a stub — a heavier neuroimaging step, distinct from the MONAI pipeline below. |
| `datasets` | **Implemented.** `MRIDataset` (PyTorch `Dataset` over a manifest/split CSV); `preprocessing_config.py` (`PreprocessingConfig`, versioned/hashable); `transforms.py` (`build_transforms` — the one preprocessing pipeline shared by training and inference, see "Preprocessing pipeline" above); `custom_transforms.py` (safe loading, quality control); `validation.py` (single-image pre-pipeline checks); `visualization.py` (preprocessing sanity-check plots). |
| `modeling` | **Implemented.** `model_config.py` (`ModelConfig`, versioned/hashable); `architectures.py` (`ConvNetClassifier`, an architecture registry); `factory.py` (`build_model`); `summary.py` (layer-by-layer shapes, parameter count, MACs/FLOPs estimate); `checkpoint.py` (self-describing save/load — weights + architecture config bundled together). Loss functions (`losses.py`) remain a stub for the training phase. |
| `training` | **Implemented.** `training_config.py` (`TrainingConfig`); `trainer.py` (`Trainer` — the full loop); `callbacks.py` (`EarlyStopping`); `visualization.py` (training-curve plots); `train.py` (CLI entry point); `mlflow_utils.py` (shared run-management, incl. nested runs); `cross_validation.py` (K-fold training + mean/std/CI summary, reusing `Trainer`); `hpo_config.py`/`hpo.py` (Optuna search over LR/batch size/optimizer/weight decay/dropout/architecture, each trial scored by CV); `reporting.py` (CV/HPO comparison reports). |
| `evaluation` | **Implemented.** `metrics.py` (accuracy/precision/recall/specificity/F1 macro+weighted+per-class, ROC-AUC, PR-AUC where applicable, binary threshold rule); `confusion_matrix.py` (computation + PNG rendering); `curves.py` (ROC/PR curve computation + PNG rendering, one-vs-rest for multiclass); `statistics.py` (t-distribution CI across folds/trials, plus non-parametric bootstrap CI for a single test set); `report.py` (`EvaluationReport` orchestration + JSON/CSV/markdown rendering); `evaluate.py` (the standalone `ad-mri evaluate` CLI, see "Evaluation" above). Back both the training loop's per-epoch validation and cross-validation/HPO/standalone-evaluation reporting alike — one implementation, not several. The citation-backed literature-comparison table (`comparison.py`) remains a stub for a later phase. |
| `explainability` | **Implemented.** `config.py` (`LimeConfig`/`ShapConfig`, versioned/hashable); `slicing.py` (2D-slice extraction/insertion for 3D volumes); `sampling.py` (deterministic background/sample-batch loading); `lime_explainer.py`/`shap_explainer.py` (real LIME/SHAP against the real model, plus `compute_global_shap_importance`); `overlays.py` (heatmap PNG rendering); `service.py` (`ExplainabilityService` — the single "image + model -> prediction + explanations" entry point); `explain.py` (CLI). See `docs/explainability.md`. |
| `inference` | **Implemented.** `model_loader.py` (`ModelLoader`/`CachedModel` — checkpoint loading + in-memory caching, keyed by checkpoint+preprocessing+device); `calibration.py` (`CalibrationConfig`, temperature scaling, offline fitting); `predictor.py` (`InferenceEngine` — the full validate→preprocess→infer→calibrate→predict pipeline, timeout-bounded, `PredictionResult`, plus loud `generate_explanation()` for `POST /explanations`); `predict.py` (CLI entry point). Deliberately independent of the (still-stub) MLflow Model Registry — loads a checkpoint directly, the same versioned artifact every other phase does. |
| `model_registry` | Thin MLflow Model Registry client (register/promote/resolve model versions) — distinct from `db`'s much narrower `model_versions` audit table below. |
| `api` | **Implemented** — see `docs/api.md`, `docs/security.md`. Routers: `predictions`, `explanations`, `models`, `meta` (versioned `/health`, `/ready`, `/version`), `monitoring` (`/monitoring/ground-truth`), `health` (unversioned probes + `/metrics`). `schemas/` (Pydantic contracts, one module per resource, each response carrying an ML-safety `disclaimer` field on the two prediction/explanation ones). `middleware/` (request ID, access log, rate limiting, `security_headers` — Phase 14). `exception_handlers.py` (centralized domain-exception → HTTP mapping, plus a Phase 14 catch-all for genuinely unmapped exceptions). `security.py` (the API-key OpenAPI scheme). `upload_utils.py` (shared secure/size-bounded upload handling). `deps.py` (DB session, API-key auth + `ApiScope`-based authorization via `require_scope`, the process-cached `InferenceEngine`). `audit.py` (best-effort audit-trail persistence, see `docs/database.md`). |
| `worker` | Celery app + tasks for async explainability/batch jobs, run in a separate container from the API. |
| `db` | **Implemented** — see `docs/database.md`, `docs/monitoring.md`. `models.py` (the five Phase 11 audit-trail tables plus Phase 13's `input_statistics`/`ground_truth_labels` and `model_versions.baseline_metrics`); `types.py` (`GUID`, a PostgreSQL/SQLite-portable UUID column type); `privacy.py` (enforced, tested guards against storing raw filenames/secrets); `repositories.py` (typed CRUD, one class per table); `audit_service.py`/`monitoring_service.py` (transactional units of work — audit trail vs. drift/quality monitoring, kept as separate concerns); `session.py` (lazy, pooled engine/session management); `migrations/` (Alembic, wired to `Settings`). |
| `storage` | **Implemented.** `object_storage.py` — the object-storage seam for large artifacts (never stored in PostgreSQL): `LocalFilesystemObjectStorage` is real; `S3ObjectStorageClient` is a documented, unimplemented seam (mirrors `model_registry.client`). |
| `monitoring` | **Implemented** — see `docs/monitoring.md`. `metrics.py` (every Prometheus series); `system_metrics.py` (CPU/memory/GPU, pull-based); `drift.py` (real-time image statistics + offline PSI/KS data-drift reports); `model_drift.py` (delayed, ground-truth-backed performance reports); `alert_state.py` (reflects the latest offline report verdict into a live gauge); `thresholds.py` (every alert threshold, once); `health_checks.py` (reusable dependency checks, used by `/readyz`). |
| `cli` | The `ad-mri` Typer CLI — `serve`, `data`, `model`, `cv`, `train`, `evaluate`, and `monitor` (`build-drift-reference`/`drift-report`/`model-drift-report`/`register-baseline`/`record-ground-truth`) are functional; `preprocess`/`explain`/`register-model` remain documented stubs. |

## Roadmap

See "Recommended Implementation Order" in
[`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md): data pipeline → baseline
training/evaluation → literature comparison → explainability → inference API hardening → MLOps →
monitoring → security/deployment → documentation. **Phase 14 (security, privacy & responsible AI)
was the last phase in this order** — see [`docs/security.md`](docs/security.md) and
[`docs/responsible_ai.md`](docs/responsible_ai.md). Anything listed as not-yet-implemented above
(the model registry, S3 object storage, Kubernetes manifests, Alertmanager delivery, Evidently
drift reports, TLS/at-rest encryption, OAuth2/per-user identity) remains exactly that: a real,
documented seam, not a hidden gap.
