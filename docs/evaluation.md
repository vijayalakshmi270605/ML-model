# Evaluation

Status: **Phase 7 — comprehensive evaluation framework implemented.** No performance number in
this document is a claim about real-world diagnostic accuracy — everything here is a methodology
description, verified with synthetic data in tests. See
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0, and
[`docs/model_selection.md`](model_selection.md) for how the test set stays untouched until this
point.

## 1. What this is (and is not) for

`ad-mri evaluate` / `python -m ad_mri.evaluation.evaluate` is the **only** command in this
repository meant to read `test.csv`. Every earlier command (`ad-mri train`, `ad-mri cv run`,
`ad-mri cv search`) only ever accepts train/val manifests — there is no `--test-manifest` flag
anywhere upstream of this one, so there is no code path that could have optimized against
test-set performance (requirement: "never use the test set during model selection").

Run it **exactly once**, after every architecture/hyperparameter decision from
[`docs/model_selection.md`](model_selection.md) is frozen and the final model has been retrained
on the full development pool (`train.csv + val.csv`).

## 2. Metrics computed

`evaluation.metrics.compute_classification_metrics` — the same function the training loop uses
for per-epoch validation metrics (Phase 5), so a test-set number is never computed by a second,
subtly-different implementation:

| Metric | Scope |
|---|---|
| Accuracy | overall |
| Precision, Recall (sensitivity) | macro, weighted, per-class |
| Specificity | macro, weighted, per-class (from the confusion matrix: `TN / (TN + FP)`, one-vs-rest) |
| F1-score | macro, weighted, per-class |
| ROC-AUC | binary (`roc_auc`) or one-vs-rest macro-averaged (`roc_auc_macro`) |
| PR-AUC (average precision) | binary (`pr_auc`) or one-vs-rest per-class + macro (`pr_auc_macro`) |
| Confusion matrix | full `[n_classes, n_classes]` matrix, rows=true, cols=predicted |

ROC-AUC/PR-AUC are computed only when every class is present at least once in the evaluated
set — scikit-learn cannot define either curve for a class with zero positive samples, so they are
omitted, never fabricated as `0.0`/`NaN`.

**Confidence intervals** (`evaluation.statistics.bootstrap_metric_ci`): a single test set is one
dataset, not a handful of independent CV folds, so the t-distribution approach used for
cross-validation summaries (`docs/model_selection.md` §2) doesn't apply. Instead, a **non-parametric
percentile bootstrap** (1000 resamples by default, reproducible via `--seed`) resamples test-set
rows with replacement and recomputes the metric each time; the reported interval is the 2.5th/97.5th
percentile of that distribution (95% by default, configurable via `--confidence-level`). Computed
for `accuracy`, `precision_macro`, `recall_macro`, `specificity_macro`, `f1_macro`, and the headline
AUC metric — the numbers a reader is most likely to act on, not every one of the ~5×n_classes
metrics in the flat dict.

## 3. Binary decision threshold

For a 2-class model, `--threshold` overrides the default argmax decision rule
(`evaluation.metrics.apply_binary_threshold`): a sample is predicted positive (`class_names[1]`)
when its predicted probability is `>= threshold`, else negative. `threshold=0.5` is mathematically
equivalent to argmax over a 2-class softmax — it's a strict generalization, not a separate code
path. Passing `--threshold` for a model with more than two classes is rejected
(`ConfigurationError`) — there is no single meaningful threshold over more than one decision
boundary.

## 4. Class labels

Every checkpoint written by `training.trainer.Trainer` (Phase 5+) bundles the exact
index→class-name mapping used during training (`extra["class_names"]`, sorted at training time by
`Trainer._resolve_label_mapping`). `ad-mri evaluate` reads that mapping back out of the checkpoint
— it never re-derives a mapping from the test manifest's own label column, which could silently
disagree in order or content with what the model was actually trained against. A checkpoint saved
without this provenance (e.g. hand-built outside the training pipeline) is rejected with a clear
`ConfigurationError` rather than guessing. Every output — confusion-matrix axes, per-class metric
keys (`precision_AD`, not `precision_1`), curve legends, the classification report, the predictions
CSV — is labeled by class **name**, never a bare integer index.

## 5. Reproducibility guarantees

- **Preprocessing match check**: the checkpoint records the content hash of the
  `PreprocessingConfig` it was trained with. `ad-mri evaluate` recomputes the hash of the
  `--preprocessing-config` it's given and refuses to run (`ConfigurationError`) if they don't
  match — evaluating with different resampling/normalization than training would silently
  invalidate every metric below.
- **Shape match check**: `preprocessing_config.target_shape` must equal `model_config.input_shape`
  (same check `Trainer` applies at construction).
- **Deterministic preprocessing only**: evaluation uses `datasets.transforms.build_transforms(...,
  train=False)` — the identical deterministic pipeline used for validation/inference, with no
  augmentation.
- **Deterministic bootstrap**: identical predictions + `--seed` always produce identical confidence
  intervals.
- **Everything is stamped with model version**: the JSON/CSV/markdown reports all carry
  `model_version`, `model_config_hash`, `checkpoint_path`, `preprocessing_version`,
  `preprocessing_config_hash`, the dataset manifest path + SHA-256, and an ISO-8601 timestamp
  (requirement: "store evaluation results with model version").

## 6. Deliverables

`ad-mri evaluate --model <checkpoint> --dataset <test.csv> [--output-dir ...]` writes, under
`--output-dir` (default `artifacts/evaluation/<checkpoint-stem>-<timestamp>`):

| File | Contents |
|---|---|
| `confusion_matrix.png` | annotated heatmap |
| `roc_curve.png` | ROC curve(s) — one for binary, one-vs-rest per class for multiclass |
| `pr_curve.png` | Precision-Recall curve(s), same convention |
| `classification_report.txt` | scikit-learn's per-class text report |
| `evaluation_report.json` | every metric, confidence interval, the confusion matrix, and the full curve point data — machine-readable, versioned |
| `evaluation_report.csv` | one row per metric: `metric, value, ci_lower, ci_upper, confidence_level` |
| `predictions.csv` | one row per evaluated sample: `subject_id, true_label, predicted_label, correct, proba_<class>...` — the raw basis every metric above was computed from, for audit/re-analysis |
| `evaluation_report.md` | human-readable summary tying everything above together |

Training/validation loss and metric curves (`training_curves.png`, requirement items 4-5) are
produced once, at training time, by `training.visualization.plot_training_curves` (Phase 5) and
saved into the training run's directory (`artifacts/checkpoints/<run>/training_curves.png`) and as
an MLflow artifact — evaluation doesn't recompute them since they describe the training run, not
the test-set pass; they are referenced by the checkpoint's run directory, not duplicated here.

## 7. Reproducible command

```bash
ad-mri evaluate \
    --model artifacts/checkpoints/final-model/best.pt \
    --dataset data/splits/<name>/test.csv \
    --preprocessing-config configs/preprocessing/default.yaml \
    --output-dir artifacts/evaluation/final-model \
    --confidence-level 0.95 \
    --n-bootstrap 1000 \
    --seed 42

# equivalently:
python -m ad_mri.evaluation.evaluate \
    --model artifacts/checkpoints/final-model/best.pt \
    --dataset data/splits/<name>/test.csv
```

Identical `--model`, `--dataset`, `--preprocessing-config`, `--threshold`, and `--seed` always
reproduce byte-identical `evaluation_report.json`/`.csv` output (verified in
`tests/unit/test_evaluation_report.py::test_render_is_deterministic_given_the_same_report`).
