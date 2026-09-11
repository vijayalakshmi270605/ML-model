# ML and system monitoring

Status: **Phase 13 — system, model, and data/model-drift monitoring implemented.** Nothing here
is a clinical claim: this system is research / decision-support tooling, **not a certified
medical device** — see [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0. A
"model drift detected" alert is a statistical statement about this deployment's own predictions
against its own registered baseline — it is not, and must never be read as, a clinical
determination about any individual scan or patient.

## 1. Architecture

Three layers, each answering a different question:

| Layer | Question it answers | Where |
|---|---|---|
| **Real-time metrics** (Prometheus, pull-based) | "What is happening right now?" | `monitoring.metrics`, scraped via `GET /metrics` |
| **Durable per-prediction records** (PostgreSQL) | "What happened, and can I look it up later?" | `input_statistics`, `ground_truth_labels` (Phase 13); `prediction_requests`/`prediction_results`/`model_versions` (Phase 11) |
| **Offline reports** (`ad-mri monitor ...` CLI) | "Has the input distribution or the model's real performance changed?" | `monitoring.drift`, `monitoring.model_drift` |

Nothing in this design assumes a request-time answer to "is this a good prediction" - see §4.

## 2. System metrics

`monitoring.system_metrics.sample_system_metrics()`, called once per `/metrics` scrape (pull-based
- see that module's docstring for why not a background thread):

| Requirement | Metric | Type |
|---|---|---|
| CPU | `ad_mri_process_cpu_percent`, `ad_mri_system_cpu_percent` | Gauge |
| Memory | `ad_mri_process_memory_bytes`, `ad_mri_system_memory_percent` | Gauge |
| GPU utilization | `ad_mri_gpu_utilization_percent{device}` | Gauge (best-effort - absent, not zero, on a CPU-only deployment) |
| Request count | `ad_mri_http_requests_total{method,path,status_code}` | Counter |
| Latency | `ad_mri_http_request_duration_seconds{method,path}` | Histogram |
| Throughput | *(derived)* `rate(ad_mri_http_requests_total[5m])` | PromQL over the Counter above - not a separate series, see `monitoring.metrics`'s module docstring for why |
| Error rate | *(derived)* `rate(...{status_code=~"5.."}[5m]) / rate(...[5m])` | PromQL, same reasoning |

## 3. Model metrics

Recorded inside `InferenceEngine.predict()` (`inference.predictor`) - the one place every
`POST /api/v1/predictions` call passes through; `POST /api/v1/explanations` runs its own,
independent forward pass and does not duplicate these (documented gap, §9):

| Requirement | Metric | Type |
|---|---|---|
| Prediction distribution | `ad_mri_class_probability{class_name}` - every class's probability, every request | Histogram |
| Class distribution | `ad_mri_predicted_class_total{predicted_class}` | Counter |
| Confidence distribution | `ad_mri_prediction_confidence` - the argmax probability | Histogram |
| Input validation failures | `ad_mri_inference_requests_total{outcome="validation_failed"}` | Counter |
| Preprocessing failures | `ad_mri_inference_requests_total{outcome="preprocessing_failed"}` | Counter |
| Model latency | `ad_mri_inference_duration_seconds` | Histogram |

`validation_failed` and `preprocessing_failed` were a single conflated `invalid_input` outcome
before this phase - split because they mean different things operationally (a validation failure
usually means a bad upload; a preprocessing failure usually means the pipeline/QC threshold
itself has an issue with an otherwise-valid file).

## 4. Data drift monitoring

**Real-time signal** (free - computed on data already in memory): `InferenceEngine._run_pipeline`
computes `monitoring.drift.compute_image_statistics()` on the *preprocessed* volume, right before
the forward pass - the exact array the model sees, not the raw upload (which may have a
completely different resolution/intensity range before preprocessing normalizes it). Recorded to
Prometheus (`ad_mri_input_mean_intensity`/`_std_intensity`/`_nonzero_fraction` - intensity
distribution and image-quality signals) and, best-effort, to `input_statistics`
(`api.monitoring.record_input_statistics`, mirroring `api.audit`'s never-fails-the-request
contract).

**Offline report** (`ad-mri monitor drift-report`, needs `train` extra - `scipy`):

1. Build a reference once, from the training distribution:
   `ad-mri monitor build-drift-reference --manifest <training manifest.csv>` - runs every
   reference image through the *same* preprocessing pipeline inference uses, saves a
   `DriftReference` (JSON: per-feature sampled values + the manifest's label distribution).
2. Compare a recent production window against it:
   `ad-mri monitor drift-report --reference <path> [--since-hours N]` - for each of
   `mean_intensity`/`std_intensity`/`p01_intensity`/`p50_intensity`/`p99_intensity`/
   `nonzero_fraction` ("feature distribution changes where feasible" - these six numeric
   summaries of the preprocessed volume are what's actually feasible without persisting raw
   arrays), computes:
   - **PSI** (Population Stability Index) - quantile-binned, industry-standard for exactly this
     "has this distribution shifted" question, no distributional assumptions.
   - **Two-sample Kolmogorov-Smirnov test** - a statistical significance check independent of
     PSI's arbitrary bin count; either one alone can flag a feature.
   - **Predicted-class distribution PSI** (categorical) - a shift in *what the model is
     predicting* is itself a drift signal, distinct from and complementary to the six intensity
     features above.

Why both PSI and KS: PSI is the standard, well-understood monitoring metric (`monitoring.thresholds`'s
0.1/0.25 bands are the conventional ones from the field PSI originates in — credit-risk-model
monitoring) but is sensitive to its bin count; a KS test gives an independent, bin-free
statistical read on the same question. A feature is flagged if *either* crosses its threshold.

## 5. Delayed model-performance monitoring (model drift)

**Ground truth is never assumed to be immediately available in production.** A brain MRI's true
diagnosis is often confirmed only after follow-up imaging, a specialist review, or a biopsy -
days to months after the prediction was made. This system's design reflects that directly:

- A prediction is **scored** (and monitored via §2-4) the moment it's made.
- A prediction is **evaluated** only once a true label is later recorded:
  - `POST /api/v1/monitoring/ground-truth` (`{request_id, true_class, label_source}`), or
  - `ad-mri monitor record-ground-truth --request-id ... --true-class ... --label-source ...`

  `label_source` is a fixed, enforced vocabulary (`clinician_review` / `follow_up_diagnosis` /
  `biopsy_confirmed` / `other`) - never free text, so this endpoint can never become a place PHI
  accidentally gets typed (`db.models.GroundTruthLabelRecord`'s docstring).
- `ad-mri monitor model-drift-report --model-version-id <id> --class-names AD,CN` joins every
  labeled prediction for that model version (`db.repositories.GroundTruthLabelRepository
  .list_labeled_predictions_for_model_version` - one query, not an application-level join),
  computes real accuracy/precision/recall/F1 (`evaluation.metrics.compute_classification_metrics`
  - the *same* function the training/evaluation pipeline uses, not a second implementation), and
  compares against the model version's registered baseline
  (`ad-mri monitor register-baseline --model-version-id <id> --from-evaluation-report
  <report.json>` - typically that version's own held-out test-set metrics from `ad-mri evaluate`).

**`MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT = 30`** (`monitoring.thresholds`) is a hard floor: with
fewer labels than that, the report returns `insufficient_samples=True` and **computes no verdict
at all**, rather than a technically-real-but-statistically-meaningless accuracy from a handful of
labels. The alternative - silently lowering the bar because labels are scarce - would trade a
true "we don't know yet" for a false "everything's fine" or a false alarm; this project's whole
disclaimer posture exists specifically to avoid exactly that kind of confident-sounding wrong
answer.

## 6. Alert thresholds

All defined once, in `ad_mri.monitoring.thresholds` (the single source of truth every other file -
`monitoring.drift`, `monitoring.model_drift`, `deploy/prometheus/alerts.yml` - either imports or,
for the PromQL file, comment-mirrors):

| Threshold | Value | Meaning |
|---|---|---|
| `PSI_WARNING_THRESHOLD` | 0.1 | Below: no significant shift. |
| `PSI_ALERT_THRESHOLD` | 0.25 | At/above: significant shift - `any_drifted` fires. |
| `KS_PVALUE_ALERT_THRESHOLD` | 0.01 | Below this p-value: statistically significant shift (independent of PSI). |
| `ACCURACY_DEGRADATION_ALERT_THRESHOLD` | 0.05 | Absolute accuracy drop (vs. baseline) that flags model drift. |
| `F1_DEGRADATION_ALERT_THRESHOLD` | 0.05 | Absolute macro-F1 drop that flags model drift. |
| `MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT` | 30 | Hard floor before any model-drift verdict is computed at all. |
| `LOW_CONFIDENCE_THRESHOLD` / `_RATE_ALERT_THRESHOLD` | 0.6 / 0.2 | >20% of predictions with confidence <0.6 over a rolling window. |
| `INPUT_FAILURE_RATE_ALERT_THRESHOLD` | 0.1 | >10% of inference requests failing validation/preprocessing. |
| `HTTP_ERROR_RATE_ALERT_THRESHOLD` | 0.05 | >5% of HTTP requests returning 5xx. |
| `INFERENCE_LATENCY_P95_ALERT_SECONDS` | 10.0 | p95 model latency. |
| `CPU_PERCENT_ALERT_THRESHOLD` / `MEMORY_PERCENT_ALERT_THRESHOLD` | 90.0 | Host resource pressure. |

Every value is an industry-standard starting point, not tuned against this project's own model.
They are plain module constants (`monitoring.thresholds`), not `core.config.Settings` fields —
tuning one for a real deployment means editing that file (and `deploy/prometheus/alerts.yml`'s
matching PromQL literal) and redeploying, not setting an environment variable; see that module's
docstring for why this was the deliberate choice.

## 7. Dashboards & metrics endpoints

- **`GET /metrics`** (unversioned, `api.routers.health`) - the Prometheus exposition endpoint,
  real since Phase 1, now carrying every metric in §2-4 plus the two drift-alert gauges below.
- **`ad_mri_data_drift_alert_active{feature}`** / **`ad_mri_model_drift_alert_active{metric}`** -
  reflect the *most recent offline report's verdict* into a live gauge
  (`monitoring.alert_state.sample_alert_state()`, called on every scrape). A short-lived CLI
  process can never itself be scraped, so its verdict is durably recorded as a `system_events`
  row (`event_type="data_drift_detected"`/`"model_drift_detected"`, written by the `drift-report`/
  `model-drift-report` commands) and picked up here by the long-running API process instead. A
  verdict older than 24h is treated as stale and cleared back to "no active alert" - schedule
  those two CLI commands to actually run periodically (cron/CI) for this to stay meaningful.
- **`deploy/prometheus/{prometheus.yml,alerts.yml}`** - a real, `promtool`-validated scrape config
  and 8 alert rules (§6's thresholds as PromQL).
- **`deploy/grafana/ad-mri-dashboard.json`** - a real, importable dashboard (17 panels: system,
  HTTP, model, drift-alert state), auto-provisioned alongside the Prometheus datasource
  (`deploy/grafana/provisioning/`).
- **Local dev**: `docker compose -f docker/docker-compose.yml --profile observability up -d`
  (`make docker-observability-up`) - Grafana at `http://localhost:3000` (anonymous viewer access
  in local dev only), Prometheus at `http://localhost:9090`.

## 8. Retraining triggers

None of these fire a retraining job automatically - **every one is a signal for a human decision,
never an automated model change** (consistent with this project's "no performance number appears
anywhere in this codebase unless it was actually produced by a logged MLflow run" posture - an
automatic retrain-and-redeploy would be exactly the kind of unreviewed model change that
principle exists to prevent). Documented triggers, roughly in order of urgency:

1. **`ModelDriftDetected` fires** (§5/§6: a monitored metric degraded ≥5 points vs. baseline,
   with ≥30 labeled samples) - the strongest possible signal, since it is measured against real
   outcomes, not a proxy. Investigate immediately; retraining (or rolling back to a previous
   model version - `docs/deployment.md` §4/§5) is the usual response once confirmed.
2. **`DataDriftDetected` fires** on a feature or the predicted-class distribution (§4/§6: PSI ≥
   0.25 or KS p-value < 0.01) - a proxy signal, available immediately (no ground truth needed).
   On its own this means "the input population has changed," not necessarily "the model is
   wrong" - but it is the earliest available warning, often weeks before enough ground truth
   accumulates to confirm §1 directly. Investigate the cause (a new scanner, a new site, a
   protocol change) before deciding whether to retrain.
3. **`LowConfidenceRate` sustained** (§6: >20% of predictions below 0.6 confidence over 15m+) -
   often co-occurs with data drift; on its own, worth investigating even without a drift alert.
4. **`HighInputFailureRate` sustained** - less a "retrain the model" trigger than a "something
   upstream changed" trigger (a new client, a broken integration) - included here because it is
   frequently the *first* visible symptom of the same root cause as #2.

A retraining decision should always be backed by re-running the standalone evaluation
(`ad-mri evaluate`) against a genuinely fresh, representative held-out set - never by these
monitoring signals alone, and never by the training-distribution PSI/KS numbers themselves
(which measure *input* shift, not model correctness).

## 9. What is explicitly out of scope

- **`/explanations`'s own internal prediction** does not feed §3's model metrics or §4's
  real-time drift signal - it runs an independent forward pass
  (`explainability.service.ExplainabilityService`), and duplicating instrumentation across two
  prediction code paths was judged not worth the added surface for a secondary endpoint. The
  primary `/predictions` path is fully instrumented.
- **Alertmanager** is not deployed (`deploy/prometheus/prometheus.yml` has no `alerting:` block) -
  a documented seam, like `deploy/k8s/`; the alert *rules* are real, routing them to
  email/Slack/PagerDuty is a follow-on integration.
- **A Prometheus Pushgateway** is not deployed - `monitoring.alert_state`'s
  system_events-round-trip approach was chosen instead specifically to avoid adding a fifth
  observability service for two gauges; revisit if more batch-job-originated metrics appear.
- **Evidently** (`evidently` is a declared, installed-only-via-`train`-extra dependency, unused
  by any code in this phase) remains a documented-but-unused seam: this phase's own PSI/KS
  implementation (`monitoring.drift`) was judged more stable and dependency-light than adopting
  Evidently's own (frequently-changing) report API under this phase's scope. Swapping it in later
  would replace `generate_data_drift_report`'s internals, not the `DriftReference`/database
  schema this doc describes.
- **Automatic retraining** - see §8's opening line. Every trigger here ends at a human decision.

## 10. Verified, not claimed

Real tooling, real output, every time: `promtool check rules`/`check config` against
`deploy/prometheus/{alerts.yml,prometheus.yml}` (8 rules, 0 errors); the Grafana dashboard JSON
parsed and cross-checked (in Python) against every metric name actually defined in
`monitoring.metrics` (zero stale/typo'd references); `docker compose config` against both the
default and `--profile observability` service sets; a full, real, end-to-end run (not mocked) of
every `ad-mri monitor` CLI command against a real tiny checkpoint, real synthetic NIfTI images,
real `InferenceEngine.predict()` calls, and a real SQLite database - `build-drift-reference` →
`register-baseline` → `drift-report` (correctly detected a real class-distribution shift) →
`model-drift-report` (correctly reported insufficient samples below the 30-label floor) →
`record-ground-truth`, every step producing real, inspected output. The full existing test suite
was re-run after every source change in this phase to confirm nothing regressed.
