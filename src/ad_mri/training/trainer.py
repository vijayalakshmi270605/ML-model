"""Training loop orchestration — the single place model training happens.

Deliberately separate from `inference/`/`api/`: nothing here is imported by
the serving path, and nothing in `inference/predictor.py` imports from this
package. Runnable from the CLI via `ad-mri train` or
`python -m ad_mri.training.train` (see `training/train.py`) — there is no
notebook-only path or hidden global state. See
docs/TECHNICAL_SPECIFICATION.md §5, §8, §9, §14, §15, §16.

`torch`/`mlflow`/`pandas` are training-phase dependencies
(requirements/train.txt), imported lazily so this module stays importable
without them.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    PreprocessingError,
    TrainingError,
)
from ad_mri.core.logging import get_logger
from ad_mri.core.seeding import set_global_seed
from ad_mri.data_pipeline.validation import compute_sha256
from ad_mri.datasets.preprocessing_config import PreprocessingConfig
from ad_mri.evaluation.confusion_matrix import compute_confusion_matrix, render_confusion_matrix
from ad_mri.evaluation.metrics import compute_classification_metrics, format_metrics
from ad_mri.modeling.checkpoint import load_checkpoint, save_checkpoint
from ad_mri.modeling.factory import build_model
from ad_mri.modeling.model_config import ModelConfig
from ad_mri.training.callbacks import EarlyStopping
from ad_mri.training.mlflow_utils import start_mlflow_run
from ad_mri.training.training_config import TrainingConfig
from ad_mri.training.visualization import plot_training_curves

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    run_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_epoch: int
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)
    final_train_metrics: dict[str, float] = field(default_factory=dict)
    final_val_metrics: dict[str, float] = field(default_factory=dict)
    mlflow_run_id: str | None = None
    training_curves_path: Path | None = None
    confusion_matrix_path: Path | None = None


class Trainer:
    """Owns the training loop: data loading, optimization, checkpointing, MLflow logging."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_manifest_path: Path,
        val_manifest_path: Path,
        preprocessing_config: PreprocessingConfig | None = None,
        run_name: str | None = None,
        resume_from: Path | None = None,
        mlflow_nested: bool = False,
    ) -> None:
        if preprocessing_config is not None and tuple(preprocessing_config.target_shape) != tuple(
            model_config.input_shape
        ):
            raise ConfigurationError(
                f"preprocessing_config.target_shape {preprocessing_config.target_shape} "
                f"!= model_config.input_shape {model_config.input_shape}."
            )

        self.model_config = model_config
        self.training_config = training_config
        self.train_manifest_path = Path(train_manifest_path)
        self.val_manifest_path = Path(val_manifest_path)
        self.preprocessing_config = preprocessing_config
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
        #: True when the caller (e.g. cross-validation/HPO orchestration) already
        #: has an MLflow run active and wants this Trainer's run nested under it,
        #: rather than opening its own top-level run.
        self.mlflow_nested = mlflow_nested

        if self.resume_from is not None and run_name is None:
            self.run_dir = self.resume_from.parent  # continue writing into the same run
        else:
            self.run_dir = self.training_config.checkpoint_dir / self.run_name
        self.best_checkpoint_path = self.run_dir / "best.pt"
        self.last_checkpoint_path = self.run_dir / "last.pt"

        self._class_names: list[str] = []

    # --- public API -----------------------------------------------------------

    def fit(self) -> TrainingResult:
        import pandas as pd

        set_global_seed(self.training_config.seed)
        self._apply_determinism()
        device = self._resolve_device()

        train_df = pd.read_csv(self.train_manifest_path)
        val_df = pd.read_csv(self.val_manifest_path)
        label_to_index, class_names = self._resolve_label_mapping(train_df, val_df)
        self._class_names = class_names
        if len(class_names) != self.model_config.num_classes:
            raise ConfigurationError(
                f"Dataset has {len(class_names)} class(es) {class_names} but "
                f"model_config.num_classes={self.model_config.num_classes}."
            )

        logger.info(
            "training_started",
            run_name=self.run_name,
            device=str(device),
            classes=class_names,
            train_rows=len(train_df),
            val_rows=len(val_df),
        )

        train_loader, val_loader = self._build_dataloaders(train_df, label_to_index)
        model = build_model(self.model_config).to(device)
        loss_fn = self._build_loss_fn(train_df, class_names, device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        scaler = self._build_grad_scaler(device)
        early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience,
            mode=self.training_config.early_stopping_mode,
            min_delta=self.training_config.early_stopping_min_delta,
        )

        history: list[dict[str, float]] = []
        start_epoch = 1
        best_epoch = 0
        if self.resume_from is not None:
            start_epoch, history, best_epoch = self._resume(
                model, optimizer, scheduler, scaler, early_stopping, device
            )

        mlflow_module, mlflow_run_context = self._start_mlflow_run()

        last_val_true: list[int] = []
        last_val_pred: list[int] = []
        epoch = start_epoch - 1
        try:
            with mlflow_run_context as active_run:
                if mlflow_module is not None and active_run is not None:
                    self._log_mlflow_params(mlflow_module)

                for epoch in range(start_epoch, self.training_config.epochs + 1):
                    train_metrics, _, _ = self._run_epoch(
                        model,
                        train_loader,
                        loss_fn,
                        optimizer,
                        scaler,
                        device,
                        class_names,
                        training=True,
                    )
                    val_metrics, last_val_true, last_val_pred = self._run_epoch(
                        model, val_loader, loss_fn, None, None, device, class_names, training=False
                    )

                    epoch_row = self._build_epoch_row(epoch, train_metrics, val_metrics)
                    if scheduler is not None:
                        self._step_scheduler(scheduler, epoch_row)

                    history.append(epoch_row)
                    self._log_epoch(epoch, train_metrics, val_metrics)
                    self._mlflow_log_metrics(mlflow_module, epoch_row, epoch)

                    metric_name = self.training_config.early_stopping_metric
                    monitored_value = epoch_row.get(metric_name)
                    if monitored_value is None:
                        raise ConfigurationError(
                            f"early_stopping_metric {metric_name!r} not "
                            f"found among computed metrics: {sorted(epoch_row)}"
                        )
                    stopping = early_stopping.step(monitored_value)

                    checkpoint_extra = self._build_checkpoint_extra(
                        epoch, optimizer, scheduler, scaler, early_stopping, history
                    )
                    if (
                        epoch % self.training_config.save_every_n_epochs == 0
                        or epoch == self.training_config.epochs
                    ):
                        save_checkpoint(
                            model,
                            self.model_config,
                            self.last_checkpoint_path,
                            extra=checkpoint_extra,
                        )
                    if stopping.is_best:
                        save_checkpoint(
                            model,
                            self.model_config,
                            self.best_checkpoint_path,
                            extra=checkpoint_extra,
                        )
                        best_epoch = epoch

                    if stopping.should_stop:
                        logger.info(
                            "early_stopping_triggered",
                            epoch=epoch,
                            patience=early_stopping.patience,
                        )
                        break

                if not self.last_checkpoint_path.exists():
                    save_checkpoint(
                        model,
                        self.model_config,
                        self.last_checkpoint_path,
                        extra=self._build_checkpoint_extra(
                            epoch, optimizer, scheduler, scaler, early_stopping, history
                        ),
                    )
                if not self.best_checkpoint_path.exists():
                    save_checkpoint(
                        model,
                        self.model_config,
                        self.best_checkpoint_path,
                        extra=self._build_checkpoint_extra(
                            epoch, optimizer, scheduler, scaler, early_stopping, history
                        ),
                    )
                    best_epoch = epoch

                curves_path = self._render_training_curves(history)
                cm_path = self._render_confusion_matrix(last_val_true, last_val_pred, class_names)
                self._mlflow_log_artifacts(mlflow_module, curves_path, cm_path)

                mlflow_run_id = (
                    active_run.info.run_id if mlflow_module is not None and active_run else None
                )
        except KeyboardInterrupt:
            logger.warning("training_interrupted_by_user", epoch=epoch)
            interrupted_path = self.run_dir / f"interrupted_epoch_{epoch}.pt"
            save_checkpoint(
                model,
                self.model_config,
                interrupted_path,
                extra=self._build_checkpoint_extra(
                    epoch, optimizer, scheduler, scaler, early_stopping, history
                ),
            )
            logger.info("interrupted_checkpoint_saved", path=str(interrupted_path))
            raise

        last_row = history[-1] if history else {}
        return TrainingResult(
            run_dir=self.run_dir,
            best_checkpoint_path=self.best_checkpoint_path,
            last_checkpoint_path=self.last_checkpoint_path,
            best_epoch=best_epoch,
            epochs_run=epoch,
            history=history,
            final_train_metrics=self._strip_prefix(last_row, "train_"),
            final_val_metrics=self._strip_prefix(last_row, "val_"),
            mlflow_run_id=mlflow_run_id,
            training_curves_path=curves_path,
            confusion_matrix_path=cm_path,
        )

    @staticmethod
    def _strip_prefix(row: dict[str, float], prefix: str) -> dict[str, float]:
        return {key[len(prefix) :]: value for key, value in row.items() if key.startswith(prefix)}

    # --- setup helpers ----------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        choice = self.training_config.device
        if choice == "cpu":
            return torch.device("cpu")
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("device='cuda' was requested but CUDA is not available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_determinism(self) -> None:
        if not self.training_config.deterministic:
            return
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _resolve_label_mapping(
        self, train_df: Any, val_df: Any
    ) -> tuple[dict[str, int], list[str]]:
        all_labels = sorted(set(train_df["label"].dropna()) | set(val_df["label"].dropna()))
        return {label: index for index, label in enumerate(all_labels)}, all_labels

    def _build_dataloaders(self, train_df: Any, label_to_index: dict[str, int]) -> tuple[Any, Any]:
        from torch.utils.data import DataLoader

        from ad_mri.data_pipeline.class_balance import build_weighted_sampler
        from ad_mri.datasets.dataset import MRIDataset
        from ad_mri.datasets.transforms import build_transforms

        train_transform = build_transforms(
            self.preprocessing_config, train=True, seed=self.training_config.seed
        )
        val_transform = build_transforms(self.preprocessing_config, train=False)

        train_dataset = MRIDataset(
            self.train_manifest_path, transform=train_transform, label_to_index=label_to_index
        )
        val_dataset = MRIDataset(
            self.val_manifest_path, transform=val_transform, label_to_index=label_to_index
        )

        sampler = None
        shuffle = True
        if self.training_config.class_imbalance_strategy == "sampler":
            sampler = build_weighted_sampler(train_df, label_column="label")
            shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.training_config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
        )
        return train_loader, val_loader

    def _build_loss_fn(self, train_df: Any, class_names: list[str], device: Any) -> Any:
        import torch
        import torch.nn as nn

        weight = None
        if self.training_config.class_imbalance_strategy == "loss_weights":
            from ad_mri.data_pipeline.class_balance import compute_class_weights

            weights_by_class = compute_class_weights(train_df, label_column="label")
            weight = torch.tensor(
                [weights_by_class[name] for name in class_names], dtype=torch.float32, device=device
            )
        return nn.CrossEntropyLoss(weight=weight)

    def _build_optimizer(self, model: Any) -> Any:
        import torch.optim as optim

        cfg = self.training_config
        if cfg.optimizer == "adam":
            return optim.Adam(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        if cfg.optimizer == "adamw":
            return optim.AdamW(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        if cfg.optimizer == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        raise ConfigurationError(f"Unknown optimizer: {cfg.optimizer!r}")  # pragma: no cover

    def _build_scheduler(self, optimizer: Any) -> Any:
        import torch.optim.lr_scheduler as lr_scheduler

        cfg = self.training_config
        if cfg.scheduler == "none":
            return None
        if cfg.scheduler == "step":
            return lr_scheduler.StepLR(
                optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
            )
        if cfg.scheduler == "cosine":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.resolve_scheduler_t_max())
        if cfg.scheduler == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=cfg.early_stopping_mode, patience=cfg.scheduler_patience
            )
        raise ConfigurationError(f"Unknown scheduler: {cfg.scheduler!r}")  # pragma: no cover

    def _step_scheduler(self, scheduler: Any, epoch_row: dict[str, float]) -> None:
        import torch.optim.lr_scheduler as lr_scheduler

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_row[self.training_config.early_stopping_metric])
        else:
            scheduler.step()

    def _build_grad_scaler(self, device: Any) -> Any:
        import torch

        use_amp = self.training_config.mixed_precision and device.type == "cuda"
        return torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- epoch loop ---------------------------------------------------------------

    def _run_epoch(
        self,
        model: Any,
        loader: Any,
        loss_fn: Any,
        optimizer: Any,
        scaler: Any,
        device: Any,
        class_names: list[str],
        *,
        training: bool,
    ) -> tuple[dict[str, float], list[int], list[int]]:
        import torch

        model.train(training)
        use_amp = self.training_config.mixed_precision and device.type == "cuda"

        total_loss = 0.0
        total_samples = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        y_proba: list[list[float]] = []

        iterator = iter(loader)
        step = 0
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except (PreprocessingError, DataValidationError) as exc:
                # A bad sample fails loud at the dataset level (Phase 3) but must
                # not crash a whole epoch of otherwise-good data — log and skip.
                logger.warning("batch_skipped_due_to_bad_sample", error=str(exc))
                continue

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            grad_context = contextlib.nullcontext() if training else torch.no_grad()
            with grad_context, torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                raise TrainingError(
                    f"Non-finite loss ({loss.item()!r}) encountered "
                    f"({'training' if training else 'validation'} step {step})."
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if self.training_config.gradient_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.training_config.gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.training_config.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.training_config.gradient_clip_norm
                        )
                    optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits.detach().float(), dim=1)
                preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_proba.extend(probs.cpu().tolist())

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            step += 1

            if step % self.training_config.log_every_n_steps == 0:
                logger.info(
                    "step_complete",
                    phase="train" if training else "val",
                    step=step,
                    loss=round(loss.item(), 6),
                )

        metrics: dict[str, float] = {
            "loss": total_loss / total_samples if total_samples else float("nan")
        }
        if total_samples:
            metrics.update(
                compute_classification_metrics(y_true, y_pred, class_names, y_proba=y_proba)
            )
        return metrics, y_true, y_pred

    # --- checkpoint / resume -------------------------------------------------------

    def _build_checkpoint_extra(
        self,
        epoch: int,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        early_stopping: EarlyStopping,
        history: list[dict[str, float]],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stopping_state": early_stopping.state_dict(),
            "history": history,
            "training_config": self.training_config.model_dump(mode="json"),
            "training_config_hash": self.training_config.compute_hash(),
            "preprocessing_config_hash": (
                self.preprocessing_config.compute_hash() if self.preprocessing_config else None
            ),
            "preprocessing_version": (
                self.preprocessing_config.version if self.preprocessing_config else None
            ),
            "train_manifest_sha256": compute_sha256(self.train_manifest_path),
            "val_manifest_sha256": compute_sha256(self.val_manifest_path),
            "class_names": self._class_names,
        }

    def _resume(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        early_stopping: EarlyStopping,
        device: Any,
    ) -> tuple[int, list[dict[str, float]], int]:
        assert self.resume_from is not None
        resumed_model, resumed_model_config, metadata = load_checkpoint(
            self.resume_from, map_location=str(device)
        )
        if resumed_model_config.compute_hash() != self.model_config.compute_hash():
            raise ConfigurationError(
                f"Resume checkpoint at {self.resume_from} was trained with a different "
                "architecture than the current model_config."
            )
        model.load_state_dict(resumed_model.state_dict())

        extra = metadata.extra
        if extra.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(extra["optimizer_state_dict"])
        if scheduler is not None and extra.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(extra["scheduler_state_dict"])
        if extra.get("scaler_state_dict") is not None:
            scaler.load_state_dict(extra["scaler_state_dict"])
        if extra.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(extra["early_stopping_state"])

        history = list(extra.get("history", []))
        start_epoch = int(extra.get("epoch", 0)) + 1
        best_epoch = int(extra.get("epoch", 0))
        logger.info("resumed_from_checkpoint", path=str(self.resume_from), start_epoch=start_epoch)
        return start_epoch, history, best_epoch

    # --- logging / artifacts --------------------------------------------------------

    def _build_epoch_row(
        self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> dict[str, float]:
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        return row

    def _log_epoch(
        self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> None:
        logger.info(
            "epoch_complete",
            epoch=epoch,
            train=format_metrics(train_metrics, prefix=""),
            val=format_metrics(val_metrics, prefix=""),
            train_loss=round(train_metrics["loss"], 6),
            val_loss=round(val_metrics["loss"], 6),
        )

    def _start_mlflow_run(self) -> tuple[Any, Any]:
        return start_mlflow_run(
            enabled=self.training_config.mlflow_enabled,
            tracking_uri=self.training_config.mlflow_tracking_uri,
            experiment_name=self.training_config.mlflow_experiment_name,
            run_name=self.run_name,
            nested=self.mlflow_nested,
        )

    def _log_mlflow_params(self, mlflow_module: Any) -> None:
        try:
            flat: dict[str, Any] = {}

            def _flatten(prefix: str, value: Any) -> None:
                if isinstance(value, dict):
                    for key, sub_value in value.items():
                        _flatten(f"{prefix}.{key}", sub_value)
                else:
                    flat[prefix] = value

            _flatten("model", self.model_config.model_dump(mode="json"))
            _flatten("training", self.training_config.model_dump(mode="json"))
            if self.preprocessing_config is not None:
                _flatten("preprocessing", self.preprocessing_config.model_dump(mode="json"))

            mlflow_module.log_params({key: str(value)[:250] for key, value in flat.items()})
            mlflow_module.log_param("model_config_hash", self.model_config.compute_hash())
            mlflow_module.log_param("training_config_hash", self.training_config.compute_hash())
            if self.preprocessing_config is not None:
                mlflow_module.log_param(
                    "preprocessing_config_hash", self.preprocessing_config.compute_hash()
                )
        except Exception as exc:
            logger.warning("mlflow_log_params_failed", error=str(exc))

    def _mlflow_log_metrics(
        self, mlflow_module: Any, epoch_row: dict[str, float], epoch: int
    ) -> None:
        if mlflow_module is None:
            return
        try:
            metrics = {k: v for k, v in epoch_row.items() if k != "epoch"}
            mlflow_module.log_metrics(metrics, step=epoch)
        except Exception as exc:
            logger.warning("mlflow_log_metrics_failed", error=str(exc))

    def _render_training_curves(self, history: list[dict[str, float]]) -> Path | None:
        try:
            return plot_training_curves(history, self.run_dir / "training_curves.png")
        except Exception as exc:
            logger.warning("training_curves_plot_failed", error=str(exc))
            return None

    def _render_confusion_matrix(
        self, y_true: list[int], y_pred: list[int], class_names: list[str]
    ) -> Path | None:
        if not y_true:
            return None
        try:
            matrix = compute_confusion_matrix(y_true, y_pred, class_names)
            return render_confusion_matrix(
                matrix, class_names, self.run_dir / "confusion_matrix.png"
            )
        except Exception as exc:
            logger.warning("confusion_matrix_render_failed", error=str(exc))
            return None

    def _mlflow_log_artifacts(
        self, mlflow_module: Any, curves_path: Path | None, cm_path: Path | None
    ) -> None:
        if mlflow_module is None:
            return
        try:
            if curves_path is not None:
                mlflow_module.log_artifact(str(curves_path))
            if cm_path is not None:
                mlflow_module.log_artifact(str(cm_path))
            if self.best_checkpoint_path.exists():
                mlflow_module.log_artifact(str(self.best_checkpoint_path))
            if self.last_checkpoint_path.exists():
                mlflow_module.log_artifact(str(self.last_checkpoint_path))
        except Exception as exc:
            logger.warning("mlflow_log_artifacts_failed", error=str(exc))
