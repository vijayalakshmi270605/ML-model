from __future__ import annotations

from pathlib import Path

from ad_mri.training.training_config import (
    TrainingConfig,
    load_training_config,
    save_training_config,
)


def test_defaults_are_valid() -> None:
    config = TrainingConfig()
    assert config.optimizer == "adamw"
    assert config.scheduler == "cosine"
    assert config.class_imbalance_strategy == "loss_weights"


def test_resolve_scheduler_t_max_defaults_to_epochs() -> None:
    config = TrainingConfig(epochs=30, scheduler_t_max=None)
    assert config.resolve_scheduler_t_max() == 30


def test_resolve_scheduler_t_max_respects_explicit_override() -> None:
    config = TrainingConfig(epochs=30, scheduler_t_max=10)
    assert config.resolve_scheduler_t_max() == 10


def test_compute_hash_is_deterministic_and_content_sensitive() -> None:
    a = TrainingConfig()
    b = TrainingConfig()
    c = TrainingConfig(learning_rate=0.01)
    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = TrainingConfig(batch_size=4, epochs=5, optimizer="sgd")
    path = tmp_path / "training.yaml"

    save_training_config(config, path)
    reloaded = load_training_config(path)

    assert reloaded.batch_size == 4
    assert reloaded.optimizer == "sgd"
    assert reloaded.compute_hash() == config.compute_hash()


def test_the_shipped_default_training_config_is_valid() -> None:
    """configs/training/default.yaml is documentation-by-example — it must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "training" / "default.yaml"
    config = load_training_config(config_path)
    assert config.epochs > 0
