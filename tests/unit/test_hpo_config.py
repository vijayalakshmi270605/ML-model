from __future__ import annotations

from pathlib import Path

from ad_mri.training.hpo_config import HPOConfig, load_hpo_config, save_hpo_config


def test_defaults_are_valid() -> None:
    config = HPOConfig()
    assert config.n_trials > 0
    assert config.n_folds >= 2
    assert config.metric_mode in ("min", "max")
    assert "adamw" in config.search_space.optimizer


def test_compute_hash_is_deterministic_and_content_sensitive() -> None:
    a = HPOConfig()
    b = HPOConfig()
    c = HPOConfig(n_trials=50)
    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = HPOConfig(n_trials=5, n_folds=3, metric_name="accuracy", metric_mode="max")
    path = tmp_path / "hpo.yaml"

    save_hpo_config(config, path)
    reloaded = load_hpo_config(path)

    assert reloaded.n_trials == 5
    assert reloaded.n_folds == 3
    assert reloaded.compute_hash() == config.compute_hash()


def test_the_shipped_default_hpo_config_is_valid() -> None:
    """configs/hpo/default.yaml is documentation-by-example — it must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "hpo" / "default.yaml"
    config = load_hpo_config(config_path)
    assert config.study_name == "ad-mri-hpo"
