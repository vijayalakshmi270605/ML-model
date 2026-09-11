from __future__ import annotations

import pytest

from ad_mri.training.callbacks import EarlyStopping


def test_first_score_is_always_best() -> None:
    es = EarlyStopping(patience=3, mode="min")
    result = es.step(1.0)
    assert result.is_best is True
    assert result.should_stop is False
    assert result.epochs_without_improvement == 0


def test_mode_min_improving_scores_reset_the_counter() -> None:
    es = EarlyStopping(patience=2, mode="min")
    es.step(1.0)
    es.step(0.9)  # improvement
    result = es.step(0.8)  # improvement
    assert result.is_best is True
    assert result.epochs_without_improvement == 0


def test_mode_min_triggers_stop_after_patience_exhausted() -> None:
    es = EarlyStopping(patience=2, mode="min")
    es.step(1.0)  # best
    r1 = es.step(1.1)  # worse, 1/2
    assert r1.should_stop is False
    r2 = es.step(1.2)  # worse, 2/2 -> stop
    assert r2.should_stop is True
    assert r2.epochs_without_improvement == 2


def test_mode_max_treats_higher_as_better() -> None:
    es = EarlyStopping(patience=2, mode="max")
    es.step(0.5)  # best
    r1 = es.step(0.4)  # worse
    assert r1.is_best is False
    r2 = es.step(0.6)  # improvement
    assert r2.is_best is True
    assert r2.epochs_without_improvement == 0


def test_min_delta_requires_a_meaningful_improvement() -> None:
    es = EarlyStopping(patience=5, mode="min", min_delta=0.1)
    es.step(1.0)
    result = es.step(0.95)  # improved, but by less than min_delta
    assert result.is_best is False


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(mode="sideways")  # type: ignore[arg-type]


def test_state_dict_round_trip_preserves_patience_counter() -> None:
    es = EarlyStopping(patience=5, mode="min")
    es.step(1.0)
    es.step(1.1)  # 1 epoch without improvement
    state = es.state_dict()

    resumed = EarlyStopping(patience=5, mode="min")
    resumed.load_state_dict(state)

    assert resumed.best_score == es.best_score
    assert resumed.epochs_without_improvement == es.epochs_without_improvement
