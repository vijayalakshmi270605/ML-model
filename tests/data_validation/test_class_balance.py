from __future__ import annotations

import pandas as pd
import pytest

from ad_mri.data_pipeline.class_balance import build_weighted_sampler, compute_class_weights


def test_compute_class_weights_balances_inverse_frequency() -> None:
    df = pd.DataFrame({"label": ["CN"] * 8 + ["AD"] * 2})
    weights = compute_class_weights(df)

    assert weights["AD"] > weights["CN"]
    # inverse-frequency formula: total / (num_classes * count)
    assert weights["CN"] == pytest.approx(10 / (2 * 8))
    assert weights["AD"] == pytest.approx(10 / (2 * 2))


def test_compute_class_weights_rejects_unknown_method() -> None:
    df = pd.DataFrame({"label": ["CN", "AD"]})
    with pytest.raises(ValueError, match="Unsupported"):
        compute_class_weights(df, method="not-a-real-method")


def test_compute_class_weights_empty_dataframe_returns_empty_dict() -> None:
    df = pd.DataFrame({"label": []})
    assert compute_class_weights(df) == {}


def test_build_weighted_sampler_returns_one_weight_per_row() -> None:
    torch = pytest.importorskip("torch")
    df = pd.DataFrame({"label": ["CN"] * 8 + ["AD"] * 2})

    sampler = build_weighted_sampler(df)

    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    assert len(sampler) == len(df)
