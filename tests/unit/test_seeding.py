from __future__ import annotations

from ad_mri.core.seeding import set_global_seed


def test_set_global_seed_runs_without_error() -> None:
    set_global_seed(42)
