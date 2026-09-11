"""Reproducibility helpers.

See docs/TECHNICAL_SPECIFICATION.md §5 — every training run seeds Python, NumPy,
and PyTorch deterministically. NumPy/PyTorch are training-phase dependencies
(requirements/train.txt), so both imports here are best-effort.
"""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Seed python `random`, and (if installed) numpy and torch, for reproducibility."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
