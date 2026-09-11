"""Custom MONAI transforms backing the deterministic preprocessing pipeline.

Imports `monai`/`numpy` at module level — safe because this module is only
ever imported lazily, from inside `transforms.py`'s functions, never from this
package's top level (which must stay importable without the `train` extra).
See docs/TECHNICAL_SPECIFICATION.md §4, §6.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from monai.transforms import LoadImaged, MapTransform

from ad_mri.core.exceptions import PreprocessingError
from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


class SafeLoadImaged(LoadImaged):
    """`LoadImaged` that wraps any load failure in a clear `PreprocessingError`.

    A raw nibabel/MONAI exception on a corrupted or unreadable file is often an
    unhelpful low-level traceback; this attaches the file path and raises a
    consistent, catchable exception type instead (requirement: "handle
    missing/invalid images").
    """

    def __call__(self, data: Any, reader: Any = None) -> Any:
        image_path = data.get(self.keys[0]) if isinstance(data, dict) else data
        try:
            return super().__call__(data, reader)
        except Exception as exc:
            raise PreprocessingError(f"Failed to load image {image_path!r}: {exc}") from exc


class QualityControld(MapTransform):
    """Optional post-preprocessing sanity checks.

    Verifies the transformed volume contains only finite values and has a
    minimum fraction of nonzero voxels (catches e.g. an accidentally
    all-background/empty crop). Controlled by
    `PreprocessingConfig.enable_quality_control` / `.qc_mode` — see
    docs/TECHNICAL_SPECIFICATION.md §4.
    """

    def __init__(
        self,
        keys: Any,
        min_nonzero_ratio: float,
        mode: str = "raise",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.min_nonzero_ratio = min_nonzero_ratio
        self.mode = mode

    def __call__(self, data: Any) -> Any:
        d = dict(data)
        for key in self.key_iterator(d):
            array = d[key]
            np_array = array.numpy() if hasattr(array, "numpy") else np.asarray(array)

            issues: list[str] = []
            if not np.all(np.isfinite(np_array)):
                issues.append("output contains non-finite values (NaN/Inf)")
            else:
                nonzero_ratio = float(np.count_nonzero(np_array)) / np_array.size
                if nonzero_ratio < self.min_nonzero_ratio:
                    issues.append(
                        f"nonzero voxel ratio {nonzero_ratio:.4f} below threshold "
                        f"{self.min_nonzero_ratio}"
                    )

            if issues:
                message = f"Quality control failed for {key!r}: {'; '.join(issues)}"
                if self.mode == "raise":
                    raise PreprocessingError(message)
                logger.warning("preprocessing_qc_failed", key=key, issues=issues)
        return d
