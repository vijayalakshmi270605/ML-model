"""MRI dataset wrapper reading a versioned manifest/split CSV.

Consumes the exact manifest schema produced by `data_pipeline.manifest` — see
docs/TECHNICAL_SPECIFICATION.md §4. `torch`/`pandas` are training-phase
dependencies (requirements/train.txt); `torch` is imported defensively so this
module stays importable without it (e.g. by tooling that only needs
`MRIDataset.label_to_index`-style static helpers), but the class is only
usable once torch is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ad_mri.core.logging import get_logger

if TYPE_CHECKING:
    # Always resolved for the type checker (mypy analyzes only this branch),
    # so MRIDataset gets a real base-class type without requiring torch to be
    # installed just to run mypy on this file.
    from torch.utils.data import Dataset as _TorchDataset
else:
    try:
        from torch.utils.data import Dataset as _TorchDataset
    except ImportError:  # torch not installed outside the training/worker images
        _TorchDataset = object

logger = get_logger(__name__)


def infer_label_mapping(df: pd.DataFrame, label_column: str = "label") -> dict[str, int]:
    """Deterministic label->index mapping: sorted unique labels, index by position.

    Sorting makes the mapping stable across runs/machines given the same label set.
    """
    unique_labels = sorted(df[label_column].dropna().unique())
    return {str(label): index for index, label in enumerate(unique_labels)}


class MRIDataset(_TorchDataset):
    """A `torch.utils.data.Dataset` over one manifest/split CSV.

    Each item is a dict `{"image": <path or transformed tensor>, "label": int,
    "subject_id": str}`. Pass a MONAI transform (see `transforms.build_transforms`)
    to load/preprocess the image; without one, "image" stays a raw file path.
    """

    def __init__(
        self,
        manifest_path: Path,
        transform: Any = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self._manifest_path = manifest_path
        self._df = pd.read_csv(manifest_path)
        self._transform = transform
        self._label_to_index = label_to_index or infer_label_mapping(self._df)
        logger.info(
            "mri_dataset_loaded",
            manifest_path=str(manifest_path),
            rows=len(self._df),
            classes=self._label_to_index,
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._df.iloc[index]
        sample: dict[str, Any] = {
            "image": row["image_path"],
            "label": self._label_to_index[str(row["label"])],
            "subject_id": row["subject_id"],
        }
        if self._transform is not None:
            from ad_mri.datasets.transforms import apply_transforms

            sample = apply_transforms(self._transform, sample)
        return sample

    @property
    def label_to_index(self) -> dict[str, int]:
        return dict(self._label_to_index)
