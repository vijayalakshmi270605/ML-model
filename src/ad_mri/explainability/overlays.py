"""Renders LIME/SHAP explanation heatmaps as overlays on the representative
MRI slice — the single rendering implementation both `lime_explainer.py` and
`shap_explainer.py` call, so the two methods' visualizations stay visually
comparable (requirement: "Comparison surfacing", TECHNICAL_SPECIFICATION §7).

`matplotlib`/`skimage` are explainability-phase dependencies
(requirements/train.txt via the `train` extra), imported lazily so this
module stays importable without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_lime_overlay(highlighted_image: Any, mask: Any, output_path: Path) -> Path:
    """Render LIME's `(temp_image, mask)` pair (from
    `ImageExplanation.get_image_and_mask`) as a superpixel-boundary overlay PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    display = highlighted_image.copy()
    display_max = display.max() if display.size and display.max() > 0 else 1.0
    boundary_image = mark_boundaries(display / display_max, mask)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(boundary_image)
    ax.set_title("LIME explanation (highlighted superpixels)")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def render_shap_overlay(background_slice: Any, attribution_slice: Any, output_path: Path) -> Path:
    """Render one 2D SHAP-attribution slice as a diverging-colormap overlay
    on its corresponding grayscale MRI slice."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt
    import numpy as np

    magnitude = max(float(np.abs(attribution_slice).max()), 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(background_slice, cmap="gray")
    im = ax.imshow(attribution_slice, cmap="coolwarm", alpha=0.55, vmin=-magnitude, vmax=magnitude)
    fig.colorbar(im, ax=ax, label="SHAP value (blue=away from class, red=toward class)")
    ax.set_title("SHAP explanation (representative slice)")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def render_global_importance_overlay(region_slice: Any, output_path: Path) -> Path:
    """Render one 2D slice of a downsampled global-importance grid
    (`shap_explainer.compute_global_shap_importance`) as a heatmap."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(region_slice, cmap="magma")
    fig.colorbar(im, ax=ax, label="Mean |SHAP value| (downsampled region)")
    ax.set_title("Aggregated SHAP importance (representative slice)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
