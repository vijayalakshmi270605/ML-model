"""CLI entry point for standalone explanation generation.

Runnable directly:

    python -m ad_mri.explainability.explain --model <checkpoint.pt> --image <scan.nii.gz>

or via the main CLI (`ad-mri explain run ...`, see `cli/main.py`) — both call
`run_explanation()` below, the single explanation entry point; there is no
second, independently-maintained copy of the argument-parsing/wiring logic.
This module contains only CLI plumbing — all explanation logic lives in
`explainability.service`/`lime_explainer`/`shap_explainer`. See
docs/explainability.md.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ad_mri.core.config import get_settings
from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import configure_logging, get_logger
from ad_mri.explainability.config import load_lime_config, load_shap_config
from ad_mri.explainability.service import ExplainabilityService, ExplanationBundle

logger = get_logger(__name__)

app = typer.Typer(help="Generate LIME/SHAP explanations for one MRI scan's prediction.")


def run_explanation(
    model_path: Path,
    image_path: Path,
    preprocessing_config_path: Path | None = None,
    methods: tuple[str, ...] = ("lime", "shap"),
    lime_config_path: Path | None = None,
    shap_config_path: Path | None = None,
    background_manifest_path: Path | None = None,
    input_id: str | None = None,
    target_class: str | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
) -> ExplanationBundle:
    """Load a checkpoint, explain one prediction, return the bundle. The
    single explanation entry point.

    Raises:
        ConfigurationError: `target_class` (by name) is not one of the
            checkpoint's known classes, or (from `ExplainabilityService`) an
            unsupported method / missing background manifest / preprocessing
            mismatch.
    """
    service = ExplainabilityService(model_path, preprocessing_config_path)

    target_class_index = None
    if target_class is not None:
        if target_class not in service.class_names:
            raise ConfigurationError(
                f"Unknown class {target_class!r}; known classes: {service.class_names}"
            )
        target_class_index = service.class_names.index(target_class)

    lime_config = None
    if "lime" in methods:
        resolved_path = lime_config_path or Path("configs/explainability/lime_default.yaml")
        lime_config = load_lime_config(resolved_path)
    shap_config = None
    if "shap" in methods:
        resolved_path = shap_config_path or Path("configs/explainability/shap_default.yaml")
        shap_config = load_shap_config(resolved_path)

    return service.explain(
        image_path,
        input_id=input_id,
        methods=methods,
        lime_config=lime_config,
        shap_config=shap_config,
        background_manifest_path=background_manifest_path,
        target_class_index=target_class_index,
        output_dir=output_dir,
        seed=seed,
    )


@app.command()
def main(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="A NIfTI/DICOM scan to explain."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    methods: str = typer.Option("lime,shap", "--methods", help="Comma-separated: lime, shap."),
    lime_config: Path | None = typer.Option(
        None, "--lime-config", help="Default: configs/explainability/lime_default.yaml"
    ),
    shap_config: Path | None = typer.Option(
        None, "--shap-config", help="Default: configs/explainability/shap_default.yaml"
    ),
    background_manifest: Path | None = typer.Option(
        None,
        "--background-manifest",
        help="Required when --methods includes 'shap': a manifest CSV (e.g. train.csv) "
        "to draw the SHAP reference/baseline sample set from.",
    ),
    input_id: str | None = typer.Option(None, "--input-id", help="Default: the image filename."),
    target_class: str | None = typer.Option(
        None, "--target-class", help="Explain this class by name. Default: the predicted class."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/explanations/<input-id>-<timestamp>"
    ),
    seed: int = typer.Option(42, "--seed", help="Reproducibility seed for both methods."),
) -> None:
    """Explain one prediction. Runnable as `python -m ad_mri.explainability.explain`
    or `ad-mri explain run`."""
    configure_logging(get_settings())
    resolved_methods = tuple(m.strip() for m in methods.split(",") if m.strip())

    bundle = run_explanation(
        model_path=model,
        image_path=image,
        preprocessing_config_path=preprocessing_config,
        methods=resolved_methods,
        lime_config_path=lime_config,
        shap_config_path=shap_config,
        background_manifest_path=background_manifest,
        input_id=input_id,
        target_class=target_class,
        output_dir=output_dir,
        seed=seed,
    )

    typer.echo(
        f"Predicted: {bundle.prediction.predicted_class} ({bundle.prediction.confidence:.4f})"
    )
    typer.echo(f"Class probabilities: {bundle.prediction.class_probabilities}")
    typer.echo(f"Explained class: {bundle.class_names[bundle.target_class_index]}")
    typer.echo(f"Output directory: {bundle.output_dir}")
    for name, path in sorted(bundle.artifacts.items()):
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
