"""CLI entry point for standalone inference.

Runnable directly:

    python -m ad_mri.inference.predict --model <checkpoint.pt> --image <scan.nii.gz>

or via the main CLI (`ad-mri predict ...`, see `cli/main.py`) — both call
`run_prediction()` below, the single inference entry point; there is no
second, independently-maintained copy of the argument-parsing/wiring logic.
This module contains only CLI plumbing — all inference logic lives in
`inference.predictor.InferenceEngine`. See docs/inference.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ad_mri.core.config import get_settings
from ad_mri.core.logging import configure_logging, get_logger
from ad_mri.inference.calibration import CalibrationConfig, load_calibration_config
from ad_mri.inference.predictor import InferenceEngine, PredictionResult

logger = get_logger(__name__)

app = typer.Typer(help="Classify one MRI scan with a trained checkpoint.")


def run_prediction(
    model_path: Path,
    image_path: Path,
    preprocessing_config_path: Path | None = None,
    calibration_config_path: Path | None = None,
    device_choice: str = "auto",
    timeout_seconds: float = 30.0,
    request_id: str | None = None,
    include_explanation: bool = False,
    explanation_methods: tuple[str, ...] | None = None,
    background_manifest_path: Path | None = None,
) -> PredictionResult:
    """Load a checkpoint, classify one image, return the result. The single
    inference entry point."""
    calibration_config = (
        load_calibration_config(calibration_config_path)
        if calibration_config_path
        else CalibrationConfig()
    )
    engine = InferenceEngine(
        checkpoint_path=model_path,
        preprocessing_config_path=preprocessing_config_path,
        calibration_config=calibration_config,
        device_choice=device_choice,
        timeout_seconds=timeout_seconds,
        explanation_background_manifest_path=background_manifest_path,
    )
    try:
        return engine.predict(
            image_path,
            request_id=request_id,
            include_explanation=include_explanation,
            explanation_methods=explanation_methods,
        )
    finally:
        engine.shutdown()


@app.command()
def main(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="A NIfTI/DICOM scan to classify."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    calibration_config: Path | None = typer.Option(
        None, "--calibration-config", help="Default: uncalibrated (plain softmax)."
    ),
    device: str = typer.Option("auto", "--device", help="'auto' | 'cpu' | 'cuda'"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    request_id: str | None = typer.Option(None, "--request-id", help="Default: a generated UUID."),
    explain: bool = typer.Option(False, "--explain", help="Also generate a LIME/SHAP explanation."),
    methods: str = typer.Option("lime,shap", "--methods", help="Used only with --explain."),
    background_manifest: Path | None = typer.Option(
        None, "--background-manifest", help="Required for a SHAP explanation."
    ),
) -> None:
    """Classify one MRI scan. Runnable as `python -m ad_mri.inference.predict`
    or `ad-mri predict`."""
    configure_logging(get_settings())
    result = run_prediction(
        model_path=model,
        image_path=image,
        preprocessing_config_path=preprocessing_config,
        calibration_config_path=calibration_config,
        device_choice=device,
        timeout_seconds=timeout_seconds,
        request_id=request_id,
        include_explanation=explain,
        explanation_methods=(
            tuple(m.strip() for m in methods.split(",") if m.strip()) if explain else None
        ),
        background_manifest_path=background_manifest,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    app()
