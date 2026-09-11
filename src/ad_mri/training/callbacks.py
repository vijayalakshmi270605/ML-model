"""Training callbacks: early stopping.

See docs/TECHNICAL_SPECIFICATION.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingResult:
    is_best: bool
    should_stop: bool
    best_score: float
    epochs_without_improvement: int


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    `mode="min"` (e.g. `val_loss`) treats a *lower* value as better;
    `mode="max"` (e.g. `val_f1`) treats a *higher* value as better.
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.epochs_without_improvement = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float) -> EarlyStoppingResult:
        """Record one epoch's monitored score.

        Returns an `EarlyStoppingResult` telling the caller whether this is a
        new best (save a "best" checkpoint) and whether training should stop.
        """
        is_best = self._is_improvement(score)
        if is_best:
            self.best_score = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return EarlyStoppingResult(
            is_best=is_best,
            should_stop=self.epochs_without_improvement >= self.patience,
            best_score=self.best_score if self.best_score is not None else score,
            epochs_without_improvement=self.epochs_without_improvement,
        )

    def state_dict(self) -> dict[str, float | int | None]:
        """For checkpointing — resuming training must resume the patience counter too."""
        return {
            "best_score": self.best_score,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(self, state: dict[str, float | int | None]) -> None:
        best_score = state.get("best_score")
        self.best_score = float(best_score) if best_score is not None else None
        self.epochs_without_improvement = int(state.get("epochs_without_improvement") or 0)
