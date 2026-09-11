"""Domain-specific exception hierarchy for the ad_mri package.

Application code should raise one of these (never a bare Exception) so API
error handlers and CLI commands can map failures to precise, typed responses.
"""


class AppError(Exception):
    """Base class for all application-raised errors."""


class ConfigurationError(AppError):
    """Raised when application configuration is invalid or unsafe to run with."""


class DataValidationError(AppError):
    """Raised when input or pipeline data fails validation."""


class PreprocessingError(AppError):
    """Raised when an MRI preprocessing step fails."""


class ModelNotFoundError(AppError):
    """Raised when a requested model/version cannot be resolved from the registry."""


class InferenceError(AppError):
    """Raised when model inference fails for a reason other than 'not implemented yet'."""


class TrainingError(AppError):
    """Raised when a training run fails in a way that must stop training (e.g. a
    NaN/Inf loss) rather than being silently skipped or absorbed."""


class ExplainabilityError(AppError):
    """Raised when generating an explanation (LIME/SHAP) fails."""
