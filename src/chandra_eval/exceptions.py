"""Project-specific exceptions."""


class ChandraEvalError(Exception):
    """Base exception for pipeline errors."""


class ConfigurationError(ChandraEvalError):
    """Raised when runtime configuration is invalid."""


class ModelResponseError(ChandraEvalError):
    """Raised when an OpenAI-compatible model response is unusable."""
