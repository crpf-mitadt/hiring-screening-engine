"""Project-specific exceptions."""


class ChandraEvalError(Exception):
    """Base exception for pipeline errors."""


class ConfigurationError(ChandraEvalError):
    """Raised when runtime configuration is invalid."""


class ModelResponseError(ChandraEvalError):
    """Raised when an OpenAI-compatible model response is unusable."""


class DocumentIngestionError(ChandraEvalError):
    """Raised when an input document cannot be read safely."""


class ScannedPdfRequiresOcrError(DocumentIngestionError):
    """Raised when PDF text extraction finds no usable embedded text."""
