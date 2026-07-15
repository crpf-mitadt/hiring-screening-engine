"""Project-specific exceptions."""


class ChandraEvalError(Exception):
    """Base exception for pipeline errors."""


class ConfigurationError(ChandraEvalError):
    """Raised when runtime configuration is invalid."""


class ModelResponseError(ChandraEvalError):
    """Raised when an OpenAI-compatible model response is unusable."""


class APICompatibilityError(ModelResponseError):
    """Raised when an API explicitly rejects a structured request feature."""


class EmptyModelResponseError(ModelResponseError):
    """Raised when a model response has no usable assistant text."""


class FinalResponseTruncatedError(ModelResponseError):
    """Raised when the final model answer is truncated."""


class StructuredExtractionError(ModelResponseError):
    """Raised when structured extraction fails after bounded retries."""


class DocumentIngestionError(ChandraEvalError):
    """Raised when an input document cannot be read safely."""


class ScannedPdfRequiresOcrError(DocumentIngestionError):
    """Raised when PDF text extraction finds no usable embedded text."""
