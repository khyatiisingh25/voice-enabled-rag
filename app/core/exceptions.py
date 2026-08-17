class PipelineError(Exception):
    """Base exception for pipeline failures."""


class PipelineTimeoutError(PipelineError):
    """Raised when the pipeline exceeds its allowed time."""


class PipelineUnavailableError(PipelineError):
    """Raised when a required downstream service is unavailable."""