"""Provider error types."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base error for provider-related failures."""

    pass


class ProviderConfigError(ProviderError):
    """Configuration validation error."""

    pass


class ProviderRequestError(ProviderError):
    """Runtime request error with retryability flag."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.cause = cause
