"""
Domain-specific exceptions for the data module.

This module provides a hierarchy of exceptions for better error handling
and more precise error reporting in data operations.
"""


class DataSourceError(Exception):
    """
    Base exception for all data source operations.

    All data-related exceptions should inherit from this class to enable
    catching all data errors with a single except clause.
    """

    pass


class DataNotAvailableError(DataSourceError):
    """
    Data legitimately doesn't exist for the requested parameters.

    Examples:
        - Requesting weekend trading data
        - Requesting future dates
        - Symbol not available in data source
        - Historical data before instrument inception
    """

    pass


class APIConnectionError(DataSourceError):
    """
    Failed to establish connection with data provider API.

    Examples:
        - Network timeout
        - DNS resolution failure
        - SSL/TLS errors
        - Service unavailable (503)
    """

    pass


class InvalidParametersError(DataSourceError):
    """
    Invalid or inconsistent request parameters provided.

    Examples:
        - Invalid date format
        - Start date after end date
        - Invalid symbol format
        - Unsupported timeframe
        - Missing required parameters
    """

    pass


class DataValidationError(DataSourceError):
    """
    Data quality validation failed.

    Examples:
        - Missing required columns (OHLCV)
        - Invalid price relationships (high < low)
        - Excessive null values
        - Duplicate timestamps
        - Data type mismatches
    """

    pass


class RateLimitError(APIConnectionError):
    """
    API rate limit exceeded.

    Raised when the data provider's rate limit is exceeded. Contains
    information about the retry-after period if available.
    """

    def __init__(self, message: str, retry_after: int = None):
        """
        Initialize with optional retry-after period.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if provided by API)
        """
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(APIConnectionError):
    """
    Authentication or authorization failed.

    Examples:
        - Invalid API key
        - Expired credentials
        - Insufficient permissions
        - Missing API key
    """

    pass
