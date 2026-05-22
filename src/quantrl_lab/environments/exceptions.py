"""
Domain-specific exceptions for environments.

This module defines a hierarchy of exceptions for the environment
module, enabling precise error handling for environment-related issues.
"""

class EnvironmentError(Exception):
    """Base exception for all environment-related errors."""

    pass


class InvalidActionError(EnvironmentError):
    """
    Raised when an agent attempts an invalid action.

    Examples:
        - Action out of bounds
        - Action not supported by current state
        - Action format incorrect
    """

    pass


class InvalidStateError(EnvironmentError):
    """
    Raised when the environment enters or is found in an invalid state.

    Examples:
        - Inconsistent internal state
        - NaN values in observation
        - Missing required state data
    """

    pass


class ConfigurationError(EnvironmentError):
    """
    Raised when environment configuration is invalid.

    Examples:
        - Missing required parameters
        - Conflicting configuration options
        - Invalid parameter values
    """

    pass


class DataFeedError(EnvironmentError):
    """
    Raised when the environment fails to get data from its source.

    Examples:
        - Data source exhausted
        - Data format mismatch
        - Missing required columns in data
    """

    pass
