"""Domain-specific exceptions raised by FinCoach."""


class FinCoachError(Exception):
    """Base class for errors callers may want to present to a user."""


class TransactionSchemaError(FinCoachError, ValueError):
    """Raised when an input file does not contain the required columns."""

