"""Validate LLM message role sequences before sending to the API."""

from __future__ import annotations

from .core import (
    MessageRole,
    Provider,
    ValidationResult,
    Violation,
    ViolationKind,
    validate,
)

__all__ = [
    "MessageRole",
    "Provider",
    "Violation",
    "ViolationKind",
    "ValidationResult",
    "validate",
]
