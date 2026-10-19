"""Exceptions for OptiGraph canonical graph persistence."""

from __future__ import annotations


class OptiGraphPersistenceError(Exception):
    """Base exception for all OptiGraph persistence operations."""


class OptiGraphLoadError(OptiGraphPersistenceError):
    """Raised when loading canonical graph fails (e.g. malformed JSON or unreadable file)."""


class OptiGraphFileNotFoundError(OptiGraphLoadError, FileNotFoundError):
    """Raised when the specified canonical graph file is not found."""


class OptiGraphValidationError(OptiGraphLoadError, ValueError):
    """Raised when canonical graph data violates schema, types, or integrity rules."""


class OptiGraphSaveError(OptiGraphPersistenceError, ValueError):
    """Raised when saving canonical graph fails or target path is unsafe."""
