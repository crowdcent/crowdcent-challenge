"""Helpers for the optional full-fidelity training stack."""

from importlib import import_module
from typing import Any


class OptionalTrainingDependencyError(ImportError):
    """Raised when GPU/research dependencies are not installed."""


def import_training_dependency(module: str) -> Any:
    try:
        return import_module(module)
    except ImportError as exc:
        raise OptionalTrainingDependencyError(
            f"Full-fidelity validation requires optional module {module!r}. "
            "Install the project's validation-training extra."
        ) from exc
