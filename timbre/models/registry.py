"""
Name -> backend registry. A backend registers a FACTORY (a zero/low-arg callable that
builds the adapter, lazily importing heavy deps only when actually invoked). The registry
itself holds no heavy state, so importing it and listing available backends is cheap and
test-friendly.

Usage:
    @register("diarizer", "pyannote")
    def _make_pyannote_diarizer():
        from timbre.models.pyannote import PyannoteDiarizer
        return PyannoteDiarizer(...)

    factory = get_factory("diarizer", "pyannote")
    diarizer = factory()
"""
from __future__ import annotations

from typing import Callable

from .base import FAMILIES

_REGISTRY: dict[tuple[str, str], Callable[..., object]] = {}


class ModelRegistry:
    """Thin namespaced view over the module-level registry (for tests / introspection)."""

    @staticmethod
    def clear() -> None:
        _REGISTRY.clear()


def register(family: str, name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator: register a backend factory under (family, name)."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family '{family}'. Known: {FAMILIES}")

    def _decorator(factory: Callable[..., object]) -> Callable[..., object]:
        _REGISTRY[(family, name)] = factory
        return factory

    return _decorator


def get_factory(family: str, name: str) -> Callable[..., object]:
    """Return the registered factory for (family, name) or raise KeyError with a hint."""
    try:
        return _REGISTRY[(family, name)]
    except KeyError:
        known = available(family)
        raise KeyError(
            f"No '{name}' registered for family '{family}'. Available: {known}"
        ) from None


def available(family: str | None = None) -> list[str]:
    """List registered backend names (optionally filtered to one family)."""
    if family is None:
        return sorted(f"{fam}:{nm}" for (fam, nm) in _REGISTRY)
    return sorted(nm for (fam, nm) in _REGISTRY if fam == family)
