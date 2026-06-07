"""
Model-adapter layer. Each model FAMILY (separator, diarizer, overlap detector, speaker
identifier, verifier, transcriber) has a small Protocol describing what the pipeline
needs from it, and backends are registered by name. Adding a new backend = write an
adapter + ``@register(...)`` it; nothing else changes (the "growable" goal).
"""
from .registry import register, get_factory, available, ModelRegistry
from . import base

__all__ = ["register", "get_factory", "available", "ModelRegistry", "base"]
