"""DQC circuits subpackage: the :class:`CircuitBuilder` framework base.

Concrete circuit builders (e.g. application-specific Grover builders) live outside
this library and subclass :class:`CircuitBuilder`.
"""
from .base import CircuitBuilder

__all__ = ["CircuitBuilder"]
