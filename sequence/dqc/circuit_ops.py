#!/usr/bin/env python3
"""Shared circuit-parsing helpers used by both placement compilers and schedulers.

A DQC ``Circuit`` op is either a ``(name, [qubits], arg?)`` tuple or a SeQUeNCe
gate object exposing ``.name`` / ``.get_all_qubits()``. These free functions
normalise both representations; they used to live on ``DQCCompilerBase``.
"""
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def gate_fields(op) -> Tuple[str, Set[int], Any]:
    """(name, qubits, arg) for a Circuit op in either representation."""
    if isinstance(op, (list, tuple)):
        return op[0], op[1], (op[2] if len(op) > 2 else None)
    return op.name, set(op.get_all_qubits()), getattr(op, "arg", None)


def parse_ops(raw_ops: List[Any]) -> List[Tuple[str, Set[int]]]:
    """List of ``(name, qubit_set)`` for the two-qubit-interaction analysis."""
    parsed = []
    for op in raw_ops:
        if isinstance(op, (list, tuple)) and len(op) >= 2 and isinstance(op[1], list):
            parsed.append((op[0], set(op[1])))
        elif hasattr(op, "name") and hasattr(op, "get_all_qubits"):
            parsed.append((op.name, set(op.get_all_qubits())))
    return parsed


def interaction_weights(parsed_ops) -> Dict[Tuple[int, int], int]:
    """Count of 2-qubit gates between each qubit pair, from parsed ops."""
    w = Counter()
    for _, qs in parsed_ops:
        if len(qs) == 2:
            i, j = sorted(qs)
            w[(i, j)] += 1
    return dict(w)


def get_interaction_weights(circuit) -> Dict[Tuple[int, int], int]:
    """Interaction-weight map for a Circuit (2-qubit-gate count per qubit pair)."""
    raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
    return interaction_weights(parse_ops(raw))


def raw_ops(circuit) -> List[Any]:
    """The circuit's op list, tolerating either ``.ops`` or ``.gates``."""
    return getattr(circuit, "ops", getattr(circuit, "gates", []))
