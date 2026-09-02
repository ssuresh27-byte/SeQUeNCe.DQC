#!/usr/bin/env python3
"""Base class for logical-circuit builders consumed by the DQC compiler.

A :class:`CircuitBuilder` produces a *topology-neutral* logical circuit (a
:class:`sequence.components.circuit.Circuit`): it knows nothing about the physical
network. Placement onto physical nodes and the network itself are handled downstream
by the architecture (:mod:`sequence.dqc.architecture`) and the compiler
(:mod:`sequence.dqc.runtime`). Concrete builders (e.g. an application's Grover
builder) live outside this library and subclass this, implementing :meth:`build`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from sequence.components.circuit import Circuit


class CircuitBuilder(ABC):
    """Abstract base for topology-neutral logical-circuit builders.

    Subclasses implement :meth:`build` to return the logical
    :class:`~sequence.components.circuit.Circuit` the DQC compiler will place and
    schedule onto the physical network.
    """

    @abstractmethod
    def build(self) -> Circuit:
        """Build and return the logical circuit.

        Returns:
            Circuit: the topology-neutral logical circuit.
        """
        raise NotImplementedError
