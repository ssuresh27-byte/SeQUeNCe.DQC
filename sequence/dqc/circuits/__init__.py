"""DQC circuits subpackage: the :class:`CircuitBuilder` framework base + builders.

Concrete circuit builders subclass :class:`CircuitBuilder` and return a topology-neutral SeQUeNCe
:class:`~sequence.components.circuit.Circuit`. Front-end builders translate a circuit written in a
common form into that:

* :class:`QutipCircuitBuilder` -- from a ``qutip_qip`` ``QubitCircuit``;
* :class:`QiskitCircuitBuilder` -- from a Qiskit ``QuantumCircuit`` (Qiskit imported lazily);
* :class:`QasmCircuitBuilder` -- from an OpenQASM 2.0 string (self-contained parser, no deps).

Application-specific builders (e.g. Grover) live outside this library.
"""
from .base import CircuitBuilder
from .qutip_builder import QutipCircuitBuilder
from .qiskit_builder import QiskitCircuitBuilder
from .qasm_builder import QasmCircuitBuilder

__all__ = ["CircuitBuilder", "QutipCircuitBuilder", "QiskitCircuitBuilder", "QasmCircuitBuilder"]
