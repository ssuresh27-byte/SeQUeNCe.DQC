#!/usr/bin/env python3
"""Qiskit-backed circuit builder: turn a Qiskit ``QuantumCircuit`` into a SeQUeNCe Circuit.

Lets DQC circuits be written in the familiar Qiskit form and consumed by the DQC compiler, which
operates on :class:`sequence.components.circuit.Circuit`. Only the gate set SeQUeNCe supports is
translated; any other Qiskit gate raises a clear error. Qiskit is imported lazily (only
:meth:`from_qasm` needs it), so this module imports fine without Qiskit installed.
"""
from __future__ import annotations

from sequence.components.circuit import Circuit
from sequence.dqc.circuits.base import CircuitBuilder


class QiskitCircuitBuilder(CircuitBuilder):
    """Translate a :class:`qiskit.QuantumCircuit` into a SeQUeNCe Circuit.

    Supported gates: ``h, x, y, z, s, sdg, t`` (1-qubit); ``cx``, ``cz``, ``swap`` (2-qubit);
    ``ccx`` (3-qubit); ``rz``/``p``/``u1``/``phase`` -> :meth:`Circuit.phase` (``rz`` up to an
    irrelevant global phase); and ``measure`` -> :meth:`Circuit.measure`. ``barrier``/``id`` are
    ignored. Any other gate raises a clear error.

    Args:
        qiskit_circuit (QuantumCircuit): the source Qiskit circuit.
    """

    _ONE_QUBIT = {"h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "sdg", "t": "t"}
    _TWO_QUBIT = {"cx": "cx", "cz": "cz", "swap": "swap"}
    _THREE_QUBIT = {"ccx": "ccx"}
    _ARG_GATES = {"rz": "phase", "p": "phase", "u1": "phase", "phase": "phase"}
    _IGNORE = {"barrier", "id"}

    def __init__(self, qiskit_circuit):
        self.qiskit_circuit = qiskit_circuit

    @classmethod
    def from_qasm(cls, qasm: str) -> "QiskitCircuitBuilder":
        """Build from an OpenQASM 2.0 string via Qiskit's parser (requires ``qiskit``).

        Args:
            qasm (str): OpenQASM 2.0 source.
        """
        from qiskit import QuantumCircuit
        return cls(QuantumCircuit.from_qasm_str(qasm))

    def build(self) -> Circuit:
        """Build the equivalent SeQUeNCe circuit.

        Returns:
            Circuit: the translated logical circuit.
        """
        qc = self.qiskit_circuit
        circ = Circuit(qc.num_qubits)

        def index(qubit) -> int:
            try:
                return qc.find_bit(qubit).index          # modern qiskit
            except Exception:
                return qc.qubits.index(qubit)            # older qiskit

        for instruction in qc.data:
            # CircuitInstruction (modern) or (operation, qargs, cargs) tuple (older).
            op = getattr(instruction, "operation", None)
            if op is None:
                op, qargs = instruction[0], instruction[1]
            else:
                qargs = instruction.qubits
            self._add(circ, op.name.lower(), [index(q) for q in qargs], list(op.params))
        return circ

    def _add(self, circ: Circuit, name: str, qubits, params) -> None:
        """Translate one Qiskit operation onto ``circ``."""
        if name in self._IGNORE:
            return
        if name == "measure":
            circ.measure(qubits[0])
        elif name in self._ONE_QUBIT:
            getattr(circ, self._ONE_QUBIT[name])(qubits[0])
        elif name in self._ARG_GATES:
            circ.phase(qubits[0], float(params[0]))
        elif name in self._TWO_QUBIT:
            gate = self._TWO_QUBIT[name]
            if gate == "swap":
                circ.swap(qubits[0], qubits[1])
            else:
                getattr(circ, gate)(qubits[0], qubits[1])   # (control, target)
        elif name in self._THREE_QUBIT:
            circ.ccx(qubits[0], qubits[1], qubits[2])
        else:
            supported = sorted(set(self._ONE_QUBIT) | set(self._TWO_QUBIT)
                               | set(self._THREE_QUBIT) | set(self._ARG_GATES))
            raise ValueError(f"QiskitCircuitBuilder: unsupported Qiskit gate '{name}'. "
                             f"Supported: {supported} (+ measure).")
