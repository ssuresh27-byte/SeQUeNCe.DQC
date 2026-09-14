#!/usr/bin/env python3
"""QuTiP-backed circuit builder: turn a ``qutip_qip`` QubitCircuit into a SeQUeNCe Circuit.

Lets DQC circuits be written in the familiar QuTiP ``QubitCircuit`` form and consumed by
the DQC compiler, which operates on :class:`sequence.components.circuit.Circuit`. Only the
gate set SeQUeNCe supports is translated; any other QuTiP gate raises a clear error.
"""
from __future__ import annotations

from sequence.components.circuit import Circuit
from sequence.dqc.circuits.base import CircuitBuilder


class QutipCircuitBuilder(CircuitBuilder):
    """Translate a :class:`qutip_qip.circuit.QubitCircuit` into a SeQUeNCe
    :class:`~sequence.components.circuit.Circuit`.

    Supported gates: SNOT(H), X, Y, Z, S, T (1-qubit); CNOT/CX, CZ, SWAP (2-qubit);
    TOFFOLI/CCX (3-qubit); and RZ/PHASEGATE/P -> ``phase`` (RZ maps up to an irrelevant
    global phase). QuTiP measurements map to :meth:`Circuit.measure`.

    Args:
        qutip_circuit (QubitCircuit): the source QuTiP circuit.
    """

    _ONE_QUBIT = {"SNOT": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "T": "t"}
    _TWO_QUBIT = {"CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap"}
    _THREE_QUBIT = {"TOFFOLI": "ccx", "CCX": "ccx"}
    _ARG_GATES = {"RZ": "phase", "PHASEGATE": "phase", "P": "phase"}

    def __init__(self, qutip_circuit):
        self.qutip_circuit = qutip_circuit

    def build(self) -> Circuit:
        """Build the equivalent SeQUeNCe circuit.

        Returns:
            Circuit: the translated logical circuit.
        """
        qc = self.qutip_circuit
        circ = Circuit(qc.N)
        for op in qc.gates:
            self._add(circ, op)
        return circ

    def _add(self, circ: Circuit, op) -> None:
        """Translate one QuTiP operation onto ``circ``."""
        name = getattr(op, "name", None)
        if name is None or op.__class__.__name__ == "Measurement":
            for t in (getattr(op, "targets", None) or []):
                circ.measure(t)
            return
        controls = list(op.controls or [])
        targets = list(op.targets or [])
        if name in self._ONE_QUBIT:
            getattr(circ, self._ONE_QUBIT[name])(targets[0])
        elif name in self._ARG_GATES:
            circ.phase(targets[0], op.arg_value)
        elif name in self._TWO_QUBIT:
            if self._TWO_QUBIT[name] == "swap":
                circ.swap(targets[0], targets[1])
            else:
                getattr(circ, self._TWO_QUBIT[name])(controls[0], targets[0])
        elif name in self._THREE_QUBIT:
            circ.ccx(controls[0], controls[1], targets[0])
        else:
            supported = sorted(set(self._ONE_QUBIT) | set(self._TWO_QUBIT)
                               | set(self._THREE_QUBIT) | set(self._ARG_GATES))
            raise ValueError(f"QutipCircuitBuilder: unsupported QuTiP gate '{name}'. "
                             f"Supported: {supported}.")
