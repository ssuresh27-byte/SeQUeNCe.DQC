#!/usr/bin/env python3
"""OpenQASM-backed circuit builder: parse an OpenQASM 2.0 program into a SeQUeNCe Circuit.

Self-contained (no external QASM dependency): parses the subset of OpenQASM 2.0 whose gates map
onto SeQUeNCe's :class:`~sequence.components.circuit.Circuit`. Lets DQC circuits be written or
exported as ``.qasm`` -- e.g. from Qiskit, or the QASM the Amaro adapter emits -- and consumed by
the DQC compiler. Any gate outside the supported set raises a clear error.
"""
from __future__ import annotations

import math
import re

from sequence.components.circuit import Circuit
from sequence.dqc.circuits.base import CircuitBuilder


class QasmCircuitBuilder(CircuitBuilder):
    """Translate an OpenQASM 2.0 program into a SeQUeNCe Circuit.

    Supported gates: ``h, x, y, z, s, sdg, t`` (1-qubit); ``cx``/``cnot``, ``cz``, ``swap``
    (2-qubit); ``ccx``/``toffoli`` (3-qubit); ``rz``/``p``/``u1``/``phase`` ``(theta)`` ->
    :meth:`Circuit.phase` (``rz`` up to an irrelevant global phase); and ``measure`` ->
    :meth:`Circuit.measure`. ``barrier``/``id`` are ignored. Multiple ``qreg`` registers are
    concatenated in declaration order, so ``q[i]`` of the k-th register maps to ``offset_k + i``.

    Args:
        qasm (str): the OpenQASM 2.0 source.
    """

    _ONE_QUBIT = {"h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "sdg", "t": "t"}
    _TWO_QUBIT = {"cx": "cx", "cnot": "cx", "cz": "cz", "swap": "swap"}
    _THREE_QUBIT = {"ccx": "ccx", "toffoli": "ccx"}
    _ARG_GATES = {"rz": "phase", "p": "phase", "u1": "phase", "phase": "phase"}
    _IGNORE = {"barrier", "id"}

    def __init__(self, qasm: str):
        self.qasm = qasm

    @classmethod
    def from_file(cls, path: str) -> "QasmCircuitBuilder":
        """Build from a ``.qasm`` file path.

        Args:
            path (str): path to an OpenQASM 2.0 file.
        """
        with open(path) as f:
            return cls(f.read())

    def build(self) -> Circuit:
        """Parse the QASM into the equivalent SeQUeNCe circuit.

        Returns:
            Circuit: the translated logical circuit.
        """
        offsets, total = self._registers()
        circ = Circuit(total)
        for stmt in self._statements():
            self._add(circ, stmt, offsets)
        return circ

    # ── parsing helpers ──────────────────────────────────────────────────────
    def _clean(self) -> str:
        """The source with block (``/* */``) and line (``//``) comments stripped."""
        s = re.sub(r"/\*.*?\*/", "", self.qasm, flags=re.S)
        return re.sub(r"//[^\n]*", "", s)

    def _statements(self):
        """Yield the executable statements (skips headers/declarations)."""
        for raw in self._clean().split(";"):
            stmt = raw.strip()
            if not stmt:
                continue
            if stmt.split()[0] in ("OPENQASM", "include", "qreg", "creg", "gate"):
                continue
            yield stmt

    def _registers(self):
        """(offsets, total): each ``qreg`` name -> its base index, and the total qubit count."""
        offsets, total = {}, 0
        for m in re.finditer(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", self._clean()):
            offsets[m.group(1)] = total
            total += int(m.group(2))
        if not offsets:
            raise ValueError("QasmCircuitBuilder: no qreg declaration found.")
        return offsets, total

    def _qubit(self, ref: str, offsets) -> int:
        """Global qubit index for a ``reg[i]`` reference."""
        m = re.fullmatch(r"(\w+)\s*\[\s*(\d+)\s*\]", ref.strip())
        if not m:
            raise ValueError(f"QasmCircuitBuilder: cannot parse qubit reference '{ref}'.")
        reg = m.group(1)
        if reg not in offsets:
            raise ValueError(f"QasmCircuitBuilder: unknown register '{reg}'.")
        return offsets[reg] + int(m.group(2))

    def _add(self, circ: Circuit, stmt: str, offsets) -> None:
        """Translate one QASM statement onto ``circ``."""
        if stmt.startswith("measure"):                       # measure q[i] -> c[j]
            m = re.match(r"measure\s+(\w+\s*\[\s*\d+\s*\])", stmt)
            if m:
                circ.measure(self._qubit(m.group(1), offsets))
            return
        name = re.match(r"[A-Za-z0-9_]+", stmt).group(0)
        if name.lower() in self._IGNORE:
            return
        m = re.match(r"[A-Za-z0-9_]+\s*(?:\(([^)]*)\))?\s+(.+)", stmt)
        if not m:
            raise ValueError(f"QasmCircuitBuilder: cannot parse statement '{stmt}'.")
        arg_str, qarg_str = m.group(1), m.group(2)
        qubits = [self._qubit(x, offsets) for x in qarg_str.split(",")]
        low = name.lower()
        if low in self._ONE_QUBIT:
            getattr(circ, self._ONE_QUBIT[low])(qubits[0])
        elif low in self._ARG_GATES:
            circ.phase(qubits[0], self._eval_arg(arg_str))
        elif low in self._TWO_QUBIT:
            gate = self._TWO_QUBIT[low]
            if gate == "swap":
                circ.swap(qubits[0], qubits[1])
            else:
                getattr(circ, gate)(qubits[0], qubits[1])   # (control, target)
        elif low in self._THREE_QUBIT:
            circ.ccx(qubits[0], qubits[1], qubits[2])
        else:
            supported = sorted(set(self._ONE_QUBIT) | set(self._TWO_QUBIT)
                               | set(self._THREE_QUBIT) | set(self._ARG_GATES))
            raise ValueError(f"QasmCircuitBuilder: unsupported gate '{name}'. "
                             f"Supported: {supported} (+ measure).")

    @staticmethod
    def _eval_arg(expr) -> float:
        """Evaluate a QASM angle expression (numbers, ``pi``, arithmetic) to a float."""
        if not expr or not expr.strip():
            return 0.0
        return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi, "PI": math.pi}))
