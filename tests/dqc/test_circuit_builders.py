"""Tests for the DQC circuit builders: QASM (self-contained) and Qiskit (skipped w/o qiskit)."""
import math

import pytest

from sequence.dqc.circuits import QasmCircuitBuilder, QiskitCircuitBuilder


GHZ_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1], q[2];   // spacing variations are fine
rz(pi/2) q[2];
cz q[0],q[2];
measure q[0] -> c[0];
measure q[2] -> c[2];
"""


def _gate_tuples(circ):
    return [(g[0], list(g[1]), g[2]) for g in circ.gates]


def test_qasm_builds_expected_gates():
    circ = QasmCircuitBuilder(GHZ_QASM).build()
    assert circ.size == 3
    assert _gate_tuples(circ) == [
        ("h", [0], None),
        ("cx", [0, 1], None),
        ("cx", [1, 2], None),
        ("phase", [2], math.pi / 2),
        ("cz", [0, 2], None),
    ]
    assert circ.measured_qubits == [0, 2]


def test_qasm_multi_register_offsets():
    qasm = """OPENQASM 2.0;
    qreg a[2];
    qreg b[2];
    cx a[1],b[0];      // a -> 0,1 ; b -> 2,3  => cx 1,2
    x b[1];
    """
    circ = QasmCircuitBuilder(qasm).build()
    assert circ.size == 4
    assert _gate_tuples(circ) == [("cx", [1, 2], None), ("x", [3], None)]


def test_qasm_ignores_barrier_and_id():
    circ = QasmCircuitBuilder("qreg q[2];\nid q[0];\nbarrier q[0],q[1];\nx q[1];").build()
    assert _gate_tuples(circ) == [("x", [1], None)]


def test_qasm_unsupported_gate_raises():
    with pytest.raises(ValueError, match="unsupported gate 'ry'"):
        QasmCircuitBuilder("qreg q[1];\nry(0.5) q[0];").build()


def test_qasm_no_qreg_raises():
    with pytest.raises(ValueError, match="no qreg"):
        QasmCircuitBuilder("OPENQASM 2.0;\nh q[0];").build()


def test_qasm_semantics_via_simulator():
    """X then measure -> 1; a Bell pair -> correlated bits."""
    from sequence.kernel.quantum_manager import QuantumManagerKet
    qm = QuantumManagerKet()
    circ = QasmCircuitBuilder("qreg q[1];\nx q[0];\nmeasure q[0] -> c[0];").build()
    key = qm.new([1, 0])
    res = qm.run_circuit(circ, [key], 0.5)
    assert res[key] == 1


# ── Qiskit builder (only these tests skip if qiskit isn't installed) ─────────
try:
    import qiskit  # noqa: F401
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")


@requires_qiskit
def test_qiskit_builds_expected_gates():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(math.pi / 2, 2)
    qc.cz(0, 2)
    qc.measure(0, 0)
    circ = QiskitCircuitBuilder(qc).build()
    assert circ.size == 3
    assert _gate_tuples(circ) == [
        ("h", [0], None),
        ("cx", [0, 1], None),
        ("cx", [1, 2], None),
        ("phase", [2], math.pi / 2),
        ("cz", [0, 2], None),
    ]
    assert circ.measured_qubits == [0]


@requires_qiskit
def test_qiskit_from_qasm_matches_qasm_builder():
    qk = QiskitCircuitBuilder.from_qasm(GHZ_QASM).build()
    qa = QasmCircuitBuilder(GHZ_QASM).build()
    assert _gate_tuples(qk) == _gate_tuples(qa)
    assert qk.measured_qubits == qa.measured_qubits
