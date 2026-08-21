"""Regression tests for the FGP-rOEE hybrid telegate+teledata scheduler.

Guards that the time-sliced partitioner produces a LOGICALLY correct plan and
RUNS correctly on 1-hop-move topologies (grid, caveman). Star (concurrent
multi-hop moves through a shared hub) is a known-open case, tracked separately.
"""
import contextlib
import io
import math

import numpy as np
import pytest

from functools import reduce
from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
import sequence.dqc.runtime as R


def _grover(n):
    marked = 11 % 2 ** n
    iters = max(1, round(math.pi / 4 * math.sqrt(2 ** n)))
    return GroverCircuitBuilder(n, [marked], iters).build_grover_circuit(), marked


def _plan_statevector(prog):
    """Classically replay the FGP plan's gates (layer order) on circ.size qubits."""
    N = prog.circuit.size
    I = np.eye(2); H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    X = np.array([[0, 1.], [1, 0]]); Z = np.array([[1, 0], [0, -1.]])
    def ph(a): return np.array([[1, 0], [0, np.exp(1j * a)]])

    def kl(ms): return reduce(np.kron, ms)
    def g1(g, q): return kl([g if i == q else I for i in range(N)])
    def ctrl(gate, c, t):
        P0 = np.array([[1, 0], [0, 0]]); P1 = np.array([[0, 0], [0, 1]])
        return (kl([P0 if i == c else I for i in range(N)])
                + kl([P1 if i == c else (gate if i == t else I) for i in range(N)]))

    gates, seen = [], set()
    for _nm, g in prog.node_ops.items():
        for op in g["local"]:
            gates.append((op["layer"], op["gate"], tuple(op["targets"]), op.get("arg")))
        for op in g["remote"]:
            k = (op["layer"], tuple(sorted(op["targets"])))
            if k in seen:
                continue
            seen.add(k)
            gates.append((op["layer"], op["gate"], tuple(op["targets"]), op.get("arg")))
    gates.sort(key=lambda x: x[0])

    sv = np.zeros(2 ** N, dtype=complex); sv[0] = 1
    for _L, nm, qs, arg in gates:
        if nm in ("h", "x", "z", "t"):
            M = {"h": H, "x": X, "z": Z, "t": ph(np.pi / 4)}[nm]; sv = g1(M, qs[0]) @ sv
        elif nm == "phase":
            sv = g1(ph(arg or 0), qs[0]) @ sv
        elif nm == "cx":
            sv = ctrl(X, qs[0], qs[1]) @ sv
        elif nm == "cz":
            sv = ctrl(Z, qs[0], qs[1]) @ sv
    return sv


def _packed_statevector(circuit):
    """Reference: replay the raw circuit."""
    class _P:  # tiny shim so _plan_statevector can reuse the machinery
        pass
    N = circuit.size
    I = np.eye(2); H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    X = np.array([[0, 1.], [1, 0]]); Z = np.array([[1, 0], [0, -1.]])
    def ph(a): return np.array([[1, 0], [0, np.exp(1j * a)]])
    def kl(ms): return reduce(np.kron, ms)
    def g1(g, q): return kl([g if i == q else I for i in range(N)])
    def ctrl(gate, c, t):
        P0 = np.array([[1, 0], [0, 0]]); P1 = np.array([[0, 0], [0, 1]])
        return (kl([P0 if i == c else I for i in range(N)])
                + kl([P1 if i == c else (gate if i == t else I) for i in range(N)]))
    sv = np.zeros(2 ** N, dtype=complex); sv[0] = 1
    for op in circuit.gates:
        nm = (op[0] if isinstance(op, (list, tuple)) else op.name).lower()
        qs = list(op[1] if isinstance(op, (list, tuple)) else op.get_all_qubits())
        arg = (op[2] if isinstance(op, (list, tuple)) and len(op) > 2 else getattr(op, "arg", None))
        if nm in ("h", "x", "z", "t"):
            sv = g1({"h": H, "x": X, "z": Z, "t": ph(np.pi / 4)}[nm], qs[0]) @ sv
        elif nm == "phase":
            sv = g1(ph(arg or 0), qs[0]) @ sv
        elif nm == "cx":
            sv = ctrl(X, qs[0], qs[1]) @ sv
        elif nm == "cz":
            sv = ctrl(Z, qs[0], qs[1]) @ sv
    return sv


def test_fgp_plan_is_logically_correct():
    """The FGP plan must reproduce the original circuit's statevector exactly."""
    circ, _ = _grover(4)
    prog = R.compile_program(circ, topo.make_caveman(1, 4, 3), partitioner="qap", scheduler="fgp", seed=0)
    assert np.allclose(np.abs(_plan_statevector(prog)) ** 2,
                       np.abs(_packed_statevector(circ)) ** 2, atol=1e-6)


def test_fgp_no_slot_collisions():
    """slot==qubit-id must give a collision-free relocation schedule."""
    from collections import defaultdict
    circ, _ = _grover(4)
    prog = R.compile_program(circ, topo.make_caveman(1, 4, 3), partitioner="qap", scheduler="fgp", seed=0)
    pos = {q: (nm, s) for nm, d in prog.data_owners.items() for q, s in d.items()}
    mbl, seen = defaultdict(list), set()
    for nm, g in prog.node_ops.items():
        for op in g.get("move", []):
            k = (op["layer"], op["qubit"])
            if k in seen:
                continue
            seen.add(k); mbl[op["layer"]].append(op)
    for L in sorted(mbl):
        for op in mbl[L]:
            pos[op["qubit"]] = (op["dest"], op["dest_slot"])
        cells = defaultdict(list)
        for q, cell in pos.items():
            cells[cell].append(q)
        assert all(len(qs) == 1 for qs in cells.values()), f"slot collision at layer {L}"


@pytest.mark.parametrize("topo_fn", [
    lambda n: topo.make_star(4, 4),
    lambda n: topo.make_grid(2, 2, 4),
    lambda n: topo.make_caveman(2, 2, 4),
])
def test_fgp_runs_correctly(topo_fn):
    """FGP executes Grover correctly on star / grid / caveman."""
    n = 4
    circ, marked = _grover(n)
    T = topo_fn(n)
    prog = R.compile_program(circ, T, partitioner="qap", scheduler="fgp", seed=0)
    with contextlib.redirect_stderr(io.StringIO()):
        r = R.run(circ, T, partitioner="qap", scheduler="fgp", data_qubits=range(n), expected=marked)
    assert r["ok"], f"measured={r['measured']} expected={marked}"
