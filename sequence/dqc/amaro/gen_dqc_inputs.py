#!/usr/bin/env python3
"""Emit a distributed-Grover QMR instance for the Amaro compiler:
  - grover_n<N>.qasm : the Grover circuit (only CX matters to Amaro's router)
  - arch_grover.json : k QPUs (cap slots each), complete intra/cross graph.
Imports the real GroverCircuitBuilder from the parent DQC project.

Run with the DQC project's python (it needs `sequence`):
    python amaro_dqc/gen_dqc_inputs.py <n_data> <qpus> <link_cost> <teleport_cost>
"""
import json, itertools, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DQC = os.path.dirname(HERE)                         # parent project = the DQC repo
sys.path.insert(0, DQC)
from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
from sequence.dqc.circuit_ops import gate_fields


def emit_qasm(n_data, marked, path):
    circ = GroverCircuitBuilder(n_data, [marked]).build_grover_circuit()
    N = circ.size
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    ncx = 0
    for op in circ.gates:
        name, qs, arg = gate_fields(op)
        if name.lower() in ("cx", "cnot") and len(qs) == 2:
            lines.append(f"cx q[{qs[0]}],q[{qs[1]}];"); ncx += 1
    open(path, "w").write("\n".join(lines) + "\n")
    return N, ncx


def emit_arch(n_qubits, qpus, cap, link_cost, teleport_cost, path, swap_cost=1.0):
    nloc = qpus * cap
    assert nloc >= n_qubits, f"{nloc} slots < {n_qubits} qubits"
    qpu_ids = [q for q in range(qpus) for _ in range(cap)]
    edges = [[a, b] for a, b in itertools.permutations(range(nloc), 2)]  # complete
    json.dump({"graph": edges, "num_qpus": qpus, "qpu_ids": qpu_ids,
               "link_cost": link_cost, "swap_cost": swap_cost,
               "teleport_cost": teleport_cost, "alg_qubits": list(range(nloc))},
              open(path, "w"), indent=1)
    return nloc


if __name__ == "__main__":
    n_data = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    qpus   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    lc     = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tc     = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    marked = (1 << n_data) - 1
    N, ncx = emit_qasm(n_data, marked, f"{HERE}/grover_n{n_data}.qasm")
    cap = -(-N // qpus)                                  # ceil
    nloc = emit_arch(N, qpus, cap, lc, tc, f"{HERE}/arch_grover.json")
    print(f"grover n_data={n_data}: {N} qubits, {ncx} CX gates -> "
          f"{qpus} QPUs x cap {cap} ({nloc} slots), link_cost={lc} teleport_cost={tc}")
