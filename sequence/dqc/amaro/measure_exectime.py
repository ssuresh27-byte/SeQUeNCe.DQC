#!/usr/bin/env python3
"""Measure end-to-end SeQUeNCe execution time (sim_ms) of the pure-TeleGate strategy
(our compilers) vs the hybrid TeleGate+TeleData strategy (Amaro-selected, realized via
FGP's teleport) for several DQC instances, on a single-hop complete-graph topology
matching the Amaro architecture."""
import logging, os, sys, itertools
logging.disable(logging.CRITICAL)
sys.path.insert(0, "/Users/sanjaysuresh/Desktop/DQC")
from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run
from sequence.dqc.compilers.fgp import FGPCompiler
_CUR = FGPCompiler._CUR

INSTANCES = [(4, 2), (5, 2), (6, 2), (4, 3), (6, 3), (4, 4)]

def complete_topo(k, cap):
    names = topo.node_names(k)
    edges = [(names[i], names[j]) for i in range(k) for j in range(i + 1, k)]
    return topo.Topology(f"complete{k}_cap{cap}", {nm: cap for nm in names}, edges)

print(f"{'n_data':6} {'QPUs':4} {'pureTG_ms':>9} {'TG':>3} | "
      f"{'hybrid_ms':>9} {'TG':>3} {'TD':>3} | {'speedup':>7}")
for n, q in INSTANCES:
    N = GroverCircuitBuilder.total_qubits_for(n)
    cap = -(-N // q)
    circ = GroverCircuitBuilder(n, [(1 << n) - 1], iterations=1).build_grover_circuit()
    T = complete_topo(q, cap)
    FGPCompiler.move_margin = int(2.0 * _CUR)      # pure TG
    a = run(circ, T, partitioner="topo-aware", scheduler="packed", data_qubits=range(n))
    FGPCompiler.move_margin = int(1.0 * _CUR)      # hybrid
    h = run(circ, T, partitioner="topo-aware", scheduler="fgp", data_qubits=range(n))
    sp = a["sim_ms"] / h["sim_ms"] if h["sim_ms"] else float("nan")
    print(f"{n:6} {q:4} {a['sim_ms']:>9.2f} {a['gates']:>3} | "
          f"{h['sim_ms']:>9.2f} {h['gates']:>3} {h['moves']:>3} | {sp:>6.2f}x", flush=True)
