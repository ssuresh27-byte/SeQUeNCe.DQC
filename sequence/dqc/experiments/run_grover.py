#!/usr/bin/env python3
"""Example: distributed Grover across compilers/schedulers and topologies.

Run from the repo root:  python examples/run_grover.py

Shows the whole pipeline: build a circuit + a topology, then let the central
controller (a real node in the network) compile with a chosen placement compiler
and scheduler and run it, over classical channels.
"""
import logging, os, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run

N_DATA, MARKED = 4, 15
circuit = GroverCircuitBuilder(N_DATA, [MARKED]).build_grover_circuit()

topologies = {
    "star":    topo.make_star(4, 3),
    "grid":    topo.make_grid(2, 2, 3),
    "caveman": topo.make_caveman(2, 2, 3),
}

print(f"Grover n={N_DATA} (marked={MARKED}); compiler=topo-aware\n")
print(f"{'topology':10} {'scheduler':10} {'ok':5} {'measured':>8} {'telegates':>9} {'moves':>6} {'sim_ms':>8}")
for tname, T in topologies.items():
    for sched in ("packed", "fgp"):
        r = run(circuit, T, partitioner="topo-aware", scheduler=sched,
                data_qubits=range(N_DATA), expected=MARKED)
        print(f"{tname:10} {sched:10} {str(r['ok']):5} {r['measured']:>8} "
              f"{r['gates']:>9} {r['moves']:>6} {r['sim_ms']:>8.2f}")
