#!/usr/bin/env python3
"""How does execution time scale with the physical distance between nodes?

n=4 Grover on a 2x2 grid. We sweep the per-hop fibre length (`link_km`): longer
links => the classical heralding round-trip of every entanglement-generation
attempt takes longer, so telegates/teledata (and thus the whole run) take longer.
We report simulated execution time (sim_ms = final sim clock) vs distance.
"""
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run

N, MARKED = 4, 15
KM = [1, 5, 10, 25, 50, 100, 200]


def main():
    circ = GroverCircuitBuilder(N, [MARKED]).build_grover_circuit()
    print(f"Grover n={N} on grid 2x2 -- execution time vs per-hop distance\n")
    print(f"{'link_km':>8} {'sim_ms':>12} {'ok':>5}")
    xs, ys = [], []
    for km in KM:
        T = topo.make_grid(2, 2, 3)
        T.link_km = km
        r = run(circ, T, scheduler="fgp", data_qubits=range(N), expected=MARKED)
        xs.append(km); ys.append(r["sim_ms"])
        print(f"{km:>8} {r['sim_ms']:>12.3f} {str(r['ok']):>5}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#C44E52")
    ax.set_xlabel("per-hop fibre length  link_km  (km)")
    ax.set_ylabel("simulated execution time  (ms)")
    ax.set_title(f"Execution time vs inter-node distance  (Grover n={N}, 2x2 grid)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("distance_sweep.png", dpi=130)
    print("\nwrote distance_sweep.png")


if __name__ == "__main__":
    main()
