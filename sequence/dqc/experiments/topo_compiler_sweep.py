#!/usr/bin/env python3
"""Communication cost across topology x compiler. Fair comparison: same node count
(6) and capacity (3) on every topology -- only the EDGE STRUCTURE differs."""
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run

N, MARKED = 6, 42            # n=6 Grover -> 10 qubits; fits 6 nodes x cap 3 = 18
circ = GroverCircuitBuilder(N, [MARKED]).build_grover_circuit()

TOPOS = {
    "star":    topo.make_star(6, 3),
    "grid":    topo.make_grid(2, 3, 3),
    "caveman": topo.make_caveman(2, 3, 3),
}
COMPILERS = [       # label, partitioner, scheduler
    ("topo-aware", "topo-aware", "packed"),
    ("random",     "random",     "packed"),
    ("qap",        "qap",        "packed"),
    ("fgp",        "topo-aware", "fgp"),
]

def hop_weighted(hops):      # sum of telegate_count * hop_distance
    return sum(h * c for h, c in hops.items())

rows = {}
print(f"{'topology':9} {'compiler':11} {'telegates':>9} {'moves':>6} "
      f"{'hop-wtd':>8} {'max_hop':>7} {'sim_ms':>9} {'steps':>6}")
for tname, T in TOPOS.items():
    for label, part, sched in COMPILERS:
        r = run(circ, T, partitioner=part, scheduler=sched,
                data_qubits=range(N), expected=MARKED)
        rows[(tname, label)] = r
        print(f"{tname:9} {label:11} {r['gates']:>9} {r['moves']:>6} "
              f"{hop_weighted(r['hops']):>8} {r['max_hop']:>7} "
              f"{r['sim_ms']:>9.1f} {r['steps']:>6}", flush=True)

# ---- Plot 1: hop-weighted communication cost, grouped bars ----
labels = [c[0] for c in COMPILERS]
tnames = list(TOPOS)
x = np.arange(len(tnames)); w = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]
for i, lab in enumerate(labels):
    vals = [hop_weighted(rows[(t, lab)]["hops"]) for t in tnames]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(tnames)
ax.set_ylabel("hop-weighted communication cost")
ax.set_xlabel("topology"); ax.set_title(f"Communication cost: topology × compiler (Grover n={N})")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
ax.legend(title="compiler", ncol=4, loc="upper center", framealpha=0.95,
          columnspacing=1.0, handletextpad=0.5)
fig.tight_layout(); fig.savefig("topo_compiler_cost.png", dpi=150)

# ---- Plot 2: execution time, grouped bars ----
fig2, ax2 = plt.subplots(figsize=(8, 5))
for i, lab in enumerate(labels):
    vals = [rows[(t, lab)]["sim_ms"] for t in tnames]
    ax2.bar(x + (i - 1.5) * w, vals, w, label=lab, color=colors[i])
ax2.set_xticks(x); ax2.set_xticklabels(tnames)
ax2.set_ylabel("simulated execution time  (ms)")
ax2.set_xlabel("topology"); ax2.set_title(f"Execution time: topology × compiler (Grover n={N})")
ax2.grid(alpha=0.3, axis="y")
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.28)
ax2.legend(title="compiler", ncol=4, loc="upper center", framealpha=0.95,
           columnspacing=1.0, handletextpad=0.5)
fig2.tight_layout(); fig2.savefig("topo_compiler_time.png", dpi=150)
print("\nwrote topo_compiler_cost.png, topo_compiler_time.png")
