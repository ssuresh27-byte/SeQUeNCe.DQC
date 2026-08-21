#!/usr/bin/env python3
"""Amaro vs our compilers on distributed Grover (n=4, 6 qubits, 2 QPUs, 16 cross
interactions). Data are the MEASURED runs (see FINDINGS.md).

Panel A: telegate/teledata mix per compiler -- only Amaro produces hybrid plans.
Panel B: total communication cost vs telegate price -- pure-telegate (ours) grows
         linearly; Amaro caps it by switching to teledata teleports."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12})

TELEPORT = 2.0

# --- measured results ---------------------------------------------------------
# ours (topo-aware / qap / fgp): pure telegate, 16 telegates, 0 teledata (any price)
# Amaro (--amaro joint solver) at link_cost = 1..5, teleport_cost = 2:
AMARO = {1: (16, 0), 2: (2, 4), 3: (0, 8), 4: (0, 8), 5: (0, 8)}  # link -> (telegates, teleports)

# Panel A: mix at a representative price (link=2, where Amaro is genuinely hybrid)
barsA = [("topo-aware", 16, 0), ("qap", 16, 0), ("fgp", 16, 0),
         ("Amaro\n(link=1)", *AMARO[1]), ("Amaro\n(link=2)", *AMARO[2]),
         ("Amaro\n(link=3)", *AMARO[3])]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

labels = [b[0] for b in barsA]
tg = np.array([b[1] for b in barsA]); td = np.array([b[2] for b in barsA])
x = np.arange(len(labels))
axA.bar(x, tg, 0.62, label="TeleGate (remote gate)", color="#4C72B0")
axA.bar(x, td, 0.62, bottom=tg, label="TeleData (teleport move)", color="#C44E52")
for i, (t, d) in enumerate(zip(tg, td)):
    if d: axA.text(i, t + d + 0.3, f"{t}+{d}", ha="center", fontsize=9)
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=11)
axA.set_ylabel("communication operations")
axA.set_title("A. Only Amaro produces hybrid TeleGate+TeleData plans")
axA.set_ylim(0, max(tg) + 8)
axA.legend(loc="upper center", ncol=2, framealpha=0.95); axA.grid(alpha=0.3, axis="y")
axA.axvline(2.5, color="grey", ls=":", lw=1)
axA.text(1.0, max(tg)+2.0, "our compilers", ha="center", fontsize=11, color="grey")
axA.text(4.0, max(tg)+2.0, "Amaro (this work)", ha="center", fontsize=11, color="grey")

# Panel B: total cost vs telegate price
links = np.array(sorted(AMARO))
pure_tg = 16 * links                                    # ours: 16 telegates always
amaro_cost = np.array([tg*L + tp*TELEPORT for L, (tg, tp) in
                       ((L, AMARO[L]) for L in links)])
axB.plot(links, pure_tg, "o-", color="#4C72B0", label="pure TeleGate (ours: fgp / topo-aware / qap)")
axB.plot(links, amaro_cost, "s-", color="#C44E52", label="Amaro (hybrid, this work)")
axB.fill_between(links, amaro_cost, pure_tg, color="#C44E52", alpha=0.12)
for L in links:
    tg_, tp_ = AMARO[L]
    axB.annotate(f"{tg_}TG+{tp_}TD", (L, amaro_cost[list(links).index(L)]),
                 textcoords="offset points", xytext=(0, -16), ha="center", fontsize=11, color="#C44E52")
axB.set_xlabel("TeleGate price  (link_cost;  teleport_cost = 2)")
axB.set_ylabel("total communication cost")
axB.set_title("B. Amaro caps cost by switching to TeleData")
axB.set_xticks(links); axB.legend(loc="upper left", framealpha=0.95); axB.grid(alpha=0.3)

fig.suptitle("Distributed Grover (n=4, 6 qubits, 2 QPUs, 16 cross-QPU interactions)", y=1.02)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amaro_vs_ours.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
