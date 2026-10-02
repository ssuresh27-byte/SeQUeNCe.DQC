#!/usr/bin/env python3
"""Placement via the Quadratic Assignment Problem (scipy FAQ solver)."""
from typing import Dict, Tuple

from sequence.dqc.partitioners.base import PartitionerBase


class QAPPartitioner(PartitionerBase):
    """Placement via the Quadratic Assignment Problem (scipy FAQ solver).

    The runtime's total telegate cost is exactly a QAP:
        minimise  sum_{i,j} weight(i,j) * hop_distance(node(i), node(j))
    with facilities = qubits, locations = node "slots" (node repeated `capacity`
    times), flow = interaction weight, distance = hop distance. Solved by
    scipy.optimize.quadratic_assignment (Fast Approximate QAP), multi-start.

    This matches the one-pair-per-telegate runtime's cost model EXACTLY (no
    entanglement reuse) -- a canonical, library-backed solver for the same
    objective the greedy TopologyAwarePartitioner targets.
    """

    def configure_topology(self, edges, capacities: Dict[str, int]):
        self._edges = list(edges)
        self._capacities = dict(capacities)
        return self

    def partition_qubits(self, interaction_weights: Dict[Tuple[int, int], int] = None,
                         seed: int = 0, starts: int = 6) -> Dict[int, str]:
        import numpy as np
        import networkx as nx
        from scipy.optimize import quadratic_assignment
        n = self.n_qubits
        names = self.node_names
        caps = {nm: int(self._capacities[nm]) for nm in names}
        n_slots = sum(caps.values())
        if n_slots < n:
            raise ValueError(f"total capacity {n_slots} < {n} qubits")

        slot_node = [nm for nm in names for _ in range(caps[nm])]   # slot -> node
        G = nx.Graph(); G.add_nodes_from(names); G.add_edges_from(self._edges)
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        def hop(a, b):
            return 0 if a == b else apsp.get(a, {}).get(b, 10 ** 6)

        D = np.array([[hop(slot_node[a], slot_node[b]) for b in range(n_slots)]
                      for a in range(n_slots)], dtype=float)
        F = np.zeros((n_slots, n_slots))                            # flow (qubits padded w/ dummies)
        for (i, j), w in (interaction_weights or {}).items():
            if i < n and j < n and i != j:
                F[i][j] += w; F[j][i] += w

        best = None
        for k in range(max(1, starts)):
            # Seed the FAQ solver's OWN rng (scipy>=1.9 ignores the legacy np.random global state),
            # so multi-start placement is reproducible: same (seed, circuit, topology) -> same plan.
            opts = {"rng": seed + k} if k == 0 else {"P0": "randomized", "rng": seed + k}
            res = quadratic_assignment(F, D, method="faq", options=opts)
            if best is None or res.fun < best.fun:
                best = res
        col = best.col_ind                                          # facility q -> slot col[q]
        self.qubit_to_node = {q: slot_node[col[q]] for q in range(n)}
        return self.qubit_to_node
