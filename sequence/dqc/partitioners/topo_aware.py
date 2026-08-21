#!/usr/bin/env python3
"""Topology-aware placement compiler (Andrés-Martínez et al. 2023)."""
from typing import Dict, Tuple

from sequence.dqc.partitioners.base import PartitionerBase


class TopologyAwarePartitioner(PartitionerBase):
    """Heterogeneous / topology-aware allocator (Andrés-Martínez, Junghans, Heunen
    et al., "Distributing circuits over heterogeneous, modular quantum computing
    network architectures", 2023 -- the pytket-dqc heterogeneous distributor).

    Places qubits on the ACTUAL physical nodes to minimise total communication
    distance

        sum over 2-qubit gates of  weight(i, j) * hop_distance(node(i), node(j))

    subject to each node's data-memory ``capacity``. So a telegate routed several
    hops is penalised by its hop count, and no node exceeds its memory. Method: a
    capacity-respecting greedy seed (natural / degree / BFS orderings) + KL/FM
    local search (single-qubit relocations and pairwise swaps), multi-start.
    """

    def configure_topology(self, edges, capacities: Dict[str, int]):
        """edges: physical (node, node) links. capacities: node_name -> max qubits."""
        self._edges = list(edges)
        self._capacities = dict(capacities)
        return self

    def partition_qubits(self, interaction_weights: Dict[Tuple[int, int], int] = None,
                         seed: int = 0) -> Dict[int, str]:
        import networkx as nx
        n = self.n_qubits
        names = self.node_names
        cap = {nm: int(self._capacities.get(nm, 0)) for nm in names}
        if sum(cap.values()) < n:
            raise ValueError(f"total node capacity {sum(cap.values())} < {n} qubits")

        G = nx.Graph(); G.add_nodes_from(names); G.add_edges_from(self._edges)
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        BIG = 10 ** 6
        def D(a, b):
            return 0 if a == b else apsp.get(a, {}).get(b, BIG)

        adj = {q: {} for q in range(n)}
        for (i, j), w in (interaction_weights or {}).items():
            if i != j and w:
                adj[i][j] = adj[i].get(j, 0) + w
                adj[j][i] = adj[j].get(i, 0) + w

        def total(assign):
            return sum(w * D(assign[i], assign[j])
                       for i in adj for j, w in adj[i].items() if i < j)

        by_deg = sorted(range(n), key=lambda q: -sum(adj[q].values()))
        orders = [list(range(n)), by_deg]
        for root in by_deg[:min(4, n)]:
            orders.append(self._bfs_order(root, adj, n))

        best_assign, best_cost = None, None
        for order in orders:
            assign, load = {}, {nm: 0 for nm in names}
            for q in order:
                cand = [nm for nm in names if load[nm] < cap[nm]]
                nm = min(cand, key=lambda nm: (
                    sum(w * D(nm, assign[nbr]) for nbr, w in adj[q].items() if nbr in assign), load[nm]))
                assign[q] = nm; load[nm] += 1
            assign = self._refine(assign, adj, names, cap, load, D)
            c = total(assign)
            if best_cost is None or c < best_cost:
                best_cost, best_assign = c, dict(assign)

        self.qubit_to_node = {q: best_assign[q] for q in range(n)}
        return self.qubit_to_node

    @staticmethod
    def _bfs_order(root, adj, n):
        """BFS over the interaction graph (heaviest neighbours first), so strongly-
        interacting qubits are placed consecutively -> onto nearby nodes."""
        seen = {root}; order = [root]; queue = [root]
        while queue:
            q = queue.pop(0)
            for nbr, _ in sorted(adj[q].items(), key=lambda kv: -kv[1]):
                if nbr not in seen:
                    seen.add(nbr); order.append(nbr); queue.append(nbr)
        for q in range(n):
            if q not in seen:
                order.append(q)
        return order

    @staticmethod
    def _cost_q(q, node, assign, adj, D):
        return sum(w * D(node, assign[nbr]) for nbr, w in adj[q].items())

    def _refine(self, assign, adj, names, cap, load, D):
        """Local search: relocate single qubits, then swap pairs, until stable."""
        improved = True
        while improved:
            improved = False
            for q in range(len(assign)):
                cur = assign[q]; cur_cost = self._cost_q(q, cur, assign, adj, D)
                best, best_delta = cur, 0
                for nm in names:
                    if nm == cur or load[nm] >= cap[nm]:
                        continue
                    delta = self._cost_q(q, nm, assign, adj, D) - cur_cost
                    if delta < best_delta:
                        best_delta, best = delta, nm
                if best != cur:
                    load[cur] -= 1; load[best] += 1; assign[q] = best; improved = True
            for q in range(len(assign)):
                for r in range(q + 1, len(assign)):
                    na, nb = assign[q], assign[r]
                    if na == nb:
                        continue
                    before = self._cost_q(q, na, assign, adj, D) + self._cost_q(r, nb, assign, adj, D)
                    assign[q], assign[r] = nb, na
                    after = self._cost_q(q, nb, assign, adj, D) + self._cost_q(r, na, assign, adj, D)
                    if after < before - 1e-9:
                        improved = True
                    else:
                        assign[q], assign[r] = na, nb
        return assign

    @staticmethod
    def cut_weight(qubit_to_node, interaction_weights):
        """Telegate count for a placement: 2-qubit gates whose endpoints cross nodes."""
        return sum(w for (i, j), w in interaction_weights.items()
                   if qubit_to_node[i] != qubit_to_node[j])

    @staticmethod
    def weighted_distance(qubit_to_node, interaction_weights, edges):
        """Total sum weight(i,j) * hop_distance -- the objective this minimises."""
        import networkx as nx
        G = nx.Graph(); G.add_edges_from(edges)
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        return sum(w * apsp.get(qubit_to_node[i], {}).get(qubit_to_node[j], 10 ** 6)
                   for (i, j), w in interaction_weights.items()
                   if qubit_to_node[i] != qubit_to_node[j])
