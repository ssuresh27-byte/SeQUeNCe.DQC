#!/usr/bin/env python3
"""Resource-constrained scheduler: telegates share a step only if their routed
entanglement paths are node-disjoint."""
from collections import defaultdict

from sequence.dqc.schedulers.base import SchedulerBase
from sequence.dqc.circuit_ops import gate_fields


class ResourceAwareScheduler(SchedulerBase):
    """Resource-constrained scheduler for distributed circuits.

    Like the packed scheduler it schedules gates as-soon-as-possible under
    qubit-dependency, but it also treats the *communication network* as a
    contended resource: two telegates may occupy the same controller step only if
    their entanglement-routing PATHS are node-disjoint. A telegate between distant
    nodes is routed (entanglement swapping) through intermediate nodes, so two
    telegates sharing any node -- as endpoints or as relays -- contend for that
    node's communication qubits and must be serialised.

    Call :meth:`set_network(edges)` to supply the physical links for routing;
    without it, it falls back to plain packed behaviour.
    """

    def set_network(self, edges):
        self._edges = list(edges)
        return self

    def compile_circuit(self, circuit):
        import networkx as nx
        raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
        q2n = self.qubit_to_node
        G = nx.Graph(); G.add_nodes_from(self.node_names)
        G.add_edges_from(getattr(self, "_edges", []))

        def path_nodes(a, b):
            try:
                return set(nx.shortest_path(G, a, b))
            except Exception:
                return {a, b}

        free = {}                       # qubit -> earliest available step
        layer_paths = defaultdict(set)  # step -> physical nodes its telegates route through
        layers = {}
        for i, op in enumerate(raw):
            _, qs, _ = gate_fields(op)
            qs = list(qs)
            earliest = max((free.get(q, 0) for q in qs), default=0)
            involved = {q2n[q] for q in qs}
            if len(involved) >= 2:       # remote telegate: needs a node-disjoint step
                pnodes = path_nodes(q2n[qs[0]], q2n[qs[1]])
                L = earliest
                while layer_paths[L] & pnodes:
                    L += 1
                layer_paths[L] |= pnodes
                layers[i] = L
            else:                        # local gate: only qubit-dependency matters
                layers[i] = earliest
            for q in qs:
                free[q] = layers[i] + 1
        return self._bucketize(raw, layers)
