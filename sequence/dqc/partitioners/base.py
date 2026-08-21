#!/usr/bin/env python3
"""Base class for placement PARTITIONERS — decide WHERE each qubit lives.

A partitioner maps logical qubits onto physical nodes, respecting per-node memory
capacity. Concrete strategies (topo-aware, random, qap) live in their
own modules and subclass :class:`PartitionerBase`, overriding ``configure_topology``
and ``partition_qubits``. The one-shot entry point is ``partition(circuit, topology)``.
A partitioner + a scheduler compose into a static COMPILER (see :mod:`compilers`).
"""
from typing import Dict, List, Tuple

from sequence.dqc.circuit_ops import get_interaction_weights


class PartitionerBase:
    """Common surface for placement partitioners."""

    def __init__(self, n_qubits: int, n_nodes: int, memory_capacity: int,
                 node_names: List[str] = None):
        self.n_qubits = n_qubits
        self.n_nodes = n_nodes
        self.memory_capacity = memory_capacity
        self.node_names = node_names or ["alice", "bob", "charlie"]
        self.qubit_to_node: Dict[int, str] = {}

    # ── to override ──────────────────────────────────────────────────────────
    def configure_topology(self, edges, capacities: Dict[str, int]):
        """Accept the physical (node, node) links + node_name -> capacity map."""
        raise NotImplementedError

    def partition_qubits(self, interaction_weights: Dict[Tuple[int, int], int] = None,
                         seed: int = 0) -> Dict[int, str]:
        """Return a {qubit_id: node_name} placement respecting capacity."""
        raise NotImplementedError

    # ── one-shot entry point ─────────────────────────────────────────────────
    def partition(self, circuit, topology, seed: int = 0) -> Dict[int, str]:
        """Place ``circuit``'s qubits onto ``topology`` and return the placement."""
        self.configure_topology(topology.edges, topology.capacities)
        iw = get_interaction_weights(circuit)
        return self.partition_qubits(iw, seed=seed)
