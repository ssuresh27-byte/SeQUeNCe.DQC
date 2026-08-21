#!/usr/bin/env python3
"""Random placement compiler -- the naive, topology-unaware baseline."""
from typing import Dict, Tuple

from sequence.dqc.partitioners.base import PartitionerBase


class RandomPartitioner(PartitionerBase):
    """Assigns each logical qubit to a random node, respecting per-node capacity."""

    def configure_topology(self, edges, capacities: Dict[str, int]):
        """Only capacities are used (edges accepted for a uniform allocator API)."""
        self._capacities = dict(capacities)
        return self

    def partition_qubits(self, interaction_weights: Dict[Tuple[int, int], int] = None,
                         seed: int = 0) -> Dict[int, str]:
        import random as _random
        rng = _random.Random(seed)
        cap = getattr(self, "_capacities", None) or {nm: self.memory_capacity for nm in self.node_names}
        load = {nm: 0 for nm in self.node_names}
        self.qubit_to_node = {}
        for q in range(self.n_qubits):
            free = [nm for nm in self.node_names if load[nm] < cap[nm]]
            if not free:
                raise ValueError("topology capacity exhausted for random placement")
            nm = rng.choice(free)
            self.qubit_to_node[q] = nm; load[nm] += 1
        return self.qubit_to_node
