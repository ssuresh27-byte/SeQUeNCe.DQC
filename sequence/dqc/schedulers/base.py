#!/usr/bin/env python3
"""Base class for SCHEDULERS -- decide WHEN each gate runs (assign a controller step).

A scheduler takes a placement ({qubit: node}) plus the circuit and produces the
per-node op buckets (local / remote / target [/ move]) with a ``layer``/``step`` on
each op, packaged as a :class:`~program.CompiledProgram`. Concrete strategies live
in their own modules and subclass :class:`SchedulerBase`, overriding
``compile_circuit``. The one-shot entry point is ``schedule(circuit, topology, placement)``.
Placement (WHERE each qubit lives) is a separate concern handled by :mod:`compilers`.
"""
from typing import Dict, List

from sequence.dqc.circuit_ops import gate_fields


class SchedulerBase:
    """Common machinery for schedulers: slot bookkeeping + op bucketing."""

    def __init__(self, n_qubits: int, n_nodes: int, memory_capacity: int,
                 node_names: List[str] = None):
        self.n_qubits = n_qubits
        self.n_nodes = n_nodes
        self.memory_capacity = memory_capacity
        self.node_names = node_names or ["alice", "bob", "charlie"]
        self.qubit_to_node: Dict[int, str] = {}
        self.data_owners: Dict[str, Dict[int, int]] = {}
        self.buckets: Dict[str, Dict[str, list]] = {}

    # ── optional hooks (routing / capacity aware schedulers override)
    def set_network(self, edges):
        return self

    def set_capacities(self, capacities: Dict[str, int]):
        return self

    def get_data_owners(self, qubit_to_node: Dict[int, str]) -> Dict[str, Dict[int, int]]:
        """{node_name: {global_qubit: local_slot}} for a static placement.
            placement -> per-node data-slot map (FGP overrides: slot == qubit-id)
        """
        data_owners = {}
        for nm in self.node_names:
            data_owners[nm] = {}
            for i, q in enumerate([q for q, n in qubit_to_node.items() if n == nm]):
                data_owners[nm][q] = i
        return data_owners

    def compile_circuit(self, circuit) -> Dict[str, Dict[str, list]]:
        """Return {node: {role: [op_dict]}} with a layer/step on each op."""
        raise NotImplementedError

    # ── shared op-bucketing (schedulers only produce the layers map) ─────────
    def _bucketize(self, raw, layers: Dict[int, int]) -> Dict[str, Dict[str, list]]:
        """Bucket each gate into its owning node(s) as local/remote ops, labelled by
        the controller step (``layer``) it runs in. The scheduler's only job is to
        produce the ``layers`` map (gate index -> step)."""
        self.buckets = {nm: {"local": [], "remote": []} for nm in self.node_names}
        for i, op in enumerate(raw):
            name, qs, arg = gate_fields(op)
            involved = {self.qubit_to_node[q] for q in qs}
            info = {"layer": layers[i], "step": layers[i], "gate": name.lower(),
                    "targets": list(qs), "arg": arg, "nodes": sorted(involved),
                    "description": "-".join(sorted(involved))}
            if len(involved) == 1:
                self.buckets[next(iter(involved))]["local"].append(info)
            else:  # remote: recorded on both control and target nodes
                for nm in involved:
                    self.buckets[nm]["remote"].append(info.copy())
        return self.buckets
