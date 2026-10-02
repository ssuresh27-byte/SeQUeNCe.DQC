#!/usr/bin/env python3
"""QubitRegistry -- the controller-owned logical<->physical<->key map for a DQC run.

One object, created and owned by the central controller (see
:meth:`sequence.dqc.controllers.base.BaseController._init_registry`), that is the single
source of truth for where every logical qubit currently lives and what quantum-manager key
represents it. The controller seeds it from the compiled program's placement, resolves each
op's concrete slots from it at dispatch, and evolves it on move ACK deltas. It is injected
into the noise-aware quantum manager (``qm.registry``) so the noise layer -- one *reader*, not
the owner -- can look up each qubit's owning node for its live fidelities/coherence.

Three related maps:

* ``qubit_to_node`` / ``qubit_to_slot`` -- logical qubit -> its current node name / local
  data-memory slot (the PLACEMENT the controller owns and evolves).
* ``qubit_to_key`` -- logical qubit -> its current quantum-manager key.
* ``node_of`` -- quantum-manager key -> owning :class:`~sequence.topology.node.DQCNode`, used
  by the noise math (fidelities/coherence read LIVE off the node). ``last_touched`` (key ->
  last sim time) backs the idle T1/T2 watermark. A key with no DQC node is ideal.
"""
from __future__ import annotations


class QubitRegistry:
    """The controller-owned logical<->physical<->key map (see module docstring)."""

    def __init__(self) -> None:
        # placement (the logical<->physical map the controller owns and evolves)
        self.qubit_to_node: dict = {}    # logical qubit -> owning node name
        self.qubit_to_slot: dict = {}    # logical qubit -> local data-memory slot on that node
        self.qubit_to_key: dict = {}     # logical qubit -> current qstate key
        # noise routing (read by the noise-aware quantum manager)
        self.node_of: dict = {}          # qstate key -> owning DQCNode
        self.last_touched: dict = {}     # qstate key -> last sim time (for idle T1/T2)

    # ── logical qubit -> physical placement (owned/evolved by the controller) ─
    def seed_placement(self, qubit_to_node: dict, data_owners: dict) -> None:
        """Seed the initial logical->physical map from the compiled program: ``qubit_to_node``
        (logical qubit -> node name) and ``data_owners`` (node name -> {qubit: slot}).

        Physical slots are assigned COMPACTLY here (lowest indices per node), NOT taken from
        ``data_owners``'s values -- those are logical hints only. This is what lets a node use just
        ``capacity + scratch`` physical data memory: each node's resident qubits occupy slots
        ``0..k-1`` regardless of their global ids. Moves then land in a free slot chosen at runtime
        (see the controller's per-step allocation)."""
        self.qubit_to_node = dict(qubit_to_node)
        self.qubit_to_slot = {}
        for node_name, slots in data_owners.items():
            for i, q in enumerate(sorted(slots)):     # compact: 0,1,2,... in qubit-id order
                self.qubit_to_slot[q] = i

    def place(self, qubit: int, node_name: str, slot: int) -> None:
        """Record that logical ``qubit`` now lives on ``node_name`` at ``slot`` (a move delta)."""
        self.qubit_to_node[qubit] = node_name
        self.qubit_to_slot[qubit] = slot

    def location(self, qubit: int):
        """(node_name, slot) of logical ``qubit``, or (None, None) if unplaced."""
        return self.qubit_to_node.get(qubit), self.qubit_to_slot.get(qubit)

    # ── logical qubit -> qstate key ───────────────────────────────────────────
    def set_key(self, qubit: int, key: int) -> None:
        """Record the current quantum-manager ``key`` for logical ``qubit``."""
        self.qubit_to_key[qubit] = key

    def key_of(self, qubit: int):
        """Current quantum-manager key for logical ``qubit`` (None if unmapped)."""
        return self.qubit_to_key.get(qubit)

    # ── key -> node routing (read by the noise math) ──────────────────────────
    def register_qubit(self, key: int, node) -> None:
        """Route qstate ``key`` to its owning DQCNode (fidelities read live from the node)."""
        self.node_of[key] = node

    def fids(self, key: int) -> tuple:
        """(one_qubit_gate_fid, two_qubit_gate_fid, measurement_fid) read live off the owning
        node; all-ideal if ``key`` has no DQC node."""
        node = self.node_of.get(key)
        if node is None:
            return (1.0, 1.0, 1.0)
        return (node.one_qubit_gate_fid, node.two_qubit_gate_fid, node.measurement_fid)

    def noise_active(self, keys) -> bool:
        """True if any of ``keys`` belongs to a registered (noisy) node."""
        return any(k in self.node_of for k in keys)

    def sim_now(self, keys):
        """Current sim time (ps) from a registered node's timeline, or None."""
        for k in keys:
            node = self.node_of.get(k)
            if node is not None:
                return node.timeline.now()
        return None
