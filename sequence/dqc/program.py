#!/usr/bin/env python3
"""The compiled-program artifact shared by the compiler stack and the runtime.

A :class:`CompiledProgram` is everything the simulation runtime needs, produced once by a
compiler (placement + op-generation): the placement, the per-node data-slot map, the per-node
op buckets (local / remote / move / swap), and the op-level dependency :class:`OpDAG`. The DAG's
canonical ASAP layering is stamped back onto the op buckets as each op's ``step`` (see
:func:`build_op_dag`); the barrier replays those waves, the adaptive path walks the DAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class Op:
    """One logical operation in the compiled DAG.

    ``kind`` is ``"local"`` (single-node gate), ``"remote"`` (cross-node two-qubit gate,
    realized as a telegate), ``"move"`` (teledata relocation), or ``"swap"``
    (two coordinated teleports exchanging occupied data slots). ``qubits`` are global
    logical indices; ``nodes`` is the owning node(s) at compile time (informational -- the
    live logical->physical map lives on the controller, not on the op).
    """
    id: int
    kind: str
    qubits: List[int]
    gate: str = None
    arg: float = None
    nodes: List[str] = None
    dest: str = None            # move only: destination node
    layer: int = 0              # compile-time ASAP layer (barrier may re-derive)


@dataclass
class OpDAG:
    """Op-level data-dependency DAG.

    ``ops`` maps id -> :class:`Op`; ``preds``/``succ`` are the dependency edges recovered
    from qubit reuse (an op depends on the most recent earlier op touching a shared qubit).
    A controller executes this directly: the barrier layers it (ASAP) into global waves,
    the adaptive path walks it by dependency.
    """
    ops: Dict[int, Op]
    preds: Dict[int, set]
    succ: Dict[int, set]

    def roots(self) -> List[int]:
        """Ids with no predecessors (ready to run first)."""
        return [i for i, p in self.preds.items() if not p]


@dataclass
class CompiledProgram:
    """Everything the runtime needs, produced once at compile time."""
    circuit: Any
    qubit_to_node: Dict[int, str]
    data_owners: Dict[str, Dict[int, int]]
    node_ops: Dict[str, Dict[str, List]]
    max_step: int
    net_layers: Set[int]
    dag: OpDAG = None           # op-level DAG (built from node_ops; see build_op_dag)


def summarize(node_ops: Dict[str, Dict[str, List]]):
    """(max_step, net_layers) for a node_ops bucket map. ``net_layers`` are the
    steps carrying real network cost -- a telegate (remote) or a teledata move --
    which the controller charges the network delay to."""
    max_step = max((op["layer"] for grp in node_ops.values()
                    for lst in grp.values() for op in lst), default=-1)
    net_layers = {op["layer"] for grp in node_ops.values()
                  for role in ("remote", "move", "swap") for op in grp.get(role, [])}
    return max_step, net_layers


def build_op_dag(node_ops: Dict[str, Dict[str, List]]) -> OpDAG:
    """Derive the op-level dependency DAG from per-node op buckets.

    Dedups the ops the buckets record on more than one node (a remote gate lives on both
    parties; a move or swap on both endpoints), orders them by their ASAP ``layer``, and
    links each op to the most recent earlier op touching a shared qubit. This is the same
    per-qubit dependency the adaptive controller recovers today, lifted to op granularity
    so it can be a first-class program artifact.
    """
    # One Op per dedup key. A remote/move op is recorded on BOTH parties, so collect every
    # source dict per key -- each gets tagged with the resulting op id so the worker can
    # reference an op individually (e.g. for per-op ACKs).
    meta: dict = {}          # dedup key -> (kind, representative op dict, qubits)
    dicts_by_key: dict = {}  # dedup key -> [source op dict, ...]
    for nm, grp in node_ops.items():
        for op in grp.get("local", []):
            key = (op["layer"], nm, tuple(op["targets"]), op["gate"])
            meta[key] = ("local", op, list(op["targets"]))
            dicts_by_key.setdefault(key, []).append(op)
        for op in grp.get("remote", []):
            key = (op["layer"], tuple(sorted(op["targets"])))
            meta[key] = ("remote", op, list(op["targets"]))
            dicts_by_key.setdefault(key, []).append(op)
        for op in grp.get("move", []):
            key = (op["layer"], op["qubit"], op["src"], op["dest"])
            meta[key] = ("move", op, [op["qubit"]])
            dicts_by_key.setdefault(key, []).append(op)

        for op in grp.get("swap", []):
            key = ("swap", op["layer"], tuple(op["qubits"]))
            meta[key] = ("swap", op, list(op["qubits"]))
            dicts_by_key.setdefault(key, []).append(op)

    entries = sorted(meta.items(), key=lambda kv: kv[1][1]["layer"])   # any valid topological order
    ops: Dict[int, Op] = {}
    key_of: Dict[int, tuple] = {}
    order: List[tuple] = []
    for i, (key, (kind, op, qubits)) in enumerate(entries):
        ops[i] = Op(id=i, kind=kind, qubits=qubits, gate=op.get("gate"), arg=op.get("arg"),
                    nodes=op.get("nodes"), dest=op.get("dest"), layer=0)
        key_of[i] = key
        for d in dicts_by_key.get(key, []):
            d["op_id"] = i                     # tag every source op dict with its DAG op id
        order.append((i, qubits))

    preds = {i: set() for i in ops}
    last: dict = {}                       # qubit -> most recent op id touching it
    last_vacate: dict = {}                # (node, slot) -> op id that last freed that data cell
    for i, (key, (kind, op, qubits)) in enumerate(entries):
        for q in qubits:                  # per-qubit dependency
            if q in last:
                preds[i].add(last[q])
        if kind == "move" and op.get("dest_slot") is not None:
            # slot-resource dependency: an arrival into (dest, dest_slot) must follow the move that
            # VACATED that cell, so a freed data slot is reused only after its owner has left it.
            cell = (op["dest"], op["dest_slot"])
            if cell in last_vacate:
                preds[i].add(last_vacate[cell])
        for q in qubits:
            last[q] = i
        if kind == "move" and op.get("src_slot") is not None:
            last_vacate[(op["src"], op["src_slot"])] = i
    succ = {i: set() for i in ops}
    for i, ps in preds.items():
        for p in ps:
            succ[p].add(i)

    # Canonical ASAP layer = longest dependency-chain depth, computed from the DAG itself (NOT
    # from any compile-time scheduler). This IS the layering; the barrier replays it as waves,
    # the adaptive path ignores it. Write it back onto the source op dicts as step/layer so the
    # per-step machinery (barrier broadcast, _ops_for) uses the DAG's layering.
    for i in ops:                         # ops are in topological order (entries sorted above)
        lyr = 0 if not preds[i] else 1 + max(ops[p].layer for p in preds[i])
        ops[i].layer = lyr
        for d in dicts_by_key.get(key_of[i], []):
            d["step"] = lyr
            d["layer"] = lyr
    return OpDAG(ops, preds, succ)


def build_program(circuit, placement: Dict[int, str],
                  data_owners: Dict[str, Dict[int, int]],
                  node_ops: Dict[str, Dict[str, List]]) -> CompiledProgram:
    """Assemble a CompiledProgram from a placement + slot map + op buckets.

    Builds the op-DAG FIRST -- it computes the canonical ASAP layering and stamps it back onto
    the op buckets -- so ``summarize`` (and the barrier's step machinery) read the DAG's
    layering rather than whatever order the op generator emitted."""
    dag = build_op_dag(node_ops)
    max_step, net_layers = summarize(node_ops)
    return CompiledProgram(circuit, dict(placement), data_owners, node_ops,
                           max_step, net_layers, dag)


def stream_to_node_ops(stream, node_names, qubit_to_node, caps=None):
    """Serialize an ordered op stream into per-node, per-step buckets, assigning COMPACT physical
    data slots with STEP-SAFE timing. Returns ``(buckets, data_owners)``.

    The caller's ``stream`` fixes WHERE qubits go and in WHAT ORDER (each move already emitted
    after the departure that frees its destination slot). This function is the single authority
    for PHYSICAL slots and step timing -- so every compiler (FGP, the Amaro adapter, a hand-written
    one) is race-free at exactly ``capacity`` data slots per module:

      * every qubit gets a compact local slot ``0..capacity-1``;
      * a slot vacated by a departure at step ``L`` becomes reusable only at step ``L+1`` -- so two
        same-step teleports never target a cell the hardware has not yet cleared;
      * a move whose destination has no slot free in time is SERIALIZED to a later step (its qubit's
        ``ready`` layer is pushed), and the gate that needs it follows automatically. That extra
        latency is the honest cost of relocating at minimal memory.

    ``caps`` is node -> data capacity (defaults to each node's initial occupancy = no spare).
    Stream ops: ``("gate", name, [qubits], arg)``, ``("move", qubit, dest_node, src_node)``,
    and ``("swap", qubit_a, node_a, qubit_b, node_b)``. Swaps exchange occupied slots.
    """
    import heapq
    caps = dict(caps) if caps else {}
    node_of = dict(qubit_to_node)
    slot_of: Dict[int, int] = {}
    free: Dict[str, list] = {nm: [] for nm in node_names}     # min-heap of (avail_step, slot)
    for nm in node_names:
        resident = sorted(q for q, n in qubit_to_node.items() if n == nm)
        for i, q in enumerate(resident):
            slot_of[q] = i
        for s in range(len(resident), caps.get(nm, len(resident))):
            heapq.heappush(free[nm], (0, s))                  # spare slots free from step 0
    data_owners = {nm: {} for nm in node_names}
    for q, nm in qubit_to_node.items():
        data_owners[nm][q] = slot_of[q]

    ready: Dict[int, int] = {q: 0 for q in qubit_to_node}     # next step each qubit is available
    buckets = {nm: {"local": [], "remote": [], "move": [], "swap": []} for nm in node_names}
    for op in stream:
        if op[0] == "move":
            q, dest, src = op[1], op[2], op[3]            # ("move", q, dest, src[, legacy_slot])
            if not free[dest]:
                raise ValueError(f"stream_to_node_ops: no data slot ever frees on {dest} for the "
                                 f"move of qubit {q} -- compiler emitted an over-capacity relocation.")
            avail, slot = heapq.heappop(free[dest])          # earliest-available slot on dest
            L = max(ready.get(q, 0), avail)                  # wait for both the qubit and the slot
            old_slot = slot_of[q]
            heapq.heappush(free[src], (L + 1, old_slot))     # source slot reusable NEXT step
            slot_of[q] = slot
            node_of[q] = dest
            ready[q] = L + 1
            info = {"layer": L, "step": L, "qubit": q, "dest": dest, "dest_slot": slot,
                    "src": src, "src_slot": old_slot, "nodes": sorted({src, dest})}
            buckets[src]["move"].append(dict(info))
            buckets[dest]["move"].append(dict(info))
        elif op[0] == "swap":
            _, a, na, b, nb = op
            if a == b or na == nb or node_of[a] != na or node_of[b] != nb:
                raise ValueError(f"Invalid swap placement: {op}")
            sa, sb = slot_of[a], slot_of[b]
            L = max(ready[a], ready[b])
            info = {"layer": L, "step": L, "qubits": [a, b], "nodes": [na, nb],
                    "slots": [sa, sb]}
            for nm in (na, nb):
                buckets[nm]["swap"].append(dict(info))
            node_of[a], node_of[b] = nb, na
            slot_of[a], slot_of[b] = sb, sa
            ready[a] = ready[b] = L + 1
        else:
            _, name, qs, arg = op
            L = max((ready.get(q, 0) for q in qs), default=0)
            for q in qs:
                ready[q] = L + 1
            nodes = {node_of[q] for q in qs}
            info = {"layer": L, "step": L, "gate": name.lower(), "targets": list(qs),
                    "arg": arg, "nodes": sorted(nodes)}
            if len(nodes) == 1:
                buckets[next(iter(nodes))]["local"].append(info)
            else:
                for nm in nodes:
                    buckets[nm]["remote"].append(dict(info))
    return buckets, data_owners
