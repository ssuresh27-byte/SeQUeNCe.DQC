#!/usr/bin/env python3
"""The compiled-program artifact shared by the compiler stack and the runtime.

A :class:`CompiledProgram` is everything the simulation runtime needs, produced
once by a compiler (placement) + scheduler (execution): the placement, the
per-node data-slot map, the per-node op buckets (local / remote / target / move)
each tagged with a controller ``step``, and the step bookkeeping the central
controller drives its barrier with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class Op:
    """One logical operation in the compiled DAG.

    ``kind`` is ``"local"`` (single-node gate), ``"remote"`` (cross-node two-qubit gate,
    realized as a telegate), or ``"move"`` (teledata relocation). ``qubits`` are global
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
    steps carrying real network cost -- a telegate (remote/target) or a teledata
    move -- which the controller charges the network delay to."""
    max_step = max((op["layer"] for grp in node_ops.values()
                    for lst in grp.values() for op in lst), default=-1)
    net_layers = {op["layer"] for grp in node_ops.values()
                  for role in ("remote", "target", "move") for op in grp.get(role, [])}
    return max_step, net_layers


def build_op_dag(node_ops: Dict[str, Dict[str, List]]) -> OpDAG:
    """Derive the op-level dependency DAG from per-node op buckets.

    Dedups the ops the buckets record on more than one node (a remote gate lives on both
    parties; a move on both source and dest), orders them by their ASAP ``layer``, and
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
    for i, qubits in order:
        for q in qubits:
            if q in last:
                preds[i].add(last[q])
        for q in qubits:
            last[q] = i
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


def stream_to_node_ops(stream, node_names, qubit_to_node):
    """Serialize an ordered op stream into per-node, per-step buckets.

    Placement- and strategy-agnostic mechanics shared by every compiler: it does
    NOT decide where qubits go or in what order gates run -- the caller's ``stream``
    already fixes that. It only (1) assigns each op a packed ASAP dependency layer,
    and (2) routes it to the right node's ``local`` / ``remote`` / ``move`` bucket
    from the running qubit locations (updated by ``move`` ops). A ``gate`` whose
    qubits span two modules becomes a \telegate (``remote`` on both). Any compiler
    that can emit an ordered stream -- FGP, the Amaro adapter, a hand-written one --
    reuses this to produce the ``node_ops`` a :class:`CompiledProgram` needs.

    Stream ops: ``("gate", name, [qubits], arg)`` and
    ``("move", qubit, dest_node, src_node, dest_slot)``.
    """
    free: Dict[int, int] = {}
    layers: List[int] = []
    for op in stream:
        if op[0] == "move":
            q = op[1]
            L = free.get(q, 0)
            layers.append(L)
            free[q] = L + 1
        else:
            qs = op[2]
            L = max((free.get(q, 0) for q in qs), default=0)
            layers.append(L)
            for q in qs:
                free[q] = L + 1

    buckets = {nm: {"local": [], "remote": [], "target": [], "move": []} for nm in node_names}
    loc = dict(qubit_to_node)
    for op, L in zip(stream, layers):
        if op[0] == "move":
            _, q, dest, src, dest_slot = op
            info = {"layer": L, "step": L, "qubit": q, "dest": dest, "dest_slot": dest_slot,
                    "src": src, "nodes": sorted({src, dest})}
            buckets[src]["move"].append(dict(info))
            buckets[dest]["move"].append(dict(info))
            loc[q] = dest
        else:
            _, name, qs, arg = op
            nodes = {loc[q] for q in qs}
            info = {"layer": L, "step": L, "gate": name.lower(), "targets": list(qs),
                    "arg": arg, "nodes": sorted(nodes), "description": "-".join(sorted(nodes))}
            if len(nodes) == 1:
                buckets[next(iter(nodes))]["local"].append(info)
            else:
                for nm in nodes:
                    buckets[nm]["remote"].append(dict(info))
    return buckets
