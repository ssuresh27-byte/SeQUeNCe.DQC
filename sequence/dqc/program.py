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
class CompiledProgram:
    """Everything the runtime needs, produced once at compile time."""
    circuit: Any
    qubit_to_node: Dict[int, str]
    data_owners: Dict[str, Dict[int, int]]
    node_ops: Dict[str, Dict[str, List]]
    max_step: int
    net_layers: Set[int]


def summarize(node_ops: Dict[str, Dict[str, List]]):
    """(max_step, net_layers) for a node_ops bucket map. ``net_layers`` are the
    steps carrying real network cost -- a telegate (remote/target) or a teledata
    move -- which the controller charges the network delay to."""
    max_step = max((op["layer"] for grp in node_ops.values()
                    for lst in grp.values() for op in lst), default=-1)
    net_layers = {op["layer"] for grp in node_ops.values()
                  for role in ("remote", "target", "move") for op in grp.get(role, [])}
    return max_step, net_layers


def build_program(circuit, placement: Dict[int, str],
                  data_owners: Dict[str, Dict[int, int]],
                  node_ops: Dict[str, Dict[str, List]]) -> CompiledProgram:
    """Assemble a CompiledProgram from a placement + slot map + op buckets."""
    max_step, net_layers = summarize(node_ops)
    return CompiledProgram(circuit, dict(placement), data_owners, node_ops,
                           max_step, net_layers)


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
