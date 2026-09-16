#!/usr/bin/env python3
"""DQC topology config generator -- a lean spec that expands into a DQCNetTopo config.

A :class:`DQCArchitecture` is a *config-generation utility*: it captures a lean description
of the physical network (nodes with a data-memory capacity, physical links, comm-memory size,
optional per-node local noise, and the controller policy/compiler choice) and expands it via
:meth:`sim_config` into the full SeQUeNCe :class:`~sequence.topology.dqc_net_topo.DQCNetTopo`
config the simulator consumes (a BSM node per edge + classical/quantum channels + the
controller node, in the teleport-json layout). It does NO simulation and answers NO topology
queries -- once a :class:`DQCNetTopo` is built from the config, THAT is the source of truth
(hop distances, capacities, node names; see :mod:`sequence.dqc.runtime`).

The generators (:func:`make_star`, :func:`make_grid`, :func:`make_caveman`) build on
top of SeQUeNCe's pre-packaged graph builders in :mod:`sequence.utils.graphs`, so DQC
topologies reuse the same NetworkX generators the router configs use. Any NetworkX graph can
be turned into a DQCArchitecture via :meth:`DQCArchitecture.from_nx_graph`.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import networkx as nx
from ..utils import graphs

# Human-friendly node names; falls back to node{k} beyond the pool.
_POOL = ["alice", "bob", "charlie", "david", "eve", "frank", "grace", "heidi",
         "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "rupert", "sybil"]


def node_names(n: int) -> List[str]:
    """First ``n`` human-friendly node names (``node{k}`` beyond the pool)."""
    return [_POOL[i] if i < len(_POOL) else f"node{i}" for i in range(n)]


class DQCArchitecture:
    """Physical DQC network: nodes with data-memory capacity + physical links.

    Build one either all at once (``DQCArchitecture(name, capacities, edges, node_noise=...)``)
    or node-by-node with the builder, which declares each node's per-node noise inline::

        arch = (DQCArchitecture("my_topo")
                .add_node("alice", 2, two_qubit_gate_fid=0.99, t1=1e-3)
                .add_node("bob", 2, measurement_fid=0.98)
                .add_node("charlie", 1)                 # ideal
                .add_edge("alice", "bob")
                .add_edge("bob", "charlie"))

    Attributes:
        name: identifier.
        capacities: {node_name: data-memory capacity (max data qubits)}.
        edges: list of undirected physical links (node_name pairs).
        comm_memo: base communication-memory (EPR) slots per node (may be raised
            by the compiler's provisioning for concurrent multi-hop telegates).
        node_noise: optional {node_name: {one_qubit_gate_fid, two_qubit_gate_fid,
            measurement_fid, t1, t2}} of per-node LOCAL
            (computational) hardware imperfections. Emitted inline on each DQCNode
            entry of the sim config, so each node is instantiated with its own
            fidelities; the ket-vector trajectory noise layer then reads them per
            qubit. Absent nodes default to ideal. (Physical Bell-pair fidelity is a
            link property and belongs in the memory template, not here.)
    """

    # per-node local-noise fields carried through to the DQCNode constructor
    NOISE_KEYS = ("one_qubit_gate_fid", "two_qubit_gate_fid", "measurement_fid", "t1", "t2")

    def __init__(self, name: str, capacities: Dict[str, int] = None,
                 edges: List[Tuple[str, str]] = None, comm_memo: int = 16,
                 node_noise: Dict[str, dict] = None):
        self.name = name
        self.capacities = dict(capacities or {})
        self.edges = [tuple(e) for e in (edges or [])]
        self.comm_memo = comm_memo
        self.node_noise = {nm: dict(p) for nm, p in (node_noise or {}).items()}
        self.link_km = 5.0          # physical fibre length per hop (settable to sweep)
        self.mem_efficiency = 0.9   # memory (Bell-pair) efficiency (settable to sweep)
        # Central controller declared inline in the sim config so DQCNetTopo builds+wires it:
        # its scheduling policy and (named) compiler choice. A custom compiler OBJECT can't
        # live in JSON -- pass it to run() to override the built one.
        self.controller_policy = "barrier"                                  # "barrier" | "adaptive"
        self.compiler_spec = {"partitioner": "topo-aware", "scheduler": "fgp"}

    @property
    def node_names(self) -> List[str]:
        """Node names in insertion order (derived from ``capacities``)."""
        return list(self.capacities)

    # ── incremental construction (node-centric builder) ─────────────────
    def add_node(self, name: str, data_qubits: int = 1, *,
                 one_qubit_gate_fid: float = 1.0, two_qubit_gate_fid: float = 1.0,
                 measurement_fid: float = 1.0, t1: float = None, t2: float = None) -> "DQCArchitecture":
        """Add (or update) a node with its data-memory capacity and per-node local noise.

        Noise knobs left at their ideal defaults are omitted, so a node is ideal unless a
        knob is set; re-adding a node with all-ideal knobs clears any prior noise on it.

        Args:
            name (str): node name.
            data_qubits (int): data-memory capacity (max data qubits on this node).
            one_qubit_gate_fid (float): 1-qubit gate fidelity (1.0 = ideal).
            two_qubit_gate_fid (float): 2-qubit gate fidelity (1.0 = ideal).
            measurement_fid (float): measurement/readout fidelity (1.0 = ideal).
            t1 (float): amplitude-damping (T1) time in seconds (None = off).
            t2 (float): dephasing (T2) time in seconds (None = off).

        Returns:
            DQCArchitecture: self (so calls can be chained).
        """
        self.capacities[name] = data_qubits
        noise = {}
        if one_qubit_gate_fid < 1.0:
            noise["one_qubit_gate_fid"] = one_qubit_gate_fid
        if two_qubit_gate_fid < 1.0:
            noise["two_qubit_gate_fid"] = two_qubit_gate_fid
        if measurement_fid < 1.0:
            noise["measurement_fid"] = measurement_fid
        if t1 is not None:
            noise["t1"] = t1
        if t2 is not None:
            noise["t2"] = t2
        if noise:
            self.node_noise[name] = noise
        else:
            self.node_noise.pop(name, None)
        return self

    def add_edge(self, a: str, b: str) -> "DQCArchitecture":
        """Add a physical link between two already-added nodes.

        Args:
            a (str): one endpoint (must be an added node).
            b (str): other endpoint (must be an added node).

        Returns:
            DQCArchitecture: self (so calls can be chained).
        """
        for nm in (a, b):
            if nm not in self.capacities:
                raise ValueError(f"add_edge: unknown node '{nm}'. Call add_node('{nm}', ...) first.")
        self.edges.append((a, b))
        return self

    # ── construction from a NetworkX graph (reuse sequence.utils.graphs) ──
    @classmethod
    def from_nx_graph(cls, name: str, G: nx.Graph, cap: int, comm_memo: int = 16,
                      node_noise: Dict[str, dict] = None, order=None) -> "DQCArchitecture":
        """Build a DQCArchitecture from a NetworkX graph, mapping its vertices to node
        names (in ``order`` if given, else ``G.nodes()`` order) and giving every node
        the same data-memory ``cap``. Lets any :mod:`sequence.utils.graphs` builder
        (or a custom graph) feed the DQC stack."""
        seq = list(order) if order is not None else list(G.nodes())
        names = node_names(len(seq))
        idx = {v: names[i] for i, v in enumerate(seq)}
        caps = {nm: cap for nm in names}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        return cls(name, caps, edges, comm_memo, node_noise)

    # ── persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = {"name": self.name, "capacities": self.capacities,
             "edges": [list(e) for e in self.edges], "comm_memo": self.comm_memo}
        if self.node_noise:
            d["node_noise"] = self.node_noise
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DQCArchitecture":
        with open(path) as f:
            d = json.load(f)
        return cls(d["name"], d["capacities"], d["edges"], d.get("comm_memo", 16),
                   d.get("node_noise"))

    # ── expansion to the DQCNetTopo config (the DQC parallel of utils.nx_converter) ──
    def generate_nodes(self, comm_memo: int = None) -> List[dict]:
        """DQCNode config entries (parallels :func:`sequence.utils.nx_converter.generate_nodes`).

        ``capacity`` is the LOGICAL data-qubit capacity the compiler partitions against;
        ``data_memo_size`` is the PHYSICAL memory allocation (the runtime may bump it to hold
        every qubit slot); per-node noise appears inline on each DQCNode entry.
        """
        comm = self.comm_memo if comm_memo is None else comm_memo
        return [{"name": nm, "type": "DQCNode", "seed": i + 1,
                 "memo_size": max(comm, self.capacities[nm] + 4),
                 "data_memo_size": max(1, self.capacities[nm]),
                 "capacity": self.capacities[nm],
                 "group": 0, "template": "teleportation",
                 **{k: v for k, v in self.node_noise.get(nm, {}).items() if k in self.NOISE_KEYS}}
                for i, nm in enumerate(self.node_names)]

    def generate_channels(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """Per physical edge, a BSM node + its quantum/classical channels (meet-in-the-middle),
        plus all-pairs classical channels between DQC nodes. Returns
        ``(bsm_nodes, qchannels, cchannels)``. Parallels the edge loop in
        :func:`sequence.utils.nx_converter.generate_config`.
        """
        # Fibre length per hop and the matching classical (heralding) delay: ~5e6 ps/km
        # (light in fibre ~2e5 km/s). Longer links => each herald round-trip takes longer.
        d = self.link_km
        delay = int(d * 1_000_000)
        bsm_nodes, qch, cch, seed = [], [], [], 100
        for (u, v) in self.edges:
            bsm = f"BSM_{u}_{v}"
            bsm_nodes.append({"name": bsm, "type": "BSMNode", "seed": seed,
                              "group": 0, "template": "teleportation"}); seed += 1
            for r in (u, v):
                qch.append({"source": r, "destination": bsm, "distance": d, "attenuation": 0.0002})
                cch.append({"source": r, "destination": bsm, "delay": delay})
                cch.append({"source": bsm, "destination": r, "delay": delay})
        for a in self.node_names:
            for b in self.node_names:
                if a != b:
                    cch.append({"source": a, "destination": b, "delay": delay})
        return bsm_nodes, qch, cch

    def generate_controller(self) -> Tuple[dict, List[dict]]:
        """The central controller node (scheduling policy + named compiler) and its classical
        channels to every DQC node. Returns ``(controller_node, cchannels)``; DQCNetTopo builds
        and wires it. A custom compiler OBJECT can't live in JSON -- pass it to run() to
        override the built one. A small control-plane delay keeps the barrier round-trip from
        dominating the reported sim time.
        """
        node = {"name": "controller", "type": "Controller", "seed": 0,
                "policy": self.controller_policy, "compiler": dict(self.compiler_spec)}
        cch = [ch for nm in self.node_names for ch in
               ({"source": "controller", "destination": nm, "delay": 1},
                {"source": nm, "destination": "controller", "delay": 1})]
        return node, cch

    def sim_config(self, comm_memo: int = None) -> dict:
        """Expand this architecture into the full DQCNetTopo config the simulator consumes
        (teleport-json layout). The DQC parallel of
        :func:`sequence.utils.nx_converter.generate_config`: assemble the DQC nodes, per-edge
        BSM nodes + channels, and the controller from the converter helpers above.
        """
        nodes = self.generate_nodes(comm_memo)
        bsm_nodes, qch, cch = self.generate_channels()
        ctrl_node, ctrl_cch = self.generate_controller()
        return {"templates": {"teleportation": {"MemoryArray": {"fidelity": 1,
                                                                "efficiency": self.mem_efficiency}}},
                "nodes": nodes + bsm_nodes + [ctrl_node],
                "qchannels": qch, "cchannels": cch + ctrl_cch,
                "stop_time": 10_000_000_000_000, "is_parallel": False}


# Back-compat / convenience aliases.
Topology = DQCArchitecture
DQCTopology = DQCArchitecture


# ── generators: build on SeQUeNCe's pre-packaged NetworkX graph builders ──
def make_star(n_nodes: int, cap: int, comm_memo: int = 16, node_noise=None) -> DQCArchitecture:
    """Hub-and-spoke: node 0 is the hub, adjacent to every spoke.

    Wraps :func:`sequence.utils.graphs.build_star` (``star_graph(n_nodes-1)``: one
    center + ``n_nodes-1`` outer nodes)."""
    G = graphs.build_star(n_nodes - 1)
    return DQCArchitecture.from_nx_graph(f"star{n_nodes}_cap{cap}", G, cap, comm_memo,
                                     node_noise, order=range(n_nodes))


def make_grid(rows: int, cols: int, cap: int, comm_memo: int = 16, node_noise=None) -> DQCArchitecture:
    """R×C mesh: node at (r,c) adjacent to Manhattan-distance-1 neighbours.

    Wraps :func:`sequence.utils.graphs.build_grid`; vertices ``(r, c)`` are named
    row-major (``r*cols + c``)."""
    G = graphs.build_grid(rows, cols)
    order = [(r, c) for r in range(rows) for c in range(cols)]
    return DQCArchitecture.from_nx_graph(f"grid{rows}x{cols}_cap{cap}", G, cap, comm_memo,
                                     node_noise, order=order)


def make_caveman(caves: int, size: int, cap: int, comm_memo: int = 16, node_noise=None) -> DQCArchitecture:
    """Connected caveman: ``caves`` cliques of ``size``, chained by one bridge each.

    Note: this keeps full cliques + a path of bridges between consecutive cliques,
    which differs from NetworkX ``connected_caveman_graph`` (used by
    :func:`sequence.utils.graphs.build_caveman`), which rewires one intra-clique edge
    into a ring. Kept for behavioural stability of existing DQC sweeps."""
    names = node_names(caves * size)
    edges = []
    for c in range(caves):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                edges.append((names[base + i], names[base + j]))
    for c in range(caves - 1):
        edges.append((names[c * size + size - 1], names[(c + 1) * size]))
    return DQCArchitecture(f"caveman{caves}x{size}_cap{cap}", {nm: cap for nm in names}, edges,
                       comm_memo, node_noise)
