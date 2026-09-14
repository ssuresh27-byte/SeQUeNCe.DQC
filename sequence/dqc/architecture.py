#!/usr/bin/env python3
"""Physical network topologies for distributed quantum computing (DQC).

A :class:`DQCArchitecture` is the *single source of truth* for the physical network:
its nodes (each with a fixed data-memory capacity), its physical links (edges), its
communication-memory size, and optional per-node local noise. It is loaded from /
saved to a lean JSON config, and expands on demand into the full SeQUeNCe
:class:`~sequence.topology.dqc_net_topo.DQCNetTopo` config the simulator consumes
(a BSM node per edge + classical/quantum channels, in the same layout as the
teleport example configs). A compiler reads nodes + capacities + hop-distances from
it; the simulator loads :meth:`sim_config`.

The generators (:func:`make_star`, :func:`make_grid`, :func:`make_caveman`) build on
top of SeQUeNCe's pre-packaged graph builders in :mod:`sequence.utils.graphs`, so DQC
topologies live alongside and reuse the same NetworkX generators the router configs
use. Any NetworkX graph (from ``graphs`` or elsewhere) can be turned into a
DQCArchitecture via :meth:`DQCArchitecture.from_nx_graph`.
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
        self._rebuild_graph()

    @property
    def node_names(self) -> List[str]:
        """Node names in insertion order (derived from ``capacities``)."""
        return list(self.capacities)

    def _rebuild_graph(self) -> None:
        """Recompute the interaction graph + all-pairs hop distances from the current
        nodes/edges. Called after any structural mutation (:meth:`add_node`/:meth:`add_edge`)."""
        self._G = nx.Graph()
        self._G.add_nodes_from(self.capacities)
        self._G.add_edges_from(self.edges)
        self._apsp = dict(nx.all_pairs_shortest_path_length(self._G))

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
        self._rebuild_graph()
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
        self._rebuild_graph()
        return self

    # ── network queries the compiler needs ──────────────────────────────
    def hop(self, a: str, b: str) -> int:
        """Physical hop-distance between two nodes (large if unreachable)."""
        return 0 if a == b else self._apsp.get(a, {}).get(b, 10 ** 6)

    def hop_distances(self) -> Dict[str, Dict[str, int]]:
        """{node: {peer: hops}} for every ordered pair of distinct nodes."""
        return {u: {v: self.hop(u, v) for v in self.node_names if v != u}
                for u in self.node_names}

    @property
    def total_capacity(self) -> int:
        return sum(self.capacities.values())

    @property
    def diameter(self) -> int:
        return nx.diameter(self._G) if self._G.number_of_nodes() > 1 else 0

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

    # ── expansion to the SeQUeNCe DQCNetTopo config ─────────────────────
    def sim_config(self, comm_memo: int = None) -> dict:
        """Full DQCNetTopo config: a BSM node per physical edge + classical/quantum
        channels (teleport-json layout). ``data_memo_size`` is the node's declared
        hardware capacity; per-node noise appears inline on each DQCNode entry.
        """
        comm = self.comm_memo if comm_memo is None else comm_memo
        nodes = [{"name": nm, "type": "DQCNode", "seed": i + 1,
                  "memo_size": max(comm, self.capacities[nm] + 4),
                  "data_memo_size": max(1, self.capacities[nm]),
                  "group": 0, "template": "teleportation",
                  **{k: v for k, v in self.node_noise.get(nm, {}).items()
                     if k in self.NOISE_KEYS}}
                 for i, nm in enumerate(self.node_names)]
        # Fibre length per hop and the matching classical (heralding/control) delay:
        # ~5e6 ps per km (light in fibre ~2e5 km/s). Longer links => each entanglement
        # attempt's herald round-trip takes longer => execution time grows.
        d = self.link_km
        delay = int(d * 1_000_000)
        qch, cch, seed = [], [], 100
        for (u, v) in self.edges:
            bsm = f"BSM_{u}_{v}"
            nodes.append({"name": bsm, "type": "BSMNode", "seed": seed,
                          "group": 0, "template": "teleportation"}); seed += 1
            for r in (u, v):
                qch.append({"source": r, "destination": bsm, "distance": d, "attenuation": 0.0002})
                cch.append({"source": r, "destination": bsm, "delay": delay})
                cch.append({"source": bsm, "destination": r, "delay": delay})
        for a in self.node_names:
            for b in self.node_names:
                if a != b:
                    cch.append({"source": a, "destination": b, "delay": delay})
        # Central controller: a real node DQCNetTopo builds (by policy) with its (named)
        # compiler, wired to every DQC node by classical channels. A small control-plane
        # delay keeps the barrier round-trip from dominating the reported sim time.
        ctrl_delay = 1
        nodes.append({"name": "controller", "type": "Controller", "seed": seed,
                      "policy": self.controller_policy, "compiler": dict(self.compiler_spec)})
        for nm in self.node_names:
            cch.append({"source": "controller", "destination": nm, "delay": ctrl_delay})
            cch.append({"source": nm, "destination": "controller", "delay": ctrl_delay})
        return {"templates": {"teleportation": {"MemoryArray": {"fidelity": 1,
                                                                "efficiency": self.mem_efficiency}}},
                "nodes": nodes, "qchannels": qch, "cchannels": cch,
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
