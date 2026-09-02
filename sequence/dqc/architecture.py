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

    def __init__(self, name: str, capacities: Dict[str, int],
                 edges: List[Tuple[str, str]], comm_memo: int = 16,
                 node_noise: Dict[str, dict] = None):
        self.name = name
        self.capacities = dict(capacities)
        self.node_names = list(capacities)
        self.edges = [tuple(e) for e in edges]
        self.comm_memo = comm_memo
        self.node_noise = {nm: dict(p) for nm, p in (node_noise or {}).items()}
        self.link_km = 5.0          # physical fibre length per hop (settable to sweep)
        self.mem_efficiency = 0.9   # memory (Bell-pair) efficiency (settable to sweep)
        self._G = nx.Graph()
        self._G.add_nodes_from(self.node_names)
        self._G.add_edges_from(self.edges)
        self._apsp = dict(nx.all_pairs_shortest_path_length(self._G))

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

    @classmethod
    def from_sim_config(cls, config: dict, name: str = None, comm_memo: int = 16) -> "DQCArchitecture":
        """Build a DQCArchitecture from a full DQCNetTopo config (the expanded format):
        per-node capacity = its ``data_memo_size``, physical edges = the DQCNode pairs
        joined by each BSM node, and per-node noise from any inline ``f_*``/``t*`` fields."""
        caps = {n["name"]: n["data_memo_size"] for n in config["nodes"]
                if n.get("type") == "DQCNode"}
        noise = {n["name"]: {k: n[k] for k in cls.NOISE_KEYS if k in n}
                 for n in config["nodes"] if n.get("type") == "DQCNode"}
        noise = {nm: p for nm, p in noise.items() if p}
        bsm = {}
        for qc in config["qchannels"]:
            bsm.setdefault(qc["destination"], []).append(qc["source"])
        edges = set()
        for members in bsm.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.add(tuple(sorted((members[i], members[j]))))
        return cls(name or config.get("name", "loaded"), caps, sorted(edges), comm_memo, noise)

    @classmethod
    def from_file(cls, path: str) -> "DQCArchitecture":
        """Load a topology, auto-detecting the lean format (nodes/edges/capacities)
        vs. a full DQCNetTopo config."""
        import os
        with open(path) as f:
            d = json.load(f)
        if "capacities" in d and "edges" in d:
            return cls(d.get("name", os.path.basename(path)), d["capacities"],
                       d["edges"], d.get("comm_memo", 16), d.get("node_noise"))
        return cls.from_sim_config(d, name=os.path.splitext(os.path.basename(path))[0])

    # ── expansion to the SeQUeNCe DQCNetTopo config ─────────────────────
    def sim_config(self, comm_memo: int = None) -> dict:
        """Full DQCNetTopo config: a BSM node per physical edge + classical/quantum
        channels (teleport-json layout). ``data_memo_size`` is the node's declared
        hardware capacity; per-node noise appears inline on each DQCNode entry."""
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
        return {"templates": {"teleportation": {"MemoryArray": {"fidelity": 1,
                                                                "efficiency": self.mem_efficiency}}},
                "nodes": nodes, "qchannels": qch, "cchannels": cch,
                "stop_time": 10_000_000_000_000, "is_parallel": False}

    def dump_sim_config(self, path: str, comm_memo: int = None, indent: int = 2) -> dict:
        """Write the expanded DQCNetTopo config (same dict :meth:`sim_config` returns,
        teleport-json layout) to ``path`` so it can be inspected or fed straight to
        ``DQCNetTopo``. Per-node noise knobs appear inline on each DQCNode entry.
        Returns the config dict."""
        config = self.sim_config(comm_memo)
        with open(path, "w") as f:
            json.dump(config, f, indent=indent)
        return config


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


# ── attach the central controller as a real node in a built network ──────────
def wire_controller(net, controller, delay: int = 1):
    """Wire a controller node into a built ``DQCNetTopo`` network.

    Creates bidirectional classical channels between ``controller`` and EVERY DQC
    node, so the controller can only exchange messages with nodes it is connected to
    (here, all of them), and registers those nodes on the controller. The controller
    must already be a :class:`~sequence.topology.node.ClassicalNode` created on
    ``net.tl``. ``delay`` (ps) is the per-hop control-channel latency; keep it small so
    the barrier round-trip doesn't dominate the reported sim time.
    """
    from sequence.topology.dqc_net_topo import DQCNetTopo
    from sequence.components.optical_channel import ClassicalChannel
    tl = net.tl
    dqc_nodes = list(net.nodes[DQCNetTopo.DQC_NODE])
    for nd in dqc_nodes:
        c_out = ClassicalChannel(f"cc_{controller.name}_{nd.name}", tl, 0, delay=delay)
        c_out.set_ends(controller, nd.name)        # controller -> node
        c_in = ClassicalChannel(f"cc_{nd.name}_{controller.name}", tl, 0, delay=delay)
        c_in.set_ends(nd, controller.name)         # node -> controller
    controller.set_nodes(dqc_nodes)
    return controller
