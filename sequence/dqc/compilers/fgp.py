#!/usr/bin/env python3
"""FGPCompiler -- Fine-Grained (time-sliced) Partitioning with relaxed OEE.

The literature hybrid telegate + teledata compiler (Baker, Duckering, Hoffman,
Chong, "Time-sliced quantum circuit partitioning for modular architectures",
ACM Computing Frontiers 2020 -- FGP-rOEE). Unlike the static two-stage
:class:`~compilers.pipeline.PipelineCompiler`, this is MONOLITHIC: it chooses a
qubit->module assignment PER TIME SLICE (rOEE local search over a lookahead-
weighted cut, capacity-bound). Between slices a qubit whose module changed is
teleported (a teledata "move"); residual cross-module gates within a slice fall
back to telegates. So it does placement and scheduling together -- there are no
swappable inner partitioner/scheduler layers.

It seeds slice 0 from a placement partitioner (default topo-aware) -- a good seed
matters because with the tuned ``move_margin`` the compiler makes few/no moves on
dense circuits and effectively runs telegates on the seed placement. Data slots
are addressed as slot == qubit-id so relocations never collide.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.partitioners import get_partitioner
from sequence.dqc.program import build_program, stream_to_node_ops
from sequence.dqc.circuit_ops import gate_fields


class FGPCompiler(CompilerBase):
    """FGP-rOEE time-sliced hybrid compiler (see module docstring)."""

    lookahead = 10       # how many future slices influence the current partition
    _CUR = 1_000_000     # weight of current-slice edges (force them to localise)
    # A relocation is accepted only if its cut-gain strictly exceeds move_margin.
    # 1.25*_CUR (strictly between _CUR and 2*_CUR) means a move must localise >=2
    # current-slice gates to pay -- the "amortise only real clusters" rule (a one-off
    # remote interaction stays a telegate). 0 => localise every gate (over-moves);
    # 1.0*_CUR => also take 1-current+lookahead moves (wins on bottleneck topologies,
    # over-moves on meshes). Keep move_margin < 2*_CUR (>=2*_CUR blocks capacity moves).
    move_margin = 5 * _CUR // 4

    def __init__(self, seed_partitioner: str = "topo-aware",
                 lookahead: int = None, move_margin: float = None):
        self.seed_partitioner = seed_partitioner
        if lookahead is not None:
            self.lookahead = lookahead
        if move_margin is not None:
            self.move_margin = move_margin

    # ── compile: seed placement + time-sliced partition/schedule ─────────────
    def compile(self, circuit, topology, seed: int = 0):
        self.n_qubits = circuit.size
        self.node_names = topology.node_names
        self._edges = list(topology.edges)
        self._capacities = dict(topology.capacities)
        # slice-0 seed placement from a real partitioner
        self.qubit_to_node = get_partitioner(
            self.seed_partitioner, self.n_qubits, self.node_names).partition(
            circuit, topology, seed=seed)

        node_ops = self._compile_circuit(circuit)
        # slot == qubit-id, so a relocated qubit keeps its unique slot on any node
        data_owners = {nm: {} for nm in self.node_names}
        for q, nm in self.qubit_to_node.items():
            data_owners[nm][q] = q
        return build_program(circuit, self.qubit_to_node, data_owners, node_ops)

    def _caps(self) -> Dict[str, int]:
        return dict(self._capacities)

    # ── plan ─────────────────────────────────────────────────────────────────
    def _compile_circuit(self, circuit):
        raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
        parsed = [gate_fields(op) for op in raw]                 # (name, qubits, arg)
        slices = self._time_slices(parsed)                       # list of gate-index lists
        caps = self._caps()

        A = dict(self.qubit_to_node)                             # current assignment
        self.move_count = 0
        stream: List[tuple] = []
        for t, gate_idxs in enumerate(slices):
            W = self._lookahead_graph(parsed, slices, t)
            A_new = self._roee(A, W, caps)
            for q in A:                                          # relocations -> teleports
                if A_new[q] != A[q]:
                    stream.append(("move", q, A_new[q], A[q], q))   # dest_slot == q
                    self.move_count += 1
            A = A_new
            for gi in gate_idxs:
                name, qs, arg = parsed[gi]
                stream.append(("gate", name, list(qs), arg))
        return self._schedule_and_bucketize(stream)

    # ── time slices: packed dependency layers ──────────────────────────────────
    @staticmethod
    def _time_slices(parsed) -> List[List[int]]:
        free: Dict[int, int] = {}
        slices: Dict[int, List[int]] = defaultdict(list)
        for i, (_name, qs, _arg) in enumerate(parsed):
            L = max((free.get(q, 0) for q in qs), default=0)
            slices[L].append(i)
            for q in qs:
                free[q] = L + 1
        return [slices[t] for t in sorted(slices)]

    # ── lookahead-weighted interaction graph for slice t ────────────────────────
    def _lookahead_graph(self, parsed, slices, t) -> Dict[Tuple[int, int], float]:
        W: Dict[Tuple[int, int], float] = defaultdict(float)
        last = min(len(slices), t + 1 + self.lookahead)
        for d, tt in enumerate(range(t, last)):
            weight = self._CUR if tt == t else (self.lookahead - (d - 1))
            if weight <= 0:
                continue
            for gi in slices[tt]:
                _name, qs, _arg = parsed[gi]
                qs = list(qs)
                if len(qs) == 2:
                    i, j = sorted(qs)
                    W[(i, j)] += weight
        return W

    # ── relaxed OEE: greedy positive-gain single-moves + swaps, capacity-bound ──
    def _roee(self, A: Dict[int, str], W, caps) -> Dict[int, str]:
        A = dict(A)
        sizes = Counter(A.values())
        nbrs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (i, j), w in W.items():
            nbrs[i].append((j, w))
            nbrs[j].append((i, w))

        def move_gain(q, M):
            cur, g = A[q], 0.0
            for j, w in nbrs.get(q, ()):
                g += w * ((A[j] == M) - (A[j] == cur))
            return g

        def swap_gain(a, b):
            Ma, Mb = A[a], A[b]
            g = 0.0
            for q, other, frm, to in ((a, b, Ma, Mb), (b, a, Mb, Ma)):
                for j, w in nbrs.get(q, ()):
                    if j == other:
                        continue
                    g += w * ((A[j] == to) - (A[j] == frm))
            return g

        qubits = list(A)
        while True:
            best_gain, best = self.move_margin, None
            for q in qubits:                          # single moves into a spare-capacity module
                for M in self.node_names:
                    if M == A[q] or sizes[M] >= caps.get(M, 0):
                        continue
                    g = move_gain(q, M)
                    if g > best_gain:
                        best_gain, best = g, ("move", q, M)
            for a_i in range(len(qubits)):             # balance-preserving swaps
                a = qubits[a_i]
                for b in qubits[a_i + 1:]:
                    if A[a] == A[b]:
                        continue
                    g = swap_gain(a, b)
                    if g > best_gain:
                        best_gain, best = g, ("swap", a, b)
            if best is None:
                return A
            if best[0] == "move":
                _, q, M = best
                sizes[A[q]] -= 1; sizes[M] += 1; A[q] = M
            else:
                _, a, b = best
                A[a], A[b] = A[b], A[a]

    # ── layering (packed by qubit dependency) + bucketize ───────────────────────
    def _schedule_and_bucketize(self, stream):
        # placement/order are already fixed in ``stream``; this is just the shared
        # stream -> per-node buckets serialization (see program.stream_to_node_ops).
        return stream_to_node_ops(stream, self.node_names, self.qubit_to_node)
