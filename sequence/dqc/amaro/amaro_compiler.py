#!/usr/bin/env python3
"""AmaroCompiler: adapter that turns an Amaro-generated QMR plan into a \dqc
CompiledProgram, so an externally generated compiler executes end-to-end on the
SeQUeNCe network via runtime.run(circuit, topology, compiler=AmaroCompiler(...)).

It emits the circuit's CX gates as OpenQASM and the topology as an Amaro arch JSON,
runs the generated Amaro solver, then translates the returned plan---a per-step qubit
placement plus the CX realizations---into per-node op buckets, recovering the TeleData
moves by diffing consecutive per-step placements. Single-qubit gates (which Amaro does
not route) are interleaved locally, and slots follow the slot==qubit-id convention so the
runtime's live move bookkeeping keeps every map consistent."""
import os, sys, json, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
DQC = os.path.dirname(HERE)
for p in (HERE, DQC):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_hybrid as rh
from sequence.dqc.circuit_ops import gate_fields
from sequence.dqc.program import build_program, stream_to_node_ops


class AmaroCompiler:
    """Run the Amaro solver and execute its plan on \dqc.  Implements the general
    compiler interface: compile(circuit, topology, seed) -> CompiledProgram."""

    SPEC = os.path.join(HERE, "dqc_hybrid.qmrl")

    def __init__(self, link_cost=2.0, teleport_cost=2.0, mode="--amaro"):
        self.link_cost = link_cost
        self.teleport_cost = teleport_cost
        self.mode = mode
        self._cache = {}                       # (n, link, tp) -> node_ops etc.
        self.last = {}                         # diagnostics from the last compile

    def _run_amaro(self, circuit, node_names):
        qpus = len(node_names)
        N = circuit.size
        cap = -(-N // qpus)                     # uniform ceil capacity
        nloc = qpus * cap
        qasm = os.path.join(HERE, "_amaro_run.qasm")
        arch = os.path.join(HERE, "_amaro_run.json")
        # QASM: qreg + this circuit's CX gates (Amaro routes only CX)
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
        for op in circuit.gates:
            name, qs, _ = gate_fields(op)
            if name.lower() in ("cx", "cnot") and len(qs) == 2:
                lines.append(f"cx q[{qs[0]}],q[{qs[1]}];")
        open(qasm, "w").write("\n".join(lines) + "\n")
        # arch: qpus x cap, complete cross graph, one QPU per cap contiguous locations
        qpu_ids = [j for j in range(qpus) for _ in range(cap)]
        edges = [[a, b] for a, b in itertools.permutations(range(nloc), 2)]
        json.dump({"graph": edges, "num_qpus": qpus, "qpu_ids": qpu_ids,
                   "link_cost": self.link_cost, "swap_cost": 1.0,
                   "teleport_cost": self.teleport_cost, "alg_qubits": list(range(nloc))},
                  open(arch, "w"))
        plan = rh.run(self.SPEC, qasm, arch, self.mode)
        return plan, qpu_ids

    def compile(self, circuit, topology, seed=0):
        node_names = list(topology.node_names)
        plan, qpu_ids = self._run_amaro(circuit, node_names)

        def node_of(loc):
            return node_names[qpu_ids[int(loc)]]

        # ---- Execute AMARO'S PLAN, not a per-CX re-derivation. ----------------
        # Amaro reorders the CX gates into its own step schedule and gives, per
        # step, the placement map (qubit -> location). We (a) run the CX gates in
        # AMARO'S step order, (b) interleave each qubit's single-qubit gates in
        # circuit order, and (c) teleport (TeleData) a qubit ONLY when a gate needs
        # it on a different module than it currently sits. This realizes exactly
        # the teleports the plan implies -- walking the circuit in its ORIGINAL
        # order instead makes the placement oscillate and emits a spurious move per
        # cross gate (which made the "hybrid" plan slower than pure TeleGate).
        gate_map, plan_pos, pos = {}, {}, 0        # cx_id -> step map / plan position
        for st in plan["steps"]:
            m = {int(k): int(v) for k, v in st["map"].items()}
            for g in st["implemented_gates"]:
                gid = g["gate"]["id"]
                gate_map[gid] = m
                plan_pos[gid] = pos
                pos += 1

        init_map = {int(k): int(v) for k, v in plan["steps"][0]["map"].items()}
        cur = {q: node_of(l) for q, l in init_map.items()}     # live node per qubit
        qubit_to_node = dict(cur)
        for q in range(circuit.size):                          # any unrouted qubit -> node 0
            qubit_to_node.setdefault(q, node_names[0]); cur.setdefault(q, node_names[0])
        origin = dict(cur)

        # index circuit gates; the k-th CX has Amaro gate-id k (QASM emitted in order)
        gates = list(circuit.gates)
        is_cx = lambda nm: nm.lower() in ("cx", "cnot")
        cx_id_of, per_qubit, k = {}, {}, 0
        for i, op in enumerate(gates):
            nm, qs, _ = gate_fields(op)
            if is_cx(nm):
                cx_id_of[i] = k; k += 1
            for q in qs:
                per_qubit.setdefault(q, []).append(i)

        # global order: CX at its plan position; a 1q gate just before its qubit's
        # NEXT CX (in circuit order) -- preserves per-qubit dependency order.
        LAST = pos + 1
        def key_of(i):
            nm, qs, _ = gate_fields(gates[i])
            if is_cx(nm):
                return (plan_pos.get(cx_id_of[i], LAST), 0.0, i)
            q = next(iter(qs))
            nxt = next((plan_pos.get(cx_id_of[j], LAST) for j in per_qubit[q]
                        if j > i and is_cx(gate_fields(gates[j])[0])), None)
            return ((nxt - 0.5) if nxt is not None else LAST + i * 1e-9, 0.0, i)

        stream, n_tg, n_td = [], 0, 0
        for i in sorted(range(len(gates)), key=key_of):
            nm, qs, arg = gate_fields(gates[i])
            qs = list(qs)
            if is_cx(nm):
                m = gate_map.get(cx_id_of[i])
                if m:
                    for q in qs:                               # lazy TeleData: move only if needed
                        req = node_of(m[q]) if q in m else cur[q]
                        if cur[q] != req:
                            stream.append(("move", q, req, cur[q], q)); cur[q] = req; n_td += 1
                if len({cur[q] for q in qs}) > 1:              # stays cross-module -> TeleGate
                    n_tg += 1
            stream.append(("gate", nm, qs, arg))

        # readout reads each data qubit from its initial owner; return any that
        # ended off-origin (Amaro's Grover plan already ends at origin, so 0 here).
        for q in range(circuit.size):
            if cur.get(q) != origin.get(q):
                stream.append(("move", q, origin[q], cur[q], q)); cur[q] = origin[q]; n_td += 1

        # serialize AMARO'S ordered stream into per-node buckets (shared, neutral
        # helper -- no FGP scheduling involved; the order above IS Amaro's schedule)
        node_ops = stream_to_node_ops(stream, node_names, qubit_to_node)

        data_owners = {n: {} for n in node_names}              # slot == qubit-id
        for q, n in qubit_to_node.items():
            data_owners[n][q] = q

        self.last = {"telegates": n_tg, "teledata": n_td, "cost": plan.get("cost")}
        return build_program(circuit, qubit_to_node, data_owners, node_ops)
