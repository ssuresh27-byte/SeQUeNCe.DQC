#!/usr/bin/env python3
"""Runtime harness: build the network, run, measure.

``run(circuit, topology, ...)`` is the single entry point; it constructs a :class:`_Run` (which
holds the config for the invocation) and drives its shot loop. Per shot, :meth:`_Run._one_shot`
expands the topology to a DQCNetTopo config (which builds the DQC nodes + the central controller +
channels), has the controller compile the program and drive its barrier over classical channels,
then reads out the data qubits. The per-shot pipeline reads top-to-bottom: ``_build_config`` ->
build net -> ``ctrl.compile`` -> ``_register_noise`` -> ``_attach_agents`` -> run -> ``_measure``.

Purification never fires: ``purification_policy`` sets a LOW reservation target fidelity, so a
multi-hop pair's swap-degraded bookkept fidelity still clears it and no distillation is ever
requested -- required for correct multi-hop teledata moves. All trajectory noise is native to
the quantum manager (per-node, self-registered by each DQCNode). No monkeypatching.
"""


from __future__ import annotations
import contextlib
import io
import json
import time
from collections import Counter

import numpy as np

import sequence.dqc.dqc_program as _da
from sequence.topology.dqc_net_topo import DQCNetTopo
from sequence.components.circuit import Circuit

from sequence.dqc.dqc_program import TeleportationDQCProgram
from sequence.dqc.compilers import build_compiler
from sequence.constants import KET_VECTOR_FORMALISM, DENSITY_MATRIX_FORMALISM
from sequence.dqc.noise import KET_VECTOR_NOISE_FORMALISM, DENSITY_MATRIX_NOISE_FORMALISM

# (formalism, noise_active) -> the quantum-manager formalism id the Timeline is built with.
# Noisy runs use the per-node noise manager (a subclass of the pristine one); noiseless
# runs use the pristine manager. Set explicitly so a noisy run never leaks its formalism
# into a later noiseless run in the same process.
_FORMALISMS = {
    ("ket", False): KET_VECTOR_FORMALISM,
    ("ket", True): KET_VECTOR_NOISE_FORMALISM,
    ("density", False): DENSITY_MATRIX_FORMALISM,
    ("density", True): DENSITY_MATRIX_NOISE_FORMALISM,
}

STOP_BUDGET = 2e9   # per-step physical budget for sizing tl.stop_time


def purification_policy(_hops):
    """(memory_size, target_fidelity) per teleport reservation. Rather than forcing
    non-degrading swaps, we set a LOW target fidelity so a multi-hop pair's swap-
    degraded bookkept fidelity (0.95 per swap, metadata only -- the circuit-formalism
    state is always a perfect Bell pair) still clears the bar and no purification is
    ever requested. Reserving exactly one pair also sidesteps a resource collision
    that corrupts multi-hop teledata moves at pair-count 3."""
    return (1, 0.5)


def _program_metrics(program, topology):
    """(n_telegates, n_moves, max_hop) from a compiled program."""
    q2n = program.qubit_to_node
    hist = Counter(); seen = set()
    for grp in program.node_ops.values():
        for op in grp["remote"]:
            key = (op["layer"], tuple(sorted(op["targets"])))
            if key in seen:
                continue
            seen.add(key)
            a, b = op.get("nodes", [q2n[op["targets"][0]], q2n[op["targets"][1]]])
            hist[topology.hop(a, b)] += 1
    mv_hops, seen_mv = [], set()
    for grp in program.node_ops.values():
        for op in grp.get("move", []):
            key = (op["layer"], op["qubit"], op["src"], op["dest"])
            if key in seen_mv:
                continue
            seen_mv.add(key)
            mv_hops.append(topology.hop(op["src"], op["dest"]))
    n_tele = sum(hist.values())
    n_moves = len(seen_mv)
    max_hop = max([max((h for h in hist if h > 0), default=0)] + mv_hops)
    return n_tele, n_moves, max_hop, dict(sorted(hist.items()))


class _Run:
    """One ``run(...)`` invocation: holds the config and drives the shot loop.

    Constructing it validates + stores the arguments and computes derived config (the data qubits
    to read out, and whether any node is noisy). :meth:`execute` runs ``shots`` shots -- each shot's
    pipeline is :meth:`_one_shot` (build config -> build net -> compile -> register noise -> attach
    agents -> run timeline -> measure) -- then assembles the result dict.
    """

    def __init__(self, circuit, topology, partitioner="topo-aware", scheduler="fgp", seed=0,
                 controller="barrier", data_qubits=None, expected=None,
                 shots=1, meas_seed=9479, compiler=None, dump_config=None, formalism="ket"):
        if formalism not in ("ket", "density"):
            raise ValueError(f"formalism must be 'ket' or 'density', got {formalism!r}.")
        self.circuit = circuit
        self.topology = topology
        self.partitioner = partitioner
        self.scheduler = scheduler
        self.seed = seed
        self.controller = controller
        self.expected = expected
        self.shots = shots
        self.meas_seed = meas_seed
        self.compiler = compiler
        self.dump_config = dump_config
        self.formalism = formalism

        self.n = circuit.size
        self.data_qubits = set(range(self.n) if data_qubits is None else data_qubits)
        # Noise is active iff any node declares a non-ideal gate/measurement fidelity or a T1/T2
        # time. Computed here (before the nodes exist) straight off the per-node config specs; it
        # selects the noise-aware quantum manager (vs the faster pristine one), applied per DQCNode.
        self.noise_active = any(
            p.get("one_qubit_gate_fid", 1.0) < 1.0 or p.get("two_qubit_gate_fid", 1.0) < 1.0
            or p.get("measurement_fid", 1.0) < 1.0 or p.get("t1") is not None or p.get("t2") is not None
            for p in getattr(topology, "node_noise", {}).values())
        self._dumped = False

    # ── per-shot pipeline steps ──────────────────────────────────────────────
    def _build_config(self) -> dict:
        """The DQCNetTopo config for one shot: the topology's sim_config with the chosen controller
        policy + compiler, the state formalism, and memory sized to the circuit."""
        self.topology.controller_policy = self.controller
        self.topology.compiler_spec = {"partitioner": self.partitioner, "scheduler": self.scheduler}
        config = self.topology.sim_config()
        config["formalism"] = _FORMALISMS[(self.formalism, self.noise_active)]
        for nd in config["nodes"]:                   # generous memory: data holds every qubit slot
            if nd.get("type") == "DQCNode":
                nd["data_memo_size"] = max(nd.get("data_memo_size", 0), self.n)
                nd["memo_size"] = max(nd.get("memo_size", 0), 8 * self.n + 8)
        if self.dump_config and not self._dumped:    # write the exact simulated config once
            with open(self.dump_config, "w") as f:
                json.dump(config, f, indent=2)
            self._dumped = True
        return config

    def _register_noise(self, qn, qm, noise_seed) -> None:
        """Route each noisy node's qubits to itself in the controller's registry (compile ran
        first), so the noise layer reads their live fidelities. No-op for ideal nodes."""
        if self.noise_active:
            for nd in qn.values():
                nd.register_qubits(qm)
            qm.noise_rng = np.random.default_rng(noise_seed)

    def _attach_agents(self, qn, ctrl, hopmap) -> None:
        """Install one TeleportationDQCProgram per DQC node (executes the controller's steps).

        Agents are near-stateless executors -- they hold no placement/op state; each step's ops
        arrive already resolved (role, peer, slots) in the controller's STEP_MESSAGE."""
        for nm, nd in qn.items():
            TeleportationDQCProgram(node=nd, controller_name=ctrl.name,
                                    hop_distances=hopmap.get(nm, {}),
                                    reservation_policy=purification_policy)

    def _measure(self, qn, ctrl, tl, rng) -> int:
        """Read out the requested data qubits; return the packed measured bitstring.

        Final placement comes from the controller's registry (moves relocate qubits at run time via
        ACK deltas), not the compile-time placement."""
        reg = ctrl.registry
        meas = {}
        for gq in self.data_qubits:
            nm, slot = reg.location(gq)
            if nm is None or nm not in qn:
                continue
            nd = qn[nm]
            key = nd.get_component_by_name(nd.data_memo_arr_name)[slot].qstate_key
            c = Circuit(1); c.measure(0)
            meas[gq] = tl.quantum_manager.run_circuit(c, [key], rng.random())[key]
        return sum(b << gq for gq, b in meas.items())

    def _one_shot(self, shot_rng, noise_seed):
        """Build the network, compile + drive the barrier, and read out one shot."""
        config = self._build_config()
        with contextlib.redirect_stdout(io.StringIO()):
            net = DQCNetTopo(config); tl = net.tl; qm = tl.quantum_manager
            qn = {node.name: node for node in net.nodes[DQCNetTopo.DQC_NODE]}
            ctrl = net.controller                       # a bare executor built + wired by DQCNetTopo
            # Compilation lives here, not in the controller: build the compiler (a custom object
            # overrides the named partitioner/scheduler), run it against the live net, and install
            # the resulting plan into the controller (which seeds its registry from it).
            compiler = self.compiler or build_compiler(self.partitioner, self.scheduler)
            program = compiler.compile(self.circuit, net, seed=self.seed)
            ctrl.load(program)
            self._register_noise(qn, qm, noise_seed)
            metrics = _program_metrics(program, net)
            _da.RESERVATION_SLACK_CC_MULT = 6 * max(metrics[2], 1)
            tl.stop_time = int((program.max_step + 16) * STOP_BUDGET * 8)
            self._attach_agents(qn, ctrl, net.hop_distances())
            tl.init(); ctrl.start_execution()
            t0 = time.time(); tl.run(); wall = time.time() - t0
            measured = self._measure(qn, ctrl, tl, shot_rng)
            return measured, ctrl.current, program, metrics, wall, tl.now() / 1e9

    def execute(self) -> dict:
        """Run ``shots`` shots and assemble the result dict."""
        meas_rng = np.random.default_rng(self.meas_seed)
        successes, hist_shots, last = 0, Counter(), None
        for shot_idx in range(max(1, self.shots)):
            measured, reached, program, metrics, wall, sim_ms = self._one_shot(
                meas_rng, self.meas_seed + 1000 + shot_idx)
            hist_shots[measured] += 1
            if self.expected is not None and measured == self.expected and reached >= program.max_step:
                successes += 1
            last = (measured, reached, program, metrics, wall, sim_ms)

        measured, reached, program, (n_tele, n_moves, max_hop, hist), wall, sim_ms = last
        result = {"measured": measured, "expected": self.expected,
                  "gates": n_tele, "moves": n_moves, "max_hop": max_hop, "hops": hist,
                  "steps": program.max_step, "nodes_used": len(set(program.qubit_to_node.values())),
                  "sim_ms": sim_ms, "wall": wall, "reached": reached}
        if self.shots > 1 or self.noise_active:
            result.update(shots=max(1, self.shots), successes=successes,
                          success_prob=successes / max(1, self.shots),
                          measured_hist=dict(sorted(hist_shots.items())))
        else:
            result["ok"] = (self.expected is not None and measured == self.expected
                            and reached >= program.max_step)
        return result


def run(circuit, topology, partitioner="topo-aware", scheduler="fgp", seed=0,
        controller="barrier", data_qubits=None, expected=None,
        shots=1, meas_seed=9479, compiler=None, dump_config=None,
        formalism="ket") -> dict:
    """Compile + simulate ``circuit`` on ``topology`` and read out the data qubits.

    ``partitioner`` + ``scheduler`` select a built-in compiler (static pipeline, or the
    monolithic FGP hybrid when ``scheduler="fgp"``). Alternatively, pass a ready-made
    ``compiler`` object to run ANY custom or external compiler: it need only implement
    ``compile(circuit, topology, seed) -> CompiledProgram``. This is the general
    extension point---a user's own partitioner/scheduler, a whole new compiler, or an
    external tool's output wrapped in an adapter (e.g. :class:`AmaroCompiler`)---all
    execute end-to-end on the physically simulated network through this one hook.
    ``controller`` is "barrier" or "adaptive".

    Noise: noise is entirely per-node -- declared on the topology (each DQCNode's
    fidelities/T1/T2); a node with no noise params is ideal. When any node is noisy,
    set ``shots``>1 to run trajectory noise (per-shot random Paulis); the result then
    reports ``success_prob`` (fraction of shots that measured ``expected``) and a
    ``measured_hist``. With no noisy node and ``shots=1`` it's the deterministic single
    run (``measured`` / ``ok``).

    ``formalism``: state representation for the simulation -- "ket" (state-vector; noise
    is per-shot random Paulis, cheap, needs many shots) or "density" (density matrix;
    noise is deterministic CPTP channels, exact in one shot, costlier). When any per-node
    noise is present, the matching noise manager is selected automatically
    (QuantumManagerKetNoise / QuantumManagerDensityNoise).

    ``dump_config``: optional path to write the EXACT expanded DQCNetTopo config that
    gets simulated (teleport-json layout, per-node noise inline on each DQCNode), so it
    can be inspected. Written once, before the first shot.
    """
    return _Run(circuit, topology, partitioner=partitioner, scheduler=scheduler, seed=seed,
                controller=controller, data_qubits=data_qubits, expected=expected, shots=shots,
                meas_seed=meas_seed, compiler=compiler, dump_config=dump_config,
                formalism=formalism).execute()
