# File: base.py
"""Shared controller plumbing for driving the per-node DQC apps.

A DQC controller is a :class:`~sequence.topology.node.ClassicalNode` wired into the
network topology: it owns the compiler (circuit + topology -> CompiledProgram) and
talks to each node's DQC app (:class:`~sequence.dqc.dqc_program.DQCProgram`) purely
over classical channels -- broadcasting *step* messages and hearing back *ACK*
messages, never calling node methods directly.

:class:`BaseController` factors out everything that is NOT an orchestration policy:
the classical-node wiring + shared run state, ``compile`` / ``set_nodes``, and the
two primitives for talking to a DQC app -- :meth:`send_step` (dispatch one step to
one node) and :meth:`_is_ack` (recognise a step-completion ACK). What it deliberately
leaves to subclasses is *when* steps are sent and *how* completions advance the run:
:class:`~barrier.BarrierController` replays the plan as a lock-step barrier,
:class:`~adaptive_central_node.AdaptiveController` as a dataflow graph.
"""
from __future__ import annotations

import time

from sequence.topology.node import ClassicalNode
from sequence.message import Message

from sequence.dqc.dqc_program import DQCMessage, DQCMsgType


def _session_identity(op) -> int:
    """A unique per-session reservation identity, shared by BOTH endpoints of a network op.

    Uses the op-DAG op id (``op_id`` -- build_op_dag tags the SAME id onto both parties' copies of
    a remote/move op) offset by 1 so it is always >= 1 (identity 0 means "auto-assign" to the RSVP
    layer). Both endpoints derive the same value from their own op copy, so each can bind
    ``identity -> app`` before the reservation travels (see ``_DualAppRouter``). Falls back to a
    deterministic hash of the op's defining fields if ``op_id`` is absent."""
    oid = op.get("op_id")
    if oid is not None:
        return oid + 1
    if "qubit" in op:                        # move: (step, qubit, dest)
        key = (op.get("step"), op.get("qubit"), op.get("dest"))
    else:                                    # remote gate: (step, sorted targets)
        key = (op.get("step"), tuple(sorted(op.get("targets", op.get("qubits", [])))))
    return (hash(key) & 0x7FFFFFFF) or 1


class BaseController(ClassicalNode):
    """Controller base: classical-node wiring, the compiled plan, and DQC-app messaging.

    Subclasses add the orchestration policy by implementing :meth:`_start` (kick off
    the run) and :meth:`receive_message` (react to ACKs).

    Args:
        name (str): controller node name (message sender label + channel key).
        timeline (Timeline): the network timeline (from ``DQCNetTopo``).
        compiler: a compiler (``compilers`` package) -- circuit+topology -> program.
            Either a static BasicCompiler (partitioner + scheduler) or the monolithic
            FGPCompiler. Run once at :meth:`compile`.
        dt (float): delay after a network (telegate/teledata) step before the next
            dispatch.
        local_dt (float): delay after a local-only step (default ``dt``; ~0 so sim-time
            reflects real network cost rather than a flat local-gate barrier).
    """

    def __init__(self, name: str, timeline, compiler=None,
                 dt: float = 0.0, local_dt: float = None):
        super().__init__(name, timeline)          # ClassicalNode -> registers on timeline
        self.compiler = compiler
        self.dt = dt
        self.local_dt = dt if local_dt is None else local_dt

        # filled by compile() / set_nodes()
        self.program = None
        self.max_step = -1
        self.net_layers = set()
        self.qnodes = []                          # DQC node objects we orchestrate
        self.registry = None                      # logical<->physical map (see _init_registry)

        # run-time state shared by every policy
        self.current = 0                          # highest completed step + 1
        self._completed = False

    # ── compile: run the compiler this controller owns ───────────────────────
    def compile(self, circuit, topology, seed: int = 0):
        """Compile ``circuit`` on ``topology`` into the full plan (CompiledProgram)."""
        self.program = self.compiler.compile(circuit, topology, seed=seed)
        self.max_step = self.program.max_step
        self.net_layers = self.program.net_layers
        self._init_registry()
        self._on_compiled()
        return self.program

    def _init_registry(self) -> None:
        """Create the controller-owned :class:`~sequence.dqc.registry.QubitRegistry`, seed it with
        the program's initial logical->physical placement, and inject it into the quantum
        manager so the noise layer reads the SAME map the controller evolves. This is the one
        source of truth for where each logical qubit lives (node/slot) and its qstate key."""
        from sequence.dqc.registry import QubitRegistry
        self.registry = QubitRegistry()
        self.registry.seed_placement(self.program.qubit_to_node, self.program.data_owners)
        qm = self.timeline.quantum_manager
        if hasattr(qm, "registry"):               # noise-aware managers hold a registry
            qm.registry = self.registry           # inject: noise + agents share the controller's map

    def _on_compiled(self) -> None:
        """Hook run after ``compile`` (subclasses may derive extra plan state)."""

    def set_nodes(self, qnodes):
        """The DQC node objects this controller drives (must be channel-connected)."""
        self.qnodes = list(qnodes)
        return self

    # ── talk to the per-node DQC apps ────────────────────────────────────────
    def send_step(self, node_name: str, step: int) -> None:
        """Dispatch controller ``step`` to one node's DQC app over the classical channel,
        carrying that node's ops for the step (resolved from the compiled program).

        Args:
            node_name (str): name of the DQC node to run ``step``.
            step (int): controller step index to execute.
        """
        msg = DQCMessage(DQCMsgType.STEP_MESSAGE, receiver="qpu_agent",
                         step=step, node=self.name, ops=self._ops_for(node_name, step))
        self.send_message(node_name, msg)         # ClassicalNode: routes via channel

    def _ops_for(self, node_name: str, step: int):
        """This node's ops for ``step`` ({"local"/"remote"/"move": [op, ...]}) from the compiled
        program, with concrete physical slots resolved from the registry so the worker doesn't
        resolve placement itself. Returns None when there's no program (hand-crafted barrier
        tests fall back to the ops the agent was constructed with).

        Copies each op dict (never mutates ``program.node_ops``) and annotates it -- from the
        registry -- with everything the worker needs so it never consults placement itself:
        this node's ROLE and the PEER node, plus concrete slots. Local ops get ``slots``;
        remote ops get ``role`` ("control"/"target"), ``peer``, ``ctrl_slot``, ``tgt_slot``;
        move ops get ``role`` ("source"/"dest"), ``peer``, ``src_slot`` (``dest_slot`` is fixed
        by the compiler). Returns None when there's no program (hand-crafted barrier tests fall
        back to the ops the agent was constructed with)."""
        if self.program is None:
            return None
        grp = self.program.node_ops.get(node_name, {})
        out = {kind: [dict(op) for op in grp.get(kind, []) if op.get("step") == step]
               for kind in ("local", "remote", "move")}
        # Stamp a unique per-session reservation identity on every network op (independent of the
        # registry): both endpoints derive the same value and bind identity -> app for routing.
        for op in out["remote"] + out["move"]:
            op["identity"] = _session_identity(op)
        reg = self.registry
        if reg is not None:
            for op in out["local"]:
                op["slots"] = [reg.qubit_to_slot.get(q) for q in op.get("targets", [])]
            for op in out["remote"]:
                qs = op.get("qubits", op.get("targets", []))
                if len(qs) == 2:
                    ctrl_q, tgt_q = qs
                    op["ctrl_slot"] = reg.qubit_to_slot.get(ctrl_q)
                    op["tgt_slot"] = reg.qubit_to_slot.get(tgt_q)
                    ctrl_node, tgt_node = reg.qubit_to_node.get(ctrl_q), reg.qubit_to_node.get(tgt_q)
                    if node_name == ctrl_node:
                        op["role"], op["peer"] = "control", tgt_node
                    elif node_name == tgt_node:
                        op["role"], op["peer"] = "target", ctrl_node
            for op in out["move"]:
                q, dest = op.get("qubit"), op.get("dest")
                op["src_slot"] = reg.qubit_to_slot.get(q)
                src_node = reg.qubit_to_node.get(q)
                if node_name == src_node:
                    op["role"], op["peer"] = "source", dest
                elif node_name == dest:
                    op["role"], op["peer"] = "dest", src_node
        return out

    @staticmethod
    def _is_ack(msg: Message) -> bool:
        """True if ``msg`` is a step-completion ACK from a DQC app."""
        return isinstance(msg, DQCMessage) and msg.msg_type is DQCMsgType.ACK

    def _apply_deltas(self, msg: Message) -> None:
        """Fold a node's reported placement changes (from a move ACK) into the controller's
        registry, so the controller owns the logical->physical map updates. No-op if there's
        no registry or no deltas."""
        if self.registry is None:
            return
        for d in getattr(msg, "deltas", ()) or ():
            self.registry.place(d["qubit"], d["node"], d["slot"])
            self.registry.set_key(d["qubit"], d["key"])

    # ── run lifecycle ─────────────────────────────────────────────────────────
    def start_execution(self) -> None:
        """Kick off the run (call after ``timeline.init()``). Stamps the wall-clock
        start used by the progress readout, then hands off to the policy's
        :meth:`_start`."""
        self._t_start = time.time()
        self._start()

    def _start(self) -> None:
        """Begin dispatching (subclass policy)."""
        raise NotImplementedError

    def receive_message(self, src: str, msg: Message) -> None:
        """React to an ACK from a DQC app (subclass policy)."""
        raise NotImplementedError
