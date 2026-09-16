"""Per-node QPU execution agents for distributed quantum computing (DQC).

A QPU agent runs on one quantum node. It registers on the node, receives compiled
*step* messages broadcast by the central controller, and executes that step's work
on the node's QPU:

* local (single-node) gates run immediately;
* remote (two-qubit) gates and qubit relocations ("moves") are executed by the
  subclass and complete asynchronously, so the agent *defers* the controller ACK
  for a step until every remote op it launched for that step reports completion.

:class:`DQCProgram` is the abstract base: it owns the controller/node contract
(registration, step dispatch, the deferred-ACK barrier, local-gate execution, and
slot resolution) and leaves *how* a remote gate / move is physically realized to
subclasses via :meth:`~DQCProgram._run_remote` / :meth:`~DQCProgram._run_move`.
:class:`TeleportationDQCProgram` implements those with entanglement teleportation
(:class:`TelegateApp` for remote gates, :class:`TeledataApp` for moves).

Contents
--------
DQCMsgType / DQCMessage
    Controller <-> agent step / ACK messages.
DQCProgram
    Abstract barriered per-node executor (controller/node contract).
TeleportationDQCProgram
    Concrete agent: remote gates via telegate, moves via teledata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.utils import log
from sequence.app.teleportation import TelegateApp, TeledataApp


# ── Reservation / timing tuning ─────────────────────────────────────────────
# All telegate timing is derived from the network's classical-channel (cc) delay
# so it stays correct if the topology's distances/delays change.
#
# TelegateApp now releases its resources via *early expire* the instant a gate
# completes (see TelegateApp._early_expire, mirroring sequence's
# base_teleport_app): the RSVP rules for the session are expired and the comm
# memory is reset to RAW. This removes the old UPPER-bound fragility — a finished
# session no longer holds its comm memory until the clock reaches end_time, so
# consecutive telegates on a node can't overlap into timecard exhaustion.
#
# The window's LOWER bound still matters: it must be LONGER than one full
# teleported-CNOT (entanglement gen + corrections), or the rules expire mid-gate
# and the memory resets to RAW. The window cannot be made arbitrarily large
# either: early expire only fires at completion (it has to — expiring sooner
# would wipe the EPR pair before the gate consumes it), so an over-wide window
# still lets the RSVP layer's continued entanglement generation/scheduling
# corrupt the in-flight gate (observed at n_data=5 with a 1000x window). 100x
# comfortably brackets a single gate while staying clear of that regime.
TELEGATE_WINDOW_CC_MULT = 400    # reservation window = this * cc_delay
RESERVATION_SLACK_CC_MULT = 2    # reservation opens this * cc_delay after "now"
# A local (single-node) gate takes a small but nonzero time. It is charged per
# local-only controller step (gates within one step parallelise across qubits);
# a distributed telegate is ~100x slower (entanglement generation + classical
# comm), so local gates are "free" whenever a step also carries a telegate. A
# no-op step (a node idle while others work) costs nothing.
LOCAL_GATE_TIME = 1e6            # ps; reasonable local gate duration (~1 microsecond)
DEFAULT_CC_DELAY = 5e8           # ps; fallback if a peer has no direct cc channel


# ── Controller <-> agent messages ───────────────────────────────────────────
class DQCMsgType(Enum):
    """Message types for DQC communications."""
    STEP_MESSAGE = auto()   # Controller -> Node: broadcast step to execute
    ACK = auto()  # Node -> Controller: acknowledge completion of a step


class DQCMessage(Message):
    """DQC message payloads.

    STEP_MESSAGE:
    • step: Controller step index to execute on this node
    • node: Sender (controller) name
    • ops: this node's ops for the step ({"local"/"remote"/"move": [op, ...]}), resolved by
      the controller from the compiled program to concrete addresses (role, peer, slots).

    ACK:
    • step: Controller step index completed on this node
    • node: Sender node name (for controller bookkeeping)
    • deltas: placement changes this node completed for the step (list of
      {"qubit", "node", "slot", "key"}), which the controller folds into its registry
      so it OWNS the logical->physical map updates (empty for non-move steps).
    """

    def __init__(self, msg_type: DQCMsgType, receiver: str, **kwargs):
        super().__init__(msg_type=msg_type, receiver=receiver)

        if msg_type is DQCMsgType.STEP_MESSAGE:
            self.step = kwargs['step']
            self.node = kwargs['node']
            self.ops = kwargs.get('ops', None)
            self.string = f"DQCMessage(type={msg_type}, moving to step={self.step}, sending from controller to node={self.node})"

        elif msg_type is DQCMsgType.ACK:
            self.step = kwargs['step']
            self.node = kwargs['node']
            self.deltas = kwargs.get('deltas', ())
            self.string = f"DQCMessage(type={msg_type}, step={self.step} is complete, sending from {self.node} to controller)"

    def __str__(self):
        return self.string


# ── QPU agent (abstract base) ───────────────────────────────────────────────
class DQCProgram(ABC):
    """Barriered (non-pipelined) per-node QPU executor — the controller contract.

    Registers on its node (``node.protocols``), receives compiled step messages
    from the central controller, runs each step's work on this node's QPU, and
    ACKs the controller. Subclasses decide *how* a remote (two-qubit) gate or a
    qubit move is physically executed by implementing :meth:`_run_remote` and
    :meth:`_run_move`; the base handles everything else.

    Behavior
    --------
    • Each controller step is executed *to completion* on this node before an ACK.
    • Local-only steps execute immediately and ACK right away.
    • A step containing a remote gate/move defers its ACK until the subclass
      reports completion (via :meth:`_ack_deferred_unit`).
    • No queuing or pipelining of future steps is performed.

    The agent is a near-stateless executor: it holds no placement map. Each STEP_MESSAGE from
    the controller carries this node's ops for the step already resolved to concrete addresses
    (role, peer, slots) from the controller's registry; the agent just executes them and ACKs
    (reporting any move relocation as a delta for the controller to fold into its registry).

    Args:
        node: The underlying :class:`~sequence.topology.node.DQCNode`.
        controller_name (str, optional): node name of the central controller; ACKs are sent to
            it over this node's classical channel (default ``"controller"``).

    Attributes:
        name (str): protocol receiver label used to route StepMessages (``"qpu_agent"``).
        _pending (Dict[int, int]): step -> count of remote ops on this node still in
            flight; the step's ACK is deferred until the count hits 0.
    """

    def __init__(self, node, controller_name: str = "controller"):
        self.name = "qpu_agent"
        self.protocol_type = "qpu_agent"  # Required by sequence framework
        self.node = node
        self.tl = node.timeline
        self.controller_name = controller_name
        self.data_array = node.data_memo_arr_name

        # The controller-owned registry, injected on the quantum manager (None on a plain/ideal
        # formalism). The agent only reads a qubit's current key off it when running a local gate.
        self.registry = getattr(self.tl.quantum_manager, "registry", None)

        # Steps awaiting a remote op before ACKing, mapped to the number still in flight for that
        # step (ACKed when the count hits 0), plus the move deltas to report on that step's ACK
        # so the CONTROLLER folds them into its registry (it owns the logical->physical map).
        self._pending: Dict[int, int] = {}
        self._pending_deltas: Dict[int, list] = {}
        self.local_gate_time = LOCAL_GATE_TIME

        # Register in node.protocols so the controller's StepMessages route here
        # (the node dispatches received messages by matching protocol.name).
        node.protocols.append(self)

    # ── controller step handling ────────────────────────────────────────────
    def received_message(self, src: str, msg: Message) -> bool:
        """Execute the controller's step on this node, then ACK.

        A packed step may carry several ops of each kind. Local gates run
        synchronously; remote gates and moves start an async op and *defer* the
        ACK until it completes. A step with no work here, or only local work, ACKs
        right away (see :meth:`_ack_local`).

        Args:
            src (str): sender (the controller).
            msg (Message): step message carrying ``step`` and this node's resolved ``ops``.

        Returns:
            bool: always ``True`` (message handled).
        """
        step = msg.step
        ops = getattr(msg, "ops", None) or {}        # the controller sends this node's resolved ops
        handlers = (("local", self._run_local), ("remote", self._run_remote), ("move", self._run_move))
        if not any(ops.get(kind) for kind, _ in handlers):
            self._ack_local(step, "no-op")
            return True
        # `deferred` becomes True once any remote op is launched, so the ACK waits
        # for its pending counter to drain (see _ack_deferred_unit). Otherwise the
        # step is local-only.
        deferred = False
        for kind, handler in handlers:
            for op in ops.get(kind, []):
                deferred = handler(op, step) or deferred
        if not deferred:
            self._ack_local(step, "local-only")
        return True

    def _run_local(self, op: dict, step: int) -> bool:
        """Run one local gate immediately; local work never defers the ACK."""
        log.logger.debug(f"[{self.node.name}] local step={step} gate={op.get('gate')} targets={op.get('targets')}")
        self._do_local(op)
        return False

    @abstractmethod
    def _run_remote(self, op: dict, step: int) -> bool:
        """Execute this node's half of a remote (two-qubit) gate for ``step``.

        The op is delivered to both parties; this node owns the control or the
        target qubit. Must call :meth:`_defer` and return ``True`` if this node is
        a party (so the step's ACK waits for completion), else return ``False``.
        """

    @abstractmethod
    def _run_move(self, op: dict, step: int) -> bool:
        """Relocate a qubit for ``step`` (this node is the source or the dest).

        Must call :meth:`_defer` and return ``True`` if this node is a party, else
        return ``False``.
        """

    # ── local gate execution ────────────────────────────────────────────────
    def _do_local(self, op: Dict[str, Any]) -> None:
        """Execute a local (single-node) gate operation."""
        gate = str(op.get("gate", "")).lower()

        # Targets can be under 'targets', 'qubits', or 'qubit'
        qs = op.get("targets", [])
        if not qs:
            qs = op.get("qubits", [])
        if not qs and "qubit" in op:
            qs = [op["qubit"]]

        if not qs:
            log.logger.warning(f"[qpu_agent:{self.node.name}] _do_local with no targets: {op}")
            return

        # Physical slots come resolved from the controller's registry (in the instruction). Keys
        # are read live off the slots (authoritative), and the registry's logical->key view is
        # kept current.
        arr = self.node.components[self.data_array]   # MemoryArray
        slots = op["slots"]
        keys: List[int] = []
        for q, s in zip(qs, slots):
            k = arr.memories[s].qstate_key
            keys.append(k)
            if self.registry is not None:
                self.registry.set_key(q, k)

        # slot -> local circuit index (0..len(slots)-1)
        slot_to_idx = {slot: i for i, slot in enumerate(slots)}

        # Circuit acts on exactly these |slots| qubits, in the order of 'keys'
        circ = Circuit(len(slots))

        log.logger.debug(f"[qpu_agent:{self.node.name}] _do_local gate={gate}, qs={qs}, slots={slots}, keys={keys}")

        # ---- Apply gates using local circuit indices ----
        # Single-qubit gates apply to every target; two-qubit gates take
        # (control, target) = the first two slots; 'phase' also takes an angle.
        idx = [slot_to_idx[s] for s in slots]
        if gate in ("h", "x", "y", "z", "s", "sdg", "t", "measure"):
            for i in idx:
                getattr(circ, gate)(i)
        elif gate == "phase":
            for i in idx:
                circ.phase(i, op.get("arg", 0.0))
        elif gate in ("cx", "cz") and len(idx) >= 2:
            getattr(circ, gate)(idx[0], idx[1])   # (control, target)
        else:
            log.logger.debug(f"[{self.node.name}] unsupported local gate '{gate}' op={op}")
            return

        # ---- Run circuit and remap any changed qstate keys back onto the slots ----
        rnd = self.node.get_generator().random()
        res: Dict[int, int] = self.tl.quantum_manager.run_circuit(circ, keys, rnd)
        if res:   # empty res => unitary-only, in-place; keys unchanged
            for gq, slot, old_key in zip(qs, slots, keys):
                new_key = res.get(old_key, old_key)
                if new_key != old_key:
                    arr.memories[slot].qstate_key = new_key
                    if self.registry is not None:
                        self.registry.set_key(gq, new_key)

    # ── ACK & controller messaging ──────────────────────────────────────────
    def _defer(self, step: int) -> None:
        """Register one more in-flight remote op for ``step``; the step's ACK is
        held until every such op has completed (see ``_ack_deferred_unit``)."""
        self._pending[step] = self._pending.get(step, 0) + 1
        log.logger.info(f"[qpu_agent:{self.node.name}] deferring step {step} (pending now: {self._pending})")

    def _ack_deferred_unit(self, step: int) -> None:
        """Mark one deferred unit of ``step`` complete; ACK when the count hits 0.

        Called from a subclass's completion callbacks: a step ACKs only once every
        in-flight remote op on this node is done.
        """
        if step is not None and step in self._pending:
            self._pending[step] -= 1
            if self._pending[step] <= 0:
                del self._pending[step]
                deltas = self._pending_deltas.pop(step, ())
                ack = DQCMessage(DQCMsgType.ACK, receiver="controller", step=step,
                                 node=self.node.name, deltas=deltas)
                self._send_to_controller(ack)
                log.logger.info(f"[qpu_agent:{self.node.name}] deferred ACK sent for step={step}")

    def _ack_local(self, step: int, reason: str = "") -> None:
        """ACK a local/no-op step, charging a local-gate time to real work.

        A step with actual local gates on this node ("local-only") takes one
        local-gate duration; a "no-op" step (this node idle while others work)
        ACKs immediately (0). Since the controller advances only when EVERY node
        has ACKed, a step's duration is the slowest node's work: a telegate time
        if the step carries one (that ACK is deferred to remote-op completion),
        else one local-gate time if any node has local work, else 0.
        """
        delay = self.local_gate_time if reason == "local-only" else 0
        ack = DQCMessage(DQCMsgType.ACK, receiver="controller", step=step, node=self.node.name)
        ev = Event(int(self.tl.now() + delay),
                   Process(self, "_send_to_controller", [ack]))
        self.tl.schedule(ev)
        log.logger.debug(f"[qpu_agent:{self.node.name}] ACK for step={step} scheduled "
                         f"after delay={delay} ({reason})")

    def _send_to_controller(self, msg: Message):
        """Send an ACK to the controller over this node's classical channel. The
        controller is a real topology node, so this is an ordinary message send
        (routed by ``controller_name``); no direct object reference is needed."""
        if self.controller_name in getattr(self.node, "cchannels", {}):
            log.logger.debug(f"[qpu_agent:{self.node.name}] ACK -> {self.controller_name} via channel: {msg}")
            self.node.send_message(self.controller_name, msg)
        else:
            log.logger.warning(f"[qpu_agent:{self.node.name}] No classical channel to controller "
                               f"'{self.controller_name}'; ACK dropped: {msg}")


# ── Teleportation agent (concrete) ──────────────────────────────────────────
class TeleportationDQCProgram(DQCProgram):
    """QPU agent that realizes remote gates and moves with entanglement teleportation.

    Remote two-qubit gates run via :class:`TelegateApp` (teleported CNOT/CZ) and
    qubit moves via :class:`TeledataApp` (state teleportation). Both apps register on
    the :class:`~sequence.topology.node.DQCNode`, which routes each reservation/memory
    callback to the owning app; each reports completion through a callback that releases
    the step's deferred ACK.

    Remote CZ is normalized to CNOT.

    Args (in addition to :class:`DQCProgram`):
        hop_distances (Dict[str, int], optional): hop distance to each peer.
        reservation_policy (Callable[[int], tuple], optional): hop count ->
            (memory_size, target_fidelity) for entanglement reservations.

    Attributes:
        tgate (TelegateApp): executes teleported gates on this node.
        tdata (TeledataApp): executes qubit moves on this node.
    """

    def __init__(self,
                 node,
                 controller_name: str = "controller",
                 hop_distances: Optional[Dict[str, int]] = None,
                 reservation_policy: Optional[Callable[[int], tuple]] = None):
        super().__init__(node, controller_name=controller_name)

        # Physical hop-distance to each peer + policy mapping hop count ->
        # (memory_size, target_fidelity). A multi-hop pair is degraded by every
        # swap, so deeper paths reserve MORE raw pairs to purify back up to the
        # target fidelity; a direct (1-hop) pair needs none. Default keeps the
        # single-pair / perfect-fidelity behaviour when no policy is supplied.
        self.hop_distances = hop_distances or {}
        self.reservation_policy = reservation_policy or (lambda hops: (1, 1.0))

        # Representative classical-channel delay (ps) read from the network config;
        # sizes the reservation windows (see _reservation_for / _peer_cc_delay).
        cc_delays = [getattr(ch, "delay", 0) for ch in node.cchannels.values()
                     if getattr(ch, "delay", 0) and getattr(ch, "delay", 0) > 0]
        self.cc_delay = float(min(cc_delays)) if cc_delays else DEFAULT_CC_DELAY

        # One TelegateApp + one TeledataApp per node. Each self-registers on the node via
        # App.__init__ -> node.set_app(); the DQCNode keeps an app registry and routes each
        # reservation/memory callback to the owning app -- so both coexist (and more can be
        # added for concurrency) without a DQC-specific router.
        self.tgate = TelegateApp(self.node)
        self.tdata = TeledataApp(self.node)

        # Teledata move bookkeeping.
        #   _expected_moves: dest_slot -> (step, global_q) for moves landing here.
        #   _move_by_srcslot: source local slot -> (step, global_q, dest) for moves
        #                     leaving here (matched on the source-completion hook).
        self._expected_moves: Dict[int, tuple] = {}
        self._move_by_srcslot: Dict[int, tuple] = {}

        # Completion callbacks -- each releases the step's deferred ACK (via
        # _ack_deferred_unit) when its remote op finishes locally:
        #   telegate        -> the control/target gate completed
        #   teledata dest   -> the moved state has landed here
        #   teledata source -> the qubit has left this node
        self.tgate.telegate_complete = self._on_telegate_complete
        self.tdata.on_complete = self._on_teledata_complete
        self.tdata.on_source_complete = self._on_teledata_source_complete

    # ── remote (telegate) gates ─────────────────────────────────────────────
    def _run_remote(self, op: dict, step: int) -> bool:
        """Start this node's half of a teleported CNOT/CZ. The op is delivered to
        both parties, so this node is the control owner (start the telegate) or the
        target owner (lock its slot). Returns True (ACK deferred) if we are a party.
        """
        qs = op.get("qubits", op.get("targets", []))
        if len(qs) != 2:
            log.logger.warning(f"[{self.node.name}] malformed remote op @step={step}: {qs}")
            return False
        ctrl_q, tgt_q = qs
        # The controller resolved this node's role + peer + concrete slots from its registry.
        role, peer = op.get("role"), op.get("peer")
        if role == "control":
            self.node.bind_app_peer(peer, self.tgate)   # telegate reservation callbacks -> tgate
            self._defer(step)
            self._start_telegate_control(op, ctrl_q, op["ctrl_slot"], peer, tgt_q,
                                         step=step, tgt_slot=op["tgt_slot"])
            return True
        if role == "target":
            self.node.bind_app_peer(peer, self.tgate)   # incoming from the control node
            self._lock_target_slot(tgt_q, step, slot=op["tgt_slot"])
            self._defer(step)
            return True
        return False

    def _lock_target_slot(self, tgt_q: int, step: int, slot: int) -> None:
        """Target side of a telegate: tell the TelegateApp which local data slot (resolved by
        the controller) to land the corrected qubit in for ``step``."""
        self.tgate._current_step = step
        self.tgate.set_target_slot_for_step(step, slot)

    def _start_telegate_control(self, op: Dict[str, Any],
                                ctrl_q: int, ctrl_slot: int,
                                peer_nm: str, tgt_q: int,
                                step: int, tgt_slot: int = None) -> None:
        """Start a teleported CNOT where this node owns the control qubit.

        The target node locks its own local slot upon receiving the same
        controller step. The ACK for ``step`` is deferred and sent by the
        :class:`TelegateApp` completion callback (:meth:`_on_telegate_complete`).

        Args:
            op (Dict[str, Any]): Operation descriptor for the remote op.
            ctrl_q (int): Global index of the control qubit owned by this node.
            ctrl_slot (int): Local data-memory slot index for ``ctrl_q``.
            peer_nm (str): Peer (target-owner) node name.
            tgt_q (int): Global index of the target qubit owned by ``peer_nm``.
            step (int): Controller step index.

        Side Effects:
            Starts the teleported CNOT via :class:`TelegateApp` over a reservation
            window sized to the peer's distance.
        """
        gate_typ = str(op.get('gate', 'cx')).lower()   # 'cx' or 'cz'; forwarded to the telegate

        arr = self.node.components[self.data_array]
        key = arr.memories[ctrl_slot].qstate_key       # for the log below

        self.tgate._current_step = step

        t0, t1, mem_size, fidelity, hops = self._reservation_for(peer_nm)

        log.logger.info(f"Executing REMOTE gate={gate_typ} step={step} targets={op['targets']}; CONTROL on {self.node.name}(q={ctrl_q},slot={ctrl_slot},key={key}) → TARGET {peer_nm}(q={tgt_q},slot={tgt_slot}); hops={hops} mem_size={mem_size} fid={fidelity}; t=[{t0},{t1}]")

        self.tgate.start(
            responder=peer_nm,
            start_t=t0,
            end_t=t1,
            memory_size=mem_size,
            fidelity=fidelity,
            control_src=ctrl_slot,  # control (local)
            target_src=tgt_slot,    # target (peer)
            gate_type=gate_typ,     # 'cx' (teleported CNOT) or 'cz' (teleported CZ)
        )

        log.logger.info(f"TeleGateCNOT started step={step}: control(local q={ctrl_q} slot={ctrl_slot}) → target({peer_nm} q={tgt_q} slot={tgt_slot}); t=[{t0},{t1}]")

    # ── teledata moves (qubit relocation) ───────────────────────────────────
    def _run_move(self, op: dict, step: int) -> bool:
        """Teledata move: start the teleport if we own the qubit, or reserve the
        landing slot if it is being teleported to us. Returns True if we are a party.
        """
        q, dest, dest_slot = op["qubit"], op["dest"], op["dest_slot"]
        # The controller resolved this node's role + peer + concrete source slot from its registry.
        role, peer = op.get("role"), op.get("peer")
        if role == "source":
            self.node.bind_app_peer(peer, self.tdata)   # teledata reservation callbacks -> tdata
            self._defer(step)
            self._start_teleport_source(q, op["src_slot"], dest, dest_slot, step=step)
            return True
        if role == "dest":
            self.node.bind_app_peer(peer, self.tdata)   # incoming from the source node
            self._expected_moves[dest_slot] = (step, q)
            self._defer(step)
            return True
        return False

    def _start_teleport_source(self, q: int, src_slot: int,
                               dest: str, dest_slot: int, step: int) -> None:
        """Initiate a teledata move of global qubit ``q`` from this (source) node to
        ``dest``'s ``dest_slot``. Defers the step ACK until Bob acknowledges."""
        self._move_by_srcslot[src_slot] = (step, q, dest, dest_slot)

        t0, t1, mem_size, fidelity, hops = self._reservation_for(dest)

        log.logger.info(f"[qpu_agent:{self.node.name}] TELEPORT q={q} slot={src_slot} → {dest} slot={dest_slot}; "
                        f"hops={hops} mem_size={mem_size} fid={fidelity}; t=[{t0},{t1}]")
        self.tdata.start(responder=dest, start_t=t0, end_t=t1, memory_size=mem_size,
                         fidelity=fidelity, data_src=src_slot, dest_slot=dest_slot)

    # ── completion callbacks (release deferred ACKs) ────────────────────────
    def _on_telegate_complete(self, data_key: int, role: str = "unknown"):
        """Release a step's deferred ACK when its teleported gate finishes locally.

        Installed as ``TelegateApp.telegate_complete``. Under the barrier only one
        step is network-active at a time, so an in-flight telegate belongs to the
        highest pending step; fall back to the telegate's current step if nothing
        is pending.

        Args:
            data_key (int): quantum-manager key of the completed local data qubit.
            role (str): completing party's role (kept for the callback signature).
        """
        step = max(self._pending) if self._pending else getattr(self.tgate, "_current_step", None)
        self._ack_deferred_unit(step)

    def _on_teledata_complete(self, data_key: int) -> None:
        """DEST side: the moved state is now in a local data slot. Report the relocation
        as a delta on the step's ACK (the controller folds it into its registry -- the one
        owner of the logical->physical map) and release the deferred step ACK."""
        arr = self.node.components[self.data_array]
        slot = next((i for i, m in enumerate(arr.memories) if m.qstate_key == data_key), None)
        if slot is None or slot not in self._expected_moves:
            # Not an agent-managed move (e.g. a bare TeledataApp test) -> ignore.
            return
        step, q = self._expected_moves.pop(slot)
        self._pending_deltas.setdefault(step, []).append(
            {"qubit": q, "node": self.node.name, "slot": slot, "key": data_key})
        log.logger.info(f"[qpu_agent:{self.node.name}] MOVE arrived: q={q} now here at slot={slot} (step={step})")
        self._ack_deferred_unit(step)

    def _on_teledata_source_complete(self, protocol) -> None:
        """SOURCE side: Bob acknowledged the teleport, so ``q`` has left this node. The
        controller's registry already relocates ``q`` (via the dest node's ACK delta); the
        source just releases the deferred step ACK."""
        src_slot = getattr(protocol, "data_memory_index", None)
        mv = self._move_by_srcslot.pop(src_slot, None)
        if mv is None:
            return
        step, q, dest = mv[0], mv[1], mv[2]
        log.logger.info(f"[qpu_agent:{self.node.name}] MOVE departed: q={q} left slot={src_slot} → {dest} (step={step})")
        self._ack_deferred_unit(step)

    # ── reservation helpers ─────────────────────────────────────────────────
    def _peer_cc_delay(self, peer: str) -> float:
        """Classical-channel delay (ps) to ``peer``, read from the network config.

        Falls back to this node's representative cc delay (and ultimately
        ``DEFAULT_CC_DELAY``) if there is no direct channel to ``peer``.
        """
        ch = self.node.cchannels.get(peer)
        delay = getattr(ch, "delay", 0) if ch is not None else 0
        return float(delay) if delay and delay > 0 else self.cc_delay

    def _reservation_for(self, peer: str) -> tuple:
        """Reservation window + size for an entanglement session to ``peer``.

        Returns ``(start_t, end_t, memory_size, fidelity, hops)``. The window is
        derived from the peer's classical-channel delay (opens after a couple of
        cc hops; sized generously — TelegateApp's early expire frees the comm
        memory the instant the gate completes, so an over-long window can't starve
        the next session). The size follows the injected reservation policy: deeper
        multi-hop paths lose fidelity to each swap, so they reserve more raw pairs
        and settle for an achievable target fidelity (default (1, 1.0) = 1-hop).

        Args:
            peer (str): name of the remote node for this session.
        """
        cc_delay = self._peer_cc_delay(peer)
        t0 = int(self.tl.now() + RESERVATION_SLACK_CC_MULT * cc_delay)
        t1 = int(t0 + TELEGATE_WINDOW_CC_MULT * cc_delay)
        hops = self.hop_distances.get(peer, 1)
        mem_size, fidelity = self.reservation_policy(hops)
        return t0, t1, mem_size, fidelity, hops
