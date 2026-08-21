"""Barriered distributed quantum computing (DQC) node app.

This module provides a minimal, barriered (non-pipelined) DQC application that
runs on a quantum node. The app executes controller-broadcast steps that may
contain either local (single-node) gates or remote (teleported) two-qubit gates.

Remote two-qubit gates are delegated to a companion :class:`TelegateApp`
implementation which handles the entanglement/BSM workflow and notifies this
app when the local half of the operation completes. Until that completion, the
ACK for the corresponding controller step is deferred.

Classes
-------
DQCMsgType
    Enum for DQC message types using auto() pattern.
DQCMessage
    DQC message payloads with structured message types.

DQCApp
    Barriered executor for per-step circuit fragments on a quantum node.
"""


from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
from types import MethodType

from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.utils import log
from sequence.app.teleportation import TelegateApp, TeledataApp
from sequence.topology.node import DQCNode
from enum import Enum, auto


class _DualAppRouter:
    """Routes a node's single app-callback slot to BOTH a TelegateApp and a
    TeledataApp.

    A :class:`~sequence.topology.node.DQCNode` has one ``node.app`` slot, and
    memory/reservation callbacks go through it. To run telegate *and* teledata on
    the same node (needed for a hybrid compiler), we register this router as the
    node's app and dispatch each callback to the right sub-app:

    * reservation callbacks are routed by the reservation's ``app_label`` (set to
      the initiating app's name and carried to the responder on the RSVP message);
    * ``get_memory`` is routed by which sub-app has that memory index mapped to a
      reservation (each app only maps its own, thanks to the label routing above).
    """

    def __init__(self, tgate: TelegateApp, tdata: TeledataApp):
        self.tgate = tgate
        self.tdata = tdata
        self.name = "dqc_app_router"

    def _route_by_label(self, reservation):
        label = getattr(reservation, "app_label", "") or ""
        return self.tdata if "teledata" in label else self.tgate

    def get_reservation_result(self, reservation, result: bool) -> None:
        self._route_by_label(reservation).get_reservation_result(reservation, result)

    def get_other_reservation(self, reservation) -> None:
        self._route_by_label(reservation).get_other_reservation(reservation)

    def get_memory(self, info) -> None:
        if info.index in self.tdata.memo_to_reservation:
            self.tdata.get_memory(info)
        elif info.index in self.tgate.memo_to_reservation:
            self.tgate.get_memory(info)

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

class DQCMsgType(Enum):
    """Message types for DQC communications."""
    STEP_MESSAGE = auto()   # Controller -> Node: broadcast step to execute
    ACK = auto()  # Node -> Controller: acknowledge completion of a step

class DQCMessage(Message):
    """DQC message payloads.
    
    ACK:
    • step: Controller step index completed on this node
    • node: Sender node name (for controller bookkeeping)
    """

    def __init__(self, msg_type: DQCMsgType, receiver: str, **kwargs):
        super().__init__(msg_type=msg_type, receiver=receiver)

        if msg_type is DQCMsgType.STEP_MESSAGE:
            self.step = kwargs['step']
            self.node = kwargs['node']
            self.string = f"DQCMessage(type={msg_type}, moving to step={self.step}, sending from controller to node={self.node})"
        
        elif msg_type is DQCMsgType.ACK:
            self.step = kwargs['step']
            self.node = kwargs['node']
            self.string = f"DQCMessage(type={msg_type}, step={self.step} is complete, sending from {self.node} to controller)"
    
    def __str__(self):
        return self.string

# Legacy alias for backward compatibility
class DQCApp:
    """Barriered (non-pipelined) DQC node application.

    TODO: Make sure each DQCApp has a unique name
    Behavior
    --------
    • Each controller step is executed *to completion* on this node before an
      ACK is sent.
    • If a step contains a teleported CNOT (remote op), the ACK is **deferred**
      until :class:`TelegateApp` reports local completion on this node.
    • Local-only steps are executed immediately and ACKed right away.
    • No queuing or pipelining of future steps is performed.

    Conventions
    -----------
    • Remote CZ is normalized to CNOT.
    • For remote ops, ``op['targets'] = [control, target]`` using **global**
      qubit indices.

    Args:
        node: The underlying :class:`~sequence.topology.node.QuantumNode`.
        qubit_to_node (Dict[int, str]): Map from global qubit index to owning
            node name.
        local_ops (List[dict]): This node's local ops keyed by step
            (each op dict contains at least ``step``, ``gate``, and ``targets``).
        remote_ops (List[dict]): This node's remote (teleported) ops keyed by step.
        data_owned (Dict[int, int]): Map from **global** qubit index to this
            node's **local** data-memory slot index.
        data_array (str): Component name of the data memory array (e.g., ``"data_mem"``).
        dt (float, optional): Base time budget for remote-op reservation windows
            (picoseconds). Defaults to ``1e6``.
        peer_slots (Optional[Dict[str, Dict[int, int]]], optional): Nested map
            ``{peer_node: {global_q: peer_local_slot}}`` indicating, for each
            peer and global qubit, the slot index in the peer's data memory.

    Attributes:
        name (str): Protocol receiver label used to route StepMessages (``"dqc_node"``).
        tgate (TelegateApp): Helper app executing teleported CNOT on this node.
        busy_slots (set[int]): Data-memory slots currently locked by in-flight
            remote operations.
        key_to_slot (Dict[int, int]): Reverse map from a quantum-manager key to
            the local slot (used to unlock on completion).
        _pending (Dict[int, int]): Controller step -> count of telegates on this
            node still in flight; the step's ACK is deferred until the count hits 0.
    """

    def __init__(self,
                 node,
                 qubit_to_node: Dict[int, str],
                 local_ops:  List[dict],
                 remote_ops: List[dict],
                 data_owned: Dict[int, int],
                 dt: float = 1e6,
                 peer_slots: Optional[Dict[str, Dict[int, int]]] = None,
                 target_ops: Optional[List[dict]] = None,
                 controller: Optional[object] = None,
                 hop_distances: Optional[Dict[str, int]] = None,
                 reservation_policy: Optional[Callable[[int], tuple]] = None,
                 move_ops: Optional[List[dict]] = None):
        self.name = "dqc_node"
        self.protocol_type = "dqc_app"  # Required by sequence framework
        self.node = node
        self.tl = node.timeline

        # Physical hop-distance from this node to each peer, and a policy mapping
        # hop count -> (memory_size, target_fidelity) for the telegate's entanglement
        # reservation. A multi-hop pair is degraded by every swap, so deeper paths
        # must reserve MORE raw pairs (larger memory_size) to purify back up to the
        # target fidelity; a direct (1-hop) pair needs none. Default keeps the old
        # single-pair / perfect-fidelity behaviour when no policy is supplied.
        self.hop_distances = hop_distances or {}
        self.reservation_policy = reservation_policy or (lambda hops: (1, 1.0))

        self.qubit_to_node = qubit_to_node
        # Group ops by controller step. A packing compiler (DQCCompiler) can place
        # several independent gates on the SAME step, so a node may hold >1 op of a
        # given role in one step (e.g. hub as control of two telegates, or a grid
        # node that is the target of one gate and the control of another). Keying by
        # step must therefore map to a LIST -- a `{step: op}` dict would silently drop
        # all but the last, which is what broke DQCCompiler runs.
        def _group_by_step(ops):
            grouped: Dict[int, List[dict]] = {}
            for op in ops:
                grouped.setdefault(op['step'], []).append(op)
            return grouped

        self.local_map = _group_by_step(local_ops)
        self.remote_map = _group_by_step(remote_ops)
        self.target_map = _group_by_step(target_ops or [])
        # Teledata "move" ops: {step, qubit, dest, dest_slot} -- teleport a qubit's
        # STATE to another node. Held on whichever node is a party (current owner =
        # source, or dest). After a move the qubit's location bookkeeping is updated
        # in place (data_owned / qubit_to_node / peer_slots are shared across apps).
        self.move_map = _group_by_step(move_ops or [])
        self.data_owned = data_owned
        self.data_array = node.data_memo_arr_name
        self.dt = dt
        self.peer_slots = peer_slots or {}
        self.controller = controller

        # Representative classical-channel delay (ps) read from the network config.
        # Used to size the local-op ACK delay; the reservation window uses the
        # peer-specific delay (see _peer_cc_delay).
        cc_delays = [getattr(ch, "delay", 0) for ch in node.cchannels.values()
                     if getattr(ch, "delay", 0) and getattr(ch, "delay", 0) > 0]
        self.cc_delay = float(min(cc_delays)) if cc_delays else DEFAULT_CC_DELAY
        self.local_gate_time = LOCAL_GATE_TIME

        # One TelegateApp + one TeledataApp per node (remote-CNOT + qubit-move
        # executors). Each calls node.set_app() in its ctor, so the LAST one wins
        # that single slot; we then install a router that dispatches to both.
        self.tgate = TelegateApp(self.node)
        self.tdata = TeledataApp(self.node)
        self._router = _DualAppRouter(self.tgate, self.tdata)
        self.node.set_app(self._router)

        # Teledata move bookkeeping.
        #   _expected_moves: dest_slot -> (step, global_q) for moves landing here.
        #   _move_by_srcslot: source local slot -> (step, global_q, dest) for moves
        #                     leaving here (matched on the source-completion hook).
        self._expected_moves: Dict[int, tuple] = {}
        self._move_by_srcslot: Dict[int, tuple] = {}
        self._wrap_teledata_complete()
        self.tdata.on_source_complete = self._on_move_source_complete

        # Slot/Key bookkeeping
        self.busy_slots: set[int] = set()
        self.key_to_slot: Dict[int, int] = {}

        # Steps awaiting local teleported-gate completion before ACKing, mapped to
        # the number of telegates on this node still in flight for that step. A step
        # is ACKed only when its count returns to zero, so a node hosting two
        # telegates in one packed step waits for BOTH before releasing the barrier.
        self._pending: Dict[int, int] = {}

        # Ensure slot unlocking on Telegate completion
        self._wrap_telegate_complete()

        # When Telegate finishes locally, send the deferred ACK for that step.
        def _telegate_done(role: str, data_key: int, step: Optional[int]):
            """Internal callback: send deferred ACK when Telegate completes.

            Args:
                role (str): ``"control"`` or ``"target"`` (local role).
                data_key (int): Quantum-manager key of the completed local data qubit.
                step (Optional[int]): Controller step to ACK (if still pending).

            Side Effects:
                Decrements the step's pending-telegate count and, once it reaches
                zero, sends the deferred :class:`DQCMessage` ACK to the controller.
            """
            log.logger.info(f"[dqc_app:{self.node.name}] ===== TELEGATE DONE CALLBACK =====")
            log.logger.info(f"[dqc_app:{self.node.name}] role={role}, data_key={data_key}, step={step}")
            log.logger.info(f"[dqc_app:{self.node.name}] pending={self._pending}")

            if step is not None and step in self._pending:
                self._pending[step] -= 1
                if self._pending[step] <= 0:
                    del self._pending[step]
                    ack = DQCMessage(DQCMsgType.ACK, receiver="controller", step=step, node=self.node.name)
                    self._send_to_controller(ack)
                    log.logger.info(f"[dqc_app:{self.node.name}] Deferred ACK sent for step={step} after all telegates completed")
                else:
                    log.logger.info(f"[dqc_app:{self.node.name}] Telegate done for step={step}; {self._pending[step]} still in flight")
            else:
                log.logger.warning(f"[dqc_app:{self.node.name}] Telegate completed but step {step} not pending")
        
        # Make _telegate_done accessible as a method
        self._telegate_done = _telegate_done

        # Set up telegate completion callback
        self.tgate.telegate_complete = self._telegate_complete_wrapper

        log.logger.info(f"DQCApp initialized (barriered) for node={getattr(node, 'name', 'node')}")

        # Attach DQCApp to node for telegate access
        node.dqc_app = self

        # Register for StepMessages
        node.protocols.append(self)

    def _wrap_telegate_complete(self) -> None:
        """Wrap :meth:`TelegateApp.gate_complete` to unlock local data slots.

        The wrapped method:
          1) calls the original ``gate_complete`` on :class:`TelegateApp`
             (allowing it to capture results / log state), then
          2) removes the local slot from ``busy_slots`` using the recorded
             ``key_to_slot`` reverse map.

        Side Effects:
            Mutates ``tgate.gate_complete`` to include unlocking.
        """
        orig = getattr(self.tgate, 'gate_complete', None)

        def _wrapper(_self, role: str, data_key: int, _orig=orig, _dqc=self):
            if _orig is not None:
                _orig(role, data_key)
            slot = _dqc.key_to_slot.pop(data_key, None)
            if slot is not None:
                if slot in _dqc.busy_slots:
                    _dqc.busy_slots.remove(slot)
                log.logger.info(f"[dqc_app:{_dqc.node.name}] TeleGate complete (role={role}, key={data_key}) → unlocked slot={slot}")
            else:
                log.logger.debug(f"[dqc_app:{_dqc.node.name}] TeleGate complete but no slot mapping for key={data_key}")

        self.tgate.gate_complete = MethodType(_wrapper, self.tgate)

    def _peer_cc_delay(self, peer: str) -> float:
        """Classical-channel delay (ps) to ``peer``, read from the network config.

        Falls back to this node's representative cc delay (and ultimately
        ``DEFAULT_CC_DELAY``) if there is no direct channel to ``peer``.
        """
        ch = self.node.cchannels.get(peer)
        delay = getattr(ch, "delay", 0) if ch is not None else 0
        return float(delay) if delay and delay > 0 else self.cc_delay

    def _defer(self, step: int) -> None:
        """Register one more in-flight telegate for ``step``; the step's ACK is
        held until every such telegate has completed (see ``_telegate_done``)."""
        self._pending[step] = self._pending.get(step, 0) + 1
        log.logger.info(f"[dqc_app:{self.node.name}] deferring step {step} (pending now: {self._pending})")

    def _ack_deferred_unit(self, step: int) -> None:
        """Mark one deferred unit of ``step`` complete; ACK when the count hits 0.

        Shared by telegate and teledata completions: a step ACKs only once every
        in-flight remote op on this node (telegate half or teleport half) is done.
        """
        if step is not None and step in self._pending:
            self._pending[step] -= 1
            if self._pending[step] <= 0:
                del self._pending[step]
                ack = DQCMessage(DQCMsgType.ACK, receiver="controller", step=step, node=self.node.name)
                self._send_to_controller(ack)
                log.logger.info(f"[dqc_app:{self.node.name}] deferred ACK sent for step={step}")

    def _ack_local(self, step: int, reason: str = "") -> None:
        """ACK a local/no-op step, charging a local-gate time to real work.

        A step with actual local gates on this node ("local-only") takes one
        local-gate duration; a "no-op" step (this node idle while others work)
        ACKs immediately (0). Since the controller advances only when EVERY node
        has ACKed, a step's duration is the slowest node's work: a telegate time
        if the step carries one (that ACK is deferred to Telegate completion),
        else one local-gate time if any node has local work, else 0.
        """
        delay = self.local_gate_time if reason == "local-only" else 0
        ack = DQCMessage(DQCMsgType.ACK, receiver="controller", step=step, node=self.node.name)
        ev = Event(int(self.tl.now() + delay),
                   Process(self, "_send_to_controller", [ack]))
        self.tl.schedule(ev)
        log.logger.debug(f"[dqc_app:{self.node.name}] ACK for step={step} scheduled "
                         f"after delay={delay} ({reason})")

    def _send_to_controller(self, msg: Message):
        """Send an ACK to the controller. Preferred path: route over this node's
        classical channel to the controller (the controller is a real topology
        node). Fallback (e.g. paper_replication, whose controller is not a network
        node): call its ``received_message`` directly."""
        ctrl_name = getattr(self.controller, "name", "controller")
        if ctrl_name in getattr(self.node, "cchannels", {}):
            log.logger.debug(f"[dqc_app:{self.node.name}] ACK -> {ctrl_name} via channel: {msg}")
            self.node.send_message(ctrl_name, msg)
        elif self.controller is not None:
            log.logger.debug(f"[dqc_app:{self.node.name}] ACK -> controller (direct): {msg}")
            self.controller.received_message(self.node.name, msg)
        else:
            log.logger.warning(f"[dqc_app:{self.node.name}] No controller instance available")

    def _telegate_complete_wrapper(self, data_key: int, role: str = "unknown"):
        """Wrapper for telegate completion that sends ACK to controller."""
        log.logger.info(f"[dqc_app:{self.node.name}] ===== TELEGATE COMPLETE WRAPPER =====")
        log.logger.info(f"[dqc_app:{self.node.name}] telegate_complete_wrapper called: data_key={data_key}, role={role}")

        # Use the step from deferred_steps instead of relying on telegate's _current_step
        # This ensures we ACK the correct step that was deferred
        deferred_step = None
        if self._pending:
            # Under the controller barrier only one step is network-active at a
            # time, so every in-flight telegate belongs to the same (max) step.
            deferred_step = max(self._pending)
            log.logger.info(f"[dqc_app:{self.node.name}] Using deferred step: {deferred_step}")
        else:
            # Fallback to telegate's current step if no deferred steps
            deferred_step = getattr(self.tgate, '_current_step', None)
            log.logger.info(f"[dqc_app:{self.node.name}] No deferred steps, using telegate current step: {deferred_step}")

        # Call the DQC app's completion callback directly
        self._telegate_done(role, data_key, deferred_step)

    def received_message(self, src: str, msg: Message) -> bool:
        """Handle broadcast step messages from the controller.

        The controller sends :class:`Message` objects with ``receiver="dqc_node"``
        and a ``step`` attribute. This handler executes any local and/or remote
        work for that step on this node and sends an ACK when appropriate.

        ACK policy:
            • No-op step on this node → immediate ACK.
            • Local-only step → ACK after executing local ops.
            • Remote step → defer ACK until local teleported-gate completion.

        Args:
            src (str): Sender name (the controller).
            msg (Message): Step message (must carry ``step`` and target ``dqc_node``).

        Returns:
            bool: ``True`` if the message was handled; ``False`` otherwise.
        """
        log.logger.debug(f"[dqc_app:{self.node.name}] Received message from {src}: type={type(msg).__name__}, msg_type={getattr(msg, 'msg_type', 'no msg_type')}")
        
        
        log.logger.info(f"[dqc_app:{self.node.name}] ===== RECEIVED STEP {msg.step} MESSAGE FROM {src} =====")

        step = msg.step
        now_ps = int(self.tl.now())
        has_local = step in self.local_map
        has_remote = step in self.remote_map
        has_target = step in self.target_map
        has_move = step in self.move_map
        # No work on this node for this step → ACK immediately
        if not has_local and not has_remote and not has_target and not has_move:
            log.logger.info(f"[{self.node.name}@t={now_ps}] no op at step={step}")
            self._ack_local(step, "no-op")
            return True
        # A packed step may carry several ops of the same role on this node; run
        # them ALL. `deferred` records whether we launched any telegate, in which
        # case the step's ACK waits for the pending counter to drain to zero.
        deferred = False

        # 1) Run local work immediately (synchronous). One node can hold several
        #    independent local gates in a packed step.
        if has_local:
            for op in self.local_map[step]:
                op_l = dict(op); op_l["_kind"] = "local"
                log.logger.info(f"[{self.node.name}@t={now_ps}] LOCAL step={step} gate={op_l.get('gate')} targets={op_l.get('targets')}")
                self._do_local(op_l)

        # 2) Handle remote work (teleported CNOT). A node can be the control of one
        #    telegate and the target of another in the same step -- start each.
        if has_remote:
            for op in self.remote_map[step]:
                op_r = dict(op); op_r["_kind"] = "remote"
                qs = op_r.get('qubits', op_r.get('targets', []))
                if len(qs) != 2:
                    log.logger.warning(f"malformed remote op @step={step}: targets={qs}")
                    continue

                ctrl_q, tgt_q = qs
                ctrl_owner = self.qubit_to_node[ctrl_q]
                tgt_owner = self.qubit_to_node[tgt_q]
                peer_nm = tgt_owner

                log.logger.info(f"[dqc_app:{self.node.name}] REMOTE OP: ctrl_q={ctrl_q}, tgt_q={tgt_q}, ctrl_owner={ctrl_owner}, tgt_owner={tgt_owner}, peer_nm={peer_nm}")

                if self.node.name == ctrl_owner:
                    # Control: lock slot and START TeleGate; ACK sent on completion.
                    ctrl_slot = self._local_slot(ctrl_q)
                    self.busy_slots.add(ctrl_slot)
                    self._defer(step); deferred = True
                    log.logger.info(f"[dqc_app:{self.node.name}] Starting telegate CONTROL for step {step}")
                    self._start_telegate_control(op_r, ctrl_q, ctrl_slot, peer_nm, tgt_q, step=step)
                elif self.node.name == tgt_owner:
                    # Target: lock slot and register the responder slot for this step.
                    tgt_slot = self._local_slot(tgt_q)
                    self.busy_slots.add(tgt_slot)
                    self._defer(step); deferred = True
                    arr = self.node.components[self.data_array]
                    key = arr.memories[tgt_slot].qstate_key
                    self.key_to_slot[key] = tgt_slot
                    try:
                        self.tgate._current_step = step
                        self.tgate.set_target_slot_for_step(step, tgt_slot)
                    except Exception:
                        pass
                    log.logger.info(
                        "REMOTE step=%d (target-side lock): locking slot=%d on %s for q=%d (key=%s)",
                        step, tgt_slot, self.node.name, tgt_q, key
                    )
                # else: not a party to this telegate -> nothing to do on this node.

        # 3) Handle target operations (responder side, separate target list).
        if has_target:
            for op in self.target_map[step]:
                op_t = dict(op); op_t["_kind"] = "target"
                gate = str(op_t.get('gate', '')).lower()
                qs = op_t.get('qubits', op_t.get('targets', []))
                if len(qs) != 2:
                    continue
                ctrl_q, tgt_q = qs
                if self.node.name == self.qubit_to_node[tgt_q]:
                    tgt_slot = self._local_slot(tgt_q)
                    self.busy_slots.add(tgt_slot)
                    self.tgate.set_target_slot_for_step(step, tgt_slot)
                    self.tgate._current_step = step
                    self._defer(step); deferred = True
                    log.logger.info(f"[{self.node.name}@t={now_ps}] TARGET step={step} gate={gate} targets={qs} slot={tgt_slot}")

        # 3.5) Handle teledata moves. This node is the SOURCE if it currently owns
        #      the qubit, or the DEST if the qubit is being teleported to it.
        if has_move:
            for op in self.move_map[step]:
                q = op["qubit"]
                dest = op["dest"]
                dest_slot = op["dest_slot"]
                owner = self.qubit_to_node[q]
                if self.node.name == owner:
                    src_slot = self._local_slot(q)
                    self._defer(step); deferred = True
                    self._start_teleport_source(q, src_slot, dest, dest_slot, step=step)
                elif self.node.name == dest:
                    # Reserve the landing slot and wait for the state to arrive.
                    self.busy_slots.add(dest_slot)
                    self._expected_moves[dest_slot] = (step, q)
                    self._defer(step); deferred = True
                    log.logger.info(f"[{self.node.name}@t={now_ps}] MOVE dest step={step}: q={q} → slot={dest_slot}")
                # else: not a party to this move.

        # 4) If we launched telegates, their completion drives the ACK; otherwise
        #    this was local-only/no-op work, so ACK after a local-op delay.
        if not deferred:
            self._ack_local(step, "local-only")
        return True
    # ───────────────────────── helpers ─────────────────────────

    def _local_slot(self, global_q: int) -> int:
        """Resolve the local data-memory slot for a global qubit index.

        Args:
            global_q (int): Global qubit index.

        Returns:
            int: Local slot index in this node's data memory.

        Raises:
            KeyError: If this node does not own ``global_q`` according to ``data_owned``.
        """
        if global_q not in self.data_owned:
            msg = (f"{self.node.name}: global_q={global_q} not owned by this node. "
                   f"data_owned keys={list(self.data_owned.keys())}")
            log.logger.error(msg)
            raise KeyError(msg)
        return self.data_owned[global_q]

    def _remote_slot(self, peer_node: str, global_q: int) -> int:
        """Resolve the peer node's local data-memory slot for a global qubit.

        Args:
            peer_node (str): Peer node name.
            global_q (int): Global qubit index.

        Returns:
            int: Peer node's local slot index for ``global_q``.

        Raises:
            KeyError: If ``peer_slots`` lacks an entry for the given peer or qubit.
        """
        try:
            return self.peer_slots[peer_node][global_q]
        except Exception:
            msg = (f"{self.node.name}: missing peer slot for node='{peer_node}', "
                   f"global_q={global_q}. Pass peer_slots={{node:{{q:slot}}}} when constructing DQCApp.")
            log.logger.error(msg)
            raise KeyError(msg)
    # ───────────────────────── local ops ─────────────────────────

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
            log.logger.warning(f"[dqc_app:{self.node.name}] _do_local with no targets: {op}")
            return

        # Map global qubit IDs → local memory slots on this node
        arr = self.node.components[self.data_array]   # MemoryArray
        slots: List[int] = [self._local_slot(q) for q in qs]

        # Get the QState keys corresponding to those slots
        keys: List[int] = [arr.memories[s].qstate_key for s in slots]

        # Local circuit indices: 0..(len(slots)-1)
        # slot -> local circuit index
        slot_to_idx = {slot: i for i, slot in enumerate(slots)}

        # Circuit acts on exactly these |slots| qubits, in the order of 'keys'
        circ = Circuit(len(slots))

        log.logger.debug(
            f"[_do_local:{self.node.name}] gate={gate}, qs={qs}, "
            f"slots={slots}, keys={keys}"
        )

        # ---- Apply gates using local indices ----

        if gate == "h":
            for s in slots:
                circ.h(slot_to_idx[s])

        elif gate == "x":
            for s in slots:
                circ.x(slot_to_idx[s])

        elif gate == "y":
            for s in slots:
                circ.y(slot_to_idx[s])

        elif gate == "z":
            for s in slots:
                circ.z(slot_to_idx[s])

        elif gate == "s":
            for s in slots:
                circ.s(slot_to_idx[s])

        elif gate == "sdg":
            for s in slots:
                circ.sdg(slot_to_idx[s])

        elif gate == "t":
            for s in slots:
                circ.t(slot_to_idx[s])

        elif gate == "phase":
            # Use Circuit's phase method for phase rotation
            theta = op.get("arg", 0.0)
            for s in slots:
                circ.phase(slot_to_idx[s], theta)

        elif gate == "cx" and len(slots) >= 2:
            # Two-qubit CNOT: first target is control, second is target (per your compiler)
            ctrl_slot, tgt_slot = slots[0], slots[1]
            circ.cx(slot_to_idx[ctrl_slot], slot_to_idx[tgt_slot])

        elif gate == "cz" and len(slots) >= 2:
            ctrl_slot, tgt_slot = slots[0], slots[1]
            circ.cz(slot_to_idx[ctrl_slot], slot_to_idx[tgt_slot])

        elif gate == "measure":
            # If you ever push measurement as a local op
            for s in slots:
                circ.measure(slot_to_idx[s])

        else:
            log.logger.debug(
                f"[dqc_app:{self.node.name}] Unsupported local gate '{gate}' "
                f"with op={op}"
            )
            return

        # ---- Run circuit on the given keys ----
        rnd = self.node.get_generator().random()
        res: Dict[int, int] = self.tl.quantum_manager.run_circuit(circ, keys, rnd)

        log.logger.debug(
            f"[_do_local:{self.node.name}] run_circuit returned mapping: {res}"
        )

        # ---- Update MemoryArray qstate_keys if needed ----
        # Convention: if res is empty, keys unchanged; otherwise, remap.
        if res:
            for slot, old_key in zip(slots, keys):
                new_key = res.get(old_key, old_key)
                if new_key != old_key:
                    log.logger.debug(
                        f"[_do_local:{self.node.name}] updating slot={slot} "
                        f"key {old_key} → {new_key}"
                    )
                    arr.memories[slot].qstate_key = new_key
        else:
            # Unitary-only, in-place update; keep keys as-is.
            pass



    # ───────────────────────── remote op (control-side start) ─────────────────────────

    def _start_telegate_control(self, op: Dict[str, Any],
                                ctrl_q: int, ctrl_slot: int,
                                peer_nm: str, tgt_q: int,
                                step: int) -> None:
        """Start a teleported CNOT where this node owns the control qubit.

        The target node will lock its own local slot upon receiving the same
        controller step. Slot unlocking occurs when :class:`TelegateApp`
        invokes :meth:`gate_complete` (wrapped to unlock in
        :meth:`_wrap_telegate_complete`). The ACK for ``step`` is deferred and
        sent by the :class:`TelegateApp` completion callback registered in
        :meth:`__init__`.

        Args:
            op (Dict[str, Any]): Operation descriptor for the remote op.
            ctrl_q (int): Global index of the control qubit owned by this node.
            ctrl_slot (int): Local data-memory slot index for ``ctrl_q``.
            peer_nm (str): Peer (target-owner) node name.
            tgt_q (int): Global index of the target qubit owned by ``peer_nm``.
            step (int): Controller step index.

        Side Effects:
            Schedules/starts the teleported CNOT via :class:`TelegateApp` over a
            wide reservation window and records key/slot bookkeeping to unlock
            later on completion.
        """
        gate_typ = str(op.get('gate', 'cx')).lower()   # 'cx' or 'cz'; forwarded to the telegate

        # We need our local data key to unlock on completion
        arr = self.node.components[self.data_array]
        key = arr.memories[ctrl_slot].qstate_key
        self.key_to_slot[key] = ctrl_slot

        self.tgate._current_step = step

        tgt_slot = self._remote_slot(peer_nm, tgt_q)

        # Reservation window derived from the peer's classical-channel delay.
        # Opens after a couple of cc hops, and is sized generously: TelegateApp's
        # early expire (TelegateApp._early_expire) frees the comm memory the
        # moment the gate completes, so a window longer than the gate can no
        # longer starve the next telegate (no timecard exhaustion).
        cc_delay = self._peer_cc_delay(peer_nm)
        t0 = int(self.tl.now() + RESERVATION_SLACK_CC_MULT * cc_delay)
        t1 = int(t0 + TELEGATE_WINDOW_CC_MULT * cc_delay)

        # Size the entanglement reservation to the physical distance to the peer:
        # a direct pair needs no purification, but each swap on a multi-hop path
        # degrades fidelity, so deeper paths reserve more raw pairs (memory_size)
        # and settle for an achievable target fidelity. Policy is injected at
        # construction; default is (1, 1.0) -> the original 1-hop behaviour.
        hops = self.hop_distances.get(peer_nm, 1)
        mem_size, fidelity = self.reservation_policy(hops)

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

    # ───────────────────────── teledata move (qubit relocation) ─────────────────
    def _wrap_teledata_complete(self) -> None:
        """Wrap :meth:`TeledataApp.teledata_complete` (fires on the DEST when the
        moved state lands in data memory) to run the dest-side bookkeeping + ACK."""
        orig = self.tdata.teledata_complete

        def _wrapper(data_key: int, _orig=orig, _dqc=self):
            _orig(data_key)                      # let the app record its result
            _dqc._on_move_dest_complete(data_key)

        self.tdata.teledata_complete = _wrapper

    def _start_teleport_source(self, q: int, src_slot: int,
                               dest: str, dest_slot: int, step: int) -> None:
        """Initiate a teledata move of global qubit ``q`` from this (source) node to
        ``dest``'s ``dest_slot``. Defers the step ACK until Bob acknowledges."""
        self.busy_slots.add(src_slot)
        self._move_by_srcslot[src_slot] = (step, q, dest, dest_slot)

        cc_delay = self._peer_cc_delay(dest)
        t0 = int(self.tl.now() + RESERVATION_SLACK_CC_MULT * cc_delay)
        t1 = int(t0 + TELEGATE_WINDOW_CC_MULT * cc_delay)
        hops = self.hop_distances.get(dest, 1)
        mem_size, fidelity = self.reservation_policy(hops)

        log.logger.info(f"[dqc_app:{self.node.name}] TELEPORT q={q} slot={src_slot} → {dest} slot={dest_slot}; "
                        f"hops={hops} mem_size={mem_size} fid={fidelity}; t=[{t0},{t1}]")
        self.tdata.start(responder=dest, start_t=t0, end_t=t1, memory_size=mem_size,
                         fidelity=fidelity, data_src=src_slot, dest_slot=dest_slot)

    def _on_move_dest_complete(self, data_key: int) -> None:
        """DEST side: the moved state is now in a local data slot. Claim the qubit
        into this node's data map (shared, so every node sees the new location) and
        release the deferred step ACK."""
        arr = self.node.components[self.data_array]
        slot = next((i for i, m in enumerate(arr.memories) if m.qstate_key == data_key), None)
        if slot is None or slot not in self._expected_moves:
            # Not a DQCApp-managed move (e.g. a bare TeledataApp test) -> ignore.
            return
        step, q = self._expected_moves.pop(slot)
        self.qubit_to_node[q] = self.node.name   # shared dict -> visible everywhere
        self.data_owned[q] = slot                # this node's map == peer_slots[dest]
        self.busy_slots.discard(slot)
        log.logger.info(f"[dqc_app:{self.node.name}] MOVE arrived: q={q} now here at slot={slot} (step={step})")
        self._ack_deferred_unit(step)

    def _on_move_source_complete(self, protocol) -> None:
        """SOURCE side: Bob acknowledged the teleport, so ``q`` has left this node.
        Drop it from this node's data map, free the source slot, and ACK the step."""
        src_slot = getattr(protocol, "data_memory_index", None)
        mv = self._move_by_srcslot.pop(src_slot, None)
        if mv is None:
            return
        step, q, dest = mv[0], mv[1], mv[2]
        self.data_owned.pop(q, None)             # this node no longer owns q
        self.busy_slots.discard(src_slot)
        log.logger.info(f"[dqc_app:{self.node.name}] MOVE departed: q={q} left slot={src_slot} → {dest} (step={step})")
        self._ack_deferred_unit(step)
