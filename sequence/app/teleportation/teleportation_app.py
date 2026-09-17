#!/usr/bin/env python3
"""TeleportationApp -- one node-attached app that runs BOTH teleported gates and qubit moves.

A :class:`~sequence.topology.node.DQCNode` is a :class:`QuantumRouter`, so it has a SINGLE
``self.app`` callback slot (the network/resource manager's reservation + memory callbacks all
forward to it). Running a telegate *and* a teledata on one node used to need two app objects behind
a router; instead this app subsumes both. It is the BASE holding all the app logic (moved verbatim
from the old TelegateApp/TeledataApp), with a thin kind-dispatch layer on top:

* each session's KIND (telegate vs teledata) is recorded by its reservation ``identity`` -- a unique
  id the controller assigns and delivers to BOTH endpoints in the step op, so each side calls
  :meth:`expect_session` before the reservation travels;
* :meth:`get_memory` sends an arriving entangled pair to the telegate or teledata half by that kind;
* :meth:`received_message` dispatches by message type.

The two standalone protocol tests still want a single-kind app, so :class:`TelegateApp` and
:class:`TeledataApp` remain as THIN subclasses that pin the kind and expose ``start``. The DQC
worker installs :class:`TeleportationApp` directly.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..request_app import RequestApp
from ...kernel.process import Process
from ...kernel.event import Event
from ...resource_management.memory_manager import MemoryInfo
from ...topology.node import DQCNode
from ...utils import log
from ...constants import TELEGATE, TELEDATA, TELEPORT

from .teleportation_base import TeleportationProtocol
from .telegate_protocol import TelegateMsgType
from .teledata_protocol import TeledataMsgType, TeledataMessage
from .teleport_protocol import TeleportMsgType, TeleportMessage


class TeleportationApp(RequestApp):
    """Unified telegate + teledata app occupying a DQCNode's single ``self.app`` slot.

    Holds ALL the app logic for both teleported gates (the "gate" half, methods prefixed
    ``_gate_``) and qubit moves (the "move" half, methods prefixed ``_move_``), and dispatches an
    arriving pair / message to the right half by the session kind recorded in :meth:`expect_session`.

    Attributes:
        node: The DQCNode that owns this app.
        name: Application name ("teleport").
        results: List of operation results for debugging/testing.
        data_keys: List of data qubit keys from completed operations.
        telegate_protocols: List of active telegate TeleportationProtocol instances.
        teledata_protocols: List of active teledata TeleportationProtocol instances.
        _complete_cbs: List of telegate completion callback functions.
        _target_slot_by_step: Mapping of step numbers to target memory slots.
        _current_step: Current step number for telegate operations.
        on_complete: Optional teledata responder-side callback (data_key -> None).
        on_source_complete: Optional teledata initiator-side callback (protocol -> None).
    """

    DEFAULT_KIND: Optional[str] = None

    def __init__(self, node: DQCNode):
        """Initialize a unified teleportation application.

        Args:
            node (DQCNode): the DQCNode that owns this app.
        """
        super().__init__(node)                       # RequestApp -> App registers node.app
        self.name = "teleport"

        # Results/keys for debugging or tests (shared by both halves).
        self.results: List = []
        self.data_keys: List[int] = []

        # ── Gate (telegate) state ────────────────────────────────────────────
        # Multiple concurrent telegate sessions.
        self.telegate_protocols: List = []

        # Completion callbacks: cb(role:str, data_key:int, step:Optional[int]).
        self._complete_cbs: List[Callable] = []

        # Target-owner predeclares the exact target data slot for an incoming telegate so the
        # responder lands the corrected qubit in the same slot the controller assigned. Keyed by
        # the session's reservation IDENTITY (concurrency-safe: two telegates to this node in one
        # wave get distinct slots). ``_target_slot_by_step`` + ``_current_step`` are the legacy
        # step-keyed fallback the standalone tests still use (no controller to assign identities).
        self._target_slot_by_identity: Dict[int, int] = {}
        self._target_slot_by_step: Dict[int, int] = {}
        self._current_step: Optional[int] = None

        # Reservations (by unique identity) whose FIRST pair has already been
        # bound to an Alice-side protocol. Lets get_memory tell a genuine NEW
        # concurrent session (bind it to the next unstarted protocol) apart from
        # an EXTRA purification pair of an already-running session (ignore it).
        self._alice_started_reservations: set = set()

        # ── Move (teledata) state ────────────────────────────────────────────
        # Maintain multiple concurrent teledata sessions.
        self.teledata_protocols: List = []

        # ── Teleport (bare state-teleport) state ─────────────────────────────
        # Multiple concurrent bare-teleport sessions.
        self.teleport_protocols: List = []

        # Optional callback fired on the INITIATOR (Alice) once Bob acknowledges a
        # completed teleport -- i.e. the moved state is now in Bob's data memory and
        # this session's resources are released. The QPU agent uses it to drive the
        # barrier ACK / free the source slot. Signature: on_source_complete(protocol).
        self.on_source_complete: Optional[callable] = None

        # Optional callback fired on the RESPONDER (Bob) once the teleported state
        # has landed in local data memory (end of teledata_complete). Mirrors
        # on_source_complete for the destination side. Signature: on_complete(data_key).
        self.on_complete: Optional[callable] = None

        # ── Kind dispatch ────────────────────────────────────────────────────
        self._kind_by_identity: Dict[int, str] = {}   # reservation identity -> TELEGATE | TELEDATA

    # ── kind dispatch ────────────────────────────────────────────────────────
    def expect_session(self, identity: int, kind: str) -> None:
        """Record session ``identity``'s KIND so :meth:`get_memory` routes its pair to the right
        half. Called on BOTH endpoints (from the controller's op) before the reservation travels.

        Args:
            identity (int): unique reservation identity of the session.
            kind (str): ``TELEGATE`` or ``TELEDATA``.
        """
        self._kind_by_identity[int(identity)] = kind

    def _kind_of(self, reservation) -> str:
        """Return the KIND recorded for ``reservation``'s session.

        Args:
            reservation (Reservation): reservation whose session kind to look up.

        Returns:
            str: ``TELEGATE`` or ``TELEDATA`` (falls back to this class's ``DEFAULT_KIND``,
            or ``TELEGATE`` on the base where it is unset).
        """
        identity = getattr(reservation, "identity", None)
        return self._kind_by_identity.get(identity, self.DEFAULT_KIND or TELEGATE)

    def get_memory(self, info) -> None:
        """Route an entangled memory to the telegate or teledata half by the session's kind.

        Args:
            info (MemoryInfo): information about the memory state change.
        """
        reservation = self.memo_to_reservation.get(info.index)
        if reservation is None:
            return
        kind = self._kind_of(reservation)
        if kind == TELEPORT:
            self._teleport_get_memory(info)
        elif kind == TELEDATA:
            self._move_get_memory(info)
        else:
            self._gate_get_memory(info)

    def received_message(self, src: str, msg) -> None:
        """Dispatch a direct app-to-app message to the half that owns its protocol.

        Args:
            src (str): source node name.
            msg (Message): received telegate or teledata message.
        """
        if isinstance(msg, TeledataMessage):
            self._move_received(src, msg)
        elif isinstance(msg, TeleportMessage):
            self._teleport_received(src, msg)
        else:
            self._gate_received(src, msg)

    def start_gate(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float,
                   control_src: int, target_src: int, step: Optional[int] = None, gate_type: str = "cx",
                   identity: int = 0):
        """Start a telegate session from the control owner (initiator).

        Args:
            responder: Name of the responder node (target qubit owner)
            start_t: Start time for entanglement reservation
            end_t: End time for entanglement reservation
            memory_size: Size of memory for entanglement
            fidelity: Required fidelity for entanglement
            control_src: Index of control qubit memory
            target_src: Index of target qubit memory
            step: Optional step number for the operation
            gate_type: Distributed gate to apply, "cx" (remote CNOT, default) or
                "cz" (remote CZ). Bob learns the choice from Alice's message.
            identity: unique reservation id (default 0 -> auto-assigned) so concurrent
                sessions route back to this app on both endpoints.
        """
        # Reserve entanglement window
        RequestApp.start(self, responder, start_t, end_t, memory_size, fidelity, identity=identity)

        # Create Alice-side protocol on the initiator (control owner)
        log.logger.info(f"[telegate:{self.node.name}] Creating Alice protocol with remote_node_name={responder}, gate_type={gate_type}")
        protocol = TeleportationProtocol.create(owner=self.node, alice=True, control_memory_index=control_src,
                                                target_memory_index=target_src, remote_node_name=responder,
                                                gate_type=gate_type, protocol_type=TELEGATE)
        if step is not None:
            protocol._step = step
        self.telegate_protocols.append(protocol)

        # Optionally mirror step for DQCApp callback convenience
        if step is not None:
            self._current_step = step  # DQCApp also sets this

    def start_move(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float, data_src: int,
                   dest_slot: Optional[int] = None, identity: int = 0):
        """Start a teledata session (this node acts as Alice).

        Args:
            responder (str): name of the remote node receiving the teleported state.
            start_t (int): start time of the reservation (in ps).
            end_t (int): end time of the reservation (in ps).
            memory_size (int): number of memories to reserve.
            fidelity (float): target fidelity of the entangled pair.
            data_src (int): index of the local data qubit to teleport.
            dest_slot (Optional[int]): data-memory slot on the responder to land the
                state in. Defaults to ``data_src`` (mirror the source index) when None,
                preserving the original behaviour; pass it explicitly to place the
                moved qubit in a specific slot (needed by DQCApp qubit moves).
            identity (int): unique reservation id (default 0 -> auto-assigned) so concurrent
                sessions route back to this app on both endpoints.
        """
        log.logger.debug(f"[TeledataApp:{self.node.name}] start() → responder={responder}, data_src={data_src}, dest_slot={dest_slot}")

        # Reserve and generate EPR pair(s)
        RequestApp.start(self, responder, start_t, end_t, memory_size, fidelity, identity=identity)

        # Create a new protocol instance for Alice
        protocol = TeleportationProtocol.create(owner=self.node, alice=True, data_memory_index=data_src,
                                                remote_node_name=responder, protocol_type=TELEDATA)
        # Where the state should land on the responder (defaults to the source index).
        protocol.dest_memory_index = data_src if dest_slot is None else dest_slot
        self.teledata_protocols.append(protocol)

    def start_teleport(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float, data_memory_index: int):
        """Start the teleportation process.

        NOTE: only teleport one data memory qubit

        Args:
            responder (str): Name of the responder node (Bob).
            start_t (int): Start time of the teleportation (in ps).
            end_t (int): End time of the teleportation (in ps).
            memory_size (int): Size of the memory used for the teleportation.
            fidelity (float): Target fidelity of the teleportation.
            data_memory_index (int): Index of the data qubit to be teleported.
        """
        log.logger.debug(f"{self.name}: start() → responder={responder}, data_memory_index={data_memory_index}")

        # reserve and generate EPR pair
        RequestApp.start(self, responder, start_t, end_t, memory_size, fidelity)

        # init a new teleportation protocol for Alice only, and append to the list
        teleport_protocol = TeleportationProtocol.create(self.node, alice=True, data_memory_index=data_memory_index, remote_node_name=responder, protocol_type=TELEPORT)
        self.teleport_protocols.append(teleport_protocol)

    def get_reservation_result(self, reservation, result: bool):
        """Handle the reservation result from the network manager.

        Args:
            reservation (Reservation): the reservation object.
            result (bool): True if the reservation succeeded, False otherwise.
        """
        RequestApp.get_reservation_result(self, reservation, result)

    # ── gate (telegate) half ─────────────────────────────────────────────────
    def set_target_slot_for_step(self, step: int, slot: int):
        """Set the target memory slot for a specific step.

        Optional helper for the target owner (responder) to force its local slot
        to match what DQCApp locked.

        Args:
            step: Step number
            slot: Memory slot index
        """
        self._target_slot_by_step[step] = int(slot)

    def set_current_step(self, step: int):
        """Set the current step for the telegate operation.

        Args:
            step: Current step number
        """
        self._current_step = int(step)

    def set_target_slot(self, identity: int, slot: int):
        """Predeclare the target data slot for the telegate session ``identity`` (responder side).

        Preferred over :meth:`set_target_slot_for_step`: keyed by the unique reservation identity,
        so a node that is the target of several telegates in one wave lands each in its own slot.

        Args:
            identity (int): reservation identity of the incoming telegate session.
            slot (int): local data-memory slot for the corrected target qubit.
        """
        self._target_slot_by_identity[int(identity)] = int(slot)

    def add_gate_complete_cb(self, cb: Callable):
        """Register a telegate completion callback.

        Args:
            cb: Callback function with signature cb(role, data_key, step)
        """
        self._complete_cbs.append(cb)

    def _gate_get_memory(self, info):
        """Handle memory entanglement events for the telegate half.

        Called when a memory becomes entangled. Determines whether this node is
        the initiator (Alice) or responder (Bob) and executes the appropriate
        protocol stage.

        Args:
            info: Memory information object containing entanglement details
        """
        log.logger.info(f"[telegate:{getattr(info.memory.owner, 'name', 'unknown')}] get_memory called: index={info.index}, state={info.state}, memo_to_reservation={list(self.memo_to_reservation.keys())}")

        # Accept a ready pair whether it was delivered directly (ENTANGLED) or via
        # entanglement purification (PURIFIED) -- the latter happens on multi-hop
        # paths where the swapped pair is purified up to target before delivery.
        if info.index not in self.memo_to_reservation or info.state not in ("ENTANGLED", "PURIFIED"):
            log.logger.warning(f"[telegate:{getattr(info.memory.owner, 'name', 'unknown')}] get_memory returning early: index in reservation={info.index in self.memo_to_reservation}, state={info.state}")
            return

        reservation = self.memo_to_reservation[info.index]
        identity = getattr(reservation, "identity", None)
        this_node = getattr(info.memory.owner, "name", None)

        # The reservation initiator owns the CNOT control; the responder owns the target.
        control_node = reservation.initiator
        target_node = reservation.responder

        log.logger.info(f"[telegate:{this_node}] get_memory: this_node={this_node}, control={control_node}, target={target_node}")

        # Control side → Alice-role (CNOT control → comm, measure, Z-correct).
        if this_node == control_node:
            log.logger.info(f"[telegate:{this_node}] Looking for Alice protocol for target={target_node}")
            # An EXTRA (purification) pair of a session already bound to a protocol:
            # the gate runs ONCE, so ignore it (stays a clean Bell pair, torn down in
            # _early_expire). Keyed on the reservation's UNIQUE identity so we don't
            # confuse it with a genuine concurrent session to the same target.
            if identity in self._alice_started_reservations:
                return
            for protocol in list(self.telegate_protocols):
                # Bind this NEW reservation's pair to the next UNSTARTED protocol.
                # Skipping (not bailing on) already-started protocols is what lets
                # multiple concurrent telegates to the same target each get their own.
                if (protocol.alice and protocol.owner.name == control_node
                        and protocol.remote_node_name == target_node
                        and not getattr(protocol, "_alice_started", False)):
                    protocol._alice_started = True
                    self._alice_started_reservations.add(identity)
                    log.logger.info(f"[telegate:{this_node}] Found Alice protocol, calling alice_stage")
                    protocol.set_alice_comm_memory_name(getattr(info.memory, "name", None))
                    protocol.set_alice_comm_memory(info.memory)
                    protocol.set_bob_comm_memory_name(getattr(info, "remote_memo", None))
                    process = Process(protocol, "alice_stage", [reservation])
                    event = Event(self.node.timeline.now(), process)
                    self.node.timeline.schedule(event)
                    return

            log.logger.error(f"[TelegateApp:{this_node}] no Alice protocol found for target={target_node}, id={identity}")
            raise RuntimeError("Alice-side protocol missing at control owner")

        # Target side → Bob-role (CNOT comm → target, H, measure, X-correct).
        if this_node == target_node:
            # Prefer the identity-keyed slot (controller-assigned, concurrency-safe); fall back to
            # the legacy step-keyed slot for standalone tests that have no session identities.
            target_slot = self._target_slot_by_identity.get(identity)
            if target_slot is None:
                step = getattr(self, "_current_step", None)
                target_slot = self._target_slot_by_step.get(step) if isinstance(step, int) else None
            if target_slot is None:
                log.logger.error(f"[TelegateApp:{this_node}] missing target slot for identity={identity} / step={getattr(self, '_current_step', None)}; did set_target_slot run on TARGET?")
                raise RuntimeError("Responder target slot not specified")
            log.logger.info(f"[telegate:{self.node.name}] Creating Bob protocol with remote_node_name={control_node}")
            protocol = TeleportationProtocol.create(owner=self.node, alice=False, control_memory_index=None,
                                                    target_memory_index=target_slot, remote_node_name=control_node, protocol_type=TELEGATE)
            protocol.set_bob_comm_memory_name(getattr(info.memory, "name", None))
            protocol.set_bob_comm_memory(info.memory)
            protocol.set_alice_comm_memory_name(getattr(info, "remote_memo", None))
            self.telegate_protocols.append(protocol)

            return

        # Otherwise ignore (only two parties per telegate)

    def _gate_received(self, src: str, msg):
        """Handle incoming telegate messages.

        Routes messages to the appropriate protocol instance based on message
        type and protocol matching criteria.

        Args:
            src: Source node name
            msg: Received telegate message
        """
        if msg.msg_type is TelegateMsgType.A_MEAS_RESULT:
            # Bob receives measurement result from Alice
            for protocol in list(self.telegate_protocols):
                if (not protocol.alice and src == protocol.remote_node_name and
                        msg.bob_comm_memory_name == protocol.bob_comm_memory_name):
                    bob_comm_memory = protocol.bob_comm_memory
                    protocol.received_message(src, msg)
                    # Bob's half of the gate is done: free its resources now instead
                    # of waiting for the reservation window to time out.
                    self._early_expire(msg.reservation, bob_comm_memory)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching telegate protocol for msg from {src}")
        elif msg.msg_type is TelegateMsgType.ACK:
            # Initiator receives acknowledgment from responder - delegate Z correction to protocol
            log.logger.info(f"[telegate:{self.node.name}] Received ACK from {src}, b_bit={getattr(msg, 'b_bit', 'unknown')}")
            for protocol in list(self.telegate_protocols):
                if (protocol.alice and src == protocol.remote_node_name and
                        msg.bob_comm_memory_name == protocol.bob_comm_memory_name):
                    log.logger.info(f"[telegate:{self.node.name}] Found matching Alice protocol for ACK")
                    # Delegate Z correction to the protocol
                    alice_comm_memory = protocol.alice_comm_memory
                    b_bit = getattr(msg, "b_bit", 0)
                    protocol.alice_z_correction(b_bit)
                    # Alice's half of the gate is done: free its resources now.
                    self._early_expire(msg.reservation, alice_comm_memory)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching telegate protocol for ACK from {src}")

    def _emit_complete(self, role: str, data_key: int, step: Optional[int] = None):
        """Emit completion events to all registered callbacks.

        Args:
            role: Role of the completing party ("alice" or "bob")
            data_key: Key of the data qubit
            step: Optional step number
        """
        for cb in list(self._complete_cbs):
            try:
                cb(role, data_key, step)
            except Exception as e:
                log.logger.warning(f"{self.name}: on_complete callback error: {e}")

    def gate_complete(self, role: str, data_key: int):
        """Handle gate completion.

        Called by DQCApp's wrapper and/or telegate_complete. Provides a single
        source of completion emission to avoid double-calling.

        Args:
            role: Role of the completing party ("alice" or "bob")
            data_key: Key of the data qubit
        """
        step = getattr(self, "_current_step", None)
        log.logger.info(f"[telegate:{self.node.name}] gate_complete called: role={role}, data_key={data_key}, step={step}")

        # Emit completion to any registered on_complete callbacks.
        log.logger.info(f"[telegate:{self.node.name}] Emitting completion for role={role}, data_key={data_key}, step={step}")
        self._emit_complete(role, data_key, step)

    def telegate_complete(self, data_key: int, role: str = "unknown"):
        """Handle telegate operation completion.

        Called by TelegateProtocol when a telegate operation completes on either
        Alice or Bob side. Extracts the final quantum state and triggers
        completion callbacks.

        Args:
            data_key: Key of the data qubit
            role: Role of the completing party ("alice" or "bob")
        """
        log.logger.info(f"[telegate:{self.node.name}] telegate_complete called: data_key={data_key}, role={role}")
        psi = self.node.timeline.quantum_manager.get(data_key).state

        self.data_keys.append(data_key)
        self.results.append((role, data_key, psi))

        # Single emission path: through gate_complete (DQCApp wraps this to unlock)
        self.gate_complete(role, data_key)

    # ── move (teledata) half ─────────────────────────────────────────────────
    def _move_get_memory(self, info: MemoryInfo):
        """Handle memory updates and wire comm memories to the right teledata protocol.

        Args:
            info (MemoryInfo): information about the memory state change.
        """
        log.logger.debug(f"{self.name}: get_memory(idx={info.index}, state={info.state})")
        # Accept PURIFIED as well as ENTANGLED: when the reservation asks for a
        # target fidelity above the raw pair fidelity (memory_size>1), the kept
        # pair is distilled and its MemoryInfo transitions to PURIFIED, not
        # ENTANGLED. Reacting only to ENTANGLED made purified teleports stall
        # forever (never Bell-measured). Telegate already accepts both.
        if info.index in self.memo_to_reservation and info.state in ("ENTANGLED", "PURIFIED"):
            reservation = self.memo_to_reservation[info.index]

            # Decide the ROLE for this pair from the reservation, not from
            # (owner, remote_node): if this node INITIATED the pair's reservation it
            # is the sender (Alice); otherwise it is the receiver (Bob). Matching by
            # (owner, remote_node) alone is ambiguous when this node is BOTH Alice
            # (sending to X) and Bob (receiving from X) at the same time -- e.g. a
            # BIDIRECTIONAL teleport a<->b -- and would let the Alice protocol grab
            # the Bob session's pair, so neither teleport ever completes (deadlock).
            is_sender = reservation.initiator == self.node.name
            # Try to match an existing Alice-side protocol. With several concurrent
            # teleports to the SAME remote node, every un-fulfilled Alice session
            # matches on (owner, remote_node); we give each ARRIVING EPR pair to a
            # DISTINCT session by skipping ones that already claimed a pair
            # (``_alice_started``), so pair k goes to the k-th session.
            for protocol in (list(self.teledata_protocols) if is_sender else []):
                this_node = getattr(info.memory.owner, 'name', None)
                remote_node = getattr(info, 'remote_node', reservation.responder)
                if (getattr(protocol, 'alice', False)
                        and this_node == protocol.owner.name
                        and remote_node == protocol.remote_node_name
                        and not getattr(protocol, '_alice_started', False)):
                    # Alice side -- this session now owns this EPR pair
                    protocol._alice_started = True
                    protocol.set_alice_comm_memory_name(getattr(info.memory, 'name', None))
                    protocol.set_alice_comm_memory(info.memory)
                    protocol.set_bob_comm_memory_name(getattr(info, 'remote_memo', None))
                    # Defer Alice's Bell measurement to a same-time, lower-priority
                    # event so Bob's EntanglementGenerationA._entanglement_succeed()
                    # commits the EPR state first (mirrors TeleportApp.get_memory).
                    # Measuring synchronously here can race that commit.
                    time_now = self.node.timeline.now()
                    process = Process(protocol, 'alice_bell_measurement', [reservation])
                    priority = self.node.timeline.schedule_counter
                    event = Event(time_now, process, priority)
                    self.node.timeline.schedule(event)
                    break
            else:
                # Only the RESPONDER receives the teleported state. On a multi-hop
                # path the intermediate (entanglement-swap) nodes also hold comm
                # memories mapped to this reservation and see them go ENTANGLED --
                # but they are NOT teleport endpoints. Without this guard an
                # intermediate would spin up a spurious Bob-side protocol and
                # consume/corrupt the swap's comm memory (telegate already ignores
                # non-endpoints the same way).
                if self.node.name != reservation.responder:
                    return
                # Bob side: create a protocol instance and stash comm memory
                protocol = TeleportationProtocol.create(owner=self.node, alice=False,
                                                        remote_node_name=reservation.initiator, protocol_type=TELEDATA)
                protocol.set_bob_comm_memory_name(getattr(info.memory, 'name', None))
                protocol.set_bob_comm_memory(info.memory)
                protocol.set_alice_comm_memory_name(getattr(info, 'remote_memo', None))
                self.teledata_protocols.append(protocol)

    def _move_received(self, src: str, msg: TeledataMessage):
        """Route teledata messages to the matching session protocol.

        Args:
            src (str): name of the node that sent the message.
            msg (TeledataMessage): the received teledata message.
        """
        log.logger.debug(f"{self.name} received_message from {src}: {msg}")

        if msg.msg_type is TeledataMsgType.MEASUREMENT_RESULT:
            # Bob receives measurement result from Alice
            for protocol in list(self.teledata_protocols):
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    protocol.received_message(src, msg)
                    # Send ACK back to Alice and close Bob-side protocol
                    protocol.bob_acknowledge_complete(msg.reservation)
                    self._early_expire(msg.reservation, getattr(protocol, "bob_comm_memory", None))
                    self.teledata_protocols.remove(protocol)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teledata protocol for msg from {src}")

        elif msg.msg_type is TeledataMsgType.ACK:
            # Alice receives acknowledgment from Bob → close Alice-side protocol
            for protocol in list(self.teledata_protocols):
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    self._early_expire(msg.reservation, getattr(protocol, "alice_comm_memory", None))
                    self.teledata_protocols.remove(protocol)
                    # Notify the initiator-side app that this teleport fully completed
                    # (Bob has the state). DQCApp uses this to release the barrier.
                    if self.on_source_complete is not None:
                        self.on_source_complete(protocol)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teledata protocol for ACK from {src}")

    def teledata_complete(self, data_key: int):
        """Called by TeledataProtocol (on Bob) once corrections are applied.

        Args:
            data_key (int): qstate key of Bob's data memory holding the teleported state.
        """
        full_state = self.node.timeline.quantum_manager.get(data_key).state
        try:
            if hasattr(full_state, "__len__") and len(full_state) > 2:
                half = len(full_state) // 2
                psi = full_state[:half]
            else:
                psi = full_state
        except Exception:
            psi = full_state

        log.logger.info(f"{self.name}: teledata done, state={psi}")
        self.data_keys.append(data_key)
        self.results.append(psi)

        if self.on_complete is not None:
            self.on_complete(data_key)

    # ── teleport (bare state-teleport) half ──────────────────────────────────
    def _teleport_get_memory(self, info: MemoryInfo):
        """Handle memory state changes.

        Args:
            info (MemoryInfo): Information about the memory state change.
        """
        log.logger.debug(f"{self.name}: get_memory, name={info.memory.name}, state={info.state}")
        # once we see our entangled half, hand it to the protocol
        if info.index in self.memo_to_reservation:
            if info.state == "ENTANGLED":
                for teleport_protocol in self.teleport_protocols:
                    this_node = info.memory.owner.name
                    remote_node = info.remote_node
                    if this_node == teleport_protocol.owner.name and remote_node == teleport_protocol.remote_node_name:
                        # this node is Alice
                        teleport_protocol.set_alice_comm_memory_name(info.memory.name)
                        teleport_protocol.set_alice_comm_memory(info.memory)
                        teleport_protocol.set_bob_comm_memory_name(info.remote_memo)
                        reservation = self.memo_to_reservation[info.index]
                        # Let Bob first execute EntanglementGenerationA._entanglement_succeed(), then let Alice do the Bell measurement
                        time_now = self.node.timeline.now()
                        process = Process(teleport_protocol, 'alice_bell_measurement', [reservation])
                        priority = self.node.timeline.schedule_counter
                        event = Event(time_now, process, priority)
                        self.node.timeline.schedule(event)
                        break  # if no matching protocol found, go to else clause
                else:
                    # this node is Bob, create the new teleport protocol instance, then append to self.teleport_protocols
                    teleport_protocol = TeleportationProtocol.create(self.node, alice=False, remote_node_name=info.remote_node, protocol_type=TELEPORT)
                    teleport_protocol.set_bob_comm_memory_name(info.memory.name)
                    teleport_protocol.set_bob_comm_memory(info.memory)
                    teleport_protocol.set_alice_comm_memory_name(info.remote_memo)
                    self.teleport_protocols.append(teleport_protocol)

    def _teleport_received(self, src: str, msg):
        """Handle incoming teleport messages.

        Args:
            src (str): Source node name.
            msg (TeleportMessage): The teleport message received.
        """
        log.logger.debug(f"{self.name} received_message from {src}: {msg}")
        if msg.msg_type is TeleportMsgType.MEASUREMENT_RESULT:  # Bob receives measurement result from Alice
            for teleport_protocol in self.teleport_protocols:   # find the correct teleport protocol on Bob's side
                if src == teleport_protocol.remote_node_name and msg.bob_comm_memory_name == teleport_protocol.bob_comm_memory_name:
                    teleport_protocol.received_message(src, msg)
                    self.node.resource_manager.expire_rules_by_reservation(msg.reservation)                    # early release of resources
                    self.node.resource_manager.update(None, teleport_protocol.bob_comm_memory, MemoryInfo.RAW)  # release the bob comm memory
                    teleport_protocol.bob_acknowledge_complete(msg.reservation)
                    self.teleport_protocols.remove(teleport_protocol)  # remove the protocol instance, it's lifecycle is complete
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teleport protocol for msg={msg} from {src}")

        elif msg.msg_type is TeleportMsgType.ACK:              # Alice receives acknowledgment from Bob
            for teleport_protocol in self.teleport_protocols:  # find the correct teleport protocol on Alice's side
                if src == teleport_protocol.remote_node_name and msg.bob_comm_memory_name == teleport_protocol.bob_comm_memory_name:
                    self.node.resource_manager.expire_rules_by_reservation(msg.reservation)                      # expire the rules
                    self.node.resource_manager.update(None, teleport_protocol.alice_comm_memory, MemoryInfo.RAW)  # release the alice comm memory
                    self.teleport_protocols.remove(teleport_protocol)  # remove the protocol instance, it's lifecycle is complete
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teleport protocol for msg={msg} from {src}")

    def teleport_complete(self, comm_key: int):
        """Called by TeleportProtocol once Bob's qubit is corrected. comm_key holds the teleported |ψ⟩.

        Args:
            comm_key (int): The key of the comm memory where the teleported state is stored.
        """
        my_qubit = self.node.timeline.quantum_manager.get(comm_key)
        psi = my_qubit.state  # get qubit state
        log.logger.info(f"{self.name}: teleport done, state={psi}")
        self.results.append((self.node.timeline.now(), psi))  # append result (timestamp, state)

    # ── shared plumbing (telegate versions -- supersets of the teledata ones) ─
    def remove_memo_reservation_map(self, index: int) -> None:
        """Idempotent override of the base-class removal (pop with a default so a
        redundant call can't ``KeyError``). Used by :meth:`_early_expire` for its
        own on-time cleanup. The scheduled ``end_time`` removal instead goes
        through the reservation-aware :meth:`remove_memo_reservation_map_for`.
        """
        self.memo_to_reservation.pop(index, None)

    def remove_memo_reservation_map_for(self, index: int, reservation) -> None:
        """Reservation-aware removal used for the base class's ``end_time`` event.

        Every telegate on a node reuses the same comm-memory index, so all of
        their ``end_time`` removals target the same slot. When windows overlap
        (small ``dt``), an OLD reservation's ``end_time`` can fire AFTER the NEXT
        telegate has already registered its own reservation on this index. The
        stock index-only removal would then pop the *current* mapping, orphaning
        the in-flight telegate: ``get_memory`` sees the index unmapped and drops
        the entangled pair, so the gate never completes -> controller deadlock.

        Guarding on identity makes the stale removal a true no-op: we only clear
        the slot if it is still mapped to the reservation whose window ended.
        """
        if self.memo_to_reservation.get(index) is reservation:
            self.memo_to_reservation.pop(index, None)

    def schedule_reservation(self, reservation) -> None:
        """Same add/remove scheduling as the base class, but the ``end_time``
        removal is reservation-aware (see :meth:`remove_memo_reservation_map_for`)
        so a stale removal cannot orphan a newer telegate that reuses the same
        comm-memory index. The ``start_time`` add is unchanged.
        """
        if reservation.initiator == self.node.name:
            self.path = reservation.path

        for card in self.node.network_manager.get_timecards():
            if reservation in card.reservations:
                self.node.timeline.schedule(Event(
                    reservation.start_time,
                    Process(self, "add_memo_reservation_map",
                            [card.memory_index, reservation])))
                self.node.timeline.schedule(Event(
                    reservation.end_time,
                    Process(self, "remove_memo_reservation_map_for",
                            [card.memory_index, reservation])))

    def _early_expire(self, reservation, comm_memory):
        """Release this session's resources the moment its gate/teleport completes.

        Expiring the RSVP rules created by ``reservation`` stops the network from
        generating further entanglement for a session that is already finished,
        and resetting the communication memory to ``RAW`` immediately returns it
        to the free pool for the next session.

        Because completion (not the clock) frees the resources, the reservation
        window no longer has to be sized to end right after the gate — it only
        needs to be wide enough for the gate to run.

        Args:
            reservation: Reservation whose rules should be expired.
            comm_memory: This node's communication memory used by the gate
                (Alice's or Bob's half), or ``None`` if unavailable.
        """

        now = self.node.timeline.now()
        end_t = getattr(reservation, "end_time", None)
        if end_t is not None:
            log.logger.info(f"[telegate:{self.node.name}] early expire @ t={now:,} "
                            f"(reservation end_time={end_t:,}, early_by={end_t - now:,})")

        # Expire this reservation's rules now. The end_time expiry scheduled in
        # generate_load_rules is NOT cancelled -- it still fires ~one window later --
        # but by then the rule is gone from the rule manager, so expire() sees a
        # missing rule and returns early without resetting memory. That guard is what
        # stops the stale expiry from resetting a comm memory a newer telegate has
        # recycled -> orphaned qubit -> wrong answer (bug #2).
        self.node.resource_manager.expire_rules_by_reservation(reservation)
        # Free the reservation's slot in the per-memory timecards so the scheduler
        # knows the memory is available again before the window's end_time.
        self.node.network_manager.remove_reservation_from_timecards(reservation)
        # Drop this reservation's memo_to_reservation entries now, in step with the
        # rules and timecards freed above. The late end_time removal still fires but
        # is a no-op (see remove_memo_reservation_map).
        stale = [idx for idx, res in self.memo_to_reservation.items()
                 if res is reservation]
        # Reset EVERY comm memory this reservation held to RAW -- not just the gate's
        # one. With memory_size > 1 (e.g. purification over a multi-hop path) the
        # reservation reserves several comm memories; the gate only consumes one, so
        # a second/purification-measured pair can be left entangled on an unused comm
        # memory. If not reset here it lingers, gets recycled by the next telegate,
        # and accumulates into a multi-qubit state -> BSM "Unknown state" later.
        mm = self.node.resource_manager.memory_manager
        for idx in stale:
            mem = mm.memory_array[idx]
            if mm.get_info_by_memory(mem).state != MemoryInfo.RAW:
                self.node.resource_manager.update(None, mem, MemoryInfo.RAW)
            self.remove_memo_reservation_map(idx)
        # The gate's comm memory (may not be in memo_to_reservation by now) too.
        if comm_memory is not None and mm.get_info_by_memory(comm_memory).state != MemoryInfo.RAW:
            self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)

        # If this node initiated the reservation, tell the intermediate (swap) nodes
        # on the path to expire early too. Both endpoints free themselves locally
        # (above); only the in-between nodes need the message. No-op for a direct
        # 2-node link (path length <= 2).
        if getattr(reservation, "initiator", None) == self.node.name:
            self.send_expire_rules_message(reservation)

    def send_expire_rules_message(self, reservation) -> None:
        """Notify the intermediate (swap) nodes on the reservation path to expire
        their rules early.

        Only the nodes strictly between the initiator and responder are messaged;
        the two endpoints free their own resources directly in :meth:`_early_expire`.
        Each intermediate node receives an ``EARLY_EXPIRE`` message (via
        :meth:`ResourceManager.expire_remote_rules`) and tears down its rules and
        timecards for this reservation. No-op for a direct link (path length <= 2).

        Args:
            reservation (Reservation): reservation whose intermediate-node rules
                should expire early.
        """
        path = getattr(reservation, "path", None) or []
        if len(path) > 2:
            for node_name in path[1:-1]:
                log.logger.info(f"[telegate:{self.node.name}] sending EARLY_EXPIRE to intermediate node {node_name}")
                self.node.resource_manager.expire_remote_rules(node_name, reservation)


class TeleportApp(TeleportationApp):
    """Thin single-kind app for standalone bare-teleport tests (pins ``TELEPORT``)."""

    DEFAULT_KIND = TELEPORT

    def __init__(self, node: DQCNode):
        """Initialize a bare state-teleport application.

        Args:
            node (DQCNode): the DQCNode that owns this app.
        """
        super().__init__(node)
        self.name = "teleport_app"

    def start(self, *args, **kwargs):
        """Initiate a bare state teleport (sender). See :meth:`TeleportationApp.start_teleport`."""
        return self.start_teleport(*args, **kwargs)


class TelegateApp(TeleportationApp):
    """Thin single-kind app for standalone telegate protocol tests (pins ``TELEGATE``)."""

    DEFAULT_KIND = TELEGATE

    def __init__(self, node: DQCNode):
        """Initialize a telegate-only application.

        Args:
            node (DQCNode): the DQCNode that owns this app.
        """
        super().__init__(node)
        self.name = "telegate_app"

    def start(self, *args, **kwargs):
        """Initiate a teleported gate (control owner). See :meth:`TeleportationApp.start_gate`."""
        return self.start_gate(*args, **kwargs)


class TeledataApp(TeleportationApp):
    """Thin single-kind app for standalone teledata protocol tests (pins ``TELEDATA``)."""

    DEFAULT_KIND = TELEDATA

    def __init__(self, node: DQCNode):
        """Initialize a teledata-only application.

        Args:
            node (DQCNode): the DQCNode that owns this app.
        """
        super().__init__(node)
        self.name = "teledata_app"

    def start(self, *args, **kwargs):
        """Initiate a qubit move (source owner). See :meth:`TeleportationApp.start_move`."""
        return self.start_move(*args, **kwargs)
