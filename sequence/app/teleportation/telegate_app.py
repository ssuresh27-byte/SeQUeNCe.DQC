"""Telegate (remote/distributed controlled-gate) protocol and app.

Implements an entanglement-assisted distributed controlled gate across two
nodes. The gate is selectable via ``gate_type`` ("cx" for a remote CNOT, the
default, or "cz" for a remote CZ):

1. Alice: CNOT(control -> comm_A), measure(comm_A) -> a_bit, send a_bit to Bob.
2. Bob: <gate>(comm_B -> target), H(comm_B), measure(comm_B) -> b_bit, apply the
   a-dependent correction (X^a for CX, Z^a for CZ).
3. Bob: send ACK(a_bit, b_bit) to Alice.
4. Alice: apply Z^b correction to the control qubit, complete.

Both gates share the same cat-entangler (step 1) and cat-disentangler (the Z^b
correction in step 4). Only Bob's local gate and his a-dependent target
correction differ.

Ported from the DQC ``telegate`` package, re-expressed as a
:class:`TeleportationProtocol` registered under ``TELEGATE`` and grouped with its
:class:`TelegateApp`. Mirrors the teleport protocol/app pair.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

from ...components.circuit import Circuit
from ...message import Message
from ...utils import log
from ...constants import TELEGATE
from ...network_management.reservation import Reservation
from ...topology.node import DQCNode

from .teleportation_base import TeleportationProtocol


class TelegateMsgType(Enum):
    """Message types for telegate protocol communication.

    Attributes:
        A_MEAS_RESULT: Alice to Bob - sends measurement result 'a' from comm qubit
        ACK: Bob to Alice - acknowledges completion with both measurement results
    """
    A_MEAS_RESULT = auto()  # Alice to Bob: send bit 'a' (result of measuring comm_A)
    ACK = auto()  # Bob to Alice: acknowledge completion


class TelegateMessage(Message):
    """Message payloads for telegate protocol communication.

    A_MEAS_RESULT message contains:
        - reservation: Network reservation for the telegate operation
        - bob_comm_memory_name: Name of Bob's communication memory
        - a_bit: Alice's measurement of her comm qubit (0/1)
        - target_memory_index: Index of Bob's target memory
        - gate_type: Distributed gate to apply on Bob's side ("cx" or "cz")

    ACK message contains:
        - reservation: Network reservation for the telegate operation
        - bob_comm_memory_name: Name of Bob's communication memory
        - target_key: Bob's target qubit key
        - a_bit: Alice's measurement result
        - b_bit: Bob's measurement result
    """

    def __init__(self, msg_type: TelegateMsgType, **kwargs):
        super().__init__(msg_type, 'telegate_app')

        if msg_type is TelegateMsgType.A_MEAS_RESULT:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.a_bit: int = kwargs['a_bit']
            self.target_memory_index: int = kwargs.get('target_memory_index', 0)
            self.gate_type: str = kwargs.get('gate_type', 'cx')
            self.string = (f'type={TelegateMsgType.A_MEAS_RESULT}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'a_bit={self.a_bit}, target_idx={self.target_memory_index}, '
                           f'gate_type={self.gate_type}, reservation={self.reservation}')
        elif msg_type is TelegateMsgType.ACK:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.target_key: Optional[int] = kwargs.get('target_key', None)
            self.a_bit: Optional[int] = kwargs.get('a_bit', None)
            self.b_bit: Optional[int] = kwargs.get('b_bit', None)
            self.string = (f'type={TelegateMsgType.ACK}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'target_key={self.target_key}, a_bit={self.a_bit}, '
                           f'b_bit={self.b_bit}, reservation={self.reservation}')
        else:
            raise Exception(f"TelegateMessage created unknown type of message: {msg_type}")

    def __str__(self):
        return self.string


@TeleportationProtocol.register(TELEGATE)
class TelegateProtocol(TeleportationProtocol):
    """Per-session protocol for remote CNOT operations.

    Manages a single telegate operation between Alice (control qubit owner) and
    Bob (target qubit owner): quantum operations, measurements, corrections, and
    completion signaling.

    Attributes:
        owner: The DQCNode that owns this protocol instance
        alice: True if this is Alice's protocol instance, False for Bob's
        remote_node_name: Name of the remote node in this telegate operation
        control_memory_index: Index of Alice's control qubit memory
        target_memory_index: Index of Bob's target qubit memory
        alice_comm_memory_name / alice_comm_memory: Alice's communication memory
        bob_comm_memory_name / bob_comm_memory: Bob's communication memory
    """

    # Alice: CNOT (data_control → comm_A) + measure(comm_A)
    _cnot = Circuit(2)
    _cnot.cx(0, 1)

    # Bob: local controlled gate (comm_B → target). CNOT realizes a remote CX;
    # CZ realizes a remote CZ. Both share the same cat-entangler/disentangler.
    _cz = Circuit(2)
    _cz.cz(0, 1)

    _hadamard = Circuit(1)
    _hadamard.h(0)

    # Bob: Measure comm qubit
    _z_measure = Circuit(1)
    _z_measure.measure(0)

    # Alice: Z correction on control qubit (if b_bit == 1)
    _z = Circuit(1)
    _z.z(0)

    # Bob: X correction on target qubit (if a_bit == 1)
    _x = Circuit(1)
    _x.x(0)

    def __init__(self, owner: DQCNode, alice: bool = False, control_memory_index: Optional[int] = None,
                 target_memory_index: Optional[int] = None, remote_node_name: Optional[str] = None,
                 gate_type: str = "cx"):
        """Initialize a telegate protocol instance.

        Args:
            owner: The DQCNode that owns this protocol instance
            alice: True if this is Alice's protocol instance, False for Bob's
            control_memory_index: Index of Alice's control qubit memory
            target_memory_index: Index of Bob's target qubit memory
            remote_node_name: Name of the remote node in this telegate operation
            gate_type: Distributed gate to apply, "cx" (remote CNOT) or "cz"
                (remote CZ). Only meaningful on Alice's side; Bob learns it from
                the A_MEAS_RESULT message.
        """
        gate_type = gate_type.lower()
        if gate_type not in ("cx", "cz"):
            raise ValueError(f"gate_type must be 'cx' or 'cz', got {gate_type!r}")
        self.gate_type = gate_type
        if alice:
            name = (f"{owner.name}.telegate.{remote_node_name}."
                    f"ControlData[{control_memory_index}]toTargetData[{target_memory_index}]")
        else:
            name = f"{owner.name}.telegate.{remote_node_name}"
        super().__init__(owner, name, TELEGATE, alice=alice, remote_node_name=remote_node_name)

        # Data indices are meaningful on their respective sides
        self.control_memory_index = control_memory_index  # Alice side index
        self.target_memory_index = target_memory_index  # Bob side index

    def alice_stage(self, reservation: Reservation):
        """Execute Alice's stage of the telegate protocol.

        Alice performs:
        1. CNOT(control → comm_A)
        2. Measure comm_A to get a_bit
        3. Send a_bit to Bob

        Args:
            reservation: Network reservation for the telegate operation
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_stage called")
        assert self.alice_comm_memory is not None, "Alice comm memory not set"
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        data_key = data_arr[self.control_memory_index].qstate_key  # type: ignore[index]
        comm_key = self.alice_comm_memory.qstate_key

        # Step 1: CNOT(control → comm)
        control_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        comm_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State before CNOT - control qubit: {control_state_before}")
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State before CNOT - comm qubit: {comm_state_before}")

        # 1. Apply CNOT without measurement first to log intermediate state
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying CNOT gate: control qubit (key={data_key}) → communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._cnot, [data_key, comm_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] CNOT gate applied successfully")

        # Log quantum state after CNOT but before measurement
        control_state_after_cnot = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        comm_state_after_cnot = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT (before measurement) - control qubit: {control_state_after_cnot}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT (before measurement) - comm qubit: {comm_state_after_cnot}")

        # 2. Now apply measurement
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z_measure, [comm_key], rnd)
        a_bit = meas[comm_key]
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Communication qubit measurement result: a_bit={a_bit}")

        # Log quantum state after measurement
        control_state_after_measure = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State after measurement - control qubit: {control_state_after_measure}")

        # 3. Send measurement result to Bob
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Sending A_MEAS_RESULT to {self.remote_node_name}")
        msg = TelegateMessage(TelegateMsgType.A_MEAS_RESULT, bob_comm_memory_name=self.bob_comm_memory_name,
                              a_bit=a_bit, target_memory_index=self.target_memory_index,
                              gate_type=self.gate_type, reservation=reservation)
        self.owner.send_message(self.remote_node_name, msg)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] A_MEAS_RESULT sent successfully")

    def received_message(self, src: str, msg: TelegateMessage):
        """Handle incoming messages for Bob's side of the protocol.

        Args:
            src: Source node name
            msg: Received telegate message
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] received_message from {src}, msg_type={msg.msg_type}")
        if msg.msg_type is TelegateMsgType.A_MEAS_RESULT:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Processing A_MEAS_RESULT from {src}")
            self._bob_stage_with_a(msg)
        else:
            log.logger.warning(f"{self.name}: received unknown message type {msg.type} from {src}")

    def _bob_stage_with_a(self, msg: TelegateMessage):
        """Execute Bob's stage of the telegate protocol.

        Bob performs:
        1. Local controlled gate (comm_B → target): CNOT for a remote CX, CZ
           for a remote CZ.
        2. H(comm_B)
        3. Measure comm_B to get b_bit
        4. Apply the a-dependent correction to the target qubit: X^a for CX,
           Z^a for CZ.
        5. Send ACK(a_bit, b_bit) to Alice
        6. Complete protocol

        Args:
            msg: Alice's measurement result message
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] _bob_stage_with_a called")
        assert self.bob_comm_memory is not None, "Bob comm memory not set"

        # Alice picks the distributed gate; Bob adopts it for this session. The
        # cat-entangler/disentangler are identical for CX and CZ -- only Bob's
        # local gate and his a-dependent target correction differ.
        gate_type = getattr(msg, "gate_type", "cx")
        self.gate_type = gate_type

        # Fetch target data key (on Bob side)
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        t_idx = msg.target_memory_index if hasattr(msg, 'target_memory_index') else (self.target_memory_index if self.target_memory_index is not None else 0)
        target_key = data_arr[t_idx].qstate_key  # type: ignore[index]
        comm_key = self.bob_comm_memory.qstate_key

        comm_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before CNOT - comm qubit: {comm_state_before}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before CNOT - target qubit: {target_state_before}")

        # 1. Apply the local controlled gate (CNOT for remote CX, CZ for remote CZ)
        local_gate = TelegateProtocol._cz if gate_type == "cz" else TelegateProtocol._cnot
        gate_name = gate_type.upper()
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying {gate_name} gate: comm qubit (key={comm_key}) → target qubit (key={target_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(local_gate, [comm_key, target_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] {gate_name} gate applied successfully")

        # Log quantum state after CNOT
        comm_state_after = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_after = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT - comm qubit: {comm_state_after}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT - target qubit: {target_state_after}")

        # 2. Apply Hadamard gate on comm qubit
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying Hadamard gate to communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._hadamard, [comm_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Hadamard gate applied successfully")

        comm_state_after_h = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_after_h = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after Hadamard - comm qubit: {comm_state_after_h}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after Hadamard - target qubit: {target_state_after_h}")

        # 3. Measure comm qubit
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Measuring communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z_measure, [comm_key], rnd)
        b_bit = meas[comm_key]
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Communication qubit measurement result: b_bit={b_bit}")

        # Step 4: Apply the a-dependent correction on the target (uses Alice's
        # bit). A remote CX corrects with X^a; a remote CZ corrects with Z^a.
        corr_circuit = TelegateProtocol._z if gate_type == "cz" else TelegateProtocol._x
        corr_name = "Z" if gate_type == "cz" else "X"
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Alice's measurement result: a_bit={msg.a_bit}")
        if msg.a_bit:
            target_state_before_corr = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before {corr_name} correction - target qubit: {target_state_before_corr}")

            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying {corr_name} correction to target qubit (key={target_key}) based on Alice's measurement")
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(corr_circuit, [target_key], rnd)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] {corr_name} correction applied successfully")

            target_state_after_corr = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] State after {corr_name} correction - target qubit: {target_state_after_corr}")
        else:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] No {corr_name} correction needed (Alice's measurement was 0)")
            target_state_no_corr = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] State after measurement (no {corr_name} correction) - target qubit: {target_state_no_corr}")

        # Send ACK to Alice (with both bits)
        log.logger.warning(f"[telegate_protocol:{self.owner.name}] About to send ACK: a_bit={msg.a_bit}, b_bit={b_bit}, target_key={target_key}")

        self.bob_acknowledge_complete(msg.reservation, a_bit=msg.a_bit, b_bit=b_bit, target_key=target_key)

        # Inform the app that the distributed CNOT has been effected
        role = "alice" if self.alice else "bob"
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Calling telegate_complete with target_key={target_key}, role={role}")
        self.owner.app.telegate_complete(target_key, role)

        # Remove this protocol from the list after completion
        if self in self.owner.app.telegate_protocols:
            self.owner.app.telegate_protocols.remove(self)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Protocol removed from list after completion")

        log.logger.info(f"[telegate_protocol:{self.owner.name}] _bob_stage_with_a completed successfully")

    def alice_z_correction(self, b_bit: int):
        """Apply Alice's Z correction based on Bob's measurement result.

        Called when Alice receives the ACK from Bob: applies the Z^b correction
        to Alice's control qubit and completes the protocol.

        Args:
            b_bit: Bob's measurement result (0 or 1)
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_z_correction called with b_bit={b_bit}")

        # Get the control qubit key
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        ctrl_key = data_arr[self.control_memory_index].qstate_key

        if b_bit:
            control_state_before_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before Z correction - control qubit: {control_state_before_z}")

            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying Z gate to control qubit (key={ctrl_key}) based on Bob's measurement")
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z, [ctrl_key], rnd)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] Z gate applied successfully")

            control_state_after_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.warning(f"[telegate_protocol:{self.owner.name}] State after Z correction - control qubit: {control_state_after_z}")
        else:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] No Z correction needed (Bob's measurement was 0)")
            control_state_no_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.warning(f"[telegate_protocol:{self.owner.name}] State after ACK (no Z correction) - control qubit: {control_state_no_z}")

        # Complete as alice (initiator/control owner)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Calling telegate_complete with ctrl_key={ctrl_key}, role=alice")
        self.owner.app.telegate_complete(ctrl_key, role="alice")

        # Remove this protocol from the list after completion
        if self in self.owner.app.telegate_protocols:
            self.owner.app.telegate_protocols.remove(self)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Protocol removed from list after completion")

        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_z_correction completed successfully")

    def bob_acknowledge_complete(self, reservation: Reservation, a_bit: int, b_bit: int, target_key: int):
        """Send acknowledgment message to Alice with measurement results.

        Args:
            reservation: Network reservation for the telegate operation
            a_bit: Alice's measurement result
            b_bit: Bob's measurement result
            target_key: Bob's target qubit key
        """
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] bob_acknowledge_complete called: a_bit={a_bit}, b_bit={b_bit}, target_key={target_key}")

        msg = TelegateMessage(TelegateMsgType.ACK, bob_comm_memory_name=self.bob_comm_memory_name, reservation=reservation,
                              a_bit=a_bit, b_bit=b_bit, target_key=target_key)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] Sending ACK to {self.remote_node_name}")
        self.owner.send_message(self.remote_node_name, msg)

    def pretty_ket(self, vec, precision=4, tol=1e-10):
        """Convert a state vector into a pretty-printed ket string."""
        # These dumps are only ever fed to INFO/DEBUG log lines, but the calls are
        # evaluated eagerly (as f-string args), so without this guard they format
        # the whole 2^k-amplitude state on every telegate even at WARNING level.
        if not log.logger.isEnabledFor(logging.INFO):
            return "<state dump suppressed; enable INFO logging>"
        v = np.asarray(vec, dtype=complex).flatten()
        terms = []
        for k, a in enumerate(v):
            if abs(a) < tol:
                continue
            re = round(a.real, precision)
            im = round(a.imag, precision)
            # avoid "-0.0"
            if abs(re) < 10**(-precision): re = 0.0
            if abs(im) < 10**(-precision): im = 0.0

            if im == 0:
                coef = f"{re:.{precision}f}"
            elif re == 0:
                coef = f"{im:.{precision}f}i"
            else:
                sign = "+" if im > 0 else "-"
                coef = f"{re:.{precision}f} {sign} {abs(im):.{precision}f}i"
            terms.append(f"({coef}) |{k}⟩")
        return " + ".join(terms) if terms else "0"
