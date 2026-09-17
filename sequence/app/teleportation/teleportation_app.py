#!/usr/bin/env python3
"""TeleportationApp -- one node-attached app that runs BOTH teleported gates and qubit moves.

A :class:`~sequence.topology.node.DQCNode` is a :class:`QuantumRouter`, so it has a SINGLE
``self.app`` callback slot (the network/resource manager's reservation + memory callbacks all
forward to it). Running a telegate *and* a teledata on one node used to need two app objects behind
a router; instead this app subsumes both. It inherits :class:`TelegateApp` and :class:`TeledataApp`
(via cooperative ``super().__init__`` their state is all initialised on one instance) and adds only
a thin kind-dispatch layer:

* each session's KIND (telegate vs teledata) is recorded by its reservation ``identity`` -- a unique
  id the controller assigns and delivers to BOTH endpoints in the step op, so each side calls
  :meth:`expect_session` before the reservation travels;
* :meth:`get_memory` sends an arriving entangled pair to the telegate or teledata half by that kind;
* :meth:`received_message` dispatches by message type.

Everything else (reservation scheduling, early-expire cleanup, per-session multiplexing) is the
existing, tested TelegateApp/TeledataApp code, reused unchanged. TelegateApp/TeledataApp remain as
standalone classes for the isolated protocol tests; this class is what the DQC worker installs.
"""
from __future__ import annotations

from ...constants import TELEGATE, TELEDATA
from .telegate_app import TelegateApp
from .teledata_app import TeledataApp, TeledataMessage


class TeleportationApp(TelegateApp, TeledataApp):
    """Unified telegate + teledata app occupying a DQCNode's single ``self.app`` slot."""

    def __init__(self, node):
        # Cooperative MI: TelegateApp.__init__ calls super().__init__, which (in this class's MRO:
        # TeleportationApp -> TelegateApp -> TeledataApp -> RequestApp -> App) runs TeledataApp's
        # init and then RequestApp/App -- so the one instance ends up with BOTH telegate and
        # teledata state, and App.__init__ registers it as node.app. (A plain TelegateApp has
        # TeledataApp absent from its MRO, so this same code does NOT pull in teledata there.)
        TelegateApp.__init__(self, node)
        self.name = "teleport"
        node.set_app(self)                       # the single app-callback slot
        self._kind_by_identity: dict = {}        # reservation identity -> TELEGATE | TELEDATA

    def expect_session(self, identity: int, kind: str) -> None:
        """Record session ``identity``'s KIND so :meth:`get_memory` routes its pair to the right
        half. Called on BOTH endpoints (from the controller's op) before the reservation travels."""
        self._kind_by_identity[int(identity)] = kind

    def start_gate(self, *args, **kwargs):
        """Initiate a teleported gate (control owner). See :meth:`TelegateApp.start`."""
        return TelegateApp.start(self, *args, **kwargs)

    def start_move(self, *args, **kwargs):
        """Initiate a qubit move (source owner). See :meth:`TeledataApp.start`."""
        return TeledataApp.start(self, *args, **kwargs)

    def get_memory(self, info) -> None:
        """Route an entangled memory to the telegate or teledata half by the session's kind."""
        reservation = self.memo_to_reservation.get(info.index)
        identity = getattr(reservation, "identity", None) if reservation is not None else None
        kind = self._kind_by_identity.get(identity, TELEGATE)
        handler = TeledataApp.get_memory if kind == TELEDATA else TelegateApp.get_memory
        handler(self, info)

    def received_message(self, src: str, msg) -> None:
        """Dispatch a direct app-to-app message to the half that owns its protocol."""
        handler = TeledataApp.received_message if isinstance(msg, TeledataMessage) else TelegateApp.received_message
        handler(self, src, msg)
