# File: barrier.py
"""Barrier controller -- a real node in the network topology.

The controller is a :class:`~sequence.topology.node.ClassicalNode`: it exchanges
classical messages with the DQC nodes over classical channels (so it can only
reach nodes it is connected to), instead of calling their methods directly. It
owns the compiler + scheduler and, on start, RUNS them to produce the
:class:`~program.CompiledProgram`; then it drives the execution barrier --
broadcast a step to every node, wait for all ACKs (delivered over channels),
advance (charging ``dt`` to telegate/teledata steps, ``local_dt`` to local-only
steps), until the program's last step.

Shared plumbing (classical-node wiring, ``compile`` / ``set_nodes``, the DQC-app
messaging primitives) lives in :class:`~base.BaseController`; this class adds the
lock-step barrier policy.
"""
from __future__ import annotations

import sys
import time

from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.message import Message
import sequence.utils.log as log

from sequence.dqc.controllers.base import BaseController


class BarrierController(BaseController):
    """Barriered step orchestrator, wired into the topology as a classical node.

    Broadcast step N to every node, wait for ALL to ACK, then advance to N+1 --
    until the program's last step. See :class:`~base.BaseController` for the
    constructor arguments and shared state.
    """

    def __init__(self, name: str, timeline, compiler=None,
                 dt: float = 0.0, local_dt: float = None):
        super().__init__(name, timeline, compiler=compiler, dt=dt, local_dt=local_dt)
        self.waiting = set()                      # names of nodes not yet ACKed this step

    # ── barrier: receive ACKs over the channel, advance when all in ──────────
    def receive_message(self, src: str, msg: Message) -> None:
        if not self._is_ack(msg):
            return
        if msg.step != self.current:
            return
        self.waiting.discard(msg.node)
        log.logger.info("[controller] ACK from %s for step=%d; remaining=%d",
                        msg.node, msg.step, len(self.waiting))
        if not self.waiting:
            log.logger.info("[controller] ===== ALL ACKS RECEIVED FOR STEP %d =====", self.current)
            finished_step = self.current
            self.current += 1
            if self.current <= self.max_step:
                step_delay = self.dt if finished_step in self.net_layers else self.local_dt
                ev = Event(self.timeline.now() + step_delay,
                           Process(self, "_broadcast", [self.current]))
                self.timeline.schedule(ev)
            else:
                log.logger.info("[controller] All steps completed successfully!")
                self._completed = True

    def _start(self):
        """Kick off the run at step 0."""
        self._broadcast(0)

    def _broadcast(self, step: int):
        """Send ``step`` to every connected node (over its channel) and await ACKs."""
        if 0 <= step <= self.max_step:
            elapsed = time.time() - getattr(self, "_t_start", time.time())
            pct = 100.0 * step / self.max_step if self.max_step else 100.0
            sys.stderr.write(
                f"\r[grover] step {step}/{self.max_step} ({pct:5.1f}%)  "
                f"elapsed {elapsed:6.1f}s  sim_t={int(self.timeline.now()):,} ps   ")
            sys.stderr.flush()
            if step == self.max_step:
                sys.stderr.write("\n")

        if step > self.max_step or self._completed or self.current > self.max_step:
            return

        self.waiting = {nd.name for nd in self.qnodes}
        for nd in self.qnodes:
            self.send_step(nd.name, step)
        log.logger.info("[controller] Broadcast step=%d to %d nodes", step, len(self.qnodes))
