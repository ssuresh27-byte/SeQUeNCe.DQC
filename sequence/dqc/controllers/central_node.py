# File: central_node.py
"""Central controller -- a real node in the network topology.

The controller is a :class:`~sequence.topology.node.ClassicalNode`: it exchanges
classical messages with the DQC nodes over classical channels (so it can only
reach nodes it is connected to), instead of calling their methods directly. It
owns the compiler + scheduler and, on start, RUNS them to produce the
:class:`~program.CompiledProgram`; then it drives the execution barrier --
broadcast a step to every node, wait for all ACKs (delivered over channels),
advance (charging ``dt`` to telegate/teledata steps, ``local_dt`` to local-only
steps), until the program's last step.
"""
from __future__ import annotations

import sys
import time

from sequence.topology.node import ClassicalNode
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.message import Message
import sequence.utils.log as log

from sequence.dqc.dqc_app import DQCMessage, DQCMsgType


class CentralNodeController(ClassicalNode):
    """Barriered step orchestrator, wired into the topology as a classical node.

    Args:
        name: controller node name (message sender label + channel key).
        timeline: the network timeline (from ``DQCNetTopo``).
        compiler: a compiler (``compilers`` package) -- circuit+topology -> program.
            Either a static PipelineCompiler (partitioner + scheduler) or the
            monolithic FGPCompiler. Run once at ``compile``.
        dt: delay after a network (telegate/teledata) step before the next broadcast.
        local_dt: delay after a local-only step (default ``dt``; ~0 so sim-time
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

        # run-time barrier state
        self.current = 0
        self.waiting = set()                      # names of nodes not yet ACKed this step
        self._completed = False

    # ── compile: run the compiler this controller owns ───────────────────────
    def compile(self, circuit, topology, seed: int = 0):
        """Compile ``circuit`` on ``topology`` into the full plan (CompiledProgram)."""
        program = self.compiler.compile(circuit, topology, seed=seed)
        self.program = program
        self.max_step = program.max_step
        self.net_layers = program.net_layers
        return program

    def set_nodes(self, qnodes):
        """The DQC node objects this controller drives (must be channel-connected)."""
        self.qnodes = list(qnodes)
        return self

    # ── barrier: receive ACKs over the channel, advance when all in ──────────
    def receive_message(self, src: str, msg: Message) -> None:
        if not isinstance(msg, DQCMessage) or msg.msg_type != DQCMsgType.ACK:
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

    def start_execution(self):
        """Kick off the run at step 0 (call after ``timeline.init()``)."""
        self._t_start = time.time()
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
            msg = DQCMessage(DQCMsgType.STEP_MESSAGE, receiver="dqc_node",
                             step=step, node=self.name)
            self.send_message(nd.name, msg)       # ClassicalNode: routes via channel
        log.logger.info("[controller] Broadcast step=%d to %d nodes", step, len(self.qnodes))
