# File: adaptive_central_node.py
"""An ADAPTIVE (dataflow) controller -- an alternative to the lock-step barrier.

Where :class:`barrier.BarrierController` replays a fixed plan one global
step at a time (broadcast step N to everyone, wait for ALL to ACK, then N+1), this
controller executes the SAME CompiledProgram as a **dataflow graph**: it dispatches
each step as soon as its data dependencies are satisfied, to only the nodes that
participate in it, and keeps MANY steps in flight at once. It reacts to completions
("hears back when things are done") rather than marching in lock-step -- so a fast
chain doesn't wait on a slow, unrelated one, and local work overlaps network work.

Dependencies come from qubit reuse: a step that touches qubit q depends on the most
recent earlier step that also touched q (the scheduler's layers are already a valid
topological order; this just recovers the finer per-qubit DAG).

Constraint: the DQCApp attributes a completing telegate to ``max(self._pending)``
(it was written for the barrier, where a node has at most one network step active
at a time). To keep that correct we never dispatch a network (telegate/teledata)
step to a node that already has a network step in flight -- but network steps on
DISJOINT nodes, and any number of local steps, run concurrently. (Lifting this
fully would need the DQCApp to tag each telegate completion with its step.)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

from sequence.message import Message
import sequence.utils.log as log

from sequence.dqc.controllers.base import BaseController


class AdaptiveController(BaseController):
    """Dataflow controller: dispatch each step when its deps clear; many in flight.

    Executes the same CompiledProgram as :class:`~barrier.BarrierController`
    but as a dependency DAG rather than a lock-step barrier. See
    :class:`~base.BaseController` for the constructor arguments and shared state.
    """

    # ── build the per-step dependency DAG (run after compile) ─────────────────
    def _on_compiled(self):
        node_ops = self.program.node_ops
        self.participants = defaultdict(set)   # step -> node names with an op there
        self.is_net = defaultdict(bool)        # step -> touches the network?
        step_qubits = defaultdict(set)         # step -> qubits it acts on
        for nm, grp in node_ops.items():
            for role, ops in grp.items():
                for op in ops:
                    s = op["layer"]
                    self.participants[s].add(nm)
                    if role == "move":
                        step_qubits[s].add(op["qubit"]); self.is_net[s] = True
                    else:
                        step_qubits[s].update(op["targets"])
                        if role in ("remote", "target"):
                            self.is_net[s] = True

        self.all_steps = set(self.participants)
        self.preds = {s: set() for s in self.all_steps}
        last = {}                              # qubit -> most recent step touching it
        for s in sorted(self.all_steps):
            for q in step_qubits.get(s, ()):
                if q in last:
                    self.preds[s].add(last[q])
            for q in step_qubits.get(s, ()):
                last[q] = s
        self.succ = defaultdict(set)
        for s, ps in self.preds.items():
            for p in ps:
                self.succ[p].add(s)
        self.remaining = {s: len(ps) for s, ps in self.preds.items()}

        self.done = set()
        self.dispatched = set()
        self.inflight = {}                     # step -> set(nodes) not yet ACKed
        self.net_busy = set()                  # nodes currently running a network step

    # ── run ──────────────────────────────────────────────────────────────────
    def _start(self):
        self._dispatch_ready()

    def _dispatch_ready(self):
        """Dispatch every step that is now ready (deps done + network free)."""
        changed = True
        while changed:
            changed = False
            for s in sorted(self.all_steps - self.dispatched):
                if self.remaining.get(s, 0) != 0:
                    continue
                if self.is_net[s] and (self.participants[s] & self.net_busy):
                    continue                   # a participant already runs a network step
                self._dispatch(s)
                changed = True

    def _dispatch(self, step: int):
        self.dispatched.add(step)
        self.inflight[step] = set(self.participants[step])
        if self.is_net[step]:
            self.net_busy |= self.participants[step]
        for nm in self.participants[step]:
            self.send_step(nm, step)           # over the classical channel
        log.logger.info("[adaptive] dispatch step=%d to %s (in flight: %d)",
                        step, sorted(self.participants[step]), len(self.inflight))
        self._progress()

    def receive_message(self, src: str, msg: Message) -> None:
        if not self._is_ack(msg):
            return
        s = msg.step
        if s not in self.inflight:
            return
        self.inflight[s].discard(msg.node)
        if self.inflight[s]:
            return
        # step s is complete
        del self.inflight[s]
        self.done.add(s)
        self.current = max(self.current, s + 1)
        if self.is_net[s]:
            self.net_busy -= self.participants[s]
        for t in self.succ.get(s, ()):
            self.remaining[t] -= 1
        log.logger.info("[adaptive] step=%d done (%d/%d complete)",
                        s, len(self.done), len(self.all_steps))
        if len(self.done) >= len(self.all_steps):
            self._completed = True
            self._progress(final=True)
        else:
            self._dispatch_ready()

    def _progress(self, final: bool = False):
        elapsed = time.time() - getattr(self, "_t_start", time.time())
        total = len(self.all_steps)
        pct = 100.0 * len(self.done) / total if total else 100.0
        sys.stderr.write(
            f"\r[adaptive] {len(self.done)}/{total} steps ({pct:5.1f}%)  "
            f"in-flight={len(self.inflight)}  elapsed {elapsed:6.1f}s  "
            f"sim_t={int(self.timeline.now()):,} ps   ")
        sys.stderr.flush()
        if final:
            sys.stderr.write("\n")
