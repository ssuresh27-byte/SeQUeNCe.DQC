#!/usr/bin/env python3
"""Serial scheduler: exactly one gate per controller step (no parallelism)."""
from sequence.dqc.schedulers.base import SchedulerBase


class SerialScheduler(SchedulerBase):
    """Serial scheduler: exactly one gate per controller step (no parallelism)."""

    def compile_circuit(self, circuit):
        raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
        return self._bucketize(raw, {i: i for i in range(len(raw))})
