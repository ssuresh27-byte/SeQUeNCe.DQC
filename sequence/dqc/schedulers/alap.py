#!/usr/bin/env python3
"""As-late-as-possible scheduler -- the dual of the packed/ASAP scheduler."""
from sequence.dqc.schedulers.base import SchedulerBase
from sequence.dqc.circuit_ops import gate_fields


class ALAPScheduler(SchedulerBase):
    """As-late-as-possible scheduler -- the dual of the packed/ASAP scheduler.

    Each gate is placed in the LATEST step that still precedes every gate that
    depends on it (same total depth as ASAP, but gates are delayed toward the
    end). A classic list-scheduling variant: it can shorten how long a data qubit
    sits entangled-but-idle before its final use, and gives a different telegate
    packing to compare against ASAP. One gate per qubit per step (dependencies
    preserved); parallelism is otherwise unconstrained.
    """

    def compile_circuit(self, circuit):
        raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
        n = len(raw)
        # forward ASAP pass to learn the total depth T (longest dependency chain)
        free, asap = {}, {}
        for i, op in enumerate(raw):
            _, qs, _ = gate_fields(op)
            L = max((free.get(q, 0) for q in qs), default=0)
            asap[i] = L
            for q in qs:
                free[q] = L + 1
        T = (max(asap.values()) + 1) if asap else 0
        # reverse pass: each gate as late as possible, just before its successors
        next_layer, layers = {}, {}
        for i in range(n - 1, -1, -1):
            _, qs, _ = gate_fields(raw[i])
            L = min((next_layer.get(q, T) for q in qs), default=T) - 1
            layers[i] = max(L, 0)
            for q in qs:
                next_layer[q] = layers[i]
        return self._bucketize(raw, layers)
