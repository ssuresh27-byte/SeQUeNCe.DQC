#!/usr/bin/env python3
"""Packed / ASAP scheduler: independent gates share a controller step."""
from collections import defaultdict

from sequence.dqc.schedulers.base import SchedulerBase


class PackedScheduler(SchedulerBase):
    """Packed / ASAP scheduler: independent gates (no shared-qubit dependency) are
    packed into the SAME controller step, so they run in parallel."""

    def compile_circuit(self, circuit):
        raw = getattr(circuit, "ops", getattr(circuit, "gates", []))
        layers = self._assign_layers(self._build_deps(raw), len(raw), raw)
        return self._bucketize(raw, layers)

    @staticmethod
    def _build_deps(raw_ops):
        deps, last = defaultdict(set), {}
        for i, op in enumerate(raw_ops):
            qs = op[1] if isinstance(op, (list, tuple)) else set()
            for q in qs:
                if q in last:
                    deps[last[q]].add(i)
                last[q] = i
        return deps

    @staticmethod
    def _assign_layers(deps, n, raw_ops):
        indeg = [0] * n
        for s in deps.values():
            for v in s:
                indeg[v] += 1
        ready = [i for i in range(n) if indeg[i] == 0]
        layer, free, cur = {}, {}, 0
        while ready:
            nxt, done = [], []
            for i in ready:
                _, qs, *_ = raw_ops[i] if isinstance(raw_ops[i], (list, tuple)) else (None, [])
                if all(free.get(q, 0) <= cur for q in qs):
                    layer[i] = cur
                    for q in qs:
                        free[q] = cur + 1
                    done.append(i)
                else:
                    nxt.append(i)
            if not done:
                cur += 1
            else:
                for i in done:
                    for v in deps.get(i, []):
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            nxt.append(v)
                ready = nxt
        return layer
