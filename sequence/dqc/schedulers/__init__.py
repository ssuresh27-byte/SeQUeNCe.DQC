#!/usr/bin/env python3
"""Schedulers -- decide WHEN each gate runs (assign every gate a controller step).

A scheduler is one half of a static COMPILER (the other half is a partitioner). It
takes a placement + circuit and produces the per-node op buckets (via
``compile_circuit``). Each strategy lives in its own module and subclasses
:class:`SchedulerBase`. The time-sliced telegate+teledata hybrid is NOT here -- it
does placement and scheduling together, so it is a monolithic compiler in the
:mod:`compilers` package (``FGPCompiler``).
"""
from sequence.dqc.schedulers.base import SchedulerBase
from sequence.dqc.schedulers.packed import PackedScheduler
from sequence.dqc.schedulers.serial import SerialScheduler
from sequence.dqc.schedulers.alap import ALAPScheduler
from sequence.dqc.schedulers.resource_aware import ResourceAwareScheduler

SCHEDULERS = {
    "packed": PackedScheduler,
    "serial": SerialScheduler,
    "alap": ALAPScheduler,
    "resource-aware": ResourceAwareScheduler,
}


def get_scheduler(name: str, n_qubits: int, node_names, memory_capacity: int = 8):
    """Instantiate the named scheduler."""
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{name}' (use {list(SCHEDULERS)})")
    return SCHEDULERS[name](n_qubits, len(node_names), memory_capacity, node_names=node_names)


__all__ = ["SchedulerBase", "PackedScheduler", "SerialScheduler", "ALAPScheduler",
           "ResourceAwareScheduler", "SCHEDULERS", "get_scheduler"]
