#!/usr/bin/env python3
"""Compilers -- circuit + topology -> CompiledProgram (the whole offline plan).

Two shapes of compiler:
  * a static BasicCompiler = a placement PARTITIONER + a SCHEDULER;
  * a MONOLITHIC compiler that does placement + scheduling together (the FGP
    time-sliced telegate+teledata hybrid).

``build_compiler(partitioner, scheduler)`` returns the right one: a monolithic
compiler when ``scheduler`` names one (e.g. ``"fgp"``, seeded by ``partitioner``),
otherwise a ``BasicCompiler(partitioner, scheduler)``.
"""
from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.compilers.basic import BasicCompiler
from sequence.dqc.compilers.fgp import FGPCompiler

# monolithic compilers: selectable via the "scheduler" slot (they subsume scheduling)
MONOLITHIC = {"fgp": FGPCompiler}


def build_compiler(partitioner: str = "topo-aware", scheduler: str = "packed") -> CompilerBase:
    """Build a compiler. ``scheduler`` may name a monolithic compiler (e.g. "fgp"),
    in which case ``partitioner`` is used only to seed it; otherwise the two are
    composed into a static BasicCompiler."""
    if scheduler in MONOLITHIC:
        return MONOLITHIC[scheduler](seed_partitioner=partitioner)
    return BasicCompiler(partitioner=partitioner, scheduler=scheduler)


__all__ = ["CompilerBase", "BasicCompiler", "FGPCompiler", "MONOLITHIC", "build_compiler"]
