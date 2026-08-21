#!/usr/bin/env python3
"""Compilers -- circuit + topology -> CompiledProgram (the whole offline plan).

Two shapes of compiler:
  * a static PipelineCompiler = a placement PARTITIONER + a SCHEDULER;
  * a MONOLITHIC compiler that does placement + scheduling together (the FGP
    time-sliced telegate+teledata hybrid).

``build_compiler(partitioner, scheduler)`` returns the right one: a monolithic
compiler when ``scheduler`` names one (e.g. ``"fgp"``, seeded by ``partitioner``),
otherwise a ``PipelineCompiler(partitioner, scheduler)``.
"""
from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.compilers.pipeline import PipelineCompiler
from sequence.dqc.compilers.fgp import FGPCompiler

# monolithic compilers: selectable via the "scheduler" slot (they subsume scheduling)
MONOLITHIC = {"fgp": FGPCompiler}


def build_compiler(partitioner: str = "topo-aware", scheduler: str = "packed") -> CompilerBase:
    """Build a compiler. ``scheduler`` may name a monolithic compiler (e.g. "fgp"),
    in which case ``partitioner`` is used only to seed it; otherwise the two are
    composed into a static PipelineCompiler."""
    if scheduler in MONOLITHIC:
        return MONOLITHIC[scheduler](seed_partitioner=partitioner)
    return PipelineCompiler(partitioner=partitioner, scheduler=scheduler)


__all__ = ["CompilerBase", "PipelineCompiler", "FGPCompiler", "MONOLITHIC", "build_compiler"]
