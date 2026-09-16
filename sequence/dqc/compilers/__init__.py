#!/usr/bin/env python3
"""Compilers -- circuit + topology -> CompiledProgram (the whole offline plan).

Two shapes of compiler:
  * a static BasicCompiler = a placement PARTITIONER + op-generation (the layering is a
    property of the op-DAG, replayed by the barrier -- not a compiler stage);
  * a MONOLITHIC compiler that does placement + move-generation together (the FGP
    time-sliced telegate+teledata hybrid).

``build_compiler(partitioner, scheduler)`` returns the right one: a monolithic compiler
when ``scheduler`` names one (e.g. ``"fgp"``, seeded by ``partitioner``), otherwise a static
``BasicCompiler(partitioner)``. For the static case the ``scheduler`` argument no longer
selects a layering -- the DAG's canonical ASAP layering is used -- so it is ignored beyond
picking the monolithic path.
"""
from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.compilers.basic import BasicCompiler
from sequence.dqc.compilers.fgp import FGPCompiler

# monolithic compilers: selectable via the "scheduler" slot (they subsume placement+moves)
MONOLITHIC = {"fgp": FGPCompiler}


def build_compiler(partitioner: str = "topo-aware", scheduler: str = "packed") -> CompilerBase:
    """Build a compiler. ``scheduler`` may name a monolithic compiler (e.g. "fgp"), in which
    case ``partitioner`` seeds it; otherwise a static ``BasicCompiler(partitioner)`` (the
    ``scheduler`` name is ignored for the static path -- the DAG owns the layering)."""
    if scheduler in MONOLITHIC:
        return MONOLITHIC[scheduler](seed_partitioner=partitioner)
    return BasicCompiler(partitioner=partitioner)


__all__ = ["CompilerBase", "BasicCompiler", "FGPCompiler", "MONOLITHIC", "build_compiler"]
