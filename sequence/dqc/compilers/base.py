#!/usr/bin/env python3
"""Base class for COMPILERS -- circuit + topology -> CompiledProgram (offline).

Compilation is fully offline: the whole plan (placement + the ops + their dependency
DAG) is decided before execution; the controller then replays it. The compiler decides
PLACEMENT and which ops exist (local vs telegate vs move) -- it does NOT decide the
execution layering: that is the op-DAG's canonical ASAP layering (computed in
``program.build_program``), which the barrier replays as waves and the adaptive path
ignores. Two shapes:

  * :class:`~compilers.basic.BasicCompiler` -- a PARTITIONER (static placement) +
    op-generation.
  * :class:`~compilers.fgp.FGPCompiler` -- the monolithic time-sliced telegate +
    teledata hybrid, which does placement and per-slice move-generation together.
"""
from sequence.dqc.program import CompiledProgram


class CompilerBase:
    """A compiler turns a circuit + topology into a :class:`CompiledProgram`."""

    def compile(self, circuit, topology, seed: int = 0) -> CompiledProgram:
        raise NotImplementedError
