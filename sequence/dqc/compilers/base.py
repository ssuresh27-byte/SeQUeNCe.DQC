#!/usr/bin/env python3
"""Base class for COMPILERS -- circuit + topology -> CompiledProgram (offline).

Compilation is fully offline: the whole plan (placement + every step's ops) is
decided before execution; the controller then replays it. A compiler is the unit
that produces that plan. There are two shapes:

  * :class:`~compilers.basic.BasicCompiler` -- a PARTITIONER (placement) +
    a SCHEDULER (execution), the classic two-stage static path.
  * :class:`~compilers.fgp.FGPCompiler` -- the monolithic time-sliced telegate +
    teledata hybrid, which does placement and scheduling together (no swappable
    inner layers).
"""
from sequence.dqc.program import CompiledProgram


class CompilerBase:
    """A compiler turns a circuit + topology into a :class:`CompiledProgram`."""

    def compile(self, circuit, topology, seed: int = 0) -> CompiledProgram:
        raise NotImplementedError
