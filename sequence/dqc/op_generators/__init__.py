"""Op-generation -- group a placed circuit into per-node local/remote ops.

The static :class:`~sequence.dqc.compilers.basic.BasicCompiler` uses
:class:`PackedOpGenerator` purely as an op-generator (``compile_circuit`` -> per-node
operations). It does NOT decide the execution layering: that is a property of the op-DAG
(``program.build_op_dag`` computes the canonical ASAP layering, which the barrier replays).
The time-sliced telegate+teledata hybrid does placement and move-generation together, so it
is a monolithic compiler in the :mod:`compilers` package (``FGPCompiler``).
"""
from .base import OpGeneratorBase
from .packed import PackedOpGenerator

__all__ = ["OpGeneratorBase", "PackedOpGenerator"]
