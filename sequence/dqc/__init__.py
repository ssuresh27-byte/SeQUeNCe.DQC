"""Distributed Quantum Computing (DQC) support for SeQUeNCe.

Physical DQC topologies (built on :mod:`sequence.utils.graphs`) that expand into the
``DQCNetTopo`` config the simulator consumes, with per-node local (computational)
noise carried inline on each node. Per-node noise is applied natively by the
ket-vector quantum manager (trajectory noise); see
:class:`sequence.kernel.quantum_manager.ket_vector.QuantumManagerKet`.
"""
from .architecture import (
    DQCArchitecture,
    node_names,
    make_star,
    make_grid,
    make_caveman,
)

__all__ = [
    "DQCArchitecture",
    "node_names",
    "make_star",
    "make_grid",
    "make_caveman",
]
