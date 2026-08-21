#!/usr/bin/env python3
"""Partitioners -- decide WHERE each qubit lives (placement: qubit -> node).

A partitioner is one half of a static COMPILER (the other half is a scheduler; see
the :mod:`compilers` package). Each strategy lives in its own module and subclasses
:class:`PartitionerBase`. ``PARTITIONERS`` maps a name to the class;
``get_partitioner(name, ...)`` instantiates one.
"""
from sequence.dqc.partitioners.base import PartitionerBase
from sequence.dqc.partitioners.topo_aware import TopologyAwarePartitioner
from sequence.dqc.partitioners.random import RandomPartitioner
from sequence.dqc.partitioners.qap import QAPPartitioner

PARTITIONERS = {
    "topo-aware": TopologyAwarePartitioner,
    "random": RandomPartitioner,
    "qap": QAPPartitioner,
}


def get_partitioner(name: str, n_qubits: int, node_names):
    """Instantiate the named placement partitioner."""
    if name not in PARTITIONERS:
        raise ValueError(f"unknown partitioner '{name}' (use {list(PARTITIONERS)})")
    return PARTITIONERS[name](n_qubits, len(node_names), n_qubits, node_names=node_names)


__all__ = ["PartitionerBase", "TopologyAwarePartitioner", "RandomPartitioner",
           "QAPPartitioner", "PARTITIONERS", "get_partitioner"]
