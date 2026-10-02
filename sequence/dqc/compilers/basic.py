#!/usr/bin/env python3
"""BasicCompiler: static placement + op-generation (the DAG/barrier does the layering)."""
from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.partitioners import get_partitioner
from sequence.dqc.schedulers.packed import PackedScheduler
from sequence.dqc.program import build_program


class BasicCompiler(CompilerBase):
    """Static compiler: a placement PARTITIONER + op-generation.

    ``compile`` = partition the qubits onto the topology (static, unlike the FGP hybrid which
    re-partitions per time slice), then generate the per-node op buckets (which gates are local
    vs. telegates) on that fixed placement. It does NOT schedule the execution: the layering
    (which ops run in which wave) is a property of the op-DAG (``build_program`` computes the
    canonical ASAP layering from the dependency graph), and the barrier replays that. The
    op-generator here is the packed bucketizer -- its own layering is superseded by the DAG's.
    """

    def __init__(self, partitioner="topo-aware"):
        self.partitioner_name = partitioner

    def compile(self, circuit, topology, seed: int = 0):
        n = circuit.size
        names = topology.node_names
        part = get_partitioner(self.partitioner_name, n, names)
        placement = part.partition(circuit, topology, seed=seed)

        gen = PackedScheduler(n, len(names), n, node_names=names)   # op-generation (bucketing)
        gen.qubit_to_node = dict(placement)
        data_owners = gen.get_data_owners(placement)
        gen.data_owners = data_owners
        node_ops = gen.compile_circuit(circuit)                     # layering re-derived by the DAG
        return build_program(circuit, placement, data_owners, node_ops)