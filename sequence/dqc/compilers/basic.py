#!/usr/bin/env python3
"""BasicCompiler: the classic two-stage static compiler = partitioner + scheduler."""
from sequence.dqc.compilers.base import CompilerBase
from sequence.dqc.partitioners import get_partitioner
from sequence.dqc.schedulers import get_scheduler
from sequence.dqc.program import build_program


class BasicCompiler(CompilerBase):
    """Compose a placement PARTITIONER and a SCHEDULER into one compiler.

    ``compile`` = partition the qubits onto the topology, then schedule the circuit
    on that fixed placement, then assemble the CompiledProgram. Placement is static
    (unlike the FGP hybrid, which re-partitions per time slice).
    """

    def __init__(self, partitioner="topo-aware", scheduler="packed"):
        self.partitioner_name = partitioner
        self.scheduler_name = scheduler

    def compile(self, circuit, topology, seed: int = 0):
        n = circuit.size
        names = topology.node_names
        part = get_partitioner(self.partitioner_name, n, names)
        placement = part.partition(circuit, topology, seed=seed)

        sched = get_scheduler(self.scheduler_name, n, names)
        sched.qubit_to_node = dict(placement)
        sched.set_network(topology.edges)          # no-op unless routing-aware
        sched.set_capacities(topology.capacities)  # no-op for the static schedulers
        data_owners = sched.get_data_owners(placement)
        sched.data_owners = data_owners
        node_ops = sched.compile_circuit(circuit)
        return build_program(circuit, placement, data_owners, node_ops)
