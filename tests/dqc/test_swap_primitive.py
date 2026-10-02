"""Full-capacity relocation must preserve logical state, including entanglement."""
import pytest
from sequence.components.circuit import Circuit
from sequence.dqc.architecture import make_star
from sequence.dqc.compilers.fgp import FGPCompiler
from sequence.dqc.program import build_program, stream_to_node_ops
from sequence.dqc.runtime import run


class StreamCompiler:
    def __init__(self, placement, stream):
        self.placement, self.stream = placement, stream

    def compile(self, circuit, topology, seed=0):
        buckets, owners = stream_to_node_ops(self.stream, topology.node_names,
                                            self.placement, topology.capacities)
        self.program = build_program(circuit, self.placement, owners, buckets)
        return self.program


@pytest.mark.parametrize('controller', ['barrier', 'adaptive'])
def test_full_swap_preserves_phase_and_entanglement(controller):
    # Bell pair + an unequal basis pair; exchange both pairs concurrently, twice.
    placement = {0: 'alice', 1: 'alice', 2: 'bob', 3: 'bob'}
    stream = [('gate', 'h', [0], None), ('gate', 'cx', [0, 1], None),
              ('gate', 'x', [2], None),
              ('swap', 0, 'alice', 2, 'bob'), ('swap', 1, 'alice', 3, 'bob'),
              ('gate', 'z', [0], None),
              ('swap', 0, 'bob', 2, 'alice'), ('swap', 1, 'bob', 3, 'alice'),
              ('gate', 'cx', [0, 1], None), ('gate', 'h', [0], None)]
    comp = StreamCompiler(placement, stream)
    result = run(Circuit(4), make_star(2, 2, comm_memo=8), compiler=comp,
                 controller=controller, expected=5)
    assert result['ok'], result
    assert result['moves'] == 8
    assert len([op for op in comp.program.dag.ops.values() if op.kind == 'swap']) == 4


@pytest.mark.parametrize('controller', ['barrier', 'adaptive'])
def test_fgp_emits_and_executes_swaps_at_full_capacity(controller):
    c = Circuit(4)
    c.x(0)
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]:
        c.cx(a, b)
    comp = FGPCompiler(move_margin=0)
    result = run(c, make_star(2, 2, comm_memo=8), compiler=comp,
                 controller=controller, expected=3)
    assert result['ok'], result
    assert result['moves'] > 0
    assert result['moves'] == comp.move_count


def test_swap_then_reuse_vacated_slot_keeps_dag_order():
    placement = {0: 'alice', 1: 'bob', 2: 'charlie'}
    stream = [('swap', 0, 'alice', 1, 'bob'), ('move', 1, 'charlie', 'alice'),
              ('move', 2, 'alice', 'charlie'), ('gate', 'x', [2], None)]
    buckets, owners = stream_to_node_ops(stream, list(['alice', 'bob', 'charlie']),
                                        placement, dict(alice=1, bob=1, charlie=2))
    program = build_program(Circuit(3), placement, owners, buckets)
    ops = list(program.dag.ops.values())
    assert [op.kind for op in ops] == ['swap', 'move', 'move', 'local']
    assert [op.layer for op in ops] == [0, 1, 2, 3]
    for i in range(1, 4):
        assert i - 1 in program.dag.preds[i]


@pytest.mark.parametrize('controller', ['barrier', 'adaptive'])
@pytest.mark.parametrize('hops', [1, 2])
def test_swap_with_minimum_data_memory_and_later_gate(controller, hops):
    # Two distant endpoints, one data slot each, no spare data slot anywhere.
    from sequence.dqc.architecture import DQCArchitecture
    names = ['alice', 'bob'] if hops == 1 else ['alice', 'relay', 'bob']
    topo = DQCArchitecture('swap_path', {name: 1 for name in names},
                           list(zip(names, names[1:])), comm_memo=2 if hops == 1 else 8)
    placement = {0: 'alice', 1: 'bob'}
    if hops == 2:
        placement[2] = 'relay'
    stream = [('gate', 'h', [0], None), ('gate', 'x', [1], None),
              ('swap', 0, 'alice', 1, 'bob'),
              ('gate', 'h', [0], None), ('gate', 'cx', [1, 0], None)]
    result = run(Circuit(len(names)), topo, compiler=StreamCompiler(placement, stream),
                 controller=controller, expected=3)
    assert result['ok'], result
    assert result['moves'] == 2
    assert result['max_hop'] == hops


@pytest.mark.parametrize('incoming_first', [True, False])
def test_swap_holds_comm_until_source_is_free_and_waits_for_ack(incoming_first):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from sequence.dqc.worker_program import TeleportationWorkerProgram
    worker = object.__new__(TeleportationWorkerProgram)
    worker.data_array = 'data'
    worker.node = SimpleNamespace(name='alice', get_generator=lambda: SimpleNamespace(random=lambda: 0.2),
        components={'data': SimpleNamespace(memories=[SimpleNamespace(qstate_key=42)])})
    worker.tl = SimpleNamespace(quantum_manager=Mock())
    worker.app = Mock()
    worker._ack_deferred_unit = Mock()
    worker._pending_deltas = {}
    worker._pending_swaps = {11: dict(step=2, slot=0, qubit=1, outgoing=10,
        source_ready=False, source_done=False, installed=False, comm_key=None)}
    worker._swap_outgoing = {10: 11}
    protocol = SimpleNamespace(identity=10)
    if incoming_first:
        assert worker._on_teleport_complete(99, 11)
        worker.app.release_teleport.assert_not_called()
        worker.tl.quantum_manager.run_circuit.assert_not_called()
        worker._on_teleport_source_ready(protocol)
    else:
        worker._on_teleport_source_ready(protocol)
        worker.app.release_teleport.assert_not_called()
        worker._on_teleport_complete(99, 11)
    worker.app.release_teleport.assert_called_once_with(11)
    worker._ack_deferred_unit.assert_not_called()
    worker._on_teleport_source_complete(protocol)
    worker._ack_deferred_unit.assert_called_once_with(2)
    assert worker._pending_deltas[2] == [dict(qubit=1, node='alice', slot=0, key=42)]
    assert not worker._pending_swaps and not worker._swap_outgoing


@pytest.mark.parametrize('controller', ['barrier', 'adaptive'])
def test_single_swap_density_matrix(controller):
    comp = StreamCompiler({0: 'alice', 1: 'bob'}, [
        ('gate', 'h', [0], None), ('gate', 'x', [1], None),
        ('swap', 0, 'alice', 1, 'bob'), ('gate', 'h', [0], None)])
    result = run(Circuit(2), make_star(2, 1, comm_memo=2), compiler=comp,
                 controller=controller, formalism='density', expected=2)
    assert result['ok'], result
