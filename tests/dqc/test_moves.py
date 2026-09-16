"""Teledata-move tests driven through the full run() path.

FGP avoids relocations at its default move_margin, so these force moves with
move_margin=0 (localize every interacting pair -> a teledata move between slices).
They exercise the controller-owned placement path: the controller resolves each op's
concrete slots from its registry, and the registry must stay consistent as qubits
relocate -- otherwise a post-move gate would read a stale slot and the answer would be
wrong. Guards phase 4d (concrete slots + move-delta registry updates).
"""
import pytest

from sequence.components.circuit import Circuit
from sequence.dqc.architecture import make_grid
from sequence.dqc.compilers.fgp import FGPCompiler
from sequence.dqc.runtime import run


def _chain(n):
    """X(0) then CX up the chain -> every qubit ends |1>, measured == 2**n - 1."""
    c = Circuit(n); c.x(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c


@pytest.mark.parametrize("rows,cols", [(2, 2), (2, 3)])
@pytest.mark.parametrize("controller", ["barrier", "adaptive"])
def test_forced_moves_are_correct(rows, cols, controller):
    """A chain compiled with move_margin=0 relocates qubits; the result must still be
    the all-ones bitstring, i.e. every controller-resolved (post-move) slot was correct."""
    n = rows * cols
    circ = _chain(n)
    expected = (1 << n) - 1
    comp = FGPCompiler(seed_partitioner="topo-aware", move_margin=0)
    res = run(circ, make_grid(rows, cols, 2), compiler=comp, controller=controller,
              data_qubits=range(n), expected=expected, seed=0)
    assert res["moves"] > 0, "move_margin=0 should force teledata moves"
    assert res["measured"] == expected, f"{rows}x{cols}/{controller}: got {res['measured']}"
    assert res["ok"], "run did not reach the final step or measured wrong"


def test_move_then_more_gates_stays_consistent():
    """After relocations, later gates on moved qubits must resolve to the new slots.

    X(0) + two CX sweeps up a 4-chain: sweep 1 -> |1111>, sweep 2 -> q=(1,0,1,0) i.e.
    measured 0b0101 = 5. Moves are logically transparent, so the value must be 5 even
    though qubits relocate between the sweeps."""
    n = 4
    c = Circuit(n); c.x(0)
    for _ in range(2):
        for i in range(n - 1):
            c.cx(i, i + 1)
    comp = FGPCompiler(seed_partitioner="topo-aware", move_margin=0)
    res = run(c, make_grid(2, 2, 2), compiler=comp, data_qubits=range(n), expected=5, seed=0)
    assert res["moves"] > 0
    assert res["measured"] == 5, f"post-move gates resolved to wrong slots: got {res['measured']}"
