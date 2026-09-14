"""Multi-telegate execution tests for the QPU agent.

These exercise several telegates per run -- sequential (chain), concurrent
(independent pairs packed into one step), and fan-out (one control, many
targets) -- across both controllers and both static schedulers. They guard the
telegate-completion path, in particular the adaptive controller's per-node
``max(_pending)`` step attribution when multiple telegates are in flight.

All circuits use X (no H) so the measured bitstring is deterministic: every data
qubit ends in |1>, i.e. measured == 2**n - 1.
"""
import pytest

from sequence.components.circuit import Circuit
from sequence.dqc.architecture import make_star
from sequence.dqc.runtime import run


def _chain(n):
    """X(0), then CX(i, i+1) up the chain -> all qubits become 1."""
    c = Circuit(n)
    c.x(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c, n - 1                      # circuit, min telegate count


def _parallel_pairs(n):
    """Independent X(2k) + CX(2k, 2k+1) pairs -> packs into concurrent telegates."""
    assert n % 2 == 0
    c = Circuit(n)
    for k in range(0, n, 2):
        c.x(k)
        c.cx(k, k + 1)
    return c, n // 2


def _fanout(n):
    """X(0), then CX(0, j) fan-out from one control -> all targets become 1."""
    c = Circuit(n)
    c.x(0)
    for j in range(1, n):
        c.cx(0, j)
    return c, n - 1


CIRCUITS = {
    "chain4": _chain(4),
    "chain6": _chain(6),
    "parallel4": _parallel_pairs(4),
    "parallel6": _parallel_pairs(6),
    "fanout4": _fanout(4),
}


@pytest.mark.parametrize("name", list(CIRCUITS))
@pytest.mark.parametrize("scheduler", ["packed", "serial"])
@pytest.mark.parametrize("controller", ["barrier", "adaptive"])
def test_multi_telegate(name, scheduler, controller):
    circ, min_tele = CIRCUITS[name]
    n = circ.size
    expected = (1 << n) - 1
    res = run(circ, make_star(n, 1), partitioner="qap", scheduler=scheduler,
              controller=controller, expected=expected, seed=0)
    assert res["measured"] == expected, f"{name}/{scheduler}/{controller}: got {res['measured']}"
    assert res["reached"] >= res["steps"], "execution did not reach the final step"
    assert res["gates"] >= min_tele, f"expected >= {min_tele} telegates, got {res['gates']}"
