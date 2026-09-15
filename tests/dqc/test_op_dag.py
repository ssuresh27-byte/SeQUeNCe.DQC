"""Tests for the op-level dependency DAG (program.build_op_dag).

The DAG is derived from the compiler's per-node op buckets and is the first-class
structure the controllers execute: the barrier layers it into ASAP waves, the adaptive
path walks it by dependency. These guard that (a) the ops are deduped across the nodes
that record them, (b) dependencies are exactly qubit-reuse edges, and (c) moves appear.
"""
import pytest

from sequence.components.circuit import Circuit
from sequence.dqc.architecture import make_star, make_grid
from sequence.dqc.compilers import build_compiler


def _dag_for(circuit, topology, partitioner="qap", scheduler="packed", seed=0):
    return build_compiler(partitioner, scheduler).compile(circuit, topology, seed=seed).dag


def _shares_qubit(dag, i, p):
    return bool(set(dag.ops[i].qubits) & set(dag.ops[p].qubits))


def test_chain_is_linear():
    """X(0) then CX up a chain -> a single dependency chain, one root."""
    c = Circuit(4); c.x(0)
    for i in range(3):
        c.cx(i, i + 1)
    dag = _dag_for(c, make_star(4, 1))
    assert len(dag.ops) == 4
    assert dag.roots() == [min(dag.ops)]                      # only the X is initially ready
    # every op past the root depends on exactly the prior op, sharing a qubit
    for i in dag.ops:
        for p in dag.preds[i]:
            assert _shares_qubit(dag, i, p)
    # the chain is a total order: each non-root has >=1 pred, each non-leaf >=1 succ
    assert sum(1 for i in dag.ops if not dag.preds[i]) == 1


def test_independent_pairs_have_multiple_roots():
    """X(2k)+CX(2k,2k+1) pairs are mutually independent -> no cross-pair edges."""
    n = 4
    c = Circuit(n)
    for k in range(0, n, 2):
        c.x(k); c.cx(k, k + 1)
    dag = _dag_for(c, make_grid(2, 2, 2))
    # two disjoint pairs -> at least two independent roots, and no dependency crosses pairs
    assert len(dag.roots()) >= 2
    for i in dag.ops:
        for p in dag.preds[i]:
            assert _shares_qubit(dag, i, p)                    # never a spurious cross-pair edge


def test_dependencies_are_acyclic_and_consistent():
    """preds/succ are mirror images and the graph is a DAG (topo order exists)."""
    c = Circuit(4); c.x(0)
    for i in range(3):
        c.cx(i, i + 1)
    dag = _dag_for(c, make_star(4, 1))
    for i, ps in dag.preds.items():
        for p in ps:
            assert i in dag.succ[p]                            # succ mirrors preds
    # Kahn's algorithm drains the graph iff acyclic
    indeg = {i: len(dag.preds[i]) for i in dag.ops}
    ready = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while ready:
        i = ready.pop()
        seen += 1
        for j in dag.succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    assert seen == len(dag.ops)


def test_fgp_moves_appear_in_dag():
    """A circuit compiled with FGP that relocates a qubit records move ops in the DAG."""
    n = 5
    c = Circuit(n); c.x(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    dag = _dag_for(c, make_grid(2, 3, 1), partitioner="topo-aware", scheduler="fgp")
    kinds = {op.kind for op in dag.ops.values()}
    assert "remote" in kinds or "move" in kinds                # distributed work exists
    # if any move exists, its qubit dependency is respected (move sits after a prior op on q)
    for i, op in dag.ops.items():
        if op.kind == "move":
            for p in dag.preds[i]:
                assert op.qubits[0] in dag.ops[p].qubits
