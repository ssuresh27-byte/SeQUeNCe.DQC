"""Same-link concurrency regression tests for telegate + teledata.

The existing dual tests only exercise concurrency across DIFFERENT links
(Alice->Bob and Alice->Charlie). These guard the SAME-link case: two sessions
sharing one physical link (one BSM) at the same time. That path was broken by a
reservation-layer bug -- ``Reservation.__eq__``/``__hash__`` ignored ``identity``
and every app reservation defaulted to ``identity=0``, so two concurrent
same-link reservations compared equal, collapsed onto ONE comm memory, and only
one of them ever entangled. On top of that each app dropped the second session
when binding a pair to its protocol.

If any of these regress, concurrent (non-blocking) schedulers silently lose
operations, so pin them down:
  * reservation layer  -> N concurrent same-link reservations yield N pairs
  * teledata           -> two concurrent teleports land on distinct slots
  * telegate           -> two concurrent CX both apply
"""

import os
import itertools

import numpy as np
import pytest

from sequence.topology.dqc_net_topo import DQCNetTopo
from sequence.app.teleportation import TeledataApp, TelegateApp
from sequence.components.circuit import Circuit
from sequence.kernel.quantum_utils import verify_same_state_vector

# A 2-node star (alice-bob over BSM_alice_bob) with 4 data + 16 comm memories per
# node -- enough headroom for several concurrent reservations on the one link.
_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "concurrent_2node.json")
MILLISECOND = 1_000_000_000

_rng = np.random.default_rng(2025)


def _random_state(rng):
    vec = (rng.normal(size=2) + 1j * rng.normal(size=2)).astype(complex)
    n = np.linalg.norm(vec)
    return np.array([1, 0], dtype=complex) if n == 0 else vec / n


def _all_nodes(topo):
    flat = []
    for g in topo.nodes.values():
        flat.extend(list(g))
    return flat


def _single_qubit(full_state):
    """Reduce a possibly-entangled state vector to its leading single-qubit half."""
    if hasattr(full_state, "__len__") and len(full_state) > 2:
        return np.array(full_state[: len(full_state) // 2], dtype=complex)
    return np.array(full_state, dtype=complex)


def _load():
    topo = DQCNetTopo(_CFG)
    nodes = _all_nodes(topo)
    alice = next(n for n in nodes if getattr(n, "name", "") == "alice")
    bob = next(n for n in nodes if getattr(n, "name", "") == "bob")
    return topo, alice, bob


# ─────────────────────── reservation layer (root cause) ───────────────────────
@pytest.mark.parametrize("n_res", [2, 3])
def test_concurrent_same_link_reservations_yield_distinct_pairs(n_res):
    """N overlapping reservations on one link must generate N distinct EPR pairs.

    Before the identity fix this always collapsed to exactly 1 (all reservations
    aliased onto comm memory index 0)."""
    topo, alice, _bob = _load()
    tl = topo.tl

    entangled = set()
    for info in alice.resource_manager.memory_manager:
        orig = info.to_entangled

        def wrap(o=orig, name=info.memory.name):
            def f(*a, **k):
                entangled.add(name)
                return o(*a, **k)
            return f
        info.to_entangled = wrap()

    for _ in range(n_res):  # no identity passed -> auto-assigned uniquely
        alice.reserve_net_resource("bob", 1 * MILLISECOND, 200 * MILLISECOND, 1, 0.8, 1)

    tl.init()
    tl.run()
    assert len(entangled) == n_res, f"expected {n_res} distinct pairs, got {len(entangled)}"


# ──────────────────────────────── teledata ────────────────────────────────
_td_inputs = [(_random_state(_rng), _random_state(_rng)) for _ in range(4)]


@pytest.mark.parametrize("psi0,psi1", _td_inputs)
def test_teledata_concurrent_same_link(psi0, psi1):
    """Two concurrent teleports Alice->Bob on the same link land on the correct,
    distinct target slots (0 and 1)."""
    topo, alice, bob = _load()
    tl = topo.tl

    a = alice.get_component_by_name(alice.data_memo_arr_name)
    a[0].update_state(psi0.astype(complex))
    a[1].update_state(psi1.astype(complex))

    A, B = TeledataApp(alice), TeledataApp(bob)
    A.start(responder="bob", start_t=1 * MILLISECOND, end_t=200 * MILLISECOND,
            memory_size=1, fidelity=0.8, data_src=0)
    A.start(responder="bob", start_t=1 * MILLISECOND, end_t=200 * MILLISECOND,
            memory_size=1, fidelity=0.8, data_src=1)

    tl.init()
    tl.run()

    assert len(B.data_keys) == 2, f"only {len(B.data_keys)}/2 teleports completed"
    bd = bob.get_component_by_name(bob.data_memo_arr_name)
    out0 = _single_qubit(tl.quantum_manager.get(bd[0].qstate_key).state)
    out1 = _single_qubit(tl.quantum_manager.get(bd[1].qstate_key).state)
    assert verify_same_state_vector(out0, psi0)
    assert verify_same_state_vector(out1, psi1)


def test_teledata_sequential_same_link():
    """Baseline: back-to-back (non-overlapping) teleports on one link both land."""
    topo, alice, bob = _load()
    tl = topo.tl
    psi0, psi1 = _random_state(_rng), _random_state(_rng)

    a = alice.get_component_by_name(alice.data_memo_arr_name)
    a[0].update_state(psi0.astype(complex))
    a[1].update_state(psi1.astype(complex))

    A, B = TeledataApp(alice), TeledataApp(bob)
    A.start(responder="bob", start_t=1 * MILLISECOND, end_t=100 * MILLISECOND,
            memory_size=1, fidelity=0.8, data_src=0)
    A.start(responder="bob", start_t=110 * MILLISECOND, end_t=200 * MILLISECOND,
            memory_size=1, fidelity=0.8, data_src=1)

    tl.init()
    tl.run()

    bd = bob.get_component_by_name(bob.data_memo_arr_name)
    assert verify_same_state_vector(_single_qubit(tl.quantum_manager.get(bd[0].qstate_key).state), psi0)
    assert verify_same_state_vector(_single_qubit(tl.quantum_manager.get(bd[1].qstate_key).state), psi1)


# ──────────────────────────────── telegate ────────────────────────────────
@pytest.mark.parametrize("t0,t1", list(itertools.product([0, 1], repeat=2)))
def test_telegate_concurrent_same_link_both_cx_apply(t0, t1):
    """Two concurrent remote-CX Alice->Bob on the same link both apply.

    Controls are set to |1> so each CX flips its target: target t -> t^1. The
    old bug bound only the first session, leaving the other target untouched."""
    topo, alice, bob = _load()
    tl = topo.tl

    ad = alice.get_component_by_name(alice.data_memo_arr_name)
    ad[0].update_state(np.array([0, 1], dtype=complex))  # control 0 = |1>
    ad[1].update_state(np.array([0, 1], dtype=complex))  # control 1 = |1>
    bd = bob.get_component_by_name(bob.data_memo_arr_name)
    bd[0].update_state(np.array([1, 0] if t0 == 0 else [0, 1], dtype=complex))
    bd[1].update_state(np.array([1, 0] if t1 == 0 else [0, 1], dtype=complex))

    A, B = TelegateApp(alice), TelegateApp(bob)
    # Bob's per-step slot map is only a placeholder at protocol creation; the gate
    # uses the target index Alice sends per session, so one registered step is fine.
    B.set_target_slot_for_step(0, 0)
    B.set_current_step(0)
    A.start(responder="bob", start_t=1 * MILLISECOND, end_t=200 * MILLISECOND,
            memory_size=1, fidelity=0.8, control_src=0, target_src=0, step=0, gate_type="cx")
    A.start(responder="bob", start_t=1 * MILLISECOND, end_t=200 * MILLISECOND,
            memory_size=1, fidelity=0.8, control_src=1, target_src=1, step=0, gate_type="cx")

    tl.init()
    tl.run()

    def measure(slot):
        c = Circuit(1)
        c.measure(0)
        return tl.quantum_manager.run_circuit(c, [bd[slot].qstate_key], 0.5)[bd[slot].qstate_key]

    assert measure(0) == t0 ^ 1, "target 0 not flipped by its CX"
    assert measure(1) == t1 ^ 1, "target 1 not flipped by its CX"
