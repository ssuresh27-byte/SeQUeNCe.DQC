"""DQCApp teledata-move integration tests.

Exercises the teleport ("move") primitive inside the barriered DQCApp runtime:
a qubit's STATE is teleported to another node, the location bookkeeping
(qubit_to_node / data_owned / peer_slots -- shared dicts) is updated in place, and
subsequent LOCAL gates address the qubit at its new home. Also covers the router
that lets telegate + teledata coexist on one node.
"""

import contextlib
import io

import numpy as np
import pytest

from sequence.topology.dqc_net_topo import DQCNetTopo
from sequence.components.circuit import Circuit
from sequence.kernel.quantum_utils import verify_same_state_vector

import sequence.dqc.architecture as topo
from sequence.dqc.dqc_app import DQCApp
from sequence.dqc.controllers.central_node import CentralNodeController


def _net(cap=4, memo=16):
    T = topo.make_star(2, cap=cap)
    cfg = T.sim_config()
    for n in cfg["nodes"]:
        if n.get("type") == "DQCNode":
            n["memo_size"] = memo
    net = DQCNetTopo(cfg)
    qn = {n.name: n for n in net.nodes[DQCNetTopo.DQC_NODE]}
    return T, net, net.tl, qn


def _measure(tl, node, slot):
    arr = node.get_component_by_name(node.data_memo_arr_name)
    c = Circuit(1)
    c.measure(0)
    return tl.quantum_manager.run_circuit(c, [arr[slot].qstate_key], 0.5)[arr[slot].qstate_key]


def _apps(qn, T, ctrl, q2n, do, ps, per_node):
    hop = T.hop_distances()
    for name, node in qn.items():
        cfg = per_node.get(name, {})
        DQCApp(node=node, qubit_to_node=q2n, local_ops=cfg.get("local", []),
               remote_ops=cfg.get("remote", []), data_owned=do[name], peer_slots=ps,
               target_ops=cfg.get("target", []), move_ops=cfg.get("move", []),
               controller=ctrl, hop_distances=hop.get(name, {}))


def _ctrl(net, max_step, net_layers, dt=0.0, local_dt=0.0):
    """Build the controller as a real node in ``net`` (no compiler/scheduler -- these
    tests hand-craft node_ops and drive the barrier), wired to every DQC node."""
    ctrl = CentralNodeController("controller", net.tl, dt=dt, local_dt=local_dt)
    ctrl.max_step = max_step
    ctrl.net_layers = net_layers
    topo.wire_controller(net, ctrl)
    return ctrl


def test_move_then_local_gate():
    """Teleport q0 alice->bob, then a LOCAL X on q0 at its new home."""
    with contextlib.redirect_stdout(io.StringIO()):
        T, net, tl, qn = _net()
        alice, bob = qn["alice"], qn["bob"]
        alice.get_component_by_name(alice.data_memo_arr_name)[0].update_state(np.array([1, 0], dtype=complex))
        q2n = {0: "alice"}
        do = {"alice": {0: 0}, "bob": {}}
        ps = {"alice": do["alice"], "bob": do["bob"]}
        move = {"step": 0, "qubit": 0, "dest": "bob", "dest_slot": 0}
        ctrl = _ctrl(net, 1, {0})
        tl.stop_time = int(5e11)
        _apps(qn, T, ctrl, q2n, do, ps, {
            "alice": {"move": [move]},
            "bob": {"move": [move], "local": [{"step": 1, "gate": "x", "targets": [0]}]},
        })
        tl.init(); ctrl.start_execution(); tl.run()
        val = _measure(tl, bob, do["bob"][0])
    assert q2n[0] == "bob" and 0 not in do["alice"]   # bookkeeping moved
    assert do["bob"][0] == 0
    assert val == 1                                    # |0> --X--> |1> at new home
    assert ctrl.current > 1                            # all steps completed


@pytest.mark.parametrize("psi", [
    np.array([1, 0], dtype=complex),
    np.array([0.6, 0.8], dtype=complex),
    np.array([0.6, 0.8j], dtype=complex),
    np.array([1, 1j], dtype=complex) / np.sqrt(2),
])
def test_move_preserves_state(psi):
    """An arbitrary single-qubit state survives a teleport-move intact."""
    psi = psi / np.linalg.norm(psi)
    with contextlib.redirect_stdout(io.StringIO()):
        T, net, tl, qn = _net()
        alice, bob = qn["alice"], qn["bob"]
        alice.get_component_by_name(alice.data_memo_arr_name)[0].update_state(psi)
        q2n = {0: "alice"}
        do = {"alice": {0: 0}, "bob": {}}
        ps = {"alice": do["alice"], "bob": do["bob"]}
        move = {"step": 0, "qubit": 0, "dest": "bob", "dest_slot": 2}
        ctrl = _ctrl(net, 0, {0})
        tl.stop_time = int(5e11)
        _apps(qn, T, ctrl, q2n, do, ps, {"alice": {"move": [move]}, "bob": {"move": [move]}})
        tl.init(); ctrl.start_execution(); tl.run()
        arr = bob.get_component_by_name(bob.data_memo_arr_name)
        st = np.array(tl.quantum_manager.get(arr[do["bob"][0]].qstate_key).state)
        st = st[: len(st) // 2] if len(st) > 2 else st
    assert do["bob"][0] == 2
    assert verify_same_state_vector(st, psi)


def test_hybrid_telegate_and_move():
    """A telegate and a teleport in one run (router routes both apps).

    step0: telegate CX(q0@alice -> q1@bob)  => q1 flips to 1
    step1: move q0 alice -> bob slot1
    step2: LOCAL CX(q0,q1) on bob           => q0==1 flips q1 back to 0
    """
    with contextlib.redirect_stdout(io.StringIO()):
        T, net, tl, qn = _net()
        alice, bob = qn["alice"], qn["bob"]
        alice.get_component_by_name(alice.data_memo_arr_name)[0].update_state(np.array([0, 1], dtype=complex))
        bob.get_component_by_name(bob.data_memo_arr_name)[0].update_state(np.array([1, 0], dtype=complex))
        q2n = {0: "alice", 1: "bob"}
        do = {"alice": {0: 0}, "bob": {1: 0}}
        ps = {"alice": do["alice"], "bob": do["bob"]}
        tg = {"step": 0, "gate": "cx", "targets": [0, 1]}
        move = {"step": 1, "qubit": 0, "dest": "bob", "dest_slot": 1}
        loc = {"step": 2, "gate": "cx", "targets": [0, 1]}
        ctrl = _ctrl(net, 2, {0, 1})
        tl.stop_time = int(9e11)
        _apps(qn, T, ctrl, q2n, do, ps, {
            "alice": {"remote": [tg], "move": [move]},
            "bob": {"target": [tg], "move": [move], "local": [loc]},
        })
        tl.init(); ctrl.start_execution(); tl.run()
        q0v = _measure(tl, bob, do["bob"][0])
        q1v = _measure(tl, bob, do["bob"][1])
    assert q2n[0] == "bob" and q2n[1] == "bob"     # both qubits co-located
    assert q0v == 1 and q1v == 0
    assert ctrl.current > 2
