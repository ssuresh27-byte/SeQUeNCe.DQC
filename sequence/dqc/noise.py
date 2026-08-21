#!/usr/bin/env python3
"""Trajectory-noise configuration for DQC runs.

Gate / measurement / idle-T1-T2 noise is now applied NATIVELY by the ket-vector
quantum manager (``sequence.kernel.quantum_manager.ket_vector.QuantumManagerKet``):
per-node knobs are self-registered by each ``DQCNode`` (resolved per qubit), and a
global :class:`NoiseConfig` (uniform fallback) is set straight on the manager by the
runtime. No monkeypatching is involved for those.

The one thing not yet native is the PHYSICAL BELL-PAIR fidelity ``f_phys`` (a link,
not a node, property): each freshly generated Barrett-Kok pair is twirled once with a
random X/Y/Z on one qubit w.p. (1 - f_phys) (average fidelity f_phys). Until that is
folded into entanglement generation proper, :func:`install_bell_pair_noise` provides
it as a small, self-contained hook. (Initialization fidelity F_init is ignored.)
"""
from __future__ import annotations

from dataclasses import dataclass

_PAULIS = ("x", "y", "z")


@dataclass
class NoiseConfig:
    """Uniform (global) trajectory-noise knobs + the link-level Bell-pair fidelity.

    Per-node knobs override f_1q/f_2q/f_m/t1/t2 for qubits on a noisy node (declared in
    the topology's ``node_noise``); this config supplies the uniform fallback for every
    other qubit and always supplies ``f_phys``.

    f_1q/f_2q : 1-/2-qubit gate fidelity (random Pauli w.p. 1-F after each gate).
    f_m       : measurement/readout fidelity (bit flip w.p. 1-f_m).
    f_phys    : physical Bell-pair fidelity (link property; twirl each fresh pair).
    t1, t2    : amplitude-damping (T1) and dephasing (T2) times in SECONDS over idle
                time (None = off). Physical constraint T2 <= 2 T1.
    """
    f_1q: float = 1.0
    f_2q: float = 1.0
    f_m: float = 1.0
    f_phys: float = 1.0
    t1: float = None
    t2: float = None

    def is_noiseless(self) -> bool:
        return (self.f_1q >= 1.0 and self.f_2q >= 1.0 and self.f_m >= 1.0
                and self.f_phys >= 1.0 and self.t1 is None and self.t2 is None)

    def manager_kwargs(self) -> dict:
        """The uniform gate/measurement/idle knobs to set on the ket manager (global
        fallback for any qubit without per-node noise). ``f_phys`` is excluded --
        it is a link property applied at entanglement generation, not a manager knob."""
        return {"f_1q": self.f_1q, "f_2q": self.f_2q, "f_m": self.f_m,
                "t1": self.t1, "t2": self.t2}


def install_bell_pair_noise(f_phys: float, rng):
    """Twirl every freshly generated Barrett-Kok Bell pair with a random Pauli w.p.
    (1 - ``f_phys``) on the primary's qubit (average fidelity ``f_phys``). Returns an
    ``uninstall()``. ``rng`` is a numpy Generator advanced once per generated pair.

    This is the only remaining monkeypatch; gate/measurement/idle noise is native to
    the ket manager. No-op wiring when ``f_phys >= 1``."""
    from sequence.components.circuit import Circuit
    import sequence.entanglement_management.generation.barret_kok as bk

    orig_succeed = bk.BarretKokA._entanglement_succeed

    def noisy_succeed(self):
        orig_succeed(self)
        if (f_phys < 1.0 and getattr(self, "primary", False) and rng.random() > f_phys):
            qm = self.owner.timeline.quantum_manager
            p = _PAULIS[int(rng.integers(3))]
            c = Circuit(1); getattr(c, p)(0)
            # apply via the ideal path so the twirl itself isn't re-noised by gate noise
            run = getattr(qm, "_run_ideal", qm.run_circuit)
            run(c, [self.memory.qstate_key])

    bk.BarretKokA._entanglement_succeed = noisy_succeed

    def uninstall():
        bk.BarretKokA._entanglement_succeed = orig_succeed

    return uninstall
