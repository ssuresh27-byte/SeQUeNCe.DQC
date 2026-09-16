#!/usr/bin/env python3
"""DQC per-node noise managers + the uniform config.

The base ket/density quantum managers stay exactly as-pulled (a plain ideal ket manager;
a density manager with its own *global* noise model). The DQC noise here is added by
SUBCLASSING them -- :class:`QuantumManagerKetNoise` extends
:class:`~sequence.kernel.quantum_manager.ket_vector.QuantumManagerKet` and
:class:`QuantumManagerDensityNoise` extends
:class:`~sequence.kernel.quantum_manager.density_matrix.QuantumManagerDensity`. Each
overrides ``run_circuit`` to add PER-NODE noise (routing every qubit to its owning DQCNode
and reading the fidelities/coherence LIVE off the node), delegating the ideal execution to
``super().run_circuit`` and reusing the base's state/measure/Pauli helpers.

Routing is COMPOSED, not inherited: each manager holds a
:class:`~sequence.dqc.registry.QubitRegistry` (``self.registry``) -- the controller-owned
logical<->physical<->key map, injected here so the noise math can read each qubit's owning
node off ``node_of`` for its live fidelities/coherence. The noise layer is one *reader* of
that registry, not its owner. A qubit with no registered node is ideal, so the noise path is
entered only for DQC-node qubits (never for plain Node / QuantumRouter qubits).

Only the APPLICATION differs: :class:`QuantumManagerKetNoise` samples a Pauli trajectory (a
ket is a pure state), :class:`QuantumManagerDensityNoise` applies deterministic CPTP (Kraus)
channels. Noise is entirely per-node: it is declared on each DQCNode (fidelities/T1/T2); a
node with no noise params is ideal.
"""
from __future__ import annotations

import itertools
import numpy as np

from ..constants import KET_VECTOR_FORMALISM, DENSITY_MATRIX_FORMALISM
from ..kernel.quantum_manager import QuantumManager
from ..kernel.quantum_manager.ket_vector import QuantumManagerKet
from ..kernel.quantum_manager.density_matrix import QuantumManagerDensity
from ..components.circuit import Circuit
from .registry import QubitRegistry


# Formalism ids for the noise-aware managers (base ids stay on the pristine managers).
KET_VECTOR_NOISE_FORMALISM = "ket_vector_noise"
DENSITY_MATRIX_NOISE_FORMALISM = "density_matrix_noise"


# ── shared fidelity -> probability + T1/T2 math ───────────────────────────────
def gate_error_prob(infidelity: float, num_qubits: int) -> float:
    """Average gate infidelity -> error-channel probability p = (d+1)/d * infid, capped at
    1, with d = 2**num_qubits (1.5*infid for 1-qubit, 1.25*infid for 2-qubit gates)."""
    d = 2 ** num_qubits
    return min(1.0, (d + 1) / d * infidelity)


def pure_dephasing_time(t1, t2):
    """Pure-dephasing time T_phi from T1, T2 (1/Tphi = 1/T2 - 1/2T1). None if T2 is off;
    +inf if there is no pure dephasing (T2 == 2 T1)."""
    if t2 is None:
        return None
    if t1 is None:
        return t2
    inv = 1.0 / t2 - 1.0 / (2.0 * t1)
    return (1.0 / inv) if inv > 0 else float("inf")


def relaxation_prob(idle_s: float, t1: float) -> float:
    """Amplitude-damping (T1) relaxation probability over an idle interval: 1 - e^(-t/T1)."""
    return 1.0 - np.exp(-idle_s / t1)


def dephasing_z_prob(idle_s: float, t1, t2) -> float:
    """Stochastic-Z (pure-dephasing) probability over an idle interval: 0.5 (1 - e^(-t/Tphi)).
    Zero if T2 is off or there is no pure dephasing."""
    t_phi = pure_dephasing_time(t1, t2)
    if t_phi is None or t_phi == float("inf"):
        return 0.0
    return 0.5 * (1.0 - np.exp(-idle_s / t_phi))


@QuantumManager.register(KET_VECTOR_NOISE_FORMALISM)
class QuantumManagerKetNoise(QuantumManagerKet):
    """Ket manager (as-pulled) + PER-NODE trajectory noise. Overrides ``run_circuit`` to add
    idle T1/T2, per-gate sampled Paulis, and readout flips, delegating the ideal execution to
    ``super().run_circuit``. Ideal for any qubit with no registered node. Routing/logical-qubit
    state lives in the composed :class:`QubitRegistry` (``self.registry``)."""

    _PAULIS = ("x", "y", "z")

    def __init__(self, seed=None):
        super().__init__()
        self.noise_rng = np.random.default_rng(seed)
        self.registry = QubitRegistry()

    @classmethod
    def get_active_formalism(cls):
        """Report the BASE formalism so the library's string dispatch (bsm.py, entanglement
        code) treats us as a plain ket manager -- our state representation is identical. This
        is an INSTANCE-level view; ``QuantumManager.create`` reads the true global off the
        base class, so it still builds this subclass for a noisy run."""
        return KET_VECTOR_FORMALISM

    def run_circuit(self, circuit, keys, meas_samp=None, inject_gate_error=False):
        """Apply a circuit. When ``inject_gate_error`` OR any of ``keys`` is on a noisy node,
        add idle T1/T2 (before), per-gate depolarizing (after), and readout flips; otherwise
        the pristine ideal path (``super().run_circuit``)."""
        if not (inject_gate_error or self.registry.noise_active(keys)):
            return super().run_circuit(circuit, keys, meas_samp)
        now = self.registry.sim_now(keys)
        if now is not None:
            self.apply_idling_decoherence(keys, now)
        result = super().run_circuit(circuit, keys, meas_samp)
        self._apply_gate_noise(circuit, keys)
        if result:
            self._apply_readout(result)
        return result

    def _pauli(self, key: int, p: str) -> None:
        """Apply a single Pauli ``p`` on ``key`` via the ideal path (never re-noised)."""
        from ..components.circuit import Circuit
        c = Circuit(1); getattr(c, p)(0); super().run_circuit(c, [key])

    def apply_noise(self, keys, noise_type, amount, is_infidelity=True) -> None:
        """Plug-in primitive (same signature as the density manager). A ket is pure, so the
        channel is realised as a SAMPLED Pauli trajectory -- w.p. p apply a random Pauli over
        ``noise_type``'s alphabet. ``amount`` is an average gate infidelity unless
        ``is_infidelity`` is False."""
        k = len(keys)
        p = gate_error_prob(amount, k) if is_infidelity else min(1.0, amount)
        if p <= 0.0 or self.noise_rng.random() >= p:
            return
        if noise_type == "dephase":
            for key in keys:
                self._pauli(key, "z")
        elif noise_type == "bit_flip":
            for key in keys:
                self._pauli(key, "x")
        elif k == 1:
            self._pauli(keys[0], self._PAULIS[int(self.noise_rng.integers(3))])
        else:
            while True:
                a, b = int(self.noise_rng.integers(4)), int(self.noise_rng.integers(4))
                if a or b:
                    break
            if a:
                self._pauli(keys[0], self._PAULIS[a - 1])
            if b:
                self._pauli(keys[1], self._PAULIS[b - 1])

    def apply_idling_decoherence(self, keys, now_ps, t1=None, t2=None) -> None:
        """Idle T1/T2 over the time since each key was last touched (watermark in
        ``last_touched``); ``t1``/``t2`` default to the owning node's values. T1 = stochastic
        relaxation via a sanctioned measurement; T2 = stochastic Z."""
        for key in keys:
            node = self.registry.node_of.get(key)
            u1 = t1 if t1 is not None else (node.t1 if node is not None else None)
            u2 = t2 if t2 is not None else (node.t2 if node is not None else None)
            if u1 is None and u2 is None:
                continue
            idle_s = (now_ps - self.registry.last_touched.get(key, now_ps)) * 1e-12
            self.registry.last_touched[key] = now_ps
            if idle_s <= 0:
                continue
            if u1 is not None and self.noise_rng.random() < relaxation_prob(idle_s, u1):
                c = Circuit(1); c.measure(0)
                if super().run_circuit(c, [key], self.noise_rng.random()).get(key) == 1:
                    self._pauli(key, "x")
            if self.noise_rng.random() < dephasing_z_prob(idle_s, u1, u2):
                self._pauli(key, "z")

    def _apply_gate_noise(self, circuit, keys) -> None:
        """Depolarize after each gate at the owning node's gate fidelity."""
        for _name, indices, _arg in circuit.gates:
            gk = [keys[i] for i in indices]
            if len(gk) == 1:
                f = self.registry.fids(gk[0])[0]
                if f < 1.0:
                    self.apply_noise(gk, "depolarize", 1.0 - f)
            elif len(gk) == 2:
                f = min(self.registry.fids(gk[0])[1], self.registry.fids(gk[1])[1])
                if f < 1.0:
                    self.apply_noise(gk, "depolarize", 1.0 - f)

    def _apply_readout(self, result) -> None:
        """Flip each reported bit w.p. (1 - measurement_fid) of its owning node."""
        for key in list(result):
            f = self.registry.fids(key)[2]
            if f < 1.0 and self.noise_rng.random() > f:
                result[key] ^= 1


@QuantumManager.register(DENSITY_MATRIX_NOISE_FORMALISM)
class QuantumManagerDensityNoise(QuantumManagerDensity):
    """Pristine density manager + PER-NODE CPTP noise. The base density manager is ideal, so
    ALL noise lives here: overrides ``run_circuit`` to apply idle T1/T2 and a per-gate
    depolarizing channel at each owning node's fidelity, reusing the base's ideal
    ``_measure`` helper. Owns the Pauli-channel machinery and its ``_merge_state`` helper.
    Routing/logical-qubit state lives in the composed :class:`QubitRegistry` (``self.registry``)."""

    # Single-qubit Pauli matrices + the alphabet each noise_type samples over.
    _PAULI = {"I": np.eye(2, dtype=complex),
              "X": np.array([[0, 1], [1, 0]], dtype=complex),
              "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
              "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    _NOISE_ALPHABET = {"depolarize": "IXYZ", "dephase": "IZ", "bit_flip": "IX"}

    def __init__(self, seed: int | None = None):
        super().__init__()              # base density is ideal; no seed/fids
        self.noise_rng = np.random.default_rng(seed)
        self.registry = QubitRegistry()
        self.gate_1q_count = self.gate_2q_count = 0
        self.measurement_count = self.measurement_error_count = 0

    @classmethod
    def get_active_formalism(cls):
        """Report the BASE formalism so string dispatch treats us as a plain density manager
        (identical state representation). ``QuantumManager.create`` reads the true global off
        the base class, so it still builds this subclass for a noisy run."""
        return DENSITY_MATRIX_FORMALISM

    def _merge_state(self, keys: list[int]) -> tuple[np.ndarray, list[int]]:
        """Tensor the distinct density-matrix blocks touched by ``keys`` into one rho.

        DQC-specific helper (the upstream base density manager does not provide it): the
        noisy path operates gate-by-gate on the joint state, so any separable blocks the
        gate/channel spans must be merged first.

        Args:
            keys (list[int]): keys whose (possibly separable) density-matrix blocks
                should be merged into one joint state.

        Returns:
            tuple[np.ndarray, list[int]]: (rho, all_keys) where all_keys is the union of
                every involved state's keys, in block order (mirrors ``_prepare_circuit``).
        """
        old_states: list[np.ndarray] = []
        all_keys: list[int] = []
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(np.asarray(qstate.state, dtype=complex))
                all_keys += list(qstate.keys)
        rho = np.array([[1.0 + 0j]])
        for state in old_states:
            rho = np.kron(rho, state)
        return rho, all_keys

    def run_circuit(self, circuit, keys, meas_samp=None, inject_gate_error=False):
        """Apply a circuit. When ``inject_gate_error`` OR any of ``keys`` is on a noisy node,
        run gate-by-gate with a CPTP channel after each gate at the owning node's fidelity
        (plus idle T1/T2 and readout error); otherwise the pristine ideal path."""
        from ..components.circuit import Circuit
        from ..kernel.quantum_state import DensityState
        from ..kernel.quantum_manager.utils import validate_circuit_run
        if not (inject_gate_error or self.registry.noise_active(keys)):
            return super().run_circuit(circuit, keys, meas_samp)

        validate_circuit_run(circuit, keys, meas_samp)   # same contract as the base path
        now = self.registry.sim_now(keys)
        if now is not None:
            self.apply_idling_decoherence(keys, now)
        rho, all_keys = self._merge_state(keys)
        n = len(all_keys)
        pos = {key: i for i, key in enumerate(all_keys)}
        for gate in circuit.gates:
            name, indices = gate[0], gate[1]
            arg = gate[2] if len(gate) > 2 else None
            gate_keys = [keys[j] for j in indices]
            mapped = [pos[k] for k in gate_keys]
            g = Circuit(n); g.gates.append([name, mapped, arg])
            gmat = g.get_unitary_matrix()
            rho = gmat @ rho @ gmat.conj().T
            if len(gate_keys) == 1:
                self.gate_1q_count += 1
                f = self.registry.fids(gate_keys[0])[0]
            elif len(gate_keys) == 2:
                self.gate_2q_count += 1
                f = min(self.registry.fids(gate_keys[0])[1], self.registry.fids(gate_keys[1])[1])
            else:
                f = 1.0
            if f < 1.0:
                rho = self._pauli_channel(rho, n, mapped, "depolarize",
                                          gate_error_prob(1.0 - f, len(mapped)))
        if len(circuit.measured_qubits) == 0:
            new_state_obj = DensityState(rho, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj
            return {}
        measured_keys = [keys[i] for i in circuit.measured_qubits]
        results = self._measure(rho, measured_keys, all_keys, meas_samp)
        for mk in list(results):                        # readout error, per owning node
            f = self.registry.fids(mk)[2]
            if f < 1.0:
                self.measurement_count += 1
                if self.noise_rng.random() > f:
                    results[mk] ^= 1
                    self.measurement_error_count += 1
        return results

    def apply_idling_decoherence(self, keys, now_ps, t1=None, t2=None) -> None:
        """Idle T1/T2 over the time since each key was last touched; ``t1``/``t2`` default to
        the owning node's values. T1 = amplitude-damping channel; T2 = Z channel."""
        for key in keys:
            node = self.registry.node_of.get(key)
            u1 = t1 if t1 is not None else (node.t1 if node is not None else None)
            u2 = t2 if t2 is not None else (node.t2 if node is not None else None)
            if u1 is None and u2 is None:
                continue
            idle_s = (now_ps - self.registry.last_touched.get(key, now_ps)) * 1e-12
            self.registry.last_touched[key] = now_ps
            if idle_s <= 0:
                continue
            rho, all_keys = self._merge_state([key])
            n = len(all_keys)
            p = all_keys.index(key)
            if u1 is not None:
                rho = self._amp_damp(rho, n, p, relaxation_prob(idle_s, u1))
            pz = dephasing_z_prob(idle_s, u1, u2)
            if pz > 0.0:
                rho = self._pauli_channel(rho, n, [p], "dephase", pz)
            self.set(all_keys, rho)

    def _pauli_labels(self, k, noise_type="depolarize"):
        """Non-identity length-k Pauli strings over ``noise_type``'s alphabet."""
        if noise_type not in self._NOISE_ALPHABET:
            raise ValueError(f"Unknown noise_type '{noise_type}'. Use one of {sorted(self._NOISE_ALPHABET)}.")
        alphabet = self._NOISE_ALPHABET[noise_type]
        labels = ("".join(t) for t in itertools.product(alphabet, repeat=k))
        return [s for s in labels if any(c != "I" for c in s)]

    def _embed_pauli(self, label, positions, n):
        """Full 2^n operator with ``label``'s Paulis on ``positions``, identity elsewhere."""
        ops = [self._PAULI["I"]] * n
        for pauli_char, q in zip(label, positions):
            ops[q] = self._PAULI[pauli_char]
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        return full

    def _pauli_channel(self, rho, n, positions, noise_type, p):
        """rho -> (1-p) rho + (p/|S|) sum_{P in S} P rho Pd over ``noise_type``'s alphabet."""
        if p <= 0.0:
            return rho
        labels = self._pauli_labels(len(positions), noise_type)
        if not labels:
            return rho
        out = (1.0 - p) * rho
        weight = p / len(labels)
        for label in labels:
            P = self._embed_pauli(label, positions, n)
            out = out + weight * (P @ rho @ P.conj().T)
        return out

    def _amp_damp(self, rho, n, position, gamma):
        """Single-qubit amplitude-damping (T1) channel on ``position`` within an n-qubit rho."""
        if gamma <= 0.0:
            return rho
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)

        def embed(op):
            ops = [np.eye(2, dtype=complex)] * n
            ops[position] = op
            full = ops[0]
            for o in ops[1:]:
                full = np.kron(full, o)
            return full
        K0, K1 = embed(k0), embed(k1)
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    def apply_noise(self, keys, noise_type, amount, is_infidelity=True) -> None:
        """Plug-in primitive (parity with the ket manager): apply a Pauli/depolarizing CPTP
        channel to 1-2 keys within their joint density matrix. ``amount`` is an average gate
        infidelity unless ``is_infidelity`` is False."""
        if amount <= 0.0:
            return
        keys = list(keys)
        k = len(keys)
        if k not in (1, 2):
            raise ValueError("apply_noise supports 1 or 2 keys.")
        p = gate_error_prob(amount, k) if is_infidelity else min(1.0, amount)
        if p <= 0.0:
            return
        rho, all_keys = self._merge_state(keys)
        positions = [all_keys.index(key) for key in keys]
        out = self._pauli_channel(rho, len(all_keys), positions, noise_type, p)
        self.set(all_keys, out)

    def reduce_to(self, keep_keys: list[int]) -> None:
        """Partial-trace the joint state holding ``keep_keys`` down to just those keys.

        Detaches ``keep_keys`` from any other qubits currently sharing their density
        matrix (e.g. measured comm qubits left entangled after a teleported gate, which
        would corrupt the data qubits when their memory slots are later reset) and
        re-registers the reduced state. No-op if the state already contains only
        ``keep_keys``.

        Args:
            keep_keys (list[int]): keys to retain; all other qubits sharing their joint
                state are traced out.

        Returns:
            None.
        """
        keep_keys = list(keep_keys)
        rho, all_keys = self._merge_state(keep_keys)
        if len(all_keys) == len(keep_keys):
            self.set(all_keys, rho)
            return
        reduced, kept_order = self._partial_trace(rho, all_keys, keep_keys)
        self.set(kept_order, reduced)

    @staticmethod
    def _partial_trace(rho, keys: list[int], keep: list[int]):
        """Partial-trace ``rho`` (ordered by ``keys``) down to ``keep``.

        Args:
            rho (np.ndarray): joint density matrix, ordered by ``keys``.
            keys (list[int]): keys labeling rho's qubits, in order.
            keep (list[int]): subset of ``keys`` to retain.

        Returns:
            tuple[np.ndarray, list[int]]: (reduced_rho, kept_key_order).
        """
        n = len(keys)
        t = np.asarray(rho, dtype=complex).reshape([2] * n + [2] * n)
        row = [chr(ord('a') + i) for i in range(n)]
        col = [chr(ord('a') + n + i) for i in range(n)]
        for i in range(n):
            if keys[i] not in keep:
                col[i] = row[i]                      # trace this qubit
        out_row = [row[i] for i in range(n) if keys[i] in keep]
        out_col = [col[i] for i in range(n) if keys[i] in keep]
        subscript = ''.join(row) + ''.join(col) + '->' + ''.join(out_row) + ''.join(out_col)
        reduced = np.einsum(subscript, t)
        m = len(keep)
        kept_order = [key for key in keys if key in keep]
        return reduced.reshape(2 ** m, 2 ** m), kept_order

    def reset_error_statistics(self) -> None:
        self.gate_1q_count = self.gate_2q_count = 0
        self.measurement_count = self.measurement_error_count = 0

    def get_error_statistics(self) -> dict:
        return {"gate_1q_count": self.gate_1q_count, "gate_2q_count": self.gate_2q_count,
                "measurement_count": self.measurement_count,
                "measurement_error_count": self.measurement_error_count}
