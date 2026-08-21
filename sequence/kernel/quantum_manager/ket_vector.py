"""This module implements the quantum manager for ket vector states.

Each gate is applied to the state tensor using tensor contraction, without constructing a full-system operator.
This costs O(2^(k+m)), where k is the number of qubits in the combined state and m is the number of qubits
the gate acts on. Since m is usually 1 or 2, it is fixed and small, making the cost O(2^k) and avoiding full
O(4^k) operators and swap matrices.
"""
from __future__ import annotations

import math
import numpy as np

from .base import QuantumManager, swap_qubits, validate_circuit_run
from ..quantum_state import KetState, OneDimensionInput
from ...constants import KET_VECTOR_FORMALISM, SQRT_HALF

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...components.circuit import Circuit


# Fixed gate matrices, matching the kron ordering SeQUeNCe uses to build compound states.
_H = np.array([[SQRT_HALF, SQRT_HALF], [SQRT_HALF, -SQRT_HALF]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_CCX = np.eye(8, dtype=complex)
_CCX[[6, 7]] = _CCX[[7, 6]]   # flip target when both controls = 1
_ROOT_IZ = SQRT_HALF * np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=complex)
_MINUS_ROOT_IZ = SQRT_HALF * np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype=complex)
_ROOT_IY = SQRT_HALF * np.array([[1, 1], [-1, 1]], dtype=complex)
_MINUS_ROOT_IY = SQRT_HALF * np.array([[1, -1], [1, 1]], dtype=complex)

# name -> (matrix, num_qubits). Names match the gate names Circuit emits (see components/circuit.py).
_FIXED = {
    "h": (_H, 1), "x": (_X, 1), "y": (_Y, 1), "z": (_Z, 1), "s": (_S, 1), "sdg": (_SDG, 1), 
    "t": (_T, 1), "cx": (_CX, 2), "cz": (_CZ, 2), "swap": (_SWAP, 2), "ccx": (_CCX, 3),
    "root_iZ": (_ROOT_IZ, 1), "minus_root_iZ": (_MINUS_ROOT_IZ, 1),
    "root_iY": (_ROOT_IY, 1), "minus_root_iY": (_MINUS_ROOT_IY, 1),
}


def _phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _gate_matrix(name: str, arg) -> np.ndarray:
    """Return the matrix for a gate by name and argument.
    
    Args:
        name (str): The name of the gate.
        arg: The argument for the gate (if any).

    Returns:
        np.ndarray: The matrix representing the gate.
    """
    if name in _FIXED:
        return _FIXED[name][0]
    if name == "phase":
        return _phase(arg)
    raise KeyError(name)


_SUPPORTED = set(_FIXED) | {"phase"}


def _apply(tensor: np.ndarray, axes: list[int], gate: np.ndarray) -> np.ndarray:
    """Contract an m-qubit ``gate`` into ``tensor`` on ``axes`` (gate-qubit order).

    ``tensor`` is the statevector viewed as an n-axis array of shape (2,)*n -- one
    axis per qubit. This applies ``gate`` to the ``m`` qubits in ``axes`` without
    ever forming the full 2^n x 2^n operator. Cost is O(2^n), vs O(4^n) for the
    matrix path.

    Worked example -- X on qubit 1 of 3 (axes=[1], gate=2x2):
        tensor[b0][b1][b2]                       # the 8 amplitudes as a 2x2x2 grid
        moveaxis(.., [1], [0]) -> [b1][b0][b2]   # bring the target qubit's axis to the front
        reshape(2, -1) -> state_matrix (2, 4)    # rows = target qubit (0/1), columns = other qubits
        gate @ state_matrix                      # apply the gate across other-qubit configurations
        reshape + moveaxis back                  # restore the original axis order
    """
    m = len(axes)
    front = list(range(m))                       # destination axes 0..m-1
    t = np.moveaxis(tensor, axes, front)         # target qubit(s) -> front (a view; no copy)
    shp = t.shape
    # Group target and other axes into rows and columns, apply the gate, then restore the tensor shape.
    t = (gate @ t.reshape(2 ** m, -1)).reshape(shp)
    return np.moveaxis(t, front, axes)           # move the axes back where they were


@QuantumManager.register(KET_VECTOR_FORMALISM)
class QuantumManagerKet(QuantumManager):
    """Track and manage quantum states with the ket-vector formalism.

    Optional MONTE-CARLO TRAJECTORY noise: a ket is a pure state, so noise is added
    per shot as random Pauli events -- over many shots the ensemble of pure
    trajectories reproduces the mixed-state statistics at O(2^n) instead of the
    density matrix's O(4^n). When every fidelity is 1 and no per-key noise is
    registered, ``run_circuit`` takes the original zero-overhead ideal path.

    Noise is PER-QUBIT: a global fidelity (from constructor kwargs) applies to any
    qubit, but a qubit may carry its own fidelities via :meth:`set_key_noise`
    (a DQCNode registers its data/comm qubits with its local hardware params), so
    different nodes can be noisier than others. Idle T1/T2 needs sim-time, so the
    Timeline hands the manager a ``timeline`` reference; ``_noise_last`` tracks each
    qubit's last-touched time.

    Constructor noise kwargs (all default to ideal):
        f_1q (float): 1-qubit gate fidelity -- after each 1q gate, w.p. (1-f_1q) a
            random X/Y/Z on that qubit.
        f_2q (float): 2-qubit gate fidelity -- after each 2q gate, w.p. (1-f_2q) a
            random non-identity 2-qubit Pauli.
        f_m (float): measurement/readout fidelity -- each reported bit flips w.p. (1-f_m).
        t1 (float | None): amplitude-damping (relaxation) time in seconds (None = off).
        t2 (float | None): dephasing time in seconds (None = off).
    """

    _PAULIS = ("x", "y", "z")

    def __init__(self, seed: int = None, **kwargs):
        super().__init__()
        self.noise_rng = np.random.default_rng(seed)
        self.f_1q: float = self._validate(kwargs.get("f_1q", 1.0))
        self.f_2q: float = self._validate(kwargs.get("f_2q", 1.0))
        self.f_m: float = self._validate(kwargs.get("f_m", 1.0))
        self.t1 = kwargs.get("t1", None)
        self.t2 = kwargs.get("t2", None)
        # per-key overrides (populated by DQCNode registration): key -> (f_1q, f_2q, f_m, t1, t2)
        self.key_noise: dict[int, tuple] = {}
        # sim-time handle + last-touched times for idle T1/T2 (timeline set by the Timeline)
        self.timeline = None
        self._noise_last: dict[int, int] = {}

    # ── noise configuration + helpers ────────────────────────────────────────
    @staticmethod
    def _validate(value: float) -> float:
        """Return a fidelity after checking it lies in [0, 1]; raise otherwise."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Fidelity must be between 0 and 1, inclusive.")
        return value

    def _noise_enabled(self) -> bool:
        """True if any global knob is non-ideal or any per-key noise is registered."""
        return (self.f_1q < 1.0 or self.f_2q < 1.0 or self.f_m < 1.0
                or self.t1 is not None or self.t2 is not None or bool(self.key_noise))

    def set_key_noise(self, key: int, f_1q: float = 1.0, f_2q: float = 1.0,
                      f_m: float = 1.0, t1: float = None, t2: float = None) -> None:
        """Register per-qubit noise for ``key`` (overrides the global knobs)."""
        self.key_noise[key] = (self._validate(f_1q), self._validate(f_2q),
                               self._validate(f_m), t1, t2)

    def _key_params(self, key: int) -> tuple:
        """(f_1q, f_2q, f_m, t1, t2) for ``key`` -- its registered params, else global."""
        p = self.key_noise.get(key)
        return p if p is not None else (self.f_1q, self.f_2q, self.f_m, self.t1, self.t2)

    def _rand_pauli(self) -> str:
        return self._PAULIS[int(self.noise_rng.integers(3))]

    def _inject_pauli(self, key: int, p: str) -> None:
        """Apply a single Pauli ``p`` on ``key`` through the ideal circuit path."""
        from ...components.circuit import Circuit
        c = Circuit(1); getattr(c, p)(0); self._run_ideal(c, [key])

    @staticmethod
    def _t_phi(t1, t2):
        """Pure-dephasing time from T1, T2 (1/Tphi = 1/T2 - 1/2T1). None if T2 off."""
        if t2 is None:
            return None
        if t1 is None:
            return t2
        inv = 1.0 / t2 - 1.0 / (2.0 * t1)
        return (1.0 / inv) if inv > 0 else math.inf

    def _apply_idle(self, keys: list[int]) -> None:
        """Apply idle T1/T2 decoherence to each key over its idle time since last touch."""
        tl = self.timeline
        if tl is None:
            return
        now = tl.now()
        for key in keys:
            idle = now - self._noise_last.get(key, now)
            if idle > 0:
                _, _, _, t1, t2 = self._key_params(key)
                if t1 is not None or t2 is not None:
                    self._decohere(key, idle, t1, t2)
        for key in keys:
            self._noise_last[key] = now

    def _decohere(self, key: int, idle_ps: float, t1, t2) -> None:
        # T1 as a stochastic RELAXATION through the SANCTIONED measurement path (never a
        # manual amplitude edit, which corrupts entanglement bookkeeping): w.p. gamma
        # collapse-measure the qubit and, if excited, drop it to |0>. T2 is a stochastic Z.
        from ...components.circuit import Circuit
        idle_s = idle_ps * 1e-12
        if idle_s <= 0:
            return
        if t1 is not None:
            gamma = 1.0 - math.exp(-idle_s / t1)
            if gamma > 0 and self.noise_rng.random() < gamma:
                c = Circuit(1); c.measure(0)
                res = self._run_ideal(c, [key], self.noise_rng.random())
                if res.get(key) == 1:
                    self._inject_pauli(key, "x")
        t_phi = self._t_phi(t1, t2)
        if t_phi is not None and t_phi != math.inf:
            p_z = 0.5 * (1.0 - math.exp(-idle_s / t_phi))
            if self.noise_rng.random() < p_z:
                self._inject_pauli(key, "z")

    def _apply_gate_noise(self, circuit: "Circuit", keys: list[int]) -> None:
        """Depolarize after each gate, using the owning node's fidelity per qubit."""
        for _name, indices, _arg in circuit.gates:
            gk = [keys[i] for i in indices]
            if len(gk) == 1:
                f_1q = self._key_params(gk[0])[0]
                if f_1q < 1.0 and self.noise_rng.random() > f_1q:
                    self._inject_pauli(gk[0], self._rand_pauli())
            elif len(gk) == 2:
                # both keys are normally co-located (telegate does the CX locally); if they
                # somehow differ, use the noisier node's 2q fidelity.
                f_2q = min(self._key_params(gk[0])[1], self._key_params(gk[1])[1])
                if f_2q < 1.0 and self.noise_rng.random() > f_2q:
                    while True:
                        a, b = int(self.noise_rng.integers(4)), int(self.noise_rng.integers(4))
                        if a or b:
                            break
                    if a:
                        self._inject_pauli(gk[0], self._PAULIS[a - 1])
                    if b:
                        self._inject_pauli(gk[1], self._PAULIS[b - 1])

    def _apply_readout_noise(self, result: dict[int, int]) -> None:
        """Flip each reported measurement bit w.p. (1 - f_m) of its owning qubit."""
        for key in list(result.keys()):
            f_m = self._key_params(key)[2]
            if f_m < 1.0 and self.noise_rng.random() > f_m:
                result[key] ^= 1

    def new(self, state: OneDimensionInput = (complex(1), complex(0))) -> int:
        """Method to create a new ket state.

        Args:
            state (OneDimensionInput): 1D state-vector amplitudes.

        Returns:
            int: the key of the new state.
        """
        key = self._least_available
        self._least_available += 1
        self.states[key] = KetState(state, [key])
        return key

    def run_circuit(self, circuit: Circuit, keys: list[int], meas_samp: float = None) -> dict[int, int]:
        """Apply a circuit to the qubits named by ``keys`` (with trajectory noise if enabled).

        When no noise is configured this is exactly the ideal contraction path. When
        noise is enabled it wraps that path with idle T1/T2 (before the op), post-gate
        depolarizing, and readout flips -- each resolved by the qubit's owning node.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): keys of the qubits, in circuit-qubit order.
            meas_samp (float): random sample in [0, 1) for measurement (required if the
                               circuit measures any qubit).

        Returns:
            dict[int, int]: mapping of each measured key to its outcome (empty if none).
        """
        if not self._noise_enabled():
            return self._run_ideal(circuit, keys, meas_samp)
        self._apply_idle(keys)                          # idle T1/T2 before the op
        result = self._run_ideal(circuit, keys, meas_samp)
        self._apply_gate_noise(circuit, keys)           # post-gate depolarizing
        if result:
            self._apply_readout_noise(result)           # readout flips
        return result

    def _run_ideal(self, circuit: Circuit, keys: list[int], meas_samp: float = None) -> dict[int, int]:
        """Apply a circuit to the qubits named by `keys` (ideal, noiseless path).

        Each gate is applied by tensor contraction. This costs O(2^(k+m)) = O(2^k) ,
        where k is the number of qubits in the combined state and m is the number of qubits the gate acts on (1 or 2).

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): keys of the qubits to apply the circuit to, 
                              in circuit-qubit order; may span several separate KetStates.
            meas_samp (float): random sample in [0, 1) used for measurement; 
                               required when the circuit measures any qubit.

        Returns:
            dict[int, int]: mapping of each measured key to its outcome, or an
                            empty dict when the circuit performs no measurement.
        """
        unsupported = [g[0] for g in circuit.gates if g[0] not in _SUPPORTED]
        if unsupported:
            raise NotImplementedError(f"QuantumManagerKet.run_circuit received unsupported gate(s): {unsupported}")
        validate_circuit_run(circuit, keys, meas_samp)

        # Combine distinct KetStates and record their qubit order. Deduplicate by object identity if needed.
        old_states, all_keys, seen = [], [], set()
        for key in keys:
            qstate = self.states[key]
            if id(qstate) not in seen:             # skip states already pulled in
                seen.add(id(qstate))
                old_states.append(qstate.state)
                all_keys += qstate.keys
        state = np.array([1], dtype=complex)
        for s in old_states:
            state = np.kron(state, s)

        # Apply each gate by contraction
        k = len(all_keys)
        tensor = state.reshape((2,) * k)           # flat 2^k vector -> k-axis grid
        key_to_axis = {key: i for i, key in enumerate(all_keys)}
        for name, indices, arg in circuit.gates:
            # Map circuit key positions to tensor axes in gate-qubit order.
            axes = [key_to_axis[keys[i]] for i in indices]
            tensor = _apply(tensor, axes, _gate_matrix(name, arg))
        new_state = tensor.reshape(-1)             # back to a flat 2^k vector

        if len(circuit.measured_qubits) == 0:
            # No measurement: create a new KetState
            new_ket = KetState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_ket
            return {}
        else:
            # Measurement: collapse the state and return the outcomes
            meas_keys = [keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, meas_keys, all_keys, meas_samp)

    def set(self, keys: list[int], amplitudes: OneDimensionInput) -> None:
        """Set the quantum state for the given keys.

        Args:
            keys (list[int]): list of keys of the quantum state.
            amplitudes (OneDimensionInput): amplitudes to set the state to.
        """
        new_state = KetState(amplitudes, keys)
        for key in keys:
            self.states[key] = new_state

    def get_ascending_keys(self, key: int) -> KetState:
        """Method to get quantum state stored at an index.
           Reorders qubits (in-place) in ascending order of keys before returning.

        Args:
            key (int): key for quantum state.

        Returns:
            KetState: quantum state at supplied key.
        """
        state = super().get(key)
        self.reorder_qubits_ascending_keys(state)
        return state

    def reorder_qubits_ascending_keys(self, state: KetState) -> None:
        """Update the quantum state (in-place) to match the ascending order of keys.
           Meanwhile, the reordered state is also set in the quantum manager.
        
        Args:
            state (KetState): The quantum state to reorder.
        """
        target_all_keys = sorted(state.keys)
        if state.keys != target_all_keys:
            _, swap_matrix = swap_qubits(state.keys, target_all_keys)
            reordered_state = swap_matrix @ state.state
            state.state = reordered_state
            self.set(target_all_keys, reordered_state.tolist())

    def set_to_zero(self, key: int) -> None:
        """Set the qubit at the given key to the |0> state.

        Args:
            key (int): key of the qubit to set to |0>.
        """
        self.set([key], [complex(1), complex(0)])

    def set_to_one(self, key: int) -> None:
        """Set the qubit at the given key to the |1> state.

        Args:
            key (int): key of the qubit to set to |1>.
        """
        self.set([key], [complex(0), complex(1)])

    def _measure(self, state: np.ndarray, keys: list[int], all_keys: list[int], meas_samp: float) -> dict[int, int]:
        """Measure `keys` and collapse their shared state.

        The measured axes are flattened into 2^m outcomes, where m is the number of measured qubits and
        `keys[0]` is the most-significant bit. Computing branch probabilities costs O(2^k) for k total qubits,
        avoiding O(4^k) projectors. Measured keys become basis states, while remaining qubits receive the
        normalized sampled branch.

        Args:
            state (np.ndarray): flat amplitudes ordered by `all_keys`.
            keys (list[int]): keys to measure in outcome bit order.
            all_keys (list[int]): qubit order of `state`.
            meas_samp (float): random sample in [0, 1) selecting the outcome.

        Returns:
            dict[int, int]: mapping of each measured key to its outcome (0 or 1).
        """
        m = len(keys)
        k = len(all_keys)
        arr = np.asarray(state, dtype=complex).reshape((2,) * k)

        # Group measured axes into rows in `keys` order; `keys[0]` is the most-significant outcome bit.
        meas_axes = [all_keys.index(key) for key in keys]
        t = np.moveaxis(arr, meas_axes, range(m)).reshape(2 ** m, -1)

        # Compute Born probabilities and sample from their cumulative distribution.
        probs = np.sum(t.conjugate() * t, axis=1).real
        cumulative_probs = np.cumsum(probs)
        cumulative_probs[-1] = 1.0  # Prevent round-off from excluding the final outcome.
        result = int(np.searchsorted(cumulative_probs, meas_samp, side="right"))
        outcome = bin(result)[2:].zfill(m)  # Convert to an m-bit binary string.
        bits = [int(bit) for bit in outcome]

        # Reassign each measured key to its basis state
        basis = (np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))
        for key, bit in zip(keys, bits):
            self.states[key] = KetState(basis[bit], [key])
        # Reassign remaining qubits (if any) to the renormalized post-measurement branch.
        rem_keys = [key for key in all_keys if key not in keys]
        if rem_keys:
            new_state = (t[result] / np.sqrt(probs[result])).reshape(-1)
            new_obj = KetState(new_state, rem_keys)
            for rem in rem_keys:
                self.states[rem] = new_obj
        return dict(zip(keys, bits))
