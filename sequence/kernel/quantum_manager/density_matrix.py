"""
This module implements the quantum manager for density matrix states.
"""
from __future__ import annotations

from .base import QuantumManager, swap_qubits, validate_circuit_run
from ..quantum_state import DensityState, OneDimensionInput, TwoDimensionInput
from ..quantum_utils import (identity, kron, measure_entangled_state_with_cache_density, 
                             measure_multiple_with_cache_density, measure_state_with_cache_density)
from ...constants import DENSITY_MATRIX_FORMALISM

import numpy as np
from numpy import array
from typing import TYPE_CHECKING
from ...components.circuit import Circuit


@QuantumManager.register(DENSITY_MATRIX_FORMALISM)
class QuantumManagerDensity(QuantumManager):
    """Class to track and manage states with the density matrix formalism.

    Ideal (noiseless). Optional DQC per-node noise is added by SUBCLASSING this manager --
    see :class:`sequence.dqc.noise.DensityMatrixNoise` -- which keeps this base untouched.
    """

    def __init__(self):
        super().__init__()

    def new(self, state: OneDimensionInput | TwoDimensionInput = ((complex(1), complex(0)), (complex(0), complex(0)))) -> int:
        """Method to create a new density matrix state.
        
        Args:
            state (OneDimensionInput | TwoDimensionInput): 2D density matrix or 1D pure-state array.

        Returns:
            int: key of the new state.
        """
        key = self._least_available
        self._least_available += 1
        self.states[key] = DensityState(state, [key])
        return key

    def run_circuit(self, circuit: Circuit, keys: list[int], meas_samp=None) -> dict[int, int]:
        """Method to run a circuit on a given list of keys.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys to apply circuit to.
            meas_samp (float): random number between 0 and 1 used for measurement.

        Returns:
            If measurement, dict[int, int]: dictionary mapping qstate keys to measurement results.
            If non-measurement, dict: empty dictionary.
        """
        validate_circuit_run(circuit, keys, meas_samp)

        new_state, all_keys, circ_mat = self._prepare_circuit(circuit, keys)

        new_state = circ_mat @ new_state @ circ_mat.conj().T

        if len(circuit.measured_qubits) == 0:
            # set state, return no measurement result
            new_state_obj = DensityState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj
            return {}
        else:
            # measure state (state reassignment done in _measure method)
            keys = [all_keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, keys, all_keys, meas_samp)

    def _prepare_circuit(self, circuit: Circuit, keys: list[int]) -> tuple[np.ndarray, list[int], np.ndarray]:
        """Prepare state and circuit matrices for dense-qubit execution.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys for quantum states to apply circuit to.

        Returns:
            tuple: tuple containing the new state, all keys, and the circuit matrix.
                   Note: the returned circuit matrix contains any necessary swaps to align qubits of new state
        """
        old_states = []
        all_keys = []

        # go through keys and get all unique qstate objects
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(qstate.state)
                all_keys += qstate.keys

        # construct compound state; order qubits
        new_state = [1]
        for state in old_states:
            new_state = kron(new_state, state)

        # get circuit matrix; expand if necessary
        circ_mat = circuit.get_unitary_matrix()
        if circuit.size < len(all_keys):
            # pad size of circuit matrix if necessary
            diff = len(all_keys) - circuit.size
            circ_mat = kron(circ_mat, identity(2 ** diff))

        # apply any necessary swaps
        if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
            all_keys, swap_mat = swap_qubits(all_keys, keys)
            circ_mat = circ_mat @ swap_mat

        return new_state, all_keys, circ_mat

    def _merge_state(self, keys: list[int]) -> tuple[np.ndarray, list[int]]:
        """Tensor the distinct density-matrix blocks touched by `keys` into one rho.

        Args:
            keys (list[int]): keys whose (possibly separable) density-matrix blocks
                should be merged into one joint state.

        Returns:
            tuple[np.ndarray, list[int]]: (rho, all_keys) where all_keys is the union
                of every involved state's keys, in block order (mirrors
                _prepare_circuit).
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

    def reduce_to(self, keep_keys: list[int]) -> None:
        """Partial-trace the joint state holding `keep_keys` down to just those keys.

        Detaches keep_keys from any other qubits currently sharing their density matrix
        (e.g. measured comm qubits left entangled after a teleported gate) and
        re-registers the reduced state. No-op if the state already contains only keep_keys.

        Args:
            keep_keys (list[int]): keys to retain; all other qubits sharing their
                joint state are traced out.

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
    def _partial_trace(rho: np.ndarray, keys: list[int], keep: list[int]) -> tuple[np.ndarray, list[int]]:
        """Partial-trace `rho` (ordered by `keys`) down to `keep`.

        Traces out each qubit whose key is not in `keep` by summing its row==col index
        via einsum.

        Args:
            rho (np.ndarray): joint density matrix, ordered by `keys`.
            keys (list[int]): keys labeling rho's qubits, in order.
            keep (list[int]): subset of `keys` to retain.

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

    def set(self, keys: list[int], state: OneDimensionInput | TwoDimensionInput) -> None:
        """Method to set the quantum state at the given keys.

        The state argument may be a 1D pure-state vector or a 2D density matrix.

        Args:
            keys (list[int]): list of quantum manager keys to modify.
            state: quantum state to set input keys to.

        Returns:
            None.
        """
        new_state = DensityState(state, keys)
        for key in keys:
            self.states[key] = new_state

    def set_to_zero(self, key: int):
        """Set the qubit at the given key to the |0><0| state.
        
        Args:
            key (int): key of the qubit to set to |0><0|.

        Returns:
            None.
        """
        self.set([key], [[complex(1), complex(0)], [complex(0), complex(0)]])

    def set_to_one(self, key: int):
        """Set the qubit at the given key to the |1><1| state.
        
        Args:
            key (int): key of the qubit to set to |1><1|.

        Returns:
            None.
        """
        self.set([key], [[complex(0), complex(0)], [complex(0), complex(1)]])

    def get_ascending_keys(self, key: int) -> DensityState:
        """Method to get quantum state stored at an index.
           Reorders qubits (in-place) in ascending order of keys before returning.

        Args:
            key (int): key for quantum state.

        Returns:
            DensityState: quantum state at supplied key.
        """
        state = super().get(key)
        self.reorder_qubits_ascending_keys(state)
        return state

    def reorder_qubits_ascending_keys(self, state: DensityState) -> None:
        """Update the quantum state (in-place) to match the ascending order of keys.
           Meanwhile, the reordered state is also set in the quantum manager.
        
        Args:
            state (DensityState): The quantum state to reorder.

        Returns:
            None.
        """
        target_all_keys = sorted(state.keys)
        if state.keys != target_all_keys:
            _, swap_matrix = swap_qubits(state.keys, target_all_keys)
            reordered_state = swap_matrix @ state.state @ swap_matrix.conj().T
            state.state = reordered_state
            self.set(target_all_keys, reordered_state.tolist())

    def _measure(self, state: list[list[complex]], keys: list[int], all_keys: list[int], meas_samp: float) -> dict[int, int]:
        """Method to measure qubits at given keys.

        SHOULD NOT be called individually; only from circuit method (unless for unit testing purposes).
        Modifies quantum state of all qubits given by all_keys.

        Args:
            state (list[complex]): state to measure.
            keys (list[int]): list of keys to measure.
            all_keys (list[int]): list of all keys corresponding to state.
            meas_samp (float): random number between 0 and 1 used for measurement.

        Returns:
            dict[int, int]: mapping of measured keys to measurement results.
        """

        if len(keys) == 1:
            if len(all_keys) == 1:
                prob_0 = measure_state_with_cache_density(tuple(map(tuple, state)))
                if meas_samp < prob_0:
                    result = 0
                    new_state = [[1, 0], [0, 0]]
                else:
                    result = 1
                    new_state = [[0, 0], [0, 1]]

            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                state_0, state_1, prob_0 = measure_entangled_state_with_cache_density(tuple(map(tuple, state)), state_index, num_states)
                if meas_samp < prob_0:
                    new_state = array(state_0, dtype=complex)
                    result = 0
                else:
                    new_state = array(state_1, dtype=complex)
                    result = 1

        else:
            # swap states into correct position
            if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
                all_keys, swap_mat = swap_qubits(all_keys, keys)
                state = swap_mat @ state @ swap_mat.conj().T

            # calculate meas probabilities and projected states
            len_diff = len(all_keys) - len(keys)
            state_to_measure = tuple(map(tuple, state))
            new_states, probabilities = measure_multiple_with_cache_density(state_to_measure, len(keys), len_diff)

            # choose result, set as new state
            for i in range(int(2 ** len(keys))):
                if meas_samp < sum(probabilities[:i + 1]):
                    result = i
                    new_state = new_states[i]
                    break

        result_digits = [int(x) for x in bin(result)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        new_state_obj = DensityState(new_state, all_keys)
        for key in all_keys:
            self.states[key] = new_state_obj

        return dict(zip(keys, result_digits))