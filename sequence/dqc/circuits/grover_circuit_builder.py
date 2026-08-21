#!/usr/bin/env python3
"""General Grover circuit builder (topology-neutral base).

Builds an n-data-qubit Grover search circuit for distributed quantum computing,
for ANY number of data qubits, using ONLY CX as the multi-qubit gate (all other
operations are synthesised from CX and single-qubit gates). This class knows
NOTHING about network topology — it just produces the circuit.

Qubit layout (total = n_data + n_ancilla qubits):
  data     = [0 .. n_data-1]                       (the search qubits)
  ancillas = [n_data .. n_data + n_ancilla - 1]    (workspace for the MCZ V-chain)
  n_ancilla = max(0, n_data - 2)

The oracle/diffuser n-controlled-Z (phase flip on |1...1>) is a V-chain of
Toffolis through the ancillas (a0 = d0&d1, a_{i-1} = a_{i-2}&d_i, ..., CZ, then
uncompute). Only controlled gate used: CX; single-qubit gates: h, x, t, phase.

This builds ONLY the logical circuit. Placement onto physical nodes and the
network itself are handled downstream by :mod:`topology` (the physical network,
from config) and :mod:`dqc_run` (the compiler: placement/routing + scheduling).
"""

from __future__ import annotations

import numpy as np
from sequence.components.circuit import Circuit


class GroverCircuitBuilder:
    """Topology-neutral n-data-qubit Grover circuit builder. Produces just the
    logical circuit via :meth:`build_grover_circuit`; physical placement and the
    network are the compiler's (:mod:`dqc_run`) and topology's (:mod:`topology`)
    concern."""

    def __init__(self, n_data: int, marked_items: list, iterations: int = 1,
                 verbose: bool = False, total_qubits: int | None = None):
        """
        Args:
            n_data: Number of data (search) qubits. Must be >= 1.
            marked_items: List of marked items; the first is used.
            iterations: Number of Grover iterations.
            verbose: If True, print a per-gate compilation listing.
            total_qubits: Optionally size the circuit to this many qubits (>= the
                data+ancilla count); extra qubits are idle. Defaults to data+ancilla.
        """
        if n_data < 1:
            raise ValueError(f"n_data must be >= 1, got {n_data}.")

        self.n_data = n_data
        self.n_ancilla = max(0, n_data - 2)
        needed = n_data + self.n_ancilla
        if total_qubits is not None and total_qubits < needed:
            raise ValueError(
                f"total_qubits={total_qubits} is too small for {n_data} data "
                f"qubits (needs at least {needed}).")
        self.n_qubits = total_qubits if total_qubits is not None else needed
        self.marked_items = marked_items
        self.iterations = iterations
        self.verbose = verbose

    @classmethod
    def ancillas_for(cls, n_data: int) -> int:
        """Number of workspace ancillas required for `n_data` data qubits."""
        return max(0, n_data - 2)

    @classmethod
    def total_qubits_for(cls, n_data: int) -> int:
        """Total qubits (data + ancilla) required for `n_data` data qubits."""
        return n_data + cls.ancillas_for(n_data)


    # ─────────────────────────── the circuit ───────────────────────────
    def _apply_cz(self, circ, ctrl: int, targ: int) -> None:
        """Emit a CZ(ctrl, targ). Default: decompose to H·CX·H, so the only 2-qubit
        gate is CX and every remote gate is a teleported CX. Override (see
        CZGroverCircuitBuilder) to emit a *direct* CZ -> a teleported CZ."""
        circ.h(targ)
        circ.cx(ctrl, targ)
        circ.h(targ)

    def build_grover_circuit(self) -> Circuit:
        """Build and return the full Grover circuit (data qubits q0..q(n_data-1))."""
        n_data = self.n_data
        N = self.n_qubits
        iters = self.iterations
        pi = np.pi

        marked = self.marked_items[0] if self.marked_items else 0
        max_state = (1 << n_data) - 1
        if not (0 <= marked <= max_state):
            raise ValueError(
                f"marked item {marked} out of range for {n_data} data qubits "
                f"(valid 0..{max_state}).")

        data = list(range(n_data))
        anc = list(range(n_data, n_data + self.n_ancilla))
        circ = Circuit(N)

        # ---- Helpers (CX is the only multi-qubit gate) ----
        def T(q: int):
            circ.t(q)

        def Tdg(q: int):
            circ.phase(q, -pi / 4)

        # Standard Toffoli decomposition, controls=(a,b) -> target=t, CX-only.
        def toffoli(a: int, b: int, t: int):
            circ.h(t)
            circ.cx(b, t)
            Tdg(t)
            circ.cx(a, t)
            T(t)
            circ.cx(b, t)
            Tdg(t)
            circ.cx(a, t)
            T(b)
            T(t)
            circ.cx(a, b)
            circ.h(t)
            T(a)
            Tdg(b)
            circ.cx(a, b)

        # n_data-controlled-Z over `data`, phase-flipping |1...1>. The 2-qubit
        # phase flip is emitted via self._apply_cz (override to change CX vs CZ).
        def mcz():
            m = n_data
            if m == 1:
                circ.phase(data[0], pi)              # single-qubit Z
                return
            if m == 2:
                self._apply_cz(circ, data[0], data[1])
                return
            # Forward V-chain: a0 = d0 & d1, a_{i-1} = a_{i-2} & d_i.
            toffoli(data[0], data[1], anc[0])
            for i in range(2, m - 1):
                toffoli(anc[i - 2], data[i], anc[i - 1])
            self._apply_cz(circ, anc[m - 3], data[m - 1])   # phase flip iff all-ones
            for i in reversed(range(2, m - 1)):      # uncompute
                toffoli(anc[i - 2], data[i], anc[i - 1])
            toffoli(data[0], data[1], anc[0])

        # 1) Prepare |+> on data
        for q in data:
            circ.h(q)

        bits = [(marked >> i) & 1 for i in range(n_data)]   # little-endian

        for _ in range(iters):
            # ORACLE: X-mask so |marked> -> |1...1>, phase-flip, unmask.
            for q, b in zip(data, bits):
                if b == 0:
                    circ.x(q)
            mcz()
            for q, b in zip(data, bits):
                if b == 0:
                    circ.x(q)
            # DIFFUSION: H - (X-MCZ-X reflection about |0...0>) - H.
            for q in data:
                circ.h(q)
            for q in data:
                circ.x(q)
            mcz()
            for q in data:
                circ.x(q)
            for q in data:
                circ.h(q)

        if self.verbose:
            print(f"=== CIRCUIT ({n_data}-data Grover, {N} qubits, {iters} iters, "
                  f"marked={marked}) : {len(circ.gates)} gates ===")

        return circ
