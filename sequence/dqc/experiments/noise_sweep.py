#!/usr/bin/env python3
"""Coordinated z-sweep: interpolate the four fidelity knobs from the noisy endpoint
(z=0: F_1q=0.999, F_2q=0.9991, F_m=0.996, F_phys=0.990) to perfect (z=1.5: all 1.0)
and plot distributed-Grover success probability vs z (trajectory noise, ket vector).

T1/T2 are not modeled yet, so this coordinates only F_1q/F_2q/F_m/F_phys.
"""
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run
from sequence.dqc.noise import NoiseConfig

# n=6, optimal iterations (~6) so the NOISELESS success is ~1.0 -- then noise pulls
# it down and the degradation is clearly visible (vs 1-iteration, which tops out ~0.46).
N_DATA, MARKED = 6, 63
ITERS = round(math.pi / 4 * math.sqrt(2 ** N_DATA))   # = 6
SHOTS = 200
Z = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
ENDPOINT = dict(f_1q=0.999, f_2q=0.9991, f_m=0.996, f_phys=0.990)   # z = 0 (noisiest)


def fids(z):
    t = z / 1.5                                   # 0 at z=0, 1 at z=1.5 (perfect)
    return {k: v + (1.0 - v) * t for k, v in ENDPOINT.items()}


def main():
    circ = GroverCircuitBuilder(N_DATA, [MARKED], ITERS).build_grover_circuit()
    T = topo.make_grid(2, 3, 3)
    print(f"Grover n={N_DATA}, {ITERS} iterations (noiseless ~1.0), grid 2x3, {SHOTS} shots/point\n")
    print(f"{'z':>5} {'F_1q':>8} {'F_2q':>8} {'F_m':>8} {'F_phys':>8} {'success':>8}")
    zs, ps = [], []
    for z in Z:
        f = fids(z)
        r = run(circ, T, partitioner="topo-aware", scheduler="fgp",
                data_qubits=range(N_DATA), expected=MARKED,
                noise=NoiseConfig(**f), shots=SHOTS)
        p = r["success_prob"]
        zs.append(z); ps.append(p)
        print(f"{z:>5.2f} {f['f_1q']:>8.4f} {f['f_2q']:>8.4f} {f['f_m']:>8.4f} "
              f"{f['f_phys']:>8.4f} {p:>8.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(zs, ps, "o-", color="#4C72B0")
    ax.set_xlabel("z  (coordinated fidelity sweep; z=0 noisiest, z=1.5 perfect)")
    ax.set_ylabel("distributed-Grover success probability")
    ax.set_title(f"Success probability vs z  (n={N_DATA}, {ITERS} Grover iters, {SHOTS} shots/point)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(0.5, max(ps) * 1.15))
    fig.tight_layout()
    fig.savefig("noise_sweep.png", dpi=130)
    print("\nwrote noise_sweep.png")


if __name__ == "__main__":
    main()
