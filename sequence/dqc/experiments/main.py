#!/usr/bin/env python3
"""Entry point: pick a CIRCUIT and a TOPOLOGY, run it on the distributed simulator.

Edit the two lines below (the circuit builder and the topology). The central
controller -- a real node in the network -- compiles the program (a partitioner +
scheduler pipeline, or the monolithic FGP hybrid) and drives the barrier over
classical channels.

Partitioners (placement):  topo-aware | qap | random
Schedulers   (execution):  packed | serial | alap | resource-aware
    ... or scheduler="fgp" for the monolithic time-sliced telegate+teledata hybrid
        (which does placement+scheduling together, seeded by the partitioner).
"""
import logging
logging.disable(logging.CRITICAL)   # quiet the SeQUeNCe INFO logging

from sequence.dqc.circuits.grover_circuit_builder import GroverCircuitBuilder
import sequence.dqc.topology as topo
from sequence.dqc.runtime import run
from sequence.dqc.noise import NoiseConfig

# ── the two user inputs ──────────────────────────────────────────────────────
N_DATA, MARKED = 4, 15
circuit = GroverCircuitBuilder(N_DATA, [MARKED]).build_grover_circuit()   # 1) circuit
# per-node local noise lives on each DQCNode (inline in the sim-config JSON, like the
# gate/measurement fidelities SeQUeNCe puts on quantum routers). Absent nodes = ideal.
topology = topo.make_grid(2, 2, 3,                                        # 2) topology
                          node_noise={"bob": {"f_1q": 0.999, "f_2q": 0.99, "f_m": 0.996}})

# ── noiseless single run (measured / ok); dump the exact sim-config JSON to see ──
result = run(circuit, topology, partitioner="topo-aware", scheduler="fgp",
             data_qubits=range(N_DATA), expected=MARKED,
             dump_config="sim_config_dump.json")
print("noiseless:", {k: result[k] for k in ("ok", "measured", "sim_ms")})

# ── noisy run: stochastic-Pauli trajectory noise in the ket vector ───────────
# F_1q / F_2q = 1q/2q gate fidelity, F_m = measurement fidelity, F_phys = physical
# Bell-pair fidelity. Runs `shots` trajectories and reports the success probability.
noise = NoiseConfig(f_1q=0.999, f_2q=0.99, f_m=0.996, f_phys=0.99)
result = run(circuit, topology, partitioner="topo-aware", scheduler="fgp",
             data_qubits=range(N_DATA), expected=MARKED, noise=noise, shots=500)
print("noisy:    ", {k: result[k] for k in ("success_prob", "shots", "successes")})
