#!/usr/bin/env python3
"""Run the Amaro-generated hybrid compiler across several DQC instances: Grover of
different sizes on different QPU counts, sweeping the TeleGate price. Prints a table of
the telegate/teledata mix and cost the generated compiler produces for each."""
import itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_dqc_inputs as gen
import run_hybrid as rh

SPEC = os.path.join(HERE, "dqc_hybrid.qmrl")
TELEPORT = 2.0

# (n_data, qpus) instances x link_cost values
INSTANCES = [(4, 2), (5, 2), (6, 2), (4, 3), (6, 3), (4, 4)]
LINKS = [1.0, 2.0, 3.0]

print(f"{'n_data':6} {'qubits':6} {'CX':4} {'QPUs':4} {'link':4} "
      f"{'TG':>3} {'TD':>3} {'cost':>6} {'regime':>12}")
for (n, q), lc in itertools.product(INSTANCES, LINKS):
    marked = (1 << n) - 1
    qasm = os.path.join(HERE, f"grover_n{n}.qasm")
    arch = os.path.join(HERE, "arch_grover.json")
    N, ncx = gen.emit_qasm(n, marked, qasm)
    cap = -(-N // q)
    gen.emit_arch(N, q, cap, lc, TELEPORT, arch)
    res = rh.analyze(rh.run(SPEC, qasm, arch, "--amaro"), arch)
    tg, td = res["telegates"], res["teleports"]
    regime = "pure TG" if td == 0 else ("pure TD" if tg == 0 else "hybrid")
    print(f"{n:6} {N:6} {ncx:4} {q:4} {lc:<4.0f} {tg:>3} {td:>3} "
          f"{res['cost']:>6.0f} {regime:>12}", flush=True)
