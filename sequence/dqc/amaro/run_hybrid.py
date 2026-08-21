#!/usr/bin/env python3
"""Run a generated Amaro solver and parse the MIN-COST result across its threads.
Reports #telegates (remote gates), #teledata teleports (cross-QPU transitions),
#intra-QPU swaps, and total cost.

The Amaro Rust tool lives in a separate clone (it needs cargo/rust). Point to it
with the AMARO_REPO env var; defaults to ~/Desktop/qmr-compiler-generator.

    python amaro_dqc/run_hybrid.py <spec.qmrl> <circuit.qasm> <arch.json> [--amaro|--sabre|--onepass]
"""
import json, subprocess, sys, os, re

AMARO_REPO = os.environ.get(
    "AMARO_REPO", os.path.expanduser("~/Desktop/qmr-compiler-generator"))


def run(spec, qasm, arch, mode="--amaro"):
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    # spec/qasm/arch may be given relative to CWD; make absolute for the subprocess
    spec, qasm, arch = (os.path.abspath(p) for p in (spec, qasm, arch))
    out = subprocess.run(["./amaro", "run", spec, qasm, arch, mode],
                         cwd=AMARO_REPO, capture_output=True, text=True, env=env)
    results = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try: results.append(json.loads(line))
            except json.JSONDecodeError: pass
    if not results:
        print("NO RESULT. Is the spec compiled in the Amaro repo?\n"
              f"  AMARO_REPO={AMARO_REPO}\n  stderr:", out.stderr[:500]); sys.exit(1)
    return min(results, key=lambda r: r.get("cost", 1e18))


def analyze(res, arch_path):
    arch = json.load(open(arch_path)); qpu = arch["qpu_ids"]
    telegates = swaps = teleports = 0
    for st in res["steps"]:
        for g in st["implemented_gates"]:
            if g["implementation"].get("remote"): telegates += 1
    for t in res.get("transitions", []):        # "CustomTransition { edge: (Location(a), Location(b)) }"
        m = re.search(r"Location\((\d+)\).*?Location\((\d+)\)", t)
        if not m: continue
        a, b = int(m.group(1)), int(m.group(2))
        if a == b: continue                      # null transition
        if qpu[a] == qpu[b]: swaps += 1
        else: teleports += 1
    return dict(cost=res.get("cost"), telegates=telegates, teleports=teleports,
                swaps=swaps, steps=len(res["steps"]))


if __name__ == "__main__":
    spec, qasm, arch = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "--amaro"
    print(analyze(run(spec, qasm, arch, mode), arch))
