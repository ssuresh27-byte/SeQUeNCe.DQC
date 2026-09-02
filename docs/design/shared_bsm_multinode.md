# Allowing a BSM to Connect to Multiple Nodes

**Status:** design note / scoping
**Context:** "general server-centric" topology, where a single BSM is shared by *N > 2*
servers instead of a dedicated 2-port BSM per link.

---

## 1. What "connect to multiple nodes" actually means

A Bell-state measurement (BSM) is physically a **2-photon** device: it has a fixed
detector bank (2 detectors for `single_atom` / `single_heralded`, 4 for
`polarization`) and heralds entanglement between exactly the **two** photons that
land in the same coincidence window.

So "a BSM connected to multiple nodes" does **not** mean it entangles 3+ nodes at
once. It means the BSM is a **shared, time-multiplexed resource**: *N* servers are
each wired to the one BSM, and any **pair** among them can be entangled through it
in a given window. Today SeQUeNCe instead instantiates a separate 2-port BSM for
every pair (one per graph edge).

Concretely, for a 4-leaf hub:

| Model | BSM nodes | Center router? | Leaf↔leaf entanglement |
|-------|-----------|----------------|------------------------|
| **Current (server-centric special)** | 4 (one per edge) | yes — swaps | 2 hops + swap at center |
| **Shared BSM (general server-centric)** | 1 (degree-4) | no | 1 hop through shared BSM |

The good news: the **measurement layer already supports this**. The work is
concentrated in (a) the config/topology generators and (b) the entanglement-
generation *coordination*, plus removing a couple of hard-coded `== 2` assumptions.

---

## 2. What already works

### 2.1 The BSM measurement is location-agnostic
`BSM.get()` (`sequence/components/bsm.py:160-181`) collects photons keyed by
**distinct `photon.location`** within a coincidence window and fires when it holds
two. It does **not** hardcode which two nodes — whichever two distinct locations
arrive together get measured. A degree-*N* BSM is physically fine as long as only
one pair emits per window.

### 2.2 The router→BSM map is per-peer, not per-BSM
`Node.add_bsm_node` (`sequence/topology/node.py:416-423`) stores
`map_to_middle_node[peer_router] = bsm_name`. Many peers can map to the *same* BSM
name, e.g. `{B: hub, C: hub, D: hub}`. No change needed for a router to reach many
peers through one shared BSM.

### 2.3 The generation A-protocol references the BSM by name
`EntanglementGenerationA` (`generation/generation_base.py:56-69`,
`single_heralded.py`, `barret_kok.py`) uses `self.middle` (a BSM name) to index
`owner.qchannels[self.middle]` / `owner.cchannels[self.middle]`. One quantum + one
classical channel from each server to the shared BSM is enough; all sessions
through that BSM reuse them.

---

## 3. The hard-coded "2" assumptions (what blocks it)

These are the concrete places that assume a BSM has exactly two neighbors.

### 3.1 Config generation — emits one BSM **per edge**
- `sequence/utils/nx_converter.py:151-169` — `generate_config` loops over graph
  edges and creates a 2-port `BSM_{a}_{b}` with exactly two quantum channels.
- `sequence/topology/router_net_topo.py:137-174` — `_add_qconnections` only handles
  `MEET_IN_THE_MID`, building `BSM.node1.node2` and looping over `[node1, node2]`.

### 3.2 BSM-side protocol — `assert len(others) == 2`
- `generation/generation_base.py:195` — `EntanglementGenerationB.__init__` asserts
  exactly two `others`.
- `sequence/topology/node.py:240-270` — `BSMNode.__init__` docstring: *"2-member
  list of node names for adjacent quantum routers"*, passes `other_nodes` straight
  into the B protocol.

### 3.3 Result forwarding — broadcasts to a fixed pair
- `generation/single_heralded.py:246-263` — `bsm_update` loops over `self.others`
  (the 2 nodes) and sends `MEAS_RES` to each. With *N* nodes this would notify all
  *N*; only the two that emitted should be told. The `info` dict from the BSM does
  **not** currently carry *which two locations* produced the result.

### 3.4 Routing cost — assumes a BSM bridges exactly two routers
- `sequence/topology/router_net_topo.py:191-198` — `_generate_forwarding_table`
  builds `costs[bsm] = [router0, router1, distance]` and turns each BSM into a
  single weighted edge. A degree-*N* BSM would accumulate *N* routers into one
  entry and produce a malformed graph edge.

---

## 4. Changes required, by layer

### Layer A — Config schema & generators
1. **New connection type** (e.g. `"shared_bsm"` / `"star_bsm"`) describing one BSM
   plus a list of *N* attached nodes, alongside the existing `meet_in_the_middle`.
2. **`nx_converter.generate_config`**: when a node is a hub (e.g. tagged
   `node_type == 'switch'`, or via a `--architecture server-centric` flag), emit a
   **single** `BSMNode` and *N* quantum + classical channels into it, instead of the
   per-edge loop. Do **not** emit the hub as a `QuantumRouter`.
3. **`router_net_topo._add_qconnections`**: handle the new type — create one BSM,
   wire *N* QC/CC channels, and register `map_to_middle_node[peer] = bsm` for every
   pair of attached servers (all-pairs through the hub).

### Layer B — BSM node & B-protocol
4. **Relax `assert len(others) == 2`** (`generation_base.py:195`) to `>= 2`; update
   `BSMNode` (`node.py:240-270`) to accept an arbitrary neighbor list.
5. **Targeted result forwarding**: have the BSM pass the two measured
   `photon.location`s up to `bsm_update`, and have the B-protocol send `MEAS_RES`
   only to those two nodes (not the whole neighbor set). Requires threading source
   identity through `BSM.get` → `trigger` → `bsm_update`'s `info` dict
   (`single_heralded.py:246-263`).

### Layer C — Generation coordination (the contended-resource problem)
6. **Time-multiplexing / arbitration.** A shared BSM can measure only one pair per
   coincidence window. Today each pair independently negotiates an `emit_time`
   (`single_heralded.py:150-174`); with a shared BSM, overlapping windows from
   different pairs would dump 3+ photons into the same detectors and corrupt the
   herald. Options:
   - a scheduler/MAC at the BSM node that grants emission slots, or
   - resource-manager-level serialization so only one session per BSM is active per
     window.
   This is the **main new mechanism**, not just an edit.
7. **Tighten timing gates.** `valid_trigger_time(time, expected_time, resolution)`
   currently filters broadcast results by timing. With targeted forwarding (step 5)
   this becomes a correctness backstop rather than the primary addressing.

### Layer D — Routing / forwarding
8. **`_generate_forwarding_table`** (`router_net_topo.py:191-198`): treat a degree-
   *N* BSM as a clique (or star) of *N* routers when building link costs, instead of
   a single 2-router edge.

### Layer E — Resource management / Rules
9. The rule that builds `EntanglementGenerationA(owner, name, middle, other, memory)`
   already resolves `middle = map_to_middle_node[other]`, so it works for a shared
   BSM. It must, however, cooperate with the Layer-C arbitration so it doesn't start
   colliding sessions on the same BSM.

### Layer F — Tooling (secondary)
10. `sequence/utils/draw_topo.py` and the GUI assume a BSM has two neighbors; update
    for visualization only.

---

## 5. Suggested incremental path

1. **Config first (Layers A + B):** emit a single shared BSM and relax the `== 2`
   assertions. Validate the JSON shape and that the topology loads.
2. **Single-pair correctness:** confirm one pair through the shared BSM still
   generates entanglement end-to-end (no behavior change with one active pair).
3. **Targeted forwarding (Layer B step 5):** thread source locations through so
   `MEAS_RES` reaches the right pair.
4. **Arbitration (Layer C):** add slot scheduling so multiple pairs can share the
   BSM without coincidence-window collisions.
5. **Routing + tooling (Layers D + F):** fix forwarding-table cost and visualization.

Steps 1–3 are mostly mechanical edits to known lines; **step 4 is the real design
work** and is where the "multiple nodes at the same time" semantics are actually
decided (slotted/contended access to a 2-photon device).

---

## 6. Key caveat to keep in mind

A shared BSM is **2-photon hardware**. "Connecting to multiple nodes" buys you a
*shared, time-multiplexed* entanglement resource (1-hop entanglement between any
pair in the cluster, no center router/swap), **not** simultaneous multi-party
entanglement. True concurrency would require multiple detector banks / multiple BSM
units within the hub node — a separate extension.
