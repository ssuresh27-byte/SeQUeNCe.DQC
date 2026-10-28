"""Base class for op-generation -- group a placed circuit into per-node local/remote ops.

Takes a placement ({qubit: node}) + circuit and produces the per-node operations (local /
remote) via ``compile_circuit``. It does NOT decide the execution layering: that is the
op-DAG's canonical ASAP layering (see :func:`sequence.dqc.program.build_op_dag`). Placement
(WHERE each qubit lives) is a separate concern handled by :mod:`compilers`.
"""
from typing import Dict, List

from ..circuit_ops import gate_fields


class OpGeneratorBase:
    """Shared slot bookkeeping and per-node operation grouping for a placed circuit.

    Subclasses implement ``compile_circuit`` to generate operations and provisional
    layers, using ``_group_ops_by_node`` to group them by node and role. Placement is supplied
    separately; the operation DAG determines the final execution layers.

    ``qubit_to_node`` identifies which node owns each logical qubit. ``data_owners``
    is a two-layer dictionary that groups qubits by node and records their local
    memory slots, rather than simply reversing ``qubit_to_node``. For example,
    placement ``{0: "alice", 1: "bob", 2: "alice"}`` gives
    ``{"alice": {0: 0, 2: 1}, "bob": {1: 0}}``.

    ``node_ops`` describes the work each node executes, grouped as
    ``{node_name: {"local": [op_dict, ...], "remote": [op_dict, ...]}}``.
    Local gates act entirely on one node; remote gates span nodes and are copied
    into every participating node's remote operation list. Each operation records its
    gate, target qubits, provisional layer/step, argument, and participating nodes.

    Attributes:
        n_qubits (int): Number of logical qubits in the circuit.
        n_nodes (int): Number of computing nodes.
        memory_capacity (int): Per-node memory capacity supplied to the generator.
        node_names (List[str]): Computing-node names.
        qubit_to_node (Dict[int, str]): Logical qubit to owning node.
        data_owners (Dict[str, Dict[int, int]]): Node to logical-qubit-to-local-slot mapping;
                                                 ``data_owners[node_name][logical_qubit]`` is the local slot.
        node_ops (Dict[str, Dict[str, list]]): Node to role to operation-list mapping;
                                              roles are ``"local"`` and ``"remote"``.
    """

    def __init__(self, n_qubits: int, n_nodes: int, memory_capacity: int,node_names: List[str] = None):
        self.n_qubits = n_qubits
        self.n_nodes = n_nodes
        self.memory_capacity = memory_capacity
        self.node_names = node_names or ["alice", "bob", "charlie"]
        self.qubit_to_node: Dict[int, str] = {}
        self.data_owners: Dict[str, Dict[int, int]] = {}
        self.node_ops: Dict[str, Dict[str, list]] = {}

    def get_data_owners(self, qubit_to_node: Dict[int, str]) -> Dict[str, Dict[int, int]]:
        """{node_name: {global_qubit: local_slot}} for a static placement.
            placement -> per-node data-slot map (FGP overrides: slot == qubit-id)
        """
        data_owners = {}
        for nm in self.node_names:
            data_owners[nm] = {}
            for i, q in enumerate([q for q, n in qubit_to_node.items() if n == nm]):
                data_owners[nm][q] = i
        return data_owners

    def compile_circuit(self, circuit) -> Dict[str, Dict[str, list]]:
        """Return {node: {role: [op_dict]}} with a layer/step on each op."""
        raise NotImplementedError

    def _group_ops_by_node(self, raw, layers: Dict[int, int]) -> Dict[str, Dict[str, list]]:
        """Group gates by owning node and local/remote role using supplied layers.

        Subclasses provide provisional gate-index-to-layer mappings. The operation
        DAG later determines the final execution layers.
        """
        self.node_ops = {nm: {"local": [], "remote": []} for nm in self.node_names}
        for i, op in enumerate(raw):
            name, qs, arg = gate_fields(op)
            involved = {self.qubit_to_node[q] for q in qs}
            info = {"layer": layers[i], "step": layers[i], "gate": name.lower(),
                    "targets": list(qs), "arg": arg, "nodes": sorted(involved)}
            if len(involved) == 1:
                self.node_ops[next(iter(involved))]["local"].append(info)
            else:  # remote: recorded on both control and target nodes
                for nm in involved:
                    self.node_ops[nm]["remote"].append(info.copy())
        return self.node_ops
