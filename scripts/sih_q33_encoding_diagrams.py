"""Export the SIH-SUS 33-qubit Heisenberg graphs and executable Qiskit circuits."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import rustworkx as rx
from qiskit import qpy
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_coupling_map

from pqfmlib.maps.heisenberg import (
    HeisenbergProjectiveQFM,
    build_heisenberg_chain_feature_circuit,
    make_heisenberg_theta_matrix,
)


def export(out, name, qfm, mode):
    qfm.prepare_layout()
    edges = None if qfm.topology == "chain" else qfm.edges_act
    # Distinct nonzero values label PCA inputs through the production slot mapper.
    values = np.arange(1, 51)[None, :]
    slots, info = make_heisenberg_theta_matrix(
        values, 33, edges_act=edges, encoding_mode=mode, use_tanh_scaling=False,
    )
    blocks = info["num_blocks"]
    circuit, parameters, _ = build_heisenberg_chain_feature_circuit(
        33, num_blocks=blocks, edges_act=edges, encoding_mode=mode,
        R=1, alpha=qfm.alpha,
    )
    schedule = circuit.metadata["pqfmlib_heisenberg"]["edge_schedule"]
    nodes = list(qfm.phys_nodes) if qfm.phys_nodes is not None else list(range(33))
    components = ParameterVector("pc", 51)
    circuit.assign_parameters(
        {p: components[int(slot)] if slot else 0 for p, slot in zip(parameters, slots[0])},
        inplace=True,
    )
    assert set(circuit.parameters) == set(components[1:])
    assert len(schedule) == 32

    coords = [[i % 11 if i // 11 != 1 else 10 - i % 11, -(i // 11)] for i in range(33)]
    if edges is not None:
        graph = rx.PyGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from_no_data([tuple(edge) for edge in schedule])
        layout = rx.spring_layout(graph, seed=42, num_iter=300)
        coords = [list(layout[i]) for i in range(33)]
    figure = plot_coupling_map(
        33, coords, schedule, figsize=(12, 4 if edges is None else 7),
        qubit_labels=[str(q) for q in nodes], planar=False,
    )
    figure.axes[0].set_title(f"33 qubits · {name} · 32 edges")
    figure.savefig(out / f"{name}-topology.png", dpi=160, bbox_inches="tight")

    prefix = out / f"{name}-{mode}"
    with prefix.with_suffix(".qpy").open("wb") as handle:
        qpy.dump(circuit, handle)
    prefix.with_suffix(".txt").write_text(str(circuit.draw(output="text", fold=150)) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block", "logical_a", "logical_b", "physical_a", "physical_b", "XX", "YY", "ZZ"])
        per_edge = 1 if mode == "shared_feature" else 3
        for block, block_slots in enumerate(slots.reshape(blocks, -1), 1):
            for (i, j), feature_slots in zip(schedule, block_slots.reshape(-1, per_edge)):
                features = [f"PC{int(n)}" if n else "0" for n in feature_slots]
                writer.writerow([block, i, j, nodes[i], nodes[j], *(features * (3 // per_edge))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matplotlib.use("Agg")
    args.output.mkdir(parents=True, exist_ok=True)
    export(args.output, "chain", HeisenbergProjectiveQFM(name_file="sih-q33-diagram", q_enc=33, topology="chain", alpha=.2), "shared_feature")
    export(args.output, "chain", HeisenbergProjectiveQFM(name_file="sih-q33-diagram", q_enc=33, topology="chain", alpha=.1), "multi_axis")
    for layout in ("auto", "fixed"):
        qfm = HeisenbergProjectiveQFM(
            name_file="sih-q33-diagram", q_enc=33, topology="native", ideal=False, alpha=.2,
            topology_file=str(args.graphs / "sherbrooke.json"),
            use_fixed_phys_nodes=layout == "fixed",
            fixed_phys_nodes_file=str(args.graphs / "fixed_q33.json"),
        )
        export(args.output, f"native-{layout}", qfm, "shared_feature")


if __name__ == "__main__":
    main()
