#!/usr/bin/env python3
"""
visualize_cells.py

Read a model encoding JSON and render the normal and reduction cells as graphs.
Generates PNG files using Graphviz (needs system `dot` available).
"""
import argparse
import json
import os
from graphviz import Digraph


def src_label(idx: int) -> str:
    """Map source index to label: 0->Input_0,1->Input_1,2->Node_0,..."""
    if idx == 0:
        return "Input_0"
    if idx == 1:
        return "Input_1"
    return f"Node_{idx-2}"


def draw_cell(name: str, edges: list, concat: list, out_dir: str):
    edges_per_node = 2
    num_nodes = len(edges) // edges_per_node
    g = Digraph(name=name)
    g.attr(rankdir='LR')

    # add input nodes
    g.node('Input_0', shape='box')
    g.node('Input_1', shape='box')

    # add intermediate nodes
    for i in range(num_nodes):
        nid = f"Node_{i}"
        g.node(nid, shape='circle')

    # add edges with operation labels
    for i in range(num_nodes):
        to_node = f"Node_{i}"
        for j in range(edges_per_node):
            op, src = edges[i * edges_per_node + j]
            frm = src_label(src)
            g.edge(frm, to_node, label=op)

    # concat/output node
    out_node = 'Concat'
    g.node(out_node, shape='doublecircle')
    for c in concat:
        frm = src_label(c)
        g.edge(frm, out_node)

    os.makedirs(out_dir, exist_ok=True)
    # render as PNG
    filename = f"{name}"
    out_path = g.render(filename=filename, directory=out_dir, format='png', cleanup=True)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description='Visualize normal and reduction cells from a JSON encoding')
    p.add_argument('json', nargs='?', default=os.path.join(os.getcwd(), 'model_1702_acc44.48_encoding.json'), help='Path to encoding json')
    p.add_argument('--out', '-o', dest='out_dir', default=os.path.join('scripts', 'out'), help='Output directory')
    args = p.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    genotype = data.get('genotype') or {}
    normal = genotype.get('normal')
    reduce_cell = genotype.get('reduce')
    normal_concat = genotype.get('normal_concat') or data.get('normal_concat') or []
    reduce_concat = genotype.get('reduce_concat') or data.get('reduce_concat') or []

    if not normal or not reduce_cell:
        print('JSON does not contain expected genotype fields (normal/reduce).')
        return

    draw_cell('normal_cell', normal, normal_concat, args.out_dir)
    draw_cell('reduction_cell', reduce_cell, reduce_concat, args.out_dir)


if __name__ == '__main__':
    main()
