from typing import List, Dict, TextIO, Tuple, Iterator
from dataclasses import dataclass, field
from random import random, choice
import os
from copy import deepcopy

import math

from notneat_py.activation import ACTIVATION_FUNCTION_LIST, ActivationFunctionType, activation_functions


def create_graph_id_generator() -> Iterator[int]:
    graph_id = 0
    while True:
        graph_id += 1
        yield graph_id


graph_id_generator = create_graph_id_generator()


def next_graph_id():
    return next(graph_id_generator)


@dataclass
class Graph:
    graph_id: int = field(default_factory=next_graph_id)
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    nodes: List[int] = field(default_factory=list)
    edges: Dict[Tuple[int, int], float] = field(default_factory=dict)
    biases: Dict[int, float] = field(default_factory=dict)
    activation: Dict[int, ActivationFunctionType] = field(default_factory=dict)


def add_new_node(graph: Graph, activation: ActivationFunctionType = ActivationFunctionType.SIGMOID):
    n = graph.nodes[-1] + 1
    graph.nodes.append(n)
    graph.activation[n] = activation
    graph.biases[n] = 0.0
    return n


def topological_sort(graph: Graph):
    def topological_sort_util(graph: Graph, node, visited, stack):
        visited.add(node)
        for (source, target), _ in graph.edges.items():
            if source == node and target not in visited:
                topological_sort_util(graph, target, visited, stack)
        stack.append(node)

    visited = set()
    stack = []
    for node in graph.nodes:
        if node not in visited:
            topological_sort_util(graph, node, visited, stack)
    return stack[::-1]


def crossover(parent1: Graph, parent2: Graph, fitter_parent_bias: float = 0.7, weaker_parent_bias=0.5):
    child_edges = {}
    child_nodes = set()

    # Iterate over both parent edges
    edges = set(parent1.edges.keys()).union(parent2.edges.keys())
    for edge_key in edges:
        # If the edge is present in both parents, randomly pick one based on a bias towards the fitter parent
        if edge_key in parent1.edges and edge_key in parent2.edges:
            parent = parent1 if random() < fitter_parent_bias else parent2
            child_edges[edge_key] = parent.edges[edge_key]
        # If the edge is present only in the fitter parent, inherit it
        elif edge_key in parent1.edges:
            child_edges[edge_key] = parent1.edges[edge_key]
        # If the edge is present only in the second parent, inherit it with a probability of weaker_parent_bias
        elif edge_key in parent2.edges and random() < weaker_parent_bias:
            child_edges[edge_key] = parent2.edges[edge_key]

        # Add nodes for the selected edge
        child_nodes.add(edge_key[0])
        child_nodes.add(edge_key[1])

    child = Graph()
    # get input and output nodes from any parent
    child.inputs = deepcopy(parent1.inputs)
    child.outputs = deepcopy(parent1.outputs)
    # set nodes list
    child.nodes = list(child_nodes)
    # set edges dict
    child.edges = deepcopy(child_edges)
    # copy biases from parent1, if not found, parent2, if not found, set to 0.0
    for node in child.nodes:
        if node not in child.inputs:
            child.biases[node] = parent1.biases.get(node, parent2.biases.get(node, 0.0))
    # get activation functions, prefer parent1. If it doesn't exist, default to sigmoid
    for (f, t), weight in child.edges.items():
        child.activation[t] = parent1.activation.get(t, parent2.activation.get(t, ActivationFunctionType.SIGMOID))
    return child


def evaluate_graph(inputs: List[float], graph: Graph) -> List[float]:
    # Initialize the node values
    node_values: Dict[int, float] = {node: 0.0 for node in graph.nodes}

    # Set the input values
    for i, input_node in enumerate(graph.inputs):
        node_values[input_node] = inputs[i]

    # Forward propagation through the graph
    for node in graph.nodes:
        if node not in graph.inputs:
            weighted_sum = 0.0
            for source_node in graph.nodes:
                if (source_node, node) in graph.edges:
                    weight = graph.edges[(source_node, node)]
                    weighted_sum += node_values[source_node] * weight
            activation_type = graph.activation.get(node)
            if activation_type is None:
                continue
            activation_function = activation_functions[activation_type]
            node_values[node] = activation_function(weighted_sum + graph.biases.get(node, 0.0))

    # let's allow broken neural networks by returning 0 if no output is found
    output_values: List[float] = [node_values.get(output_node, 0.0) for output_node in graph.outputs]

    return output_values


def significance(diff: float) -> float:
    return 0.0 if math.isclose(diff, 0) else diff * diff if diff >= 1 else min(diff * diff, 1.0 / diff)


def graph_similarity(g1: Graph, g2: Graph):
    # it just calculates rate of change between node count, edge count and total weight
    # then averages those values.
    len_g1_nodes = len(g1.nodes)
    len_g2_nodes = len(g2.nodes)
    node_count_change = abs((len_g2_nodes - len_g1_nodes) / len_g1_nodes)
    len_g1_edges = len(g1.edges)
    len_g2_edges = len(g2.edges)
    edge_count_change = abs((len_g2_edges - len_g1_edges) / len_g1_edges)
    g1_total_weight = sum(g1.edges.values())
    g2_total_weight = sum(g2.edges.values())
    total_weight_change = abs((g2_total_weight - g1_total_weight) / g1_total_weight)
    total_change = (edge_count_change + node_count_change + total_weight_change) / 3
    return total_change


def mutate_weights(graph: Graph, mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1):
    for ft in graph.edges.keys():
        if random() < mutate_weight_probability:
            w = graph.edges[ft]
            factor = 2 * random() - 1
            amount = factor * mutate_weight_amount
            w += amount
            # no clip, let GA get rid of invalid values
            graph.edges[ft] = w


def mutate_delete_edge(graph: Graph, mutate_delete_edge_probability: float = 0.01, sort: bool = False):
    if math.isclose(mutate_delete_edge_probability, 0.0):
        return
    mutated = False
    if random() < mutate_delete_edge_probability:
        # find the weakest edge
        edge, weight = min([(edge, weight) for edge, weight in graph.edges.items() if edge[0] not in graph.inputs and edge[1] not in graph.outputs], key=lambda e: abs(e[1]), default=((None, None), None))
        mutated = graph.edges.pop(edge, None) is not None
        if mutated:
            print(f"Graph({graph.graph_id}) deleting Edge {edge} with weight {weight}")
    if mutated and sort:
        graph.nodes = topological_sort(graph)


def mutate_new_node(graph: Graph, mutate_new_node_probability: float = 0.01, sort: bool = False):
    if math.isclose(mutate_new_node_probability, 0.0):
        return
    mutated = False
    for (f, t), w in list(graph.edges.items()):
        if random() < mutate_new_node_probability:
            mutated = True
            n = add_new_node(graph, activation=ActivationFunctionType.SIGMOID)
            graph.edges[(f, n)] = w
            graph.edges[(n, t)] = w
            del graph.edges[(f, t)]
            print(f"Graph({graph.graph_id}) adding new node + edge: {f} -> {n} (new) -> {t} and disconnecting {f} -> {t}")
    if mutated and sort:
        graph.nodes = topological_sort(graph)


def mutate_activation(graph: Graph, mutate_activation_probability: float = 0.01):
    for node in graph.nodes:
        if node not in graph.inputs:
            if random() < mutate_activation_probability:
                new_activation = choice(ACTIVATION_FUNCTION_LIST)
                graph.activation[node] = new_activation


def mutate_bias(graph: Graph, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    for node in graph.nodes:
        if node not in graph.inputs:
            if random() < mutate_bias_probability:
                b = graph.biases[node]
                factor = 2 * random() - 1
                b += factor * mutate_bias_amount
                # b = max(b, -5.0)
                # b = min(b, 5.0)
                graph.biases[node] = b
                new_activation = choice(ACTIVATION_FUNCTION_LIST)
                graph.activation[node] = new_activation


def mutate(graph: Graph, mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1, mutate_delete_edge_probability: float = 0.01, mutate_new_node_probability: float = 0.01, mutate_activation_probability: float = 0.01, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    mutate_weights(graph, mutate_weight_probability=mutate_weight_probability, mutate_weight_amount=mutate_weight_amount)
    mutate_delete_edge(graph, mutate_delete_edge_probability=mutate_delete_edge_probability)
    mutate_new_node(graph, mutate_new_node_probability=mutate_new_node_probability)
    mutate_activation(graph, mutate_activation_probability=mutate_activation_probability)
    mutate_bias(graph, mutate_bias_probability=mutate_bias_probability, mutate_bias_amount=mutate_bias_amount)


def generate_neural_network(layers: List[int]) -> Graph:
    graph = Graph()
    n = 0
    len_layers = len(layers)
    for index, layer in enumerate(layers, 1):
        for _ in range(layer):
            if index == 1:
                graph.inputs.append(n)
            elif index == len_layers:
                graph.outputs.append(n)
            graph.nodes.append(n)
            if index > 1:
                graph.activation[n] = ActivationFunctionType.SIGMOID
                graph.biases[n] = 0.0
            n += 1
    n = 0
    prev_nodes = []
    for layer in layers:
        to_nodes = []
        for _ in range(layer):
            to = graph.nodes[n]
            to_nodes.append(to)
            n += 1
        for pn in prev_nodes:
            for to in to_nodes:
                graph.edges[(pn, to)] = 0.5
        prev_nodes = to_nodes
    return graph


def to_dot_file(graph: Graph, out: TextIO):
    out.write("digraph G {\n")
    for node in graph.nodes:
        out.write(f"    {node};\n")
    for (node1, node2), weight in graph.edges.items():
        out.write(f'    {node1} -> {node2} [label="{round(weight, 2)} f({graph.activation[node2].name[0]})"];\n')
    out.write("}")


if __name__ == "__main__":
    g = generate_neural_network([2, 3, 1])
    # input_values = [1.0 for _ in range(0, len(g.inputs))]
    # outputs = evaluate_graph(g, input_values=input_values)
    # print(outputs)
    # with open(f"graph.dot", "w") as f:
    #     to_dot_file(g, f)
    # os.system(f"dot graph.dot -Tpng -o graph.png")
    initial_g = deepcopy(g)
    print(f"Generating graphs...")
    for i in range(1, 21):
        mutate_weights(g)
        mutate_new_node(g)
        mutate_activation(g)
        print(i)
        with open(f"graph{i}.dot", "w") as f:
            to_dot_file(g, f)
        os.system(f"dot graph{i}.dot -Tpng -o graph{i}.png")
    child_g = crossover(initial_g, g)
    with open(f"graph_child.dot", "w") as f:
        to_dot_file(child_g, f)
    os.system(f"dot graph_child.dot -Tpng -o graph_child.png")

    # distance = graph_similarity(prev_g, g)
    # outputs = evaluate_graph(g, input_values=input_values)
    # print(f"{i} {distance=} {outputs=}")
    # prev_g = deepcopy(g)
    # graphs.append(prev_g)
    # print(f"Ranking graphs...")
    # ranked = rank_by_standard_deviation(graphs, graph_similarity)
    # for g in ranked:
    #     print(g)
    # print("Done!")
    #     with open(f"graph{i}.dot", "w") as f:
    #         to_dot_file(g, f)
    #     os.system(f"dot graph{i}.dot -Tpng -o graph{i}.png")
