import functools
from multiprocessing import Pool
from typing import List

from notneat_py.genetic import genetic_algorithm
from notneat_py.graph import significance, evaluate_graph, Graph


def fitness_truth_table(inputs, outputs, graph: Graph, verbose: bool = False) -> float:
    total_diff = 0.0
    score = 0.0
    for i, o in zip(inputs, outputs):
        r = evaluate_graph(i, graph)
        if verbose:
            print(f"Graph({graph.graph_id}) {i} -> {r} (expected: {o})")
        diff: float = sum(abs(a - b) for a, b in zip(o, r))
        total_diff += diff
        score += significance(diff)
    if verbose:
        print(f"Graph({graph.graph_id}) total diff: {total_diff} score: {score}")
    return score


def batch_fitness_truth_table(inputs, outputs, graphs: List[Graph]) -> List[float]:
    return list(map(functools.partial(fitness_truth_table, inputs, outputs), graphs))
    # with Pool() as p:
    #     return p.map(functools.partial(fitness_truth_table, inputs, outputs), graphs)


def experiment_truth_table(inputs: List[List[int]], outputs: List[List[int]], layers: List[int]):
    graph, best_fitness = genetic_algorithm(
        population_size=100,
        layers=layers,
        fitness_function=functools.partial(batch_fitness_truth_table, inputs, outputs),
        max_generations=10000,
        elite_percentage=0.3,
        target_fitness=0.3,
        mutate_weight_probability=0.2,
        mutate_weight_amount=0.1,
        mutate_delete_edge_probability=10.0 / 1_000_000,
        mutate_new_node_probability=0.1 / 1_000_000,
        mutate_activation_probability=0.1,
        mutate_bias_probability=0.2,
        mutate_bias_amount=0.1,
    )
    fitness_truth_table(inputs, outputs, graph, verbose=True)
    print(f"{best_fitness=}")
