from copy import deepcopy
from random import choices
from typing import List, Dict, Callable, Tuple, Iterable

from notneat_py.graph import Graph, generate_neural_network, crossover, mutate, evaluate_graph, to_dot_file, significance


def generate_population(size: int, layers: List[int]) -> List[Graph]:
    return [generate_neural_network(layers=layers) for _ in range(size)]


# Takes a neural network, returns a floating point representing fitnesss (lower = fitter)
FitnessFunctionType = Callable[[Graph], float]


def cluster_iterator(clusters: List[List[Graph]]):
    for cluster in clusters:
        for g in cluster:
            yield g


def evaluate_population_fitness(population: Iterable[Graph], fitness_function: FitnessFunctionType) -> Tuple[Dict[int, float], float, float, Graph]:
    fitness = {}
    best_fitness = float('+inf')
    total_fitness = 0
    total_count = 0
    fittest = None
    for g in population:
        f = fitness_function(g)
        total_fitness += f
        total_count += 1
        fitness[g.graph_id] = f
        if f < best_fitness:
            best_fitness = f
            fittest = g
    return fitness, best_fitness, total_fitness / total_count, fittest


#
# def cluster_species(population: List[Graph]) -> List[List[Graph]]:
#     # map ids to graphs
#     graphs = {g.graph_id: g for g in population}
#
#     # cluster by ids
#     def graph_similarity_by_id(id1: int, id2: int):
#         return graph_similarity(graphs[id1], graphs[id2])
#
#     clusters = agglomerative_hierarchical_clustering(list(graphs.keys()), graph_similarity_by_id)
#     # return clusters of graphs
#     return [[graphs[i] for i in cluster] for cluster in clusters]

def crossover_population(population: List[Graph], fitness: Dict[int, float], elite_percentage: float = 0.1) -> List[Graph]:
    number_of_elites = int(len(population) * elite_percentage)
    population = sorted(population, key=lambda g: fitness[g.graph_id])
    weights = [w for w in range(len(population), 0, -1)]
    crossed_over = deepcopy(population[:number_of_elites])
    remaining_to_select = len(population) - len(crossed_over)
    for _ in range(remaining_to_select):
        parent1, parent2 = choices(population, weights, k=2)
        if fitness[parent2.graph_id] > fitness[parent1.graph_id]:
            parent1, parent2 = parent2, parent1
        child = crossover(parent1, parent2)
        crossed_over.append(child)
    return crossed_over


#
# def crossover_clusters(clusters: List[List[Graph]], fitness: Dict[int, float], elite_percentage: float = 0.4):
#     new_clusters = []
#     # let's go over each cluster, select some elites,
#     for cluster in clusters:
#         crossover_cluster = []
#         number_of_elites = round(len(cluster) * elite_percentage)
#         # elites crossover without change
#         sorted_cluster = sorted(cluster, key=lambda g: fitness[g.graph_id])
#         crossover_cluster.extend(islice(sorted_cluster, number_of_elites))
#         remaining_cluster = list(sorted_cluster)
#         remaining_to_select = len(remaining_cluster)
#         for _ in range(remaining_to_select):
#             # sample can choose the same parent twice
#             parent1, parent2 = sample(remaining_cluster, 2)
#             child = crossover(parent1, parent2)
#             crossover_cluster.append(child)
#         new_clusters.append(crossover_cluster)
#     return new_clusters

def mutate_population(population: List[Graph], mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1, mutate_delete_edge_probability: float = 0.01, mutate_new_node_probability: float = 0.01, mutate_activation_probability: float = 0.01, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    for member in population:
        mutate(member, mutate_weight_probability=mutate_weight_probability, mutate_weight_amount=mutate_weight_amount, mutate_delete_edge_probability=mutate_delete_edge_probability, mutate_new_node_probability=mutate_new_node_probability, mutate_activation_probability=mutate_activation_probability, mutate_bias_probability=mutate_bias_probability, mutate_bias_amount=mutate_bias_amount)
    return population


# def mutate_clusters(clusters: List[List[Graph]]) -> List[Graph]:
#     # this mutates the clusters. Also returns a new population without any clusters
#     new_population = []
#     for cluster in clusters:
#         for g in cluster:
#             mutate(g)
#             new_population.append(g)
#     return new_population


def genetic_algorithm(population_size: int, layers: List[int], fitness_function: FitnessFunctionType, max_generations: int = 100, elite_percentage: float = 0.1, target_fitness: float = 0.01, mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1, mutate_delete_edge_probability: float = 0.01, mutate_new_node_probability: float = 0.01, mutate_activation_probability: float = 0.01, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    # initialize population
    population = generate_population(population_size, layers)
    best_fitness = float('+inf')
    fittest = None
    for generation in range(1, max_generations):
        fitness, generation_best_fitness, generation_mean_fitness, generation_fittest = evaluate_population_fitness(population, fitness_function)
        improved = generation_best_fitness < best_fitness
        fittest = fittest or generation_fittest
        if improved:
            best_fitness = generation_best_fitness
            fittest = generation_fittest
        if improved or (generation % 100 == 0):
            print(f"GENERATION {generation} FITTEST: Graph({fittest.graph_id}) best: {best_fitness} mean: {generation_mean_fitness} pop size: {len(population)}")
        target_reached = best_fitness <= target_fitness
        if target_reached:
            return fittest, best_fitness
        population = crossover_population(population, fitness, elite_percentage=elite_percentage)
        population = mutate_population(population, mutate_weight_probability=mutate_weight_probability, mutate_weight_amount=mutate_weight_amount, mutate_delete_edge_probability=mutate_delete_edge_probability, mutate_new_node_probability=mutate_new_node_probability, mutate_activation_probability=mutate_activation_probability, mutate_bias_probability=mutate_bias_probability, mutate_bias_amount=mutate_bias_amount)
    else:
        return fittest, best_fitness


def experiment_truth_table(inputs: List[List[int]], outputs: List[List[int]]):
    def fitness_truth_table(graph: Graph, verbose: bool = False, rounding: bool = False) -> float:
        total_diff = 0.0
        score = 0.0
        for i, o in zip(inputs, outputs):
            r = evaluate_graph(i, graph)
            if rounding:
                r = [float(round(o)) for o in r]
            if verbose:
                print(f"Graph({graph.graph_id}) {i} -> {r} (expected: {o})")
            diff: float = sum(abs(a - b) for a, b in zip(o, r))
            total_diff += diff
            score += significance(diff)
        # score *= len(graph.nodes) - len(graph.inputs)
        if verbose:
            print(f"Graph({graph.graph_id}) total diff: {total_diff} score: {score}")
        return score

    graph, best_fitness = genetic_algorithm(
        population_size=100,
        layers=[len(inputs[0]), 5, len(outputs[0])],
        fitness_function=fitness_truth_table,
        max_generations=10000,
        elite_percentage=0.3,
        target_fitness=0.3,
        mutate_weight_probability=0.2,
        mutate_weight_amount=0.1,
        mutate_new_node_probability=1.0 / 1_000_000,
        mutate_activation_probability=0.2,
        mutate_bias_probability=0.2,
        mutate_bias_amount=0.1,
    )
    print(graph)
    fitness_truth_table(graph=graph, verbose=True, rounding=True)
    print(f"{best_fitness=}")
    with open(f"graph.dot", "w") as f:
        to_dot_file(graph, f)
    # os.system(f"dot graph.dot -Tpng -o graph.png")
    # os.system(f"eog graph.png")


if __name__ == "__main__":
    # full adder
    inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs = [[0, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1]]

    # xor
    # inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # outputs = [[0], [1], [1], [0]]

    experiment_truth_table(inputs, outputs)
