from copy import deepcopy
from random import choices
from typing import List, Dict, Callable, Tuple, Iterable

import math

from notneat_py.graph import Graph, generate_neural_network, crossover, mutate


def generate_population(size: int, layers: List[int]) -> List[Graph]:
    return [generate_neural_network(layers=layers) for _ in range(size)]


# Takes a batch of neural networks, returns a floating point representing fitnesss (lower = fitter)
FitnessFunctionType = Callable[[List[Graph]], List[float]]


def batched(n, iterable):
    batch = []
    for i, e in enumerate(iterable, 1):
        batch.append(e)
        if i % n == 0:
            yield batch
            batch = []
    else:
        yield batch


def evaluate_population_fitness(population: Iterable[Graph], fitness_function: FitnessFunctionType, batch_size: int = 4) -> Tuple[Dict[int, float], float, float, Graph]:
    fitness = {}
    best_fitness = float('+inf')
    total_fitness = 0
    total_count = 0
    fittest = None
    for gs in batched(batch_size, population):
        fs = fitness_function(gs)
        for g, f in zip(gs, fs):
            total_fitness += f
            total_count += 1
            fitness[g.graph_id] = f
            if f < best_fitness:
                best_fitness = f
                fittest = g
    return fitness, best_fitness, total_fitness / total_count, fittest


def crossover_population(population: List[Graph], fitness: Dict[int, float], elite_percentage: float = 0.1) -> List[Graph]:
    number_of_elites = int(len(population) * elite_percentage)
    population = sorted(population, key=lambda g: fitness[g.graph_id])
    weights = [math.exp(w) for w in range(len(population), 0, -1)]
    crossed_over = deepcopy(population[:number_of_elites])
    remaining_to_select = len(population) - len(crossed_over)
    for _ in range(remaining_to_select):
        parent1, parent2 = choices(population, weights, k=2)
        if fitness[parent2.graph_id] > fitness[parent1.graph_id]:
            parent1, parent2 = parent2, parent1
        child = crossover(parent1, parent2)
        crossed_over.append(child)
    return crossed_over


def mutate_population(population: List[Graph], mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1, mutate_delete_edge_probability: float = 0.01, mutate_new_node_probability: float = 0.01, mutate_activation_probability: float = 0.01, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    for member in population:
        mutate(member, mutate_weight_probability=mutate_weight_probability, mutate_weight_amount=mutate_weight_amount, mutate_delete_edge_probability=mutate_delete_edge_probability, mutate_new_node_probability=mutate_new_node_probability, mutate_activation_probability=mutate_activation_probability, mutate_bias_probability=mutate_bias_probability, mutate_bias_amount=mutate_bias_amount)
    return population


def genetic_algorithm(population_size: int, layers: List[int], fitness_function: FitnessFunctionType, max_generations: int = 100, elite_percentage: float = 0.1, target_fitness: float = 0.01, mutate_weight_probability: float = 0.1, mutate_weight_amount: float = 0.1, mutate_delete_edge_probability: float = 0.01, mutate_new_node_probability: float = 0.01, mutate_activation_probability: float = 0.01, mutate_bias_probability: float = 0.1, mutate_bias_amount: float = 0.1):
    # initialize population
    population = generate_population(population_size, layers)
    best_fitness = float('+inf')
    fittest = None
    for generation in range(1, max_generations):
        fitness, generation_best_fitness, generation_mean_fitness, generation_fittest = evaluate_population_fitness(population, fitness_function, batch_size=8)
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
