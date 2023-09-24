import random
import time
from multiprocessing import Pool
from typing import List

import gymnasium as gym
import math

from notneat_py.genetic import genetic_algorithm
from notneat_py.graph import Graph, evaluate_graph

if __name__ == "__main__":
    # game = 'CartPole-v1'
    game = "LunarLander-v2"
    env = gym.make(game, render_mode="rgb_array")
    observation, info = env.reset()
    min_action, max_action = 0, 3
    print(env.action_space)

    outputs = [0, 1, 2, 3]


    def gym_fitness_batch(graphs: List[Graph]):
        # gym is really slow, let's try to speed it up
        with Pool() as p:
            return p.map(gym_fitness, graphs)


    def gym_fitness(graph: Graph, render: bool = False):
        rewards = 0.0
        iterations = 3  # if you don't test on multiple iterations, it might plateau on a lucky run.
        for _ in range(iterations):
            observation, info = env.reset()
            if render:
                env.render()
            while True:
                output = evaluate_graph(observation, graph)
                # make probabilistic actions
                weights = [math.exp(o) for o in output]
                action = random.choices(outputs, weights=weights, k=1)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                rewards += reward
                if terminated or truncated:
                    break
        return 1000 - rewards / iterations


    # with open("graph.pickle", "rb") as f:
    #     graph = pickle.load(f)

    # this hopefully converges towards something that resembles a flight controller around generation 1500
    graph, best_fitness = genetic_algorithm(
        population_size=40,
        layers=[8, 8, 4, 4, 4],
        fitness_function=gym_fitness_batch,
        max_generations=1700,
        elite_percentage=0.2,
        target_fitness=900,
        mutate_weight_probability=0.2,
        mutate_weight_amount=0.1,
        mutate_delete_edge_probability=100.0 / 1_000_000,
        mutate_new_node_probability=10.0 / 1_000_000,
        mutate_activation_probability=0.2,
        mutate_bias_probability=0.2,
        mutate_bias_amount=0.1,
    )

    # with open("graph.pickle", "wb") as f:
    #     pickle.dump(graph, f)

    while True:
        env = gym.make(game, render_mode="human")
        observation, info = env.reset()
        gym_fitness(graph, render=True)
        time.sleep(2)
