import pickle
import time

import gymnasium as gym
import numpy as np

from notneat_py.genetic import genetic_algorithm
from notneat_py.graph import Graph, evaluate_graph

if __name__ == "__main__":
    # game = 'CartPole-v1'
    game = "LunarLander-v2"
    env = gym.make(game, render_mode="rgb_array")
    observation, info = env.reset()
    min_action, max_action = 0, 3
    print(env.action_space)


    def gym_fitness(graph: Graph, render: bool = False):
        rewards = 0.0
        iterations = 5
        for _ in range(iterations):
            observation, info = env.reset()
            if render:
                env.render()
            while True:
                output = evaluate_graph(observation, graph)
                exp = np.exp(output)
                action_probabilities = exp / np.sum(exp)
                action = np.random.choice(4, p=action_probabilities, replace=False)
                # action = np.argmin(output)
                observation, reward, terminated, truncated, info = env.step(action)
                rewards += reward
                if terminated or truncated:
                    break
        return 1000 - rewards / iterations

    with open("graph.pickle", "rb") as f:
        graph = pickle.load(f)


    # graph, best_fitness = genetic_algorithm(
    #     population_size=30,
    #     layers=[8, 8, 4],
    #     fitness_function=gym_fitness,
    #     max_generations=1600,
    #     elite_percentage=0.1,
    #     target_fitness=500,
    #     mutate_weight_probability=0.2,
    #     mutate_weight_amount=0.1,
    #     mutate_delete_edge_probability=0.0,
    #     mutate_new_node_probability=0.0,
    #     mutate_activation_probability=0.2,
    #     mutate_bias_probability=0.2,
    #     mutate_bias_amount=0.1,
    # )
    #
    # with open("graph.pickle", "wb") as f:
    #     pickle.dump(graph, f)

    while True:
        env = gym.make(game, render_mode="human")
        observation, info = env.reset()
        gym_fitness(graph, render=True)
        time.sleep(2)
