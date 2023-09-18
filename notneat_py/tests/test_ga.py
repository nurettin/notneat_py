import unittest

from notneat_py.genetic import genetic_algorithm
from notneat_py.graph import Graph, evaluate_graph


class TestGA(unittest.TestCase):
    def test_xor(self):
        xor_table = [
            {"inputs": [0, 0], "output": 0},
            {"inputs": [0, 1], "output": 1},
            {"inputs": [1, 0], "output": 1},
            {"inputs": [1, 1], "output": 0},
        ]

        def xor_fitness_function(g: Graph):
            error = 0.0
            for row in xor_table:
                error += abs(evaluate_graph(input_values=row["inputs"], graph=g)[0] - row["output"])
            return error

        clusters, fitness = genetic_algorithm(5, [2, 1, 1], xor_fitness_function, 30)
        print(clusters)
        print(fitness)
