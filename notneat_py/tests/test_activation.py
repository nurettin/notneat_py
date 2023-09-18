import math
import unittest

from notneat_py.activation import leaky_relu, normalized_tanh, relu, sigmoid, softplus


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1), 1 / (1 + math.exp(-1)))
        self.assertLess(sigmoid(-100), 0.01)

    def test_normalized_tanh(self):
        self.assertEqual(normalized_tanh(0), 0.5)
        self.assertAlmostEqual(normalized_tanh(1), (math.tanh(1) + 1) / 2, places=5)

    def test_relu(self):
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(5), 5)
        self.assertEqual(relu(-5), 0)

    def test_leaky_relu(self):
        self.assertEqual(leaky_relu(0), 0)
        self.assertEqual(leaky_relu(5), 5)
        self.assertEqual(leaky_relu(-5), -0.05)
        self.assertEqual(leaky_relu(-5, alpha=0.1), -0.5)

    def test_softplus(self):
        self.assertAlmostEqual(softplus(0), math.log(2), places=5)
        self.assertAlmostEqual(softplus(5), math.log(1 + math.exp(5)), places=5)

if __name__ == '__main__':
    unittest.main()