from enum import Enum

import math


class ActivationFunctionType(Enum):
    SIGMOID = 1
    NORMALIZED_TANH = 2
    TANH = 3
    RELU = 4
    LEAKY_RELU = 5
    # SOFTPLUS = 6


ACTIVATION_FUNCTION_LIST = list(ActivationFunctionType)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def normalized_tanh(x):
    return (math.tanh(x) + 1) / 2


def relu(x):
    return max(0, x)


def leaky_relu(x, alpha=0.01):
    return max(alpha * x, x)


def softplus(x):
    return math.log(1 + math.exp(x))


activation_functions = {
    ActivationFunctionType.SIGMOID: sigmoid,
    ActivationFunctionType.NORMALIZED_TANH: normalized_tanh,
    ActivationFunctionType.TANH: math.tanh,
    ActivationFunctionType.RELU: relu,
    ActivationFunctionType.LEAKY_RELU: leaky_relu,
    # ActivationFunctionType.SOFTPLUS: softplus,
}
