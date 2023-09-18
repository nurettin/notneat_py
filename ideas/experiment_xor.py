import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network architecture with weights and biases
weights_input_to_hidden = np.array([[10, -10], [-10, 10]])
weights_hidden_to_output = np.array([[10], [10]])
biases_hidden = np.array([-5, 5])
bias_output = np.array([-10])

# Define the XOR inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Calculate XOR outputs for each input
for input_data in inputs:
    # Calculate Hidden Layer outputs
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Calculate Output Layer output
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    output = sigmoid(output_layer_input)

    print(f"Input: {input_data}, Output: {output[0]}")

