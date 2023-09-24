from ideas.experiment_truth_table import experiment_truth_table

if __name__ == "__main__":
    # full adder
    inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs = [[0, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1]]

    experiment_truth_table(inputs, outputs, [3, 5, 3, 2])
