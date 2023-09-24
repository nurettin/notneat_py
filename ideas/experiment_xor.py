from ideas.experiment_truth_table import experiment_truth_table

if __name__ == "__main__":
    # xor
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    experiment_truth_table(inputs, outputs, [2, 3, 4, 3, 4, 1])
