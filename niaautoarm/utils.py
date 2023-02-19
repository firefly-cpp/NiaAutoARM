def calculate_dimension_of_the_problem(
        preprocessing,
        algorithms,
        hyperparameters,
        metrics):
    return (
        len(preprocessing) +
        len(algorithms) +
        len(hyperparameters) +
        len(metrics))
