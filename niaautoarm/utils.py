def calculate_dimension_of_the_problem(
        preprocessing,
        algorithms,
        hyperparameters,
        metrics):
    return ( 2 + len(hyperparameters) + len(metrics))

