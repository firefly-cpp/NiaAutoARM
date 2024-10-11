def float_to_category(component, val):
    r"""Map float value to component (category). """
    if val == 1:
        return len(component) - 1
    return int(val * len(component))


def float_to_num(component, val):
    r"""Map float value to integer. """
    parameters = [1] * len(component)
    for i in range(len(component)):
        parameters[i] = int(int(component[i]['min'] + (int(component[i]['max']) - int(component[i]['min'])) * val[i]))
    return parameters


def threshold(component, val):
    r"""Calculate whether feature is over a threshold. """
    data = [(i,c) for i, c in enumerate(component) if val[i] > 0.5]
    if data:
        return zip(*data)
    return [], ()

def calculate_dimension_of_the_problem(
        preprocessing,
        hyperparameters,
        metrics,
        optimize_metric_weights=False,
        allow_multiple_preprocessing=False):
    r"""Calculate the dimension of the problem. """

    num_preprocessing = 1
    if allow_multiple_preprocessing:
        num_preprocessing = len(preprocessing)

    metrics_factor = 1
    if optimize_metric_weights:
        metrics_factor = 2

    return (num_preprocessing + len(hyperparameters) + metrics_factor * len(metrics) + 1)