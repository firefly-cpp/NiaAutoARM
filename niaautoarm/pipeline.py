class Pipeline:
    r"""Class representing a pipeline.
    Args:
       preprocessing (str): Selected preprocessing techniques.
       algorithm (str): Selected algorithm.
       metrics (list): Selected metrics.
       parameters (list): Hyperparameter values.
       support (float): Support value.
       confidence (float): Confidence value.
    """

    def __init__(self, preprocessing, algorithm, metrics, parameters, support, confidence):
        self.preprocessing = preprocessing
        self.algorithm = algorithm
        self.metrics = metrics
        self.parameters = parameters
        self.support = support
        self.confidence = confidence
