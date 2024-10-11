import numpy as np
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

    def __init__(self, preprocessing, algorithm, metrics, parameters, fitness, rules):
        self.preprocessing = preprocessing
        self.algorithm = algorithm
        self.metrics = metrics
        self.parameters = parameters
        self.fitness = fitness        
        self.rules = rules

        self.support = 0
        self.confidence = 0
        self.surrogate_fitness = 0

        if len(rules) > 0:
            self.support = rules.mean("support")
            self.confidence = rules.mean("confidence")

        self.num_rules = len(rules)

    def get_rules_support(self):
        return self.support
    
    def get_rules_confidence(self):
        return self.confidence
    
    def get_surrogate_fitness(self, metrics):
        surrogate_fitness = 0
        for metric in metrics:
            surrogate_fitness += self.rules.mean(metric)

        self.surrogate_fitness = surrogate_fitness / len(metrics)
        return self.surrogate_fitness       

    def __str__(self):
        return "\nPreprocessing: {}\nAlgorithm: {}\nHyperparameters: {}\nMetrics: {}\nFitness: {:.4f}\nMean Support: {:.4f}\nMean Confidence: {:.4f}\nRules: {}\n------------------".format(
            self.preprocessing, self.algorithm, self.parameters, self.metrics, self.fitness, self.support, self.confidence, self.num_rules)
    
