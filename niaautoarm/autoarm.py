from niaarm import NiaARM
from niaarm.dataset import Dataset
from niapy.problems import Problem
from niapy.algorithms.basic import DifferentialEvolution
from niapy.task import Task, OptimizationType
import numpy as np
import csv
import random

class AutoARM(Problem):
    r"""Implementation of AutoARM.
    
    Date:
        2022
    
    Attributes:
    
    """
    
    def __init__(
            self,
            dimension,
            lower,
            upper,
            preprocessing,
            algorithm_selection,
            hyperparameters,
            metrics):
        r"""Initialize instance of AutoARM.
        
        Arguments:
        
        """
        self.dim = dimension
        self.preprocessing = preprocessing
        self.algorithm_selection = algorithm_selection
        self.hyperparameters = hyperparameters
        self.metrics = metrics

        super().__init__(dimension, lower, upper)

    def float_to_category(self):
        pass # TODO

    # TODO
    def _evaluate(self, sol):
        data = Dataset("datasets/Abalone.csv")

        problem = NiaARM(data.dimension, data.features, data.transactions, metrics=('support', 'confidence'), logging=True)

        # build niapy task
        task = Task(problem=problem, max_iters=30, optimization_type=OptimizationType.MAXIMIZATION)

        # use Differential Evolution (DE) algorithm from the NiaPy library
        # see full list of available algorithms: https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
        algo = DifferentialEvolution(population_size=50, differential_weight=0.5, crossover_probability=0.9)

        # run algorithm
        best = algo.run(task=task)

        # sort rules
        problem.rules.sort()

        return random.randint(0,2)
