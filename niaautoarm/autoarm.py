from niaarm import NiaARM
from niaarm.dataset import Dataset
from niapy.problems import Problem
from niapy.algorithms.basic import DifferentialEvolution
from niapy.task import Task, OptimizationType
import numpy as np
import csv
import random
import sys

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

    # swap float to categorical attribute
    def float_to_category(self, component, val):
        if val == 1:
            return (len(component)-1)
        return int(val * len(component) + 0)

    def float_to_num(self, component, val):
        print (component)
        parameters = [1] * len(component)
        for i in range(len(component)):
            parameters[i] = int(val * int(component[i]['max']) + int(component[i]['min']))

        return parameters

    # indicate whether the attribute is part of the component
    def threshold(self, component, val):
        selected = [0] * len(component)
        for i in range(len(val)):
            if val[i] > 0.5:
                selected[i] = 1
        return selected

    def _evaluate(self, sol):

        # firstly, genotype to phenotype mapping
        print ("Solution: ", sol)
        preprocessing_component = self.preprocessing[self.float_to_category(self.preprocessing, sol[0])]
        print ("Izbrani preprocessing: ", preprocessing_component)
        algorithm_component = self.algorithm_selection[self.float_to_category(self.algorithm_selection, sol[1])]
        print ("Izbrani algorithm: ", algorithm_component)
        hyperparameter_component = self.float_to_num(self.hyperparameters, sol[2:3])
        print ("Izbrane vrednosti hyp:", hyperparameter_component)
        metrics_component = self.threshold(self.metrics, sol[4:12])
        print ("Izbrane metrics", metrics_component)

        sys.exit(1)

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
