from niaarm.rule import Rule
from niaarm.association_rule import AssociationRule
from niapy.problems import Problem
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
            features,
            transactions,
            preprocessing,
            algorithm_selection,
            hyperparameters,
            evaluation_criterions):
        r"""Initialize instance of AutoARM.
        
        Arguments:
        
        """
        self.dim = dimension
        self.features = features
        self.transactions = transactions
        self.preprocessing = preprocessing
        self.algorithm_selection = algorithm_selection
        self.hyperparameters = hyperparameters
        self.evaluation_criterions = evaluation_criterions

        self.best_fitness = np.NINF
        self.rules = []
        super().__init__(dimension, lower, upper)

    def _evaluate(self, sol):
        return random.randint(0,2)
