#from niaarm.logger import Logger
from niapy.util.factory import get_algorithm
from niaautoarm import AutoARM
from niapy.task import Task, OptimizationType
from niaautoarm.logger import Logger
from niaautoarm.stats import ArmPipelineStatistics
import pickle


__all__ = ["ArmPipelineOptimizer"]

class ArmPipelineOptimizer:
    r"""Class for running the AutoARM framework.

    Date:
        2024

    Author:
        Uroš Mlakar

    License:
        MIT
    """
    
    def __init__(self, **kwargs):
        self.data = None
        self.feature_prepocessing_techniques = None
        self.rule_mining_algorithms = None        
        self.metrics = None
        self.hyperparameters = None
        self.logger = None

        self._set_parameters(**kwargs)

    def _set_parameters(self, data, feature_prepocessing_techniques, rule_mining_algorithms, metrics, hyperparameters, log=True,log_verbose=False, log_output_file=None):

        self.data = data
        self.feature_prepocessing_techniques = feature_prepocessing_techniques        
        self.rule_mining_algorithms = rule_mining_algorithms
        self.metrics = metrics
        self.hyperparameters = hyperparameters

        if log is True:
            self.logger = Logger(log_verbose, output_file=log_output_file)

    def get_data(self):
        return self.data
    
    def get_feature_prepocessing_techniques(self):
        return self.feature_prepocessing_techniques
    
    def get_rule_mining_algorithms(self):
        return self.rule_mining_algorithms
    
    def get_logger(self):
        return self.logger    
    
    def run(self, population_size, optimization_algorithm, max_iters=10):
        
        algo = get_algorithm(optimization_algorithm)
        algo.NP = population_size

        problem = AutoARM(        
            self.data,
            self.feature_prepocessing_techniques,
            self.rule_mining_algorithms,
            self.hyperparameters,
            self.metrics,
            self.logger)
        
        task = Task(
            problem=problem,
            max_iters=max_iters,
            optimization_type=OptimizationType.MAXIMIZATION)
        
        best = algo.run(task=task)
        arm_best_pipeline = problem.get_best_pipeline()
        arm_stats = ArmPipelineStatistics(problem.get_all_pipelines(),arm_best_pipeline)
        with open("test.pickle", 'wb') as file:
            pickle.dump(arm_stats, file)

        return arm_best_pipeline