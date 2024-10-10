from niapy.util.factory import get_algorithm
from niaautoarm.autoarmproblem import AutoARMProblem
from niapy.task import Task, OptimizationType
from niaautoarm.logger import Logger
from niaautoarm.stats import ARMPipelineStatistics

__all__ = ["ArmPipelineOptimizer"]

class AutoARMOptimizer:
    r"""Class for running the AutoARM framework.

    Date:
        2024

    Author:
        Uroš Mlakar & Iztok Fister Jr.

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
    
    def run(
            self,
            optimization_algorithm,
            population_size=5, 
            max_iters=10,
            optimize_metric_weights=False,
            allow_multiple_preprocessing=False,
            output_pipeline_file=None):
        
        algo = get_algorithm(optimization_algorithm)
        algo.NP = population_size

        problem = AutoARMProblem(        
            self.data,
            self.feature_prepocessing_techniques,
            self.rule_mining_algorithms,
            self.hyperparameters,
            self.metrics,
            optimize_metric_weights,
            allow_multiple_preprocessing,
            self.logger)
        
        task = Task(
            problem=problem,
            max_iters=max_iters,
            optimization_type=OptimizationType.MAXIMIZATION)
        
        best = algo.run(task=task)
        arm_best_pipeline = problem.get_best_pipeline()
        arm_stats = ARMPipelineStatistics(problem.get_all_pipelines(),arm_best_pipeline)
        
        if output_pipeline_file is not None:
            arm_stats.dump_to_file(output_pipeline_file)

        return arm_best_pipeline