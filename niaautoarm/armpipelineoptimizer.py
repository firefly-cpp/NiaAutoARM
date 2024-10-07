#from niaarm.logger import Logger
from niapy.util.factory import get_algorithm
from niaautoarm import AutoARM
from niapy.task import Task, OptimizationType
from niaautoarm.logger import Logger

class ArmPipelineOptimizer:
    
    def __init__(self, **kwargs):
        self._data = None
        self._feature_prepocessing_techniques = None
        self._rule_mining_algorithms = None
        self._logger = None
        self._metrics = None
        self._hyperparameters = None

        self._set_parameters(**kwargs)

    def _set_parameters(self, data, feature_prepocessing_techniques, rule_mining_algorithms, metrics, hyperparameters, log=True,log_verbose=False, log_output_file=None):

        self._data = data
        self._feature_prepocessing_techniques = feature_prepocessing_techniques        
        self._rule_mining_algorithms = rule_mining_algorithms
        self._metrics = metrics

        self._hyperparameters = hyperparameters
        if log is True:
            self._logger = Logger(log_verbose, output_file=log_output_file)

    def get_data(self):
        return self._data
    
    def get_feature_prepocessing_techniques(self):
        return self._feature_prepocessing_techniques
    
    def get_rule_mining_algorithms(self):
        return self._rule_mining_algorithms
    
    def get_logger(self):
        return self._logger
    
    def run(self, population_size, optimization_algorithm):
        
        algo = get_algorithm(optimization_algorithm)
        algo.NP = population_size

        problem = AutoARM(
            self._data,
            self._feature_prepocessing_techniques,
            self._rule_mining_algorithms,
            self._hyperparameters,
            self._metrics,
            logging=True)
        
        task = Task(
            problem=problem,
            max_iters=2,
            optimization_type=OptimizationType.MAXIMIZATION)
        
        best = algo.run(task=task)
        armpipeline = problem.get_best_pipeline()

        return armpipeline