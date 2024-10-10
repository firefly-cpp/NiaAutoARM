import numpy as np
from niaarm import NiaARM
from niapy.problems import Problem
from niapy.task import Task, OptimizationType

from niaautoarm.pipeline import Pipeline
from niaautoarm.preprocessing import Preprocessing

from niaautoarm.utils import calculate_dimension_of_the_problem, float_to_category, float_to_num, threshold


class AutoARMProblem(Problem):
    r"""Definition of Auto Association Rule Mining.

    The implementation is composed of ideas found in the following papers:
    * Pečnik, L., Fister, I., & Fister, I. (2021). NiaAML2: An Improved AutoML Using Nature-Inspired Algorithms. In Advances in Swarm Intelligence: 12th International Conference, ICSI 2021, Qingdao, China, July 17–21, 2021, Proceedings, Part II 12 (pp. 243-252). Springer International Publishing.

    * Stupan, Ž., & Fister, I. (2022). NiaARM: A minimalistic framework for Numerical Association Rule Mining. Journal of Open Source Software, 7(77), 4448.

    Args:
        dataset (list): The entire dataset.
        preprocessing (list): Preprocessing components (data squashing or none).
        algorithms (list): Algorithm components (one arbitrary algorithm from niapy collection).
        hyperparameters (list): Selected hyperparameter values.
        metrics (list): Metrics component.
        logging (bool): Enable logging of fitness improvements. Default: ``False``.
    Attributes:
        rules (RuleList): A list of mined association rules.
    """

    def __init__(
            self,
            dataset,
            preprocessing_methods,
            algorithms,
            hyperparameters,
            metrics,
            optimize_metric_weights,
            allow_multiple_preprocessing,
            logger
    ):
        r"""Initialize instance of AutoARM.dataset_class

        Arguments:

        """
        # calculate the dimension of the problem
        dimension = calculate_dimension_of_the_problem(
            preprocessing_methods, hyperparameters, metrics, optimize_metric_weights=optimize_metric_weights, allow_multiple_preprocessing=allow_multiple_preprocessing)

        super().__init__(dimension, 0, 1)
        self.preprocessing_methods = preprocessing_methods
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.best_fitness = -np.inf
        self.preprocessing_instance = Preprocessing(dataset,None)

        self.logger = logger
        self.all_pipelines = []
        self.best_pipeline = None

        self.allow_multiple_preprocessing = allow_multiple_preprocessing
        self.optimize_metric_weights = optimize_metric_weights

    def get_best_pipeline(self):
        return self.best_pipeline

    def get_all_pipelines(self):
        return self.all_pipelines

    def _evaluate(self, x):
        r"""Evaluate the fitness of the pipeline.
        """

        #get the algorithm component
        algorithm_component = self.algorithms[float_to_category(
            self.algorithms, x[0])]
        
        pos_x = 1
        
        hyperparameter_component = float_to_num(self.hyperparameters, x[pos_x:pos_x + len(self.hyperparameters)])

        pos_x += len(self.hyperparameters)
        
        # get preprocessing components
        if self.allow_multiple_preprocessing:
            preprocessing_component = threshold(self.preprocessing_methods, x[pos_x:pos_x + len(self.preprocessing_methods)])
            pos_x += len(self.preprocessing_methods)
        else:
            preprocessing_component = [self.preprocessing_methods[float_to_category(
            self.preprocessing_methods, x[pos_x])]]
            pos_x += 1        

        
        metrics_indexes ,metrics_component = threshold(self.metrics, x[pos_x:pos_x + len(self.metrics)])

        if metrics_component == ():  # if no metrics are selected TODO: check for alternative solution
            return -np.inf

        pos_x += len(self.metrics)

        if self.optimize_metric_weights:
            metrics_weights = x[pos_x:]
            metrics_weights = [metrics_weights[i] for i in metrics_indexes]
            metrics_component = dict(zip(metrics_component, metrics_weights))



        self.preprocessing_instance.set_preprocessing_algorithms(preprocessing_component) #TODO can be a list of multiple preprocessing techniques, order is determined by importance in class Preprocessing
        dataset = self.preprocessing_instance.apply_preprocessing()

        problem = NiaARM(
            dataset.dimension,            
            dataset.features,
            dataset.transactions,
            metrics=metrics_component)        

        # build niapy task
        task = Task(
            problem=problem,
            max_evals=hyperparameter_component[1],
            optimization_type=OptimizationType.MAXIMIZATION)

        algorithm_component.NP = hyperparameter_component[0]
        
        _, fitness = algorithm_component.run(task=task)

        if (len(problem.rules) == 0):
            return -np.inf

        pipeline = Pipeline(preprocessing_component, algorithm_component.Name[1], metrics_component, hyperparameter_component, fitness, problem.rules)

        # store each generated and also valid pipeline in a list for post-processing
        self.all_pipelines.append(pipeline)
        
        if fitness >= self.best_fitness:

            self.best_fitness = fitness
            self.best_pipeline = pipeline

            if self.logger is not None:
                self.logger.log_pipeline(pipeline)
        else:
            print("Fitness: ", fitness) # For debugging purposes, remove later
        return fitness