import numpy as np
from niaarm import NiaARM
from niapy.problems import Problem
from niapy.task import Task, OptimizationType

from niaautoarm.pipeline import Pipeline
from niaautoarm.preprocessing import Preprocessing

from niaautoarm.utils import calculate_dimension_of_the_problem, float_to_category, float_to_num, threshold
import copy

class AutoARMProblem(Problem):
    r"""Definition of Auto Association Rule Mining.

    The implementation is composed of ideas found in the following papers:
    * Pečnik, L., Fister, I., & Fister, I. (2021). NiaAML2: An Improved AutoML Using Nature-Inspired Algorithms. In Advances in Swarm Intelligence: 12th International Conference, ICSI 2021, Qingdao, China, July 17–21, 2021, Proceedings, Part II 12 (pp. 243-252). Springer International Publishing.

    * Stupan, Ž., & Fister, I. (2022). NiaARM: A minimalistic framework for Numerical Association Rule Mining. Journal of Open Source Software, 7(77), 4448.

    Args:
        dataset (list): The entire dataset.
        preprocessing_methods (list): Preprocessing components (see Prepprocessing class).
        algorithms (list): Algorithm components (one arbitrary algorithm from niapy collection).
        hyperparameters (list): Selected hyperparameter values.
        metrics (list): Metrics component.
        optimize_metric_weights (bool)
        allow_multiple_preprocessing (bool)
        use_surrogate_fitness (bool)
        logger (Logger): Logger instacne for logging fitness improvements.
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
            use_surrogate_fitness,
            conserve_space,
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
        self.preprocessing_instance = Preprocessing(dataset, None)

        self.logger = logger
        self.all_pipelines = []
        self.best_pipeline = None

        self.allow_multiple_preprocessing = allow_multiple_preprocessing
        self.optimize_metric_weights = optimize_metric_weights
        self.use_surrogate_fitness = use_surrogate_fitness

        self.conserve_space = conserve_space

    def get_best_pipeline(self):
        return self.best_pipeline

    def get_all_pipelines(self):
        return self.all_pipelines
    
    def decode_solution(self, x):
        r"""Decode a genotype vector into a readable ARM pipeline configuration."""

        x = np.asarray(x, dtype=float)

        pos_x = 0

        # Algorithm component
        algorithm_index = float_to_category(self.algorithms, x[pos_x])
        algorithm_component = self.algorithms[algorithm_index]
        algorithm_name = algorithm_component.Name[1]
        pos_x += 1

        # Hyperparameter component
        hyperparameter_component = float_to_num(
            self.hyperparameters,
            x[pos_x:pos_x + len(self.hyperparameters)]
        )
        pos_x += len(self.hyperparameters)

        # Preprocessing component
        if self.allow_multiple_preprocessing:
            _, preprocessing_component = threshold(
                self.preprocessing_methods,
                x[pos_x:pos_x + len(self.preprocessing_methods)]
            )
            pos_x += len(self.preprocessing_methods)
        else:
            preprocessing_component = [
                self.preprocessing_methods[
                    float_to_category(self.preprocessing_methods, x[pos_x])
                ]
            ]
            pos_x += 1

        if not preprocessing_component:
            preprocessing_component = ["none"]

        preprocessing_component = list(preprocessing_component)

        # Metrics component
        metrics_indexes, metrics_component = threshold(
            self.metrics,
            x[pos_x:pos_x + len(self.metrics)]
        )
        pos_x += len(self.metrics)

        metrics_component = list(metrics_component)

        # Metric weights
        metric_weights = None

        if self.optimize_metric_weights:
            raw_weights = x[pos_x:]
            selected_weights = [raw_weights[i] for i in metrics_indexes]
            metric_weights = dict(zip(metrics_component, selected_weights))

        population_size = hyperparameter_component[0]
        max_evals = hyperparameter_component[1]

        return {
            "algorithm_index": algorithm_index,
            "algorithm_name": algorithm_name,
            "algorithm": algorithm_component,

            "hyperparameters": list(hyperparameter_component),
            "population_size": population_size,
            "max_evals": max_evals,

            "preprocessing": preprocessing_component,
            "metrics": metrics_component,
            "metric_weights": metric_weights,
        }

    def _evaluate(self, x):
        r"""Evaluate the fitness of the pipeline."""

        config = self.decode_solution(x)

        algorithm_component = config["algorithm"]
        hyperparameter_component = config["hyperparameters"]
        preprocessing_component = config["preprocessing"]
        metrics_component = config["metrics"]

        if not metrics_component:
            return -np.inf

        if self.optimize_metric_weights:
            metrics_component = config["metric_weights"]

            if sum(metrics_component.values()) == 0:
                return -np.inf

        self.preprocessing_instance.set_preprocessing_algorithms(preprocessing_component)
        dataset = self.preprocessing_instance.apply_preprocessing()

        if dataset is None:
            return -np.inf

        problem = NiaARM(
            dataset.dimension,
            dataset.features,
            dataset.transactions,
            metrics=metrics_component,
        )

        task = Task(
            problem=problem,
            max_evals=hyperparameter_component[1],
            optimization_type=OptimizationType.MAXIMIZATION,
        )

        algorithm_component.population_size = hyperparameter_component[0]

        _, fitness = algorithm_component.run(task=task)

        if len(problem.rules) == 0:
            return -np.inf

        pipeline = Pipeline(
            x,
            preprocessing_component,
            algorithm_component.Name[1],
            metrics_component,
            hyperparameter_component,
            fitness,
            problem.rules,
        )

        if self.use_surrogate_fitness:
            fitness = pipeline.get_surrogate_fitness(["support", "confidence"])

        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_pipeline = copy.deepcopy(pipeline)

            try:
                self.best_pipeline.solution_vector = np.array(x, copy=True)
                self.best_pipeline.decoded_config = self.decode_solution(x)
            except Exception:
                pass

            if self.logger is not None:
                self.logger.log_pipeline(pipeline)

        if self.conserve_space:
            pipeline.clean()

        self.all_pipelines.append(pipeline)

        return fitness