import numpy as np
from niaarm import NiaARM
from niaarm import Dataset, squash
from niapy.problems import Problem
from niapy.algorithms.basic import DifferentialEvolution, FireflyAlgorithm, ParticleSwarmAlgorithm, GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.task import Task, OptimizationType
from niaautoarm.utils import calculate_dimension_of_the_problem
import csv

def float_to_category(component, val):
    r"""Map float value to component (category). """
    if val == 1:
        return len(component) - 1
    return int(val * len(component))


def float_to_num(component, val):
    r"""Map float value to integer. """
    parameters = [1] * len(component)
    for i in range(len(component)):
        parameters[i] = int(val *
                            int(component[i]['max']) +
                            int(component[i]['min']))

    return parameters


def threshold(component, val):
    r"""Calculate whether feature is over a threshold. """
    selected = [c for i, c in enumerate(component) if val[i] > 0.5]
    return tuple(selected)


class AutoARM(Problem):
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
            preprocessing,
            algorithms,
            hyperparameters,
            metrics,
            logging=False
    ):
        r"""Initialize instance of AutoARM.

        Arguments:

        """
        # calculate the dimension of the problem
        dimension = calculate_dimension_of_the_problem(
            preprocessing, algorithms, hyperparameters, metrics)

        super().__init__(dimension, 0, 1)
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.logging = logging
        self.rules = None
        self.best_fitness = -np.inf

        # for writing pipelines in excel file
        self.all_pipelines = []

    def dump_to_file(self):
        with open("results.txt", 'w', newline='') as f:
            writer = csv.writer(f)
            for pip in self.pipelines:
                writer.writerow([pip.preprocessing, pip.algorithm, pip.metrics])

    def _evaluate(self, sol):
        # get components
        preprocessing_component = self.preprocessing[float_to_category(
            self.preprocessing, sol[0])]

        algorithm_component = self.algorithms[float_to_category(
            self.algorithms, sol[1])]

        hyperparameter_component = float_to_num(self.hyperparameters, sol[2:3])

        metrics_component = threshold(self.metrics, sol[4:10])

        # perform data squashing if selected

        if preprocessing_component == "squash_euclidean":
            self.dataset = squash(
                self.dataset,
                threshold=0.9,
                similarity='euclidean')
        elif preprocessing_component == "squash_cosine":
            self.dataset = squash(
                self.dataset,
                threshold=0.9,
                similarity='cosine')

        problem = NiaARM(
            self.dataset.dimension,
            self.dataset.features,
            self.dataset.transactions,
            metrics=metrics_component)

        # build niapy task
        task = Task(
            problem=problem,
            max_evals=hyperparameter_component[1],
            optimization_type=OptimizationType.MAXIMIZATION)

        # see full list of available algorithms:
        # https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
        if algorithm_component == "DE":
            algo = DifferentialEvolution(
                population_size=hyperparameter_component[0],
                differential_weight=0.5,
                crossover_probability=0.9)
        elif algorithm_component == "PSO":
            algo = ParticleSwarmAlgorithm(
                population_size=hyperparameter_component[0],
                min_velocity=-4.0,
                max_velocity=4.0)
        elif algorithm_component == "GA":
            algo = GeneticAlgorithm(
                population_size=hyperparameter_component[0],
                crossover=uniform_crossover,
                mutation=uniform_mutation,
                crossover_rate=0.45,
                mutation_rate=0.9)
        elif algorithm_component == "FA":
            algo = FireflyAlgorithm(
                population_size=hyperparameter_component[0],
                alpha=1.0,
                beta0=0.2,
                gamma=1.0)
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm_component}')

        _, fitness = algo.run(task=task)

        # store each pipeline in csv file for post-processing

        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.rules = problem.rules

            if self.logging:
                print(
                    f'Preprocessing: {preprocessing_component}'
                    f' - Algorithm: {algorithm_component}'
                    f' - Hyperparameters: {hyperparameter_component}'
                    f' - Metrics: {metrics_component}\n'
                    f'Fitness: {self.best_fitness:.4f}'
                    f' - Mean Support: {self.rules.mean("support"):.4f}'
                    f' - Mean Confidence: {self.rules.mean("confidence"):.4f}')

        return fitness
