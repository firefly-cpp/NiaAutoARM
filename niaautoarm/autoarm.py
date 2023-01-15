import numpy as np
from niaarm import NiaARM
from niapy.problems import Problem
from niapy.algorithms.basic import DifferentialEvolution, FireflyAlgorithm, ParticleSwarmAlgorithm, GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.task import Task, OptimizationType


def float_to_category(component, val):
    if val == 1:
        return len(component) - 1
    return int(val * len(component))


def float_to_num(component, val):
    parameters = [1] * len(component)
    for i in range(len(component)):
        parameters[i] = int(val * int(component[i]['max']) + int(component[i]['min']))

    return parameters


def threshold(component, val):
    selected = []
    for i in range(len(val)):
        if val[i] > 0.5:
            selected.append(component[i])
    return tuple(selected)


class AutoARM(Problem):
    r"""Implementation of AutoARM.
    
    Date:
        2022
    
    Attributes:
    
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
        super().__init__(11, 0, 1)
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.logging = logging
        self.rules = None
        self.best_fitness = -np.inf

    def _evaluate(self, sol):

        # firstly, genotype to phenotype mapping
        # print("Solution: ", sol)
        # TODO: add preprocessing
        # preprocessing_component = self.preprocessing[float_to_category(self.preprocessing, sol[0])]
        # print("Izbrani preprocessing: ", preprocessing_component)
        algorithm_component = self.algorithms[float_to_category(self.algorithms, sol[1])]
        # print("Izbrani algorithm: ", algorithm_component)
        hyperparameter_component = float_to_num(self.hyperparameters, sol[2:3])
        # print("Izbrane vrednosti hyp:", hyperparameter_component)
        metrics_component = threshold(self.metrics, sol[4:10])
        # print("Izbrane metrics", metrics_component)

        problem = NiaARM(self.dataset.dimension, self.dataset.features, self.dataset.transactions, metrics=metrics_component)

        # build niapy task
        task = Task(problem=problem, max_evals=hyperparameter_component[1], optimization_type=OptimizationType.MAXIMIZATION)

        # use Differential Evolution (DE) algorithm from the NiaPy library
        # see full list of available algorithms: https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
        if algorithm_component == "DE":
            algo = DifferentialEvolution(population_size=hyperparameter_component[0], differential_weight=0.5,
                                         crossover_probability=0.9)
        elif algorithm_component == "PSO":
            algo = ParticleSwarmAlgorithm(population_size=hyperparameter_component[0], min_velocity=-4.0, max_velocity=4.0)
        elif algorithm_component == "GA":
            algo = GeneticAlgorithm(population_size=hyperparameter_component[0], crossover=uniform_crossover,
                                    mutation=uniform_mutation, crossover_rate=0.45, mutation_rate=0.9)
        elif algorithm_component == "FA":
            algo = FireflyAlgorithm(population_size=hyperparameter_component[0], alpha=1.0, beta0=0.2, gamma=1.0)
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm_component}')

        _, fitness = algo.run(task=task)

        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.rules = problem.rules

            if self.logging:
                print(f'Fitness: {self.best_fitness:.4f}'
                      f' - Mean Support: {self.rules.mean("support"):.4f}'
                      f' - Mean Confidence: {self.rules.mean("confidence"):.4f}')

        return fitness
