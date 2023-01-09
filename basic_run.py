from niaautoarm import AutoARM
from niaarm.dataset import Dataset
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.task import Task, OptimizationType

# load dataset from csv
data = Dataset("datasets/Abalone.csv")

# set dimension of the problem
# components = preprocessing (1) + algorithm (1) + hyperparameters (2 (NP,FES)) +
# metrics (8, (support, confidence, lift, coverage, amplitude, inclusion, comprehensibility))
dimension = 11 # can be extended when new components are added

# define which preprocessing methods to use
preprocessing = ["FeatureSelection", "HotCodeEncoding", "Squashing"]

# feature selection algorithms
fs = ["jDEFSTH", "PCA"]

# algorithms for searching the association rules
algorithms = ["PSO", "DE", "GA", "FA"]

#define hyperparameters and their min/max values
hyperparameters = []

hyperparameter1 = {
    "parameter": "NP",
    "min": "5",
    "max": "75"
    }

hyperparameter2 = {
    "parameter": "N_FES",
    "min": "5000",
    "max": "25000"
    }
# create array
hyperparameters = [hyperparameter1, hyperparameter2]

# evaluation criterions
metrics = ["support", "confidence", "coverage", "amplitude", "inclusion", "comprehensibility"]

# Create a problem::: 
# dimension represents dimension of the problem;
# 0, 1 represents the range of search space
# features represent the list of features, while transactions depicts the list of transactions

problem = AutoARM(dimension, 0, 1, preprocessing, algorithms, hyperparameters, metrics)

# build niapy task
task = Task(
    problem=problem,
    max_iters=3,
    optimization_type=OptimizationType.MAXIMIZATION)

# use Differential Evolution (DE) algorithm
# see full list of available algorithms: https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
algo = DifferentialEvolution(population_size=50, differential_weight=0.5, crossover_probability=0.9)

# use Particle swarm Optimization (PSO) algorithm from NiaPy library
algo2 = ParticleSwarmAlgorithm(
    population_size=100,
    min_velocity=-4.0,
    max_velocity=4.0)

# run algorithm
best = algo.run(task=task)

