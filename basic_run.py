from autoarm import AutoARM
from niaarm.dataset import Dataset
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution
from niapy.task import Task, OptimizationType

# load dataset from csv
data = Dataset("datasets/Abalone.csv")

# preprocess dataset and obtain features
features = data.get_features()

# calculate dimension of the problem
dimension = 10

# obtain transaction database
transactions = data.transaction_data

# define which preprocessing methods to use
preprocessing = ["FeatureSelection", "HotCodeEncoding"]

# feature selection algorithms
fs = ["jDEFSTH", "PCA"]

# algorithms for searching the association rules
algorithms = ["PSO", "DE", "GA", "FA"]

#hyperparameters
hyperparameters = ["A", "B", "C"]

# evaluation criterions
evaluations = ["support", "confidence", "shrinkage", "coverage"]

# Create a problem::: 
# dimension represents dimension of the problem;
# 0, 1 represents the range of search space
# features represent the list of features, while transactions depicts the list of transactions

problem = AutoARM(dimension, 0, 1, features, transactions, preprocessing, algorithms, hyperparameters, evaluations)

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

