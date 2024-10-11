# Description: This is a basic example of how to run the ARMPipelineOptimizer class.
from niaarm.dataset import Dataset
from niaautoarm.armoptimizer import AutoARMOptimizer
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution, GeneticAlgorithm, FireflyAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation

import random as rnd
import numpy as np
import argparse

def parse_cli():
    cli_parser = argparse.ArgumentParser(description="Run the AutoARM framework.")
    cli_parser.add_argument("--dataset", type=str, default="datasets/Abalone.csv", help="Path to the dataset.")
    cli_parser.add_argument("--algorithm", type=str, default="ParticleSwarmAlgorithm", help="Algorithm to use for optimization of the pipelines.")
    cli_parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    cli_parser.add_argument("--run", type=int, default=1, help="Run number")
    return cli_parser.parse_args()

if __name__ == "__main__":

    cli = parse_cli()
    #rnd.seed(cli.seed)
    #np.random.seed(cli.seed)

    # load dataset from csv
    data = Dataset(cli.dataset)

    # define which preprocessing methods to use
    preprocessing = ["min_max_scaling", "squash_cosine", "none"]

    # define algorithms for searching the association rules
    algorithms = [ParticleSwarmOptimization(min_velocity=-4, max_velocity=4,seed=cli.seed),
                    DifferentialEvolution(crossover_probability=0.9, differential_weight=0.5,seed=cli.seed),
                    GeneticAlgorithm(crossover=uniform_crossover, mutation=uniform_mutation, crossover_rate=0.9, mutation_rate=0.1,seed=cli.seed), 
                    FireflyAlgorithm(alpha=1.0, beta0=0.2, gamma=1.0,seed=cli.seed)]

    # define hyperparameters and their min/max values
    hyperparameter1 = {
        "parameter": "NP",
        "min": 10,
        "max": 30
    }

    hyperparameter2 = {
        "parameter": "N_FES",
        "min": 2000,
        "max": 10000
    }
    # create array
    hyperparameters = [hyperparameter1, hyperparameter2]

    # evaluation criteria
    metrics = [
        "support",
        "confidence",
        "coverage",
        "amplitude",
        "inclusion",
        "comprehensibility"]    

    pipeline_optimizer = AutoARMOptimizer(data=data, 
                                feature_prepocessing_techniques=preprocessing,
                                rule_mining_algorithms=algorithms, 
                                metrics=metrics,
                                hyperparameters=hyperparameters,
                                log=True,
                                log_verbose=True,
                                log_output_file=None
                                )
    
    pipeline_optimizer.run(
        optimization_algorithm=cli.algorithm,
        population_size=10,
        max_evals=100,
        seed=cli.seed,
        optimize_metric_weights=True,
        allow_multiple_preprocessing=False,
        use_surrogate_fitness=False,
        output_pipeline_file="pipeline_{}_{}_{}.ppln".format(cli.algorithm,cli.dataset,cli.run))