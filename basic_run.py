# Description: This is a basic example of how to run the ARMPipelineOptimizer class.
from niaarm.dataset import Dataset
from niaautoarm.armoptimizer import AutoARMOptimizer
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution, GeneticAlgorithm, FireflyAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation


import argparse
import time

def parse_cli():
    cli_parser = argparse.ArgumentParser(description="Run the AutoARM framework.")
    cli_parser.add_argument("--dataset", type=str, default="Abalone", help="Dataset name. Dataset must be in the datasets folder.")
    cli_parser.add_argument("--algorithm", type=str, default="ParticleSwarmAlgorithm", help="Algorithm to use for optimization of the pipelines.")
    cli_parser.add_argument("--popsize", type=int, default=30, help="Population size.")
    cli_parser.add_argument("--maxfes", type=int, default=500, help="Maximum number of pipeline evaluations.")
    cli_parser.add_argument("--ow", type=bool, default=True, help="Optimize metric weights.")
    cli_parser.add_argument("--amp", type=bool, default=True, help="Allow multiple preprocessing.")
    cli_parser.add_argument("--sf", type=bool, default=True, help="Use surrogate fitness.")
    cli_parser.add_argument("--seed", type=int, default=37, help="Random seed.")
    cli_parser.add_argument("--run", type=int, default=1, help="Run number")
    return cli_parser.parse_args()

if __name__ == "__main__":

    cli = parse_cli()

    # load dataset from csv
    data = Dataset("datasets/{}.csv".format(cli.dataset))

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
    


    start_run = time.time()

    best_pipeline = pipeline_optimizer.run(
        optimization_algorithm=cli.algorithm,
        population_size=cli.popsize,
        max_evals=cli.maxfes,
        seed=cli.seed,
        optimize_metric_weights=cli.ow,
        allow_multiple_preprocessing=cli.amp,
        use_surrogate_fitness=cli.sf,
        output_pipeline_file="pipeline_{}_{}_{}_{}_{}.ppln".format(cli.algorithm,cli.dataset,cli.popsize,cli.maxfes,cli.run))
    end_run = time.time()

    print("Run time: {:.4f} seconds".format(end_run - start_run))    
    print(best_pipeline)