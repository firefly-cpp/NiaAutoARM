# Description: This is a basic example of how to run the ARMPipelineOptimizer class.
from niaarm.dataset import Dataset
from niaautoarm.armpipelineoptimizer import ARMPipelineOptimizer

if __name__ == "__main__":        

    # load dataset from csv
    data = Dataset("datasets/Abalone.csv")

    # define which preprocessing methods to use
    # data squashing is now supported
    preprocessing = ["min_max_scaling", "squash_cosine", "none"]

    # define algorithms for searching the association rules
    algorithms = ["PSO", "DE", "GA", "FA"]

    # define hyperparameters and their min/max values
    hyperparameter1 = {
        "parameter": "NP",
        "min": 5,
        "max": 15
    }

    hyperparameter2 = {
        "parameter": "N_FES",
        "min": 1000,
        "max": 2000
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

    algo = ARMPipelineOptimizer(data=data, 
                                feature_prepocessing_techniques=preprocessing,
                                rule_mining_algorithms=algorithms, metrics=metrics,
                                hyperparameters=hyperparameters,
                                log=True,
                                log_verbose=True,
                                log_output_file=None
                                )
    
    algo.run(optimization_algorithm="ParticleSwarmAlgorithm",population_size=5,max_iters=2,output_pipeline_file="results.pckl")