from niaautoarm import AutoARM
from niaarm.dataset import Dataset
from niapy.task import Task, OptimizationType
from niaautoarm.armpipelineoptimizer import ArmPipelineOptimizer

if __name__ == "__main__":        

    # load dataset from csv
    data = Dataset("datasets/Abalone.csv")

    # define which preprocessing methods to use
    # data squashing is now supported
    preprocessing = ["squash_euclid", "squash_cosine", "none"]

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

    algo = ArmPipelineOptimizer(data=data, 
                                feature_prepocessing_techniques=preprocessing,
                                rule_mining_algorithms=algorithms, metrics=metrics,
                                hyperparameters=hyperparameters,
                                log=True,
                                log_verbose=True,
                                log_output_file=None
                                )
    algo.run(5, "ParticleSwarmAlgorithm",max_iters=2)