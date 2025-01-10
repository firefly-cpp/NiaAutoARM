from niaautoarm import __version__
import unittest
from niaarm.dataset import Dataset
import os
from niaautoarm.armoptimizer import AutoARMOptimizer
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution
from niaautoarm.pipeline import Pipeline
from niaautoarm.preprocessing import Preprocessing
import pandas as pd

def test_version():
    assert __version__ == '0.2.0'

class PreprocessingTestCase(unittest.TestCase):
    def setUp(self):
        self.data = Dataset(os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/test_dataset.csv")
        
    def test_is_preprocessing_set(self):
        preprocessing = Preprocessing(self.data, ["min_max_scaling", "z_score_normalization"])
        self.assertIsNotNone(preprocessing)

    def test_preprocessing_none(self):
        preprocessing = Preprocessing(self.data, ["none"])
        preprocessed_data = preprocessing.apply_preprocessing()
        self.assertTrue(self.data.transactions.equals(preprocessed_data.transactions))

    def test_preprocessing_not_none(self):
        preprocessing = Preprocessing(self.data, ["min_max_scaling"])
        preprocessed_data = preprocessing.apply_preprocessing()
        self.assertFalse(self.data.transactions.equals(preprocessed_data.transactions))

    
class AutoARMPipelineOptimizerTestCase(unittest.TestCase):
    def setUp(self):
        self.data = Dataset(os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/test_dataset.csv")
        
    def test_pipeline_optimizer_allow_amp(self):
        print("Testing AutoARMPipelineOptimizer")
        # define which preprocessing methods to use
        pipeline_optimizer = AutoARMOptimizer(
            data=self.data, 
            feature_prepocessing_techniques=["min_max_scaling", "z_score_normalization"],
            rule_mining_algorithms=[ParticleSwarmOptimization(min_velocity=-4, max_velocity=4,seed=2), DifferentialEvolution(crossover_probability=0.9, differential_weight=0.5,seed=2)], 
            metrics=["support", "confidence"],
            hyperparameters=[{"parameter": "NP", "min": 10, "max": 30}, {"parameter": "N_FES", "min": 2000, "max": 10000}],
            conserve_space=True
            )
        best_pipeline = pipeline_optimizer.run(
            optimization_algorithm="ParticleSwarmOptimization", 
            population_size=3, 
            max_evals=3, 
            seed=1, 
            optimize_metric_weights=False, 
            allow_multiple_preprocessing=True, 
            use_surrogate_fitness=True, 
            output_pipeline_file=None
            )
        
        self.assertIsNotNone(best_pipeline)
        self.assertIsInstance(best_pipeline, Pipeline)
        self.assertTrue(best_pipeline.fitness > 0)

        self.assertTrue(
            best_pipeline.get_preprocessing() == ("min_max_scaling", "z_score_normalization") or
            best_pipeline.get_preprocessing() == ("z_score_normalization", "min_max_scaling") or
            best_pipeline.get_preprocessing() == ("min_max_scaling",) or
            best_pipeline.get_preprocessing() == ("z_score_normalization",) or
            best_pipeline.get_preprocessing() == ("none",)
        )

    def test_pipeline_optimizer_not_allow_amp(self):
        print("Testing AutoARMPipelineOptimizer")
        # define which preprocessing methods to use
        pipeline_optimizer = AutoARMOptimizer(
            data=self.data, 
            feature_prepocessing_techniques=["min_max_scaling", "z_score_normalization"],
            rule_mining_algorithms=[ParticleSwarmOptimization(min_velocity=-4, max_velocity=4,seed=2), DifferentialEvolution(crossover_probability=0.9, differential_weight=0.5,seed=2)], 
            metrics=["support", "confidence"],
            hyperparameters=[{"parameter": "NP", "min": 10, "max": 30}, {"parameter": "N_FES", "min": 2000, "max": 10000}],
            log=False,
            log_verbose=False,
            log_output_file=None,
            conserve_space=True
            )
        best_pipeline = pipeline_optimizer.run(
            optimization_algorithm="ParticleSwarmOptimization", 
            population_size=3, 
            max_evals=3, 
            seed=1, 
            optimize_metric_weights=False, 
            allow_multiple_preprocessing=False, 
            use_surrogate_fitness=True, 
            output_pipeline_file=None
            )
        

        self.assertIsNotNone(best_pipeline)
        self.assertIsInstance(best_pipeline, Pipeline)
        self.assertTrue(best_pipeline.fitness > 0)
        self.assertTrue(best_pipeline.get_preprocessing() == ("min_max_scaling",) or best_pipeline.get_preprocessing() == ("z_score_normalization",) or best_pipeline.get_preprocessing() == ("none",))


        