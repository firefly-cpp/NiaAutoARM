import os
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from niaarm.dataset import Dataset

from niaautoarm.autoarmproblem import AutoARMProblem

from niaautoarm.explain.counterfactual import (
    find_counterfactuals,
    compute_mean_degradation,
    compute_max_degradation,
    compute_improvement_potential,
)


@dataclass
class XAICounterfactualSummary:
    base_score: float
    epsilon: float
    num_degrading_counterfactuals: int
    num_improving_counterfactuals: int
    mean_degradation_D_P: Optional[float]
    max_degradation_Dmax_P: Optional[float]
    improvement_potential_G_P: Optional[float]


class XAIArmPipeline:
    """
    Counterfactual explanation wrapper for a saved NiaAutoARM pipeline.
    """


    def __init__(
        self,
        pipeline_path: str,
        dataset_path: str,
        *,
        min_delta: float = 0.01,
        max_neighbors: Optional[int] = 10,
        hyperparameter_step: float = 0.05,
        metric_weight_step: float = 0.05,
        cache_dir: Optional[str] = None,
        generate_degrading_cfs: bool = True,
        generate_improving_cfs: bool = False,
    ):
        self.pipeline_path = pipeline_path
        self.dataset_path = dataset_path

        self.min_delta = min_delta
        self.max_neighbors = max_neighbors
        self.hyperparameter_step = hyperparameter_step
        self.metric_weight_step = metric_weight_step

        self.cache_dir = cache_dir

        self.generate_degrading_cfs = generate_degrading_cfs
        self.generate_improving_cfs = generate_improving_cfs

        self.loaded_object = None
        self.best_pipeline = None
        self.solution_vector = None
        self.run_config = None
        self.problem = None

        self.degrading_result = None
        self.improving_result = None

    def load_pipeline(self):
        with open(self.pipeline_path, "rb") as file:
            self.loaded_object = pickle.load(file)

        self.best_pipeline = self._extract_best_pipeline(self.loaded_object)
        self.solution_vector = self._extract_solution_vector(self.best_pipeline)
        self.run_config = self._extract_run_config(self.best_pipeline)

        return self

    def _extract_best_pipeline(self, obj):
        if hasattr(obj, "best_pipeline"):
            return obj.best_pipeline

        if isinstance(obj, dict) and "best_pipeline" in obj:
            return obj["best_pipeline"]

        # In case the file itself is already a Pipeline object.
        if hasattr(obj, "get_solution_vector") or hasattr(obj, "solution_vector"):
            return obj

        raise ValueError("Could not extract best_pipeline from saved object.")

    def _extract_solution_vector(self, pipeline):
        if hasattr(pipeline, "get_solution_vector"):
            x = pipeline.get_solution_vector()
        elif hasattr(pipeline, "solution_vector"):
            x = pipeline.solution_vector
        else:
            raise ValueError("Pipeline does not contain solution_vector.")

        if x is None:
            raise ValueError("Pipeline solution_vector is None.")

        return np.asarray(x, dtype=float)

    def _extract_run_config(self, pipeline) -> Dict[str, Any]:
        if hasattr(pipeline, "get_run_config"):
            config = pipeline.get_run_config()
        elif hasattr(pipeline, "run_config"):
            config = pipeline.run_config
        else:
            raise ValueError("Pipeline does not contain run_config.")

        required = ["fps", "rma", "metrics", "hyperparameters", "ow", "amp", "sf"]
        missing = [key for key in required if key not in config]

        if missing:
            raise ValueError(f"Pipeline run_config is missing keys: {missing}")

        return config

    def build_problem(self):
        if self.run_config is None:
            raise RuntimeError("Call load_pipeline() before build_problem().")

        dataset = Dataset(self.dataset_path)

        self.problem = AutoARMProblem(
            dataset=dataset,
            preprocessing_methods=self.run_config["fps"],
            algorithms=self.run_config["rma"],
            hyperparameters=self.run_config["hyperparameters"],
            metrics=self.run_config["metrics"],
            optimize_metric_weights=self.run_config["ow"],
            allow_multiple_preprocessing=self.run_config["amp"],
            use_surrogate_fitness=self.run_config["sf"],
            conserve_space=True,
            logger=None,
        )

        return self

    def generate_counterfactuals(self, *, force: bool = False):
        if self.problem is None:
            raise RuntimeError("Call build_problem() before generate_counterfactuals().")

        if self.generate_degrading_cfs:
            self.degrading_result = self._generate_or_load_counterfactuals(
                mode="degrade",
                force=force,
            )

        if self.generate_improving_cfs:
            self.improving_result = self._generate_or_load_counterfactuals(
                mode="improve",
                force=force,
            )

        return self

    def _generate_or_load_counterfactuals(self, *, mode: str, force: bool = False):
        cache_path = self._cache_path(mode)

        if cache_path is not None and os.path.exists(cache_path) and not force:
            with open(cache_path, "rb") as file:
                return pickle.load(file)

        result = find_counterfactuals(
            self.problem,
            self.solution_vector,
            mode=mode,
            min_delta=self.min_delta,
            max_neighbors=self.max_neighbors,
            hyperparameter_step=self.hyperparameter_step,
            metric_weight_step=self.metric_weight_step,
        )

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, "wb") as file:
                pickle.dump(result, file)

        return result

    def _cache_path(self, mode: str):
        if self.cache_dir is None:
            return None

        base = os.path.splitext(os.path.basename(self.pipeline_path))[0]

        return os.path.join(
            self.cache_dir,
            f"{base}_{mode}_cfs.pkl",
        )

    def summarize(self) -> XAICounterfactualSummary:
        if self.degrading_result is None and self.improving_result is None:
            raise RuntimeError("No counterfactuals available. Call generate_counterfactuals().")

        if self.degrading_result is not None:
            base_score = self.degrading_result["base_score"]
        else:
            base_score = self.improving_result["base_score"]

        num_degrading = (
            len(self.degrading_result["all"])
            if self.degrading_result is not None
            else 0
        )

        num_improving = (
            len(self.improving_result["all"])
            if self.improving_result is not None
            else 0
        )

        mean_degradation = (
            compute_mean_degradation(self.degrading_result)
            if self.degrading_result is not None
            else None
        )

        max_degradation = (
            compute_max_degradation(self.degrading_result)
            if self.degrading_result is not None
            else None
        )

        improvement_potential = (
            compute_improvement_potential(self.improving_result)
            if self.improving_result is not None
            else None
        )

        return XAICounterfactualSummary(
            base_score=base_score,
            epsilon=self.min_delta,
            num_degrading_counterfactuals=num_degrading,
            num_improving_counterfactuals=num_improving,
            mean_degradation_D_P=mean_degradation,
            max_degradation_Dmax_P=max_degradation,
            improvement_potential_G_P=improvement_potential,
        )

    def to_dict(self):
        summary = self.summarize()

        return {
            "pipeline_path": self.pipeline_path,
            "dataset_path": self.dataset_path,
            "base_score": summary.base_score,
            "epsilon": summary.epsilon,
            "num_degrading_counterfactuals": summary.num_degrading_counterfactuals,
            "num_improving_counterfactuals": summary.num_improving_counterfactuals,
            "mean_degradation_D_P": summary.mean_degradation_D_P,
            "max_degradation_Dmax_P": summary.max_degradation_Dmax_P,
            "improvement_potential_G_P": summary.improvement_potential_G_P,
        }

xai = (
    XAIArmPipeline(
        pipeline_path="pipelines_testek/pipeline_ParticleSwarmAlgorithm_Abalone_3_3_1.ppln",
        dataset_path="datasets/Wine.csv",
        min_delta=0.01,
        cache_dir="saved_counterfactuals",
        generate_degrading_cfs=True,
        generate_improving_cfs=True,
    )
    .load_pipeline()
    .build_problem()
    .generate_counterfactuals()
)

summary = xai.summarize()
print(summary)

print(xai.to_dict())