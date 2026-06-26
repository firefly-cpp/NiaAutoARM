import copy
import statistics
from typing import Any, Dict, Optional

import numpy as np

from niaautoarm.utils import float_to_category, float_to_num, threshold



# ---------------------------------------------------------------------
# Safe evaluation
# ---------------------------------------------------------------------

def evaluate_without_side_effects(problem, x: np.ndarray) -> float:
    """
    Evaluate x using problem._evaluate(x), but restore the internal state
    afterwards.

    This is needed because the current AutoARMProblem._evaluate() updates:
        - best_fitness
        - best_pipeline
        - all_pipelines
        - algorithm.population_size
        - logger output

    Counterfactual evaluation should not overwrite the original optimized
    pipeline or pollute the stored pipeline list.
    """

    old_best_fitness = problem.best_fitness
    old_best_pipeline = copy.deepcopy(problem.best_pipeline)
    old_logger = problem.logger
    old_all_pipelines_len = len(problem.all_pipelines)

    old_population_sizes = [
        getattr(algorithm, "population_size", None)
        for algorithm in problem.algorithms
    ]

    try:
        problem.logger = None
        fitness = problem._evaluate(np.asarray(x, dtype=float))
        return float(fitness)

    finally:
        problem.best_fitness = old_best_fitness
        problem.best_pipeline = old_best_pipeline
        problem.logger = old_logger

        problem.all_pipelines = problem.all_pipelines[:old_all_pipelines_len]

        for algorithm, old_population_size in zip(
            problem.algorithms,
            old_population_sizes,
        ):
            if old_population_size is not None:
                algorithm.population_size = old_population_size


# ---------------------------------------------------------------------
# Genotype layout
# ---------------------------------------------------------------------

def get_solution_layout(problem) -> Dict[str, Optional[slice]]:
    """
    Return slices of the genotype vector corresponding to each pipeline
    component.
    """

    pos = 0
    layout = {}

    layout["algorithm"] = slice(pos, pos + 1)
    pos += 1

    layout["hyperparameters"] = slice(pos, pos + len(problem.hyperparameters))
    pos += len(problem.hyperparameters)

    if problem.allow_multiple_preprocessing:
        layout["preprocessing"] = slice(
            pos,
            pos + len(problem.preprocessing_methods)
        )
        pos += len(problem.preprocessing_methods)
    else:
        layout["preprocessing"] = slice(pos, pos + 1)
        pos += 1

    layout["metrics"] = slice(pos, pos + len(problem.metrics))
    pos += len(problem.metrics)

    if problem.optimize_metric_weights:
        layout["metric_weights"] = slice(pos, problem.dimension)
    else:
        layout["metric_weights"] = None

    return layout


def category_center(index: int, n_categories: int) -> float:
    """
    Return a value in [0, 1) that decodes to the selected category index.
    """

    if n_categories <= 1:
        return 0.5

    step = 1.0 / n_categories
    return min(0.999999, index * step + step / 2.0)


def binary_on() -> float:
    return 0.75


def binary_off() -> float:
    return 0.25


# ---------------------------------------------------------------------
# Neighbor generation
# ---------------------------------------------------------------------

def generate_vector_neighbors(
    problem,
    x: np.ndarray,
    *,
    hyperparameter_step: float = 0.05,
    metric_weight_step: float = 0.05,
):
    """
    Generate local one-step counterfactual candidates in genotype space.

    Each generated neighbor modifies exactly one type of pipeline component:
        - algorithm
        - hyperparameter
        - preprocessing
        - metric
        - metric_weight, only if enabled
    """

    x = np.asarray(x, dtype=float)
    layout = get_solution_layout(problem)
    base_config = problem.decode_solution(x)

    # ---------------------------------------------------------------
    # 1. Algorithm perturbations
    # ---------------------------------------------------------------
    current_algorithm_index = base_config["algorithm_index"]
    n_algorithms = len(problem.algorithms)

    for algorithm_index in range(n_algorithms):
        if algorithm_index == current_algorithm_index:
            continue

        x_new = np.array(x, copy=True)
        x_new[0] = category_center(algorithm_index, n_algorithms)

        yield {
            "action": f"algorithm -> {problem.algorithms[algorithm_index].Name[1]}",
            "edit_type": "algorithm",
            "x": x_new,
        }

    # ---------------------------------------------------------------
    # 2. Hyperparameter perturbations
    # ---------------------------------------------------------------
    h_slice = layout["hyperparameters"]

    for i in range(h_slice.start, h_slice.stop):
        hp_index = i - h_slice.start
        hp_name = problem.hyperparameters[hp_index]["parameter"]

        for delta in (-hyperparameter_step, hyperparameter_step):
            x_new = np.array(x, copy=True)
            x_new[i] = np.clip(x_new[i] + delta, 0.0, 0.999999)

            if np.allclose(x_new, x):
                continue

            yield {
                "action": f"adjust {hp_name}",
                "edit_type": "hyperparameter",
                "x": x_new,
            }

    # ---------------------------------------------------------------
    # 3. Preprocessing perturbations
    # ---------------------------------------------------------------
    p_slice = layout["preprocessing"]

    if problem.allow_multiple_preprocessing:
        for i, method in enumerate(problem.preprocessing_methods):
            x_new = np.array(x, copy=True)

            current_value = x_new[p_slice][i]
            x_new[p_slice][i] = (
                binary_off()
                if current_value >= 0.5
                else binary_on()
            )

            yield {
                "action": f"toggle preprocessing {method}",
                "edit_type": "preprocessing",
                "x": x_new,
            }

    else:
        current_method = (
            base_config["preprocessing"][0]
            if base_config["preprocessing"]
            else None
        )

        n_methods = len(problem.preprocessing_methods)

        for method_index, method in enumerate(problem.preprocessing_methods):
            if method == current_method:
                continue

            x_new = np.array(x, copy=True)
            x_new[p_slice.start] = category_center(method_index, n_methods)

            yield {
                "action": f"preprocessing -> {method}",
                "edit_type": "preprocessing",
                "x": x_new,
            }

    # ---------------------------------------------------------------
    # 4. Metric perturbations
    # ---------------------------------------------------------------
    m_slice = layout["metrics"]

    for i, metric in enumerate(problem.metrics):
        x_new = np.array(x, copy=True)

        current_value = x_new[m_slice][i]
        x_new[m_slice][i] = (
            binary_off()
            if current_value >= 0.5
            else binary_on()
        )

        yield {
            "action": f"toggle metric {metric}",
            "edit_type": "metric",
            "x": x_new,
        }

    # ---------------------------------------------------------------
    # 5. Metric-weight perturbations
    # ---------------------------------------------------------------
    if problem.optimize_metric_weights:
        w_slice = layout["metric_weights"]

        for i in range(w_slice.start, w_slice.stop):
            for delta in (-metric_weight_step, metric_weight_step):
                x_new = np.array(x, copy=True)
                x_new[i] = np.clip(x_new[i] + delta, 0.0, 0.999999)

                if np.allclose(x_new, x):
                    continue

                yield {
                    "action": f"adjust metric weight position {i}",
                    "edit_type": "metric_weight",
                    "x": x_new,
                }


# ---------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------

def config_signature(config: Dict[str, Any]):
    """
    Create a hashable signature of a decoded pipeline configuration.

    Used to remove duplicate counterfactuals that decode to the same pipeline.
    """

    metric_weights = config.get("metric_weights")

    if metric_weights is not None:
        weight_signature = tuple(sorted(metric_weights.items()))
    else:
        weight_signature = None

    return (
        config["algorithm_name"],
        tuple(config["hyperparameters"]),
        tuple(sorted(config["preprocessing"])),
        tuple(sorted(config["metrics"])),
        weight_signature,
    )


# ---------------------------------------------------------------------
# Counterfactual search
# ---------------------------------------------------------------------

def find_counterfactuals(
    problem,
    x: np.ndarray,
    *,
    mode: str = "degrade",
    min_delta: float = 0.01,
    max_neighbors: Optional[int] = None,
    hyperparameter_step: float = 0.05,
    metric_weight_step: float = 0.05,
) -> Dict[str, Any]:
    """
    Find local one-step counterfactuals.

    Parameters
    ----------
    problem:
        AutoARMProblem instance.

    x:
        Original genotype vector.

    mode:
        "degrade" accepts candidates with:
            delta <= -min_delta

        "improve" accepts candidates with:
            delta >= min_delta

    min_delta:
        Significance threshold epsilon.

    max_neighbors:
        Optional limit on the number of generated neighbors.

    Returns
    -------
    dict
        Counterfactual result dictionary containing:
            - mode
            - base_score
            - base_config
            - best
            - all
    """

    if mode not in {"degrade", "improve"}:
        raise ValueError("mode must be either 'degrade' or 'improve'.")

    x = np.asarray(x, dtype=float)

    base_score = evaluate_without_side_effects(problem, x)
    base_config = problem.decode_solution(x)

    accepted = []
    seen = set()

    neighbors = generate_vector_neighbors(
        problem,
        x,
        hyperparameter_step=hyperparameter_step,
        metric_weight_step=metric_weight_step,
    )

    for index, neighbor in enumerate(neighbors):
        if max_neighbors is not None and index >= max_neighbors:
            break

        x_new = neighbor["x"]
        new_score = evaluate_without_side_effects(problem, x_new)

        if not np.isfinite(new_score):
            continue

        delta = new_score - base_score

        if mode == "degrade":
            is_valid = delta <= -min_delta
        else:
            is_valid = delta >= min_delta

        if not is_valid:
            continue

        new_config = problem.decode_solution(x_new)
        signature = config_signature(new_config)

        if signature in seen:
            continue

        seen.add(signature)

        accepted.append({
            "action": neighbor["action"],
            "edit_type": neighbor["edit_type"],
            "x": np.array(x_new, copy=True),
            "score": float(new_score),
            "delta": float(delta),
            "abs_delta": float(abs(delta)),
            "config": new_config,
        })

    if mode == "degrade":
        accepted.sort(key=lambda cf: cf["delta"])
    else:
        accepted.sort(key=lambda cf: -cf["delta"])

    return {
        "mode": mode,
        "epsilon": float(min_delta),
        "base_score": float(base_score),
        "base_config": base_config,
        "best": accepted[0] if accepted else None,
        "all": accepted,
    }


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------

def compute_mean_degradation(result: Dict[str, Any]) -> Optional[float]:
    """
    Compute D(P), the mean absolute degradation over degrading counterfactuals.
    """

    if result is None or not result["all"]:
        return None

    return statistics.mean(abs(cf["delta"]) for cf in result["all"])


def compute_max_degradation(result: Dict[str, Any]) -> Optional[float]:
    """
    Compute D_max(P), the largest absolute degradation.
    """

    if result is None or not result["all"]:
        return None

    return max(abs(cf["delta"]) for cf in result["all"])


def compute_improvement_potential(result: Dict[str, Any]) -> Optional[float]:
    """
    Compute G(P), the largest positive improvement.
    """

    if result is None or not result["all"]:
        return None

    return max(cf["delta"] for cf in result["all"])


# ---------------------------------------------------------------------
# Text explanation
# ---------------------------------------------------------------------

def explain_counterfactual_result(result: Dict[str, Any]) -> str:
    """
    Produce a human-readable explanation of a counterfactual result.
    """

    lines = []

    lines.append(f"Mode: {result['mode']}")
    lines.append(f"Epsilon: {result.get('epsilon', 'N/A')}")
    lines.append(f"Base score: {result['base_score']:.6f}")

    base = result["base_config"]

    lines.append("Base configuration:")
    lines.append(f"  algorithm: {base['algorithm_name']}")
    lines.append(f"  population_size: {base['population_size']}")
    lines.append(f"  max_evals: {base['max_evals']}")
    lines.append(f"  preprocessing: {base['preprocessing']}")
    lines.append(f"  metrics: {base['metrics']}")

    if base.get("metric_weights") is not None:
        lines.append(f"  metric_weights: {base['metric_weights']}")

    if result["best"] is None:
        lines.append("")
        lines.append("No valid counterfactual found.")
        return "\n".join(lines)

    best = result["best"]
    cfg = best["config"]

    lines.append("")
    lines.append("Best counterfactual:")
    lines.append(f"  action: {best['action']}")
    lines.append(f"  edit_type: {best['edit_type']}")
    lines.append(f"  score: {best['score']:.6f}")
    lines.append(f"  delta: {best['delta']:+.6f}")
    lines.append(f"  abs_delta: {best['abs_delta']:.6f}")
    lines.append("  configuration:")
    lines.append(f"    algorithm: {cfg['algorithm_name']}")
    lines.append(f"    population_size: {cfg['population_size']}")
    lines.append(f"    max_evals: {cfg['max_evals']}")
    lines.append(f"    preprocessing: {cfg['preprocessing']}")
    lines.append(f"    metrics: {cfg['metrics']}")

    if cfg.get("metric_weights") is not None:
        lines.append(f"    metric_weights: {cfg['metric_weights']}")

    return "\n".join(lines)