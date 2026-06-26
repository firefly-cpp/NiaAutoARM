
from niaautoarm.XAIArmPipeline import XAIArmPipeline


pipeline_path = "path_to_pipeline"
xai = (
    XAIArmPipeline(
        pipeline_path=pipeline_path,
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