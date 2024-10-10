import csv
import pickle

class ARMPipelineStatistics:
    def __init__(self, all_pipelines, best_pipeline):
        self.pipelines = all_pipelines
        self.best_pipeline = best_pipeline


    def dump_to_file(self, output_pipeline_file):
        with open(output_pipeline_file, 'wb') as file:
            pickle.dump(self, file)

    def _calculate_most_frequent_preprocessing_technique(self):
        return max(set([pip.preprocessing for pip in self.pipelines]), key=[pip.preprocessing for pip in self.pipelines].count)
    

    def _calculate_most_frequent_algorithm(self):
        return max(set([pip.algorithm for pip in self.pipelines]), key=[pip.algorithm for pip in self.pipelines].count)
    
    def _calculateMost_frequent_algorithm_by_quartile_range(self):
        algorithms = [pip.algorithm for pip in self.pipelines]
        algorithms.sort()
        q1 = algorithms[int(len(algorithms) * 0.25)]
        q2 = algorithms[int(len(algorithms) * 0.5)]
        q3 = algorithms[int(len(algorithms) * 0.75)]
        return max(set([q1, q2, q3]), key=[q1, q2, q3].count)

