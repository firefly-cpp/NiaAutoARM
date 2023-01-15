from niaarm.dataset import Dataset
from niaaml.preprocessing.feature_selection import ParticleSwarmOptimization
from niaaml.data import CSVDataReader


class FeatureSelection:
    """
    Class for working with Feature selection methods
    
    Args:
        dataset (Dataset): Dataset object.
    """

    def __init__(self, dataset):
        """
        Initialization method for FeatureSelection class
        
        Args:
            dataset: A full dataset
        """
        self.dataset = dataset

        self.features = []
        self.transactions = []

    def load_dataset(self):
        data = Dataset(self.dataset)
        self.features = data.features
        self.transactions = data.transactions

    def feature_selection_threshold(self):
        # prepare data reader using csv file
        data_reader = CSVDataReader(
            src=self.dataset,
            has_header=True,
            contains_classes=True,
        )

        print(data_reader.get_y())
        # instantiate feature selection algorithm
        fs = ParticleSwarmOptimization()

        # set parameters of the instantiated algorithm
        fs.set_parameters(C1=1.5, C2=2.0)

        features_mask = fs.select_features(data_reader.get_x(), data_reader.get_y())

        return features_mask
