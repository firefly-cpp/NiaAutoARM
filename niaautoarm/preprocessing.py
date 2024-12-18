from niaarm.dataset import Dataset
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import numpy as np
from pandas.api.types import is_float_dtype, is_integer_dtype

class Preprocessing:
    r"""Preprocessing class for preprocessing the dataset.

    Attributes:
        dataset (Dataset): The dataset to be preprocessed.
        preprocessing_algorithms (list): List of preprocessing algorithms to be used.
    """


    def __init__(self,dataset, prepocessing_algorithms : list):
        self.dataset = dataset
        self.preprocessing_algorithms = prepocessing_algorithms

        self._order = {'min_max_scaling': 1, 'z_score_normalization' : 1, 'squash_euclidean' : 1, 'squash_cosine' : 1, 'yeo_johnson' : 1,
                        'remove_highly_correlated_features' : 2,
                        'discretization_equal_width' : 3, 'discretization_equal_frequency' : 3, 'discretization_kmeans' : 3,
                        'none' : 4} #TODO : Use just one of the numbered values for each rank
        

    def has_preprocessing_failed(self,dataset):
        if dataset is None:
            return True
        if dataset.transactions.empty:
            return True
        if dataset.transactions.isnull().values.any():
            return True
        return False

    def set_preprocessing_algorithms(self, preprocessing_algorithms):
        self.preprocessing_algorithms = preprocessing_algorithms

    def get_preprocessing_algorithms(self):
        return self.preprocessing_algorithms    
    
    def apply_preprocessing(self):
        dataset = self.dataset
        self._reorder_preprocessing_algorithms()

        for preprocessing_algorithm in self.preprocessing_algorithms:
            try:
                dataset = Dataset(self._apply_preprocessing_algorithm(preprocessing_algorithm,dataset))
            except:
                return None
            if self.has_preprocessing_failed(dataset):
                return None
        
        return dataset


    def _apply_preprocessing_algorithm(self, preprocessing_algorithm, dataset):
        if preprocessing_algorithm == 'min_max_scaling':
            return self._min_max_scaling(dataset)
        
        elif preprocessing_algorithm == 'z_score_normalization':
            return self._z_score_normalization(dataset)
        
        elif preprocessing_algorithm == 'discretization_equal_width':
            return self._discretization_equal_width(dataset)
        
        elif preprocessing_algorithm == 'squash_euclidean':
            Warning('This method is very slow, need to optimize !!!')
            return self.squash(dataset, threshold=0.95, similarity='euclidean') #Very slow, need to optimize !!!
        
        elif preprocessing_algorithm == 'squash_cosine':
            return self.squash(dataset, threshold=0.95, similarity='cosine')
        
        elif preprocessing_algorithm == 'discretization_equal_frequency':
            return self.discretization_equal_frequency(dataset,q=5)
        
        elif preprocessing_algorithm == 'discretization_kmeans':
            return self.discretization_kmeans(dataset,n_clusters=4)
        
        elif preprocessing_algorithm == 'remove_highly_correlated_features':
            return self.remove_highly_correlated_features(dataset,threshold=0.95)
        
        elif preprocessing_algorithm == 'yeo_johnson':
            return self.yeo_johnson(dataset)
        
        elif preprocessing_algorithm == 'none':
            return dataset.transactions
        
        else:
            raise ValueError('Unknown preprocessing algorithm: {}'.format(self.preprocessing_algorithm))
        

    def _reorder_preprocessing_algorithms(self):
        #print(self.preprocessing_algorithms)
        self.preprocessing_algorithms = sorted(self.preprocessing_algorithms, key=lambda x: self._order[x])
        # Remove multiple occurences of the same rank


    def _min_max_scaling(self,dataset):
        '''Scale float data to have a minimum of 0 and a maximum of 1'''

        scaled_transactions = dataset.transactions.copy()
        min_max_scaler = MinMaxScaler()
        for head in dataset.header:   
            if dataset.transactions[head].dtype == 'float':
                scaled_transactions[head] = min_max_scaler.fit_transform(dataset.transactions[head].values.reshape(-1, 1))
                
        return scaled_transactions

    def _z_score_normalization(self,dataset):
        '''Scale float data to have a mean of 0 and a standard deviation of 1'''

        scaled_transactions = dataset.transactions.copy()
        scaler = StandardScaler()

        for head in dataset.header:   
            if dataset.transactions[head].dtype == 'float':
                scaled_transactions[head] = scaler.fit_transform(dataset.transactions[head].values.reshape(-1, 1))
                
        return scaled_transactions

    def _discretization_equal_width(self, dataset, bins=10):
        '''Discretize float data into equal width bins'''

        discretized_transactions = dataset.transactions.copy()
        for head in dataset.header:
            if dataset.transactions[head].dtype == 'float':
                discretized_transactions[head] = pd.cut(dataset.transactions[head], bins=bins, labels=False)

        return discretized_transactions
    
    def yeo_johnson(self,dataset):
        '''Apply Yeo-Johnson transformation to the dataset'''

        transformed_transactions = dataset.transactions.copy()
        transformer = PowerTransformer()

        for head in dataset.header:
            if dataset.transactions[head].dtype == 'float':
                transformed_transactions[head] = transformer.fit_transform(dataset.transactions[head].values.reshape(-1, 1))

        return transformed_transactions
    
    def _euclidean(self,u, v, features):
        dist = 0
        for f in features:
            if f.dtype == 'cat':
                weight = 1 / len(f.categories)
                if u[f.name] != v[f.name]:
                    dist += weight * weight
            else:
                weight = 1 / (f.max_val - f.min_val)
                dist += (u[f.name] - v[f.name]) * (u[f.name] - v[f.name]) * weight * weight

        return 1 - (dist ** 0.5)


    def _cosine_similarity(self,u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


    def _mean_or_mode(self,column):
        if is_float_dtype(column):
            return column.mean()
        elif is_integer_dtype(column):
            return round(column.mean())
        else:
            return column.mode()


    def squash(self,dataset, threshold, similarity='euclidean'):
        """Squash dataset.

        Args:
            dataset (Dataset): Dataset to squash.
            threshold (float): Similarity threshold. Should be between 0 and 1.
            similarity (str): Similarity measure for comparing transactions (euclidean or cosine). Default: 'euclidean'.

        Returns:
            Dataset: Squashed dataset.

        """
        transactions = dataset.transactions
        transactions_dummies = pd.get_dummies(dataset.transactions).to_numpy()
        num_transactions = len(transactions)
    
        squashed = np.zeros(num_transactions, dtype=bool)
        squashed_transactions = pd.DataFrame(columns=transactions.columns, dtype=int)

        for pos in range(num_transactions):
            if squashed[pos]:
                continue

            squashed_set = transactions.iloc[pos:pos + 1]
            squashed[pos] = True

            for i in range(pos + 1, num_transactions):
                if squashed[i]:
                    continue
                if similarity == 'euclidean':
                    distance = self._euclidean(transactions.iloc[pos], transactions.iloc[i], dataset.features)
                    
                else:
                    distance = self._cosine_similarity(transactions_dummies[pos], transactions_dummies[i])

                if distance >= threshold:
                    squashed_set = pd.concat([squashed_set, transactions.iloc[i:i + 1]], ignore_index=True)
                    squashed[i] = True

            if not squashed_set.empty:
                squashed_transaction = squashed_set.agg(self._mean_or_mode)
                squashed_transactions = pd.concat([squashed_transactions, squashed_transaction], ignore_index=True)

        return squashed_transactions
    
    def discretization_equal_frequency(self,dataset,q=5):
        '''Discretize float data into equal frequency bins'''

        discretized_transactions = dataset.transactions.copy()
        for head in dataset.header:
            if dataset.transactions[head].dtype == 'float':
                discretized_transactions[head] = pd.qcut(dataset.transactions[head], q=q, labels=False)

        return discretized_transactions

    def discretization_kmeans(self,dataset,n_clusters=4):
        '''Discretize float data using KMeans clustering'''

        disretized_transactions = dataset.transactions.copy()
        for head in dataset.header:
            if dataset.transactions[head].dtype == 'float':
                disretized_transactions[head] = KMeans(n_init='auto',n_clusters=n_clusters).fit_predict(dataset.transactions[head].values.reshape(-1, 1))  
        
        return disretized_transactions

    def remove_highly_correlated_features(self,dataset,threshold=0.95):
        '''Remove highly correlated features'''
        uncorrelated_transactions = dataset.transactions.copy()
        correlation_matrix = uncorrelated_transactions.corr(numeric_only=True).abs()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] >= threshold:
                    colname = correlation_matrix.columns[i]
                    uncorrelated_transactions = uncorrelated_transactions.drop(colname, axis=1)

        return uncorrelated_transactions

