import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from niaarm.dataset import Dataset
from sklearn.cluster import KMeans

def min_max_scaling(dataset):
    '''Scale float data to have a minimum of 0 and a maximum of 1'''

    scaled_transactions = dataset.transactions.copy()
    min_max_scaler = MinMaxScaler()
    for head in dataset.header:   
        if dataset.transactions[head].dtype == 'float':
            scaled_transactions[head] = min_max_scaler.fit_transform(dataset.transactions[head].values.reshape(-1, 1))
            
    return Dataset(scaled_transactions)

def z_score_normalization(dataset):
    '''Scale float data to have a mean of 0 and a standard deviation of 1'''

    scaled_transactions = dataset.transactions.copy()
    scaler = StandardScaler()

    for head in dataset.header:   
        if dataset.transactions[head].dtype == 'float':
            scaled_transactions[head] = scaler.fit_transform(dataset.transactions[head].values.reshape(-1, 1))
            
    return Dataset(scaled_transactions)

def discretization_equal_width(dataset, bins=10):
    '''Discretize float data into equal width bins'''

    discretized_transactions = dataset.transactions.copy()
    for head in dataset.header:
        if dataset.transactions[head].dtype == 'float':
            discretized_transactions[head] = pd.cut(dataset.transactions[head], bins=bins, labels=False)

    return Dataset(discretized_transactions)

def discretization_equal_frequency(dataset,q=5):
    '''Discretize float data into equal frequency bins'''

    discretized_transactions = dataset.transactions.copy()
    for head in dataset.header:
        if dataset.transactions[head].dtype == 'float':
            discretized_transactions[head] = pd.qcut(dataset.transactions[head], q=q, labels=False)

    return Dataset(discretized_transactions)

def discretization_kmeans(dataset,n_clusters=4):
    '''Discretize float data using KMeans clustering'''

    disretized_transactions = dataset.transactions.copy()
    for head in dataset.header:
        if dataset.transactions[head].dtype == 'float':
            disretized_transactions[head] = KMeans(n_init='auto',n_clusters=n_clusters).fit_predict(dataset.transactions[head].values.reshape(-1, 1))  
    
    return Dataset(disretized_transactions)

def remove_highly_correlated_features(dataset,threshold=0.95):
    '''Remove highly correlated features'''
    uncorrelated_transactions = dataset.transactions.copy()
    correlation_matrix = uncorrelated_transactions.corr(numeric_only=True).abs()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] >= threshold:
                colname = correlation_matrix.columns[i]
                uncorrelated_transactions = uncorrelated_transactions.drop(colname, axis=1)

    print(uncorrelated_transactions)
