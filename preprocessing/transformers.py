# load dependencies
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from warnings import filterwarnings
    

class Z_Score(BaseEstimator, TransformerMixin):
    """Compute Z-scores for each column in given dataframe"""

    def __init__(self):
        pass

    def transform(self, X, y=None):
        mu = X.mean(axis = 0)
        sigma = X.std(axis = 0)
        X = (X - mu) / sigma
        return X

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class MaxFeatureIndex(BaseEstimator, TransformerMixin):
    
    """Find max feature index for each job id from given dataframe"""

    def __init__(self):
        pass


    def transform(self, X, y = None):

        max_feature_index = X.values.argmax(axis = 1)
        X['max_feature_index'] = max_feature_index
        return X[['max_feature_index']]

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class MaxFeatureAbsMeanDiff(BaseEstimator, TransformerMixin):
    """
    
    Find absolute difference of value of max feature index 
    from mean value of that feature for each job_id
    
    """
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self


    def transform(self, X, y = None):
        max_feature_index_values = X.values.argmax(axis = 1)
        unique_indexes = np.unique(max_feature_index_values)
        mean_by_feature_index = X.iloc[:, unique_indexes].mean(axis = 0)
        mean_by_feature_index = pd.DataFrame(mean_by_feature_index).set_index(unique_indexes).squeeze()

        X['max_feat_idx'] = max_feature_index_values
        X['max_value'] = X.max(axis = 1)
        X['mean_by_feature_index'] = X['max_feat_idx'].map(mean_by_feature_index)
        X['diff'] = abs(X['max_value'] - X['mean_by_feature_index'])
        return X[['diff']] 

class CustomNormalizer(BaseEstimator, TransformerMixin):
    """
    Create a custom normalizer for a dataframe
    """

    def __init__(self, custom_normalization_func):
        self.custom_normalization_func = custom_normalization_func
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        return self.custom_normalization_func(X)
        
    
# def save_transformer_state(pipe_obj, filepath_name):
#     """
#     Saves pipeline in .pkl/.joblib file for further usage
#     """
#     joblib.dump(pipe_obj, filepath_name)


# def load_transformer(filepath_name):
#     """
#     Loads pkl/joblib file with transformer pipeline
#     """
    
#     loaded_pipe = joblib.load(filepath_name)
#     return loaded_pipe

