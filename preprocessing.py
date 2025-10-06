"""
Preprocessing module 
Placeholder to support loading legacy pickled preprocessors
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

class DataPreprocessor:
    """
    Data Preprocessor class for backward compatibility
    Wrapper around sklearn scalers
    """
    def __init__(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

# Make the commonly used preprocessing classes available
__all__ = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'DataPreprocessor']
