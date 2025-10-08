"""
Data Preprocessing Module
Wrapper for sklearn preprocessors
"""

from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np


class DataPreprocessor:
    """
    Wrapper class for data preprocessing
    Uses RobustScaler by default (better for outliers)
    """
    
    def __init__(self, scaler_type='robust'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: 'robust' or 'standard'
        """
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
    
    def fit(self, X):
        """Fit the scaler"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        """Transform the data"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform"""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform"""
        return self.scaler.inverse_transform(X)
