"""
Data Preprocessing Module
Handles data cleaning, temporal undersampling, and preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter


class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        
    def temporal_undersample(self, df, target_col='target', majority_class=0, ratio=0.3):
        """
        Perform temporal undersampling to balance classes while preserving time order
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            majority_class: The majority class to undersample
            ratio: Ratio of majority to minority samples (0.3 = 30% of original majority)
            
        Returns:
            Undersampled DataFrame preserving temporal order
        """
        print(f"\n{'='*60}")
        print("TEMPORAL UNDERSAMPLING")
        print(f"{'='*60}")
        
        # Get class distribution
        class_counts = df[target_col].value_counts().sort_index()
        print(f"Original class distribution:\n{class_counts}")
        
        # Separate classes
        majority_df = df[df[target_col] == majority_class]
        minority_dfs = [df[df[target_col] == cls] for cls in df[target_col].unique() if cls != majority_class]
        
        # Calculate how many majority samples to keep
        minority_count = sum([len(mdf) for mdf in minority_dfs])
        target_majority_count = int(minority_count / ratio) if ratio > 0 else len(majority_df)
        
        # Temporal undersampling: sample uniformly across time
        if len(majority_df) > target_majority_count:
            # Get indices at regular intervals to preserve temporal distribution
            step = len(majority_df) // target_majority_count
            selected_indices = majority_df.index[::step][:target_majority_count]
            majority_df = majority_df.loc[selected_indices]
        
        # Combine all classes
        undersampled_df = pd.concat([majority_df] + minority_dfs)
        
        # Sort by index to maintain temporal order
        undersampled_df = undersampled_df.sort_index()
        
        # Report new distribution
        new_class_counts = undersampled_df[target_col].value_counts().sort_index()
        print(f"\nUndersampled class distribution:\n{new_class_counts}")
        print(f"Reduction: {len(df)} → {len(undersampled_df)} samples ({len(undersampled_df)/len(df)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return undersampled_df
    
    def calculate_class_weights(self, y):
        """
        Calculate class weights for imbalanced learning
        
        Args:
            y: Target array
            
        Returns:
            Dictionary of class weights
        """
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        # Inverse frequency weighting
        class_weights = {
            cls: total_samples / (n_classes * count) 
            for cls, count in class_counts.items()
        }
        
        print(f"\n{'='*60}")
        print("CLASS WEIGHTS")
        print(f"{'='*60}")
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Class weights: {class_weights}")
        print(f"{'='*60}\n")
        
        return class_weights
    
    def split_temporal_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        Split data temporally (no shuffle to preserve time order)
        
        Args:
            df: DataFrame with features and target
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            train_df, val_df, test_df
        """
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"\n{'='*60}")
        print("TEMPORAL DATA SPLIT")
        print(f"{'='*60}")
        print(f"Total samples: {n_samples}")
        print(f"Training:   {len(train_df)} samples ({len(train_df)/n_samples*100:.1f}%)")
        print(f"Validation: {len(val_df)} samples ({len(val_df)/n_samples*100:.1f}%)")
        print(f"Test:       {len(test_df)} samples ({len(test_df)/n_samples*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return train_df, val_df, test_df
    
    def fit_scaler(self, X_train, scaler_type='robust'):
        """
        Fit scaler on training data
        
        Args:
            X_train: Training features
            scaler_type: 'standard' or 'robust'
        """
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(X_train)
        self.feature_columns = X_train.columns.tolist()
        
        print(f"✓ Fitted {scaler_type} scaler on {len(X_train)} training samples")
        
    def transform(self, X):
        """
        Transform features using fitted scaler
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features as DataFrame
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def prepare_lstm_sequences(self, X, y, sequence_length=30):
        """
        Prepare sequences for LSTM model
        
        Args:
            X: Features
            y: Target
            sequence_length: Length of each sequence
            
        Returns:
            X_sequences (3D array), y_sequences
        """
        X_sequences = []
        y_sequences = []
        
        # Convert to numpy for faster processing
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.values if isinstance(y, pd.Series) else y
        
        # Create sequences
        for i in range(sequence_length, len(X_values)):
            X_sequences.append(X_values[i-sequence_length:i])
            y_sequences.append(y_values[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✓ Created {len(X_sequences)} sequences of length {sequence_length}")
        print(f"  Shape: {X_sequences.shape} (samples, timesteps, features)")
        
        return X_sequences, y_sequences
    
    def add_lag_features(self, df, feature_cols, lags=[1, 2, 3, 5]):
        """
        Add lagged versions of features
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in feature_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Drop NaN rows created by lagging
        df = df.dropna()
        
        print(f"✓ Added {len(feature_cols) * len(lags)} lag features")
        
        return df
    
    def remove_outliers(self, df, columns=None, n_std=4):
        """
        Remove extreme outliers using z-score method
        
        Args:
            df: DataFrame
            columns: Columns to check for outliers (None = all numeric)
            n_std: Number of standard deviations for threshold
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        initial_len = len(df)
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= (n_std * std)]
        
        removed = initial_len - len(df)
        
        if removed > 0:
            print(f"✓ Removed {removed} outliers ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def prepare_data_pipeline(self, df, feature_cols, target_col='target',
                             train_ratio=0.7, val_ratio=0.15,
                             undersample=True, undersample_ratio=0.3,
                             scaler_type='robust',
                             sequence_length=30):
        """
        Complete data preparation pipeline
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            undersample: Whether to apply temporal undersampling
            undersample_ratio: Ratio for undersampling
            scaler_type: Type of scaler ('standard' or 'robust')
            sequence_length: Sequence length for LSTM
            
        Returns:
            Dictionary containing prepared datasets
        """
        print(f"\n{'='*60}")
        print("DATA PREPARATION PIPELINE")
        print(f"{'='*60}\n")
        
        # 1. Temporal split
        train_df, val_df, test_df = self.split_temporal_data(df, train_ratio, val_ratio)
        
        # 2. Undersample training data (optional)
        if undersample:
            train_df = self.temporal_undersample(train_df, target_col, ratio=undersample_ratio)
        
        # 3. Calculate class weights
        class_weights = self.calculate_class_weights(train_df[target_col])
        
        # 4. Separate features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # 5. Fit and transform with scaler
        print("Scaling features...")
        self.fit_scaler(X_train, scaler_type)
        X_train_scaled = self.transform(X_train)
        
        # Only transform validation set if it exists
        if len(X_val) > 0:
            X_val_scaled = self.transform(X_val)
        else:
            X_val_scaled = X_val  # Empty DataFrame
            
        X_test_scaled = self.transform(X_test)
        
        # 6. Prepare LSTM sequences
        print("\nPreparing LSTM sequences...")
        X_train_lstm, y_train_lstm = self.prepare_lstm_sequences(
            X_train_scaled, y_train, sequence_length
        )
        
        # Only prepare validation sequences if validation set exists
        if len(X_val) > 0:
            X_val_lstm, y_val_lstm = self.prepare_lstm_sequences(
                X_val_scaled, y_val, sequence_length
            )
        else:
            X_val_lstm = np.array([]).reshape(0, sequence_length, X_train_scaled.shape[1])
            y_val_lstm = np.array([])
            
        X_test_lstm, y_test_lstm = self.prepare_lstm_sequences(
            X_test_scaled, y_test, sequence_length
        )
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")
        
        return {
            # Original DataFrames
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            
            # Regular features (for CatBoost, Random Forest)
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            
            # LSTM sequences
            'X_train_lstm': X_train_lstm,
            'y_train_lstm': y_train_lstm,
            'X_val_lstm': X_val_lstm,
            'y_val_lstm': y_val_lstm,
            'X_test_lstm': X_test_lstm,
            'y_test_lstm': y_test_lstm,
            
            # Metadata
            'class_weights': class_weights,
            'feature_columns': feature_cols,
            'scaler': self.scaler
        }
