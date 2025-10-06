"""
Machine Learning Models Module
Contains CatBoost, Random Forest, and LSTM models with custom configurations
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical
import joblib
import json
import os


class CatBoostModel:
    """CatBoost model with class weighting"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, class_weights=None, iterations=1000, depth=6, learning_rate=0.03, 
                    l2_leaf_reg=3, bagging_temperature=1.0, random_strength=1.0):
        """Build CatBoost model with regularization"""
        
        # Convert class weights to CatBoost format
        if class_weights:
            # CatBoost expects weights as a list or array for each sample
            # We'll use auto_class_weights instead
            class_weights_mode = 'Balanced'
        else:
            class_weights_mode = None
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,  # L2 regularization
            bagging_temperature=bagging_temperature,  # Bayesian bagging intensity
            random_strength=random_strength,  # Randomness for scoring splits
            loss_function='MultiClass',
            eval_metric='TotalF1',
            auto_class_weights=class_weights_mode,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            task_type='CPU'
        )
        
        print("✓ CatBoost model built")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model"""
        print("\nTraining CatBoost...")
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = (X_val, y_val)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                use_best_model=True,
                plot=False
            )
        else:
            # Train without validation set
            self.model.fit(
                X_train, y_train,
                plot=False
            )
        
        print("✓ CatBoost training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = self.model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model"""
        self.model.save_model(path)
        print(f"✓ CatBoost model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        print(f"✓ CatBoost model loaded from {path}")


class RandomForestModel:
    """Random Forest model with class weighting"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, class_weights=None, n_estimators=200, max_depth=15, 
                    min_samples_split=10, min_samples_leaf=5, max_features='sqrt'):
        """Build Random Forest model with regularization"""
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,  # Regularization: higher = less overfitting
            min_samples_leaf=min_samples_leaf,    # Regularization: higher = less overfitting
            max_features=max_features,            # Limit features per split
            class_weight='balanced' if class_weights else None,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("✓ Random Forest model built")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        self.model.fit(X_train, y_train)
        
        print("✓ Random Forest training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, path)
        print(f"✓ Random Forest model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(path)
        print(f"✓ Random Forest model loaded from {path}")


class LSTMModel:
    """LSTM model for sequence prediction"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build(self, input_shape, lstm_units=[128, 64], dropout=0.3, recurrent_dropout=0.0, l2_reg=0.0):
        """
        Build LSTM model with regularization
        
        Args:
            input_shape: (timesteps, features)
            lstm_units: List of LSTM layer units
            dropout: Dropout rate for dropout layers
            recurrent_dropout: Dropout rate for recurrent connections
            l2_reg: L2 regularization strength
        """
        from tensorflow.keras import regularizers
        
        model = keras.Sequential()
        
        # First LSTM layer with regularization
        model.add(layers.LSTM(
            lstm_units[0],
            input_shape=input_shape,
            return_sequences=len(lstm_units) > 1,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
        ))
        
        # Additional LSTM layers with regularization
        for i, units in enumerate(lstm_units[1:]):
            return_seq = i < len(lstm_units) - 2
            model.add(layers.LSTM(
                units, 
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
            ))
        
        # Dense layers with L2 regularization
        model.add(layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
        ))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(
            32, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
        ))
        model.add(layers.Dropout(dropout / 2))
        
        # Output layer
        model.add(layers.Dense(self.n_classes, activation='softmax'))
        
        self.model = model
        
        print("✓ LSTM model built with regularization")
        print(f"  Architecture: LSTM{lstm_units} → Dense[64, 32] → Softmax({self.n_classes})")
        print(f"  Regularization: Dropout={dropout}, Recurrent Dropout={recurrent_dropout}, L2={l2_reg}")
        
        return self
    
    def compile(self, learning_rate=0.001, class_weights=None):
        """Compile LSTM model"""
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # For weighted loss, we'll use sample weights during training
        # Use SparseCategoricalAccuracy since we have integer labels
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Removed F1Score to avoid shape mismatch
        )
        
        print("✓ LSTM model compiled")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             class_weights=None, epochs=50, batch_size=32):
        """Train LSTM model"""
        print("\nTraining LSTM...")
        
        # Create sample weights from class weights
        sample_weights = None
        if class_weights:
            sample_weights = np.array([class_weights[y] for y in y_train])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None and len(X_val) > 0 else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None and len(X_val) > 0 else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train with or without validation
        if X_val is not None and y_val is not None and len(X_val) > 0:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                sample_weight=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
        else:
            # Train without validation
            self.history = self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
        
        print("✓ LSTM training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X, verbose=0)
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
        print(f"✓ LSTM model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"✓ LSTM model loaded from {path}")


class EnsembleModel:
    """Ensemble of CatBoost, Random Forest, and LSTM models"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.catboost = CatBoostModel(n_classes)
        self.random_forest = RandomForestModel(n_classes)
        self.lstm = LSTMModel(n_classes)
        self.weights = None
        
    def set_weights(self, catboost_weight=0.4, rf_weight=0.3, lstm_weight=0.3):
        """Set ensemble weights"""
        total = catboost_weight + rf_weight + lstm_weight
        self.weights = {
            'catboost': catboost_weight / total,
            'rf': rf_weight / total,
            'lstm': lstm_weight / total
        }
        print(f"✓ Ensemble weights set: {self.weights}")
    
    def predict_proba(self, X_regular, X_lstm):
        """
        Get ensemble prediction probabilities
        
        Args:
            X_regular: Features for CatBoost and Random Forest
            X_lstm: Sequences for LSTM
            
        Returns:
            Weighted average probabilities
        """
        if self.weights is None:
            self.set_weights()
        
        # Get predictions from each model
        catboost_proba = self.catboost.predict_proba(X_regular)
        rf_proba = self.random_forest.predict_proba(X_regular)
        lstm_proba = self.lstm.predict_proba(X_lstm)
        
        # Weighted average
        ensemble_proba = (
            self.weights['catboost'] * catboost_proba +
            self.weights['rf'] * rf_proba +
            self.weights['lstm'] * lstm_proba
        )
        
        return ensemble_proba
    
    def predict(self, X_regular, X_lstm):
        """Get ensemble predictions"""
        proba = self.predict_proba(X_regular, X_lstm)
        return np.argmax(proba, axis=1)
    
    def save(self, save_dir):
        """Save all models"""
        os.makedirs(save_dir, exist_ok=True)
        
        self.catboost.save(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.save(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.lstm.save(os.path.join(save_dir, 'lstm_model.h5'))
        
        # Save weights
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f)
        
        print(f"✓ Ensemble models saved to {save_dir}")
    
    def load(self, save_dir):
        """Load all models"""
        self.catboost.load(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.load(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.lstm.load(os.path.join(save_dir, 'lstm_model.h5'))
        
        # Load weights
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'r') as f:
            self.weights = json.load(f)
        
        print(f"✓ Ensemble models loaded from {save_dir}")
