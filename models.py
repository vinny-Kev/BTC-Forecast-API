"""
Machine Learning Models Module
Contains CatBoost, Random Forest, and Logistic Regression models with ensemble support
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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


class EnsembleModel:
    """
    Ensemble of CatBoost, Random Forest, and Logistic Regression models
    Supports both weighted averaging and stacking with meta-learner
    """
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.catboost = CatBoostModel(n_classes)
        self.random_forest = RandomForestModel(n_classes)
        self.logistic = None  # Will be loaded if exists
        self.weights = None
        self.meta_learner = None  # Meta-learner for stacking
        self.use_stacking = False
        
    def set_weights(self, catboost_weight=0.5, rf_weight=0.25, logistic_weight=0.25):
        """Set ensemble weights for weighted averaging"""
        total = catboost_weight + rf_weight + logistic_weight
        self.weights = {
            'catboost': catboost_weight / total,
            'rf': rf_weight / total,
            'logistic': logistic_weight / total
        }
        print(f"✓ Ensemble weights set: {self.weights}")
    
    def predict_proba(self, X_regular):
        """
        Get ensemble prediction probabilities
        
        Args:
            X_regular: Features for CatBoost, Random Forest, and Logistic Regression
            
        Returns:
            Ensemble probabilities (weighted average or stacked)
        """
        # Get base model predictions
        catboost_proba = self.catboost.predict_proba(X_regular)
        rf_proba = self.random_forest.predict_proba(X_regular)
        
        if self.logistic is not None:
            logistic_proba = self.logistic.predict_proba(X_regular)
        else:
            # If no logistic model, use equal weights for catboost and rf
            logistic_proba = np.zeros_like(catboost_proba)
        
        # If using stacking with meta-learner
        if self.use_stacking and self.meta_learner is not None:
            # Stack base model predictions as features
            meta_features = np.hstack([catboost_proba, rf_proba, logistic_proba])
            # Get final prediction from meta-learner
            ensemble_proba = self.meta_learner.predict_proba(meta_features)
        else:
            # Use weighted averaging
            if self.weights is None:
                self.set_weights()
            
            ensemble_proba = (
                self.weights['catboost'] * catboost_proba +
                self.weights['rf'] * rf_proba +
                self.weights['logistic'] * logistic_proba
            )
        
        return ensemble_proba
    
    def predict(self, X_regular):
        """Get ensemble predictions"""
        proba = self.predict_proba(X_regular)
        return np.argmax(proba, axis=1)
    
    def save(self, save_dir):
        """Save all models"""
        os.makedirs(save_dir, exist_ok=True)
        
        self.catboost.save(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.save(os.path.join(save_dir, 'random_forest_model.pkl'))
        
        if self.logistic is not None:
            joblib.dump(self.logistic, os.path.join(save_dir, 'logistic_model.pkl'))
        
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, os.path.join(save_dir, 'meta_learner.pkl'))
        
        # Save weights
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f)
        
        print(f"✓ Ensemble models saved to {save_dir}")
    
    def load(self, save_dir):
        """Load all models"""
        # Load base models
        self.catboost.load(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.load(os.path.join(save_dir, 'random_forest_model.pkl'))
        
        # Load logistic model if exists
        logistic_path = os.path.join(save_dir, 'logistic_model.pkl')
        if os.path.exists(logistic_path):
            loaded = joblib.load(logistic_path)
            # Handle both dict format (with 'model' key) and direct model format
            if isinstance(loaded, dict) and 'model' in loaded:
                self.logistic = loaded['model']
                print(f"✓ Logistic Regression model loaded from {logistic_path} (dict format)")
            else:
                self.logistic = loaded
                print(f"✓ Logistic Regression model loaded from {logistic_path}")
        else:
            print("⚠ No logistic model found, using 2-model ensemble")
        
        # Load meta-learner if exists (for stacking)
        meta_learner_path = os.path.join(save_dir, 'meta_learner.pkl')
        if os.path.exists(meta_learner_path):
            self.meta_learner = joblib.load(meta_learner_path)
            self.use_stacking = True
            print(f"✓ Meta-learner loaded from {meta_learner_path}")
            print("✓ Using stacking ensemble (base models → meta-learner)")
        else:
            print("⚠ No meta-learner found, using weighted averaging")
        
        # Load weights
        weights_path = os.path.join(save_dir, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.weights = json.load(f)
        else:
            self.set_weights()
        
        print(f"✓ Ensemble models loaded from {save_dir}")
