"""
Machine Learning Models Module
Contains CatBoost, Random Forest, and Logistic Regression models for Bitcoin prediction
No LSTM - removed for better performance and reliability
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os


class CatBoostModel:
    """CatBoost Gradient Boosting model"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, iterations=500, learning_rate=0.1, depth=6, l2_leaf_reg=3, 
              random_strength=1, bagging_temperature=1, class_weights=None):
        """Build CatBoost model with regularization"""
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            verbose=False,
            random_seed=42,
            class_weights=class_weights,
            loss_function='MultiClass' if self.n_classes > 2 else 'Logloss'
        )
        
        print("✓ CatBoost model built")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model"""
        print("\nTraining CatBoost...")
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=50 if eval_set else None
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
    """Random Forest ensemble model"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, class_weights=None, n_estimators=50, max_depth=8, 
                    min_samples_split=20, min_samples_leaf=10, max_features='sqrt'):
        """Build Random Forest model with reduced complexity to prevent overfitting"""
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,        # Reduced from 200 to 50
            max_depth=max_depth,              # Reduced from 15 to 8  
            min_samples_split=min_samples_split,  # Increased from 10 to 20
            min_samples_leaf=min_samples_leaf,    # Increased from 5 to 10
            max_features=max_features,            # Keep sqrt for regularization
            class_weight='balanced' if class_weights else None,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("✓ Random Forest model built with reduced complexity")
        print(f"  Architecture: {n_estimators} trees, max_depth={max_depth}, regularization enhanced")
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


class LogisticRegressionModel:
    """Logistic Regression model for baseline and ensemble balance"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        
    def build(self, class_weights=True, C=1.0, max_iter=1000, solver='lbfgs'):
        """Build Logistic Regression model"""
        
        self.model = LogisticRegression(
            C=C,                              # Regularization strength (lower = more regularization)
            max_iter=max_iter,
            solver=solver,                    # lbfgs works well for small datasets
            class_weight='balanced' if class_weights else None,
            random_state=42,
            multi_class='ovr'                 # One-vs-rest for multi-class
        )
        
        print("✓ Logistic Regression model built")
        print(f"  Regularization: C={C}, solver={solver}")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Logistic Regression model with feature scaling"""
        print("\nTraining Logistic Regression...")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        print("✓ Logistic Regression training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance (coefficients)"""
        if self.n_classes == 2:
            coeffs = self.model.coef_[0]
        else:
            # For multi-class, use average absolute coefficients
            coeffs = np.mean(np.abs(self.model.coef_), axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        print(f"✓ Logistic Regression model saved to {path}")
    
    def load(self, path):
        """Load model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"✓ Logistic Regression model loaded from {path}")


class EnsembleModel:
    """Ensemble of CatBoost, Random Forest, and Logistic Regression models with optional Meta-Learner"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.catboost = CatBoostModel(n_classes)
        self.random_forest = RandomForestModel(n_classes)
        self.logistic = LogisticRegressionModel(n_classes)
        self.weights = None
        self.meta_learner = None  # LSTM meta-learner for stacking
        self.use_stacking = False
        
    def set_weights(self, catboost_weight=0.5, rf_weight=0.25, logistic_weight=0.25):
        """Set ensemble weights - 3 model ensemble"""
        total = catboost_weight + rf_weight + logistic_weight
        self.weights = {
            'catboost': catboost_weight / total,
            'rf': rf_weight / total,
            'logistic': logistic_weight / total
        }
        print(f"✓ Ensemble weights set: {self.weights}")
        print(f"  CatBoost: {catboost_weight/total:.1%}, RF: {rf_weight/total:.1%}, Logistic: {logistic_weight/total:.1%}")
    
    def predict_proba(self, X_regular):
        """
        Get ensemble prediction probabilities
        
        Args:
            X_regular: Features for CatBoost, Random Forest, and Logistic Regression
            
        Returns:
            Weighted average probabilities (or meta-learner output if stacking is enabled)
        """
        # Get base model predictions
        catboost_proba = self.catboost.predict_proba(X_regular)
        rf_proba = self.random_forest.predict_proba(X_regular)
        logistic_proba = self.logistic.predict_proba(X_regular)
        
        # If using stacking with meta-learner
        if self.use_stacking and self.meta_learner is not None:
            # Stack base predictions as meta-features
            meta_features = np.hstack([catboost_proba, rf_proba, logistic_proba])
            
            # Get meta-learner prediction
            return self.meta_learner.predict_proba(meta_features)
        
        # Otherwise, use weighted averaging
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
        
        # Save individual models
        self.catboost.save(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.save(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.logistic.save(os.path.join(save_dir, 'logistic_model.pkl'))
        
        # Save meta-learner if exists
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, os.path.join(save_dir, 'meta_learner.pkl'))
            print(f"✓ Meta-learner saved")
        
        # Save weights
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f)
        
        print(f"✓ Ensemble models saved to {save_dir}")
    
    def load(self, save_dir):
        """Load all models"""
        self.catboost.load(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.load(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.logistic.load(os.path.join(save_dir, 'logistic_model.pkl'))
        
        # Load meta-learner if exists
        meta_learner_path = os.path.join(save_dir, 'meta_learner.pkl')
        if os.path.exists(meta_learner_path):
            self.meta_learner = joblib.load(meta_learner_path)
            self.use_stacking = True
            print(f"✓ Meta-learner loaded from {meta_learner_path}")
            print(f"  Using stacked ensemble architecture")
        else:
            self.use_stacking = False
            print(f"  No meta-learner found, using weighted averaging")
        
        # Load weights
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'r') as f:
            self.weights = json.load(f)
        
        print(f"✓ Ensemble models loaded from {save_dir}")
        print(f"  Weights: {self.weights}")