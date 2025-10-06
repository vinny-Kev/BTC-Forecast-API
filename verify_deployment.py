#!/usr/bin/env python3
"""
Pre-deployment verification script
Tests all critical components before pushing to production
"""

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    
    try:
        import prediction_api
        print("✅ prediction_api imports successfully")
    except Exception as e:
        print(f"❌ prediction_api import failed: {e}")
        return False
    
    try:
        from models import EnsembleModel
        print("✅ EnsembleModel imports successfully")
    except Exception as e:
        print(f"❌ EnsembleModel import failed: {e}")
        return False
    
    try:
        from data_scraper import DataScraper
        print("✅ DataScraper imports successfully (lazy loading)")
    except Exception as e:
        print(f"❌ DataScraper import failed: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imports successfully")
    except Exception as e:
        print(f"❌ FeatureEngineer import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from prediction_api import get_latest_model_dir, load_models
        import os
        
        model_dir = get_latest_model_dir()
        if not model_dir:
            print("❌ No model directory found")
            return False
        
        print(f"✅ Found model directory: {model_dir}")
        
        # Check required files exist
        required_files = [
            'catboost_model.cbm',
            'random_forest_model.pkl', 
            'logistic_model.pkl',
            'ensemble_weights.json',
            'preprocessor.pkl',
            'feature_columns.pkl'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_ensemble_structure():
    """Test that ensemble has correct structure"""
    print("\nTesting ensemble structure...")
    
    try:
        from models import EnsembleModel
        
        ensemble = EnsembleModel(n_classes=3)
        
        # Check it has the right components
        if hasattr(ensemble, 'catboost'):
            print("✅ Has CatBoost model")
        else:
            print("❌ Missing CatBoost model")
            return False
            
        if hasattr(ensemble, 'random_forest'):
            print("✅ Has Random Forest model")
        else:
            print("❌ Missing Random Forest model")
            return False
            
        if hasattr(ensemble, 'logistic'):
            print("✅ Has Logistic Regression model")
        else:
            print("❌ Missing Logistic Regression model")
            return False
        
        # Check it doesn't have LSTM
        if hasattr(ensemble, 'lstm'):
            print("❌ Still has LSTM model (should be removed)")
            return False
        else:
            print("✅ LSTM model correctly removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble structure test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🔍 Pre-deployment Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_loading, 
        test_ensemble_structure
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Safe to deploy.")
        return 0
    else:
        print("❌ Some tests failed. Do not deploy.")
        return 1

if __name__ == "__main__":
    exit(main())