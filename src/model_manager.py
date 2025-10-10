import threading
import os
import json
import numpy as np
import tensorflow as tf
import pickle
import joblib
import logging

logger = logging.getLogger("model_manager")


class ModelCandidate:
    def __init__(self, model, feature_columns, preprocessor, metadata, path):
        self.model = model
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.path = path


class ModelManager:
    """Load, validate, and atomically swap models for serving.

    This manager focuses on transformer-style models and their supporting
    artifacts (feature_columns, preprocessor, metadata). It keeps the active
    candidate in memory and provides thread-safe swap operations.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._active: ModelCandidate | None = None
        self._previous: ModelCandidate | None = None

    def get_active(self) -> ModelCandidate | None:
        with self._lock:
            return self._active

    def load_candidate(self, model_dir: str) -> ModelCandidate:
        """Load model and artifacts from a model directory. Raises on missing files."""
        logger.info(f"ModelManager: loading candidate from {model_dir}")

        # metadata
        metadata_path = os.path.join(model_dir, 'transformer_regressor_v2_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing transformer metadata: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
        feature_columns = []
        if os.path.exists(feature_columns_path):
            try:
                with open(feature_columns_path, 'rb') as f:
                    feature_columns = pickle.load(f)
            except Exception:
                try:
                    feature_columns = joblib.load(feature_columns_path)
                except Exception as e:
                    logger.warning(f"Failed to load feature_columns: {e}")

        # preprocessor
        preprocessor = None
        for candidate in ('preprocessor.pkl', 'scaler.pkl', 'preprocessor.joblib'):
            ppath = os.path.join(model_dir, candidate)
            if os.path.exists(ppath):
                try:
                    preprocessor = joblib.load(ppath)
                    break
                except Exception:
                    try:
                        with open(ppath, 'rb') as f:
                            preprocessor = pickle.load(f)
                            break
                    except Exception:
                        logger.warning(f"Failed to load preprocessor from {ppath}")

        # model file
        model_path = os.path.join(model_dir, 'transformer_regressor_v2_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Transformer model file not found: {model_path}")

        # Try to import custom objects if available
        custom_objs = {}
        try:
            # local import; training code may have used src.models_transformer_regression_v2
            from src.models_transformer_regression_v2 import PositionalEncoding, TransformerBlock  # noqa: F401
            custom_objs = {
                'PositionalEncoding': PositionalEncoding,
                'TransformerBlock': TransformerBlock,
            }
        except Exception:
            # no custom objects available; continue and let load_model raise if needed
            logger.debug("ModelManager: no custom objects module available or failed to import")

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, safe_mode=False)

        return ModelCandidate(model=model, feature_columns=feature_columns, preprocessor=preprocessor, metadata=metadata, path=model_dir)

    def validate_candidate(self, candidate: ModelCandidate, raise_on_error: bool = True) -> bool:
        """Run a simple predict with a zero input to confirm shape compatibility."""
        logger.info(f"ModelManager: validating candidate from {candidate.path}")
        try:
            seq_len = int(candidate.metadata.get('sequence_length', 1))
            n_features = len(candidate.feature_columns) if candidate.feature_columns else candidate.metadata.get('n_features') or 1
            dummy = np.zeros((1, seq_len, n_features), dtype=float)
            # wrap calls in try/except to convert errors to informative logs
            _ = candidate.model.predict(dummy)
            return True
        except Exception as e:
            logger.warning(f"ModelManager: validation failed: {e}")
            if raise_on_error:
                raise
            return False

    def swap_in(self, candidate: ModelCandidate):
        """Atomically make candidate the active model."""
        with self._lock:
            logger.info(f"ModelManager: swapping in model from {candidate.path}")
            self._previous = self._active
            self._active = candidate

    def reload_latest(self, model_dir: str):
        """Convenience: load, validate, and swap the candidate from model_dir."""
        candidate = self.load_candidate(model_dir)
        self.validate_candidate(candidate)
        self.swap_in(candidate)
