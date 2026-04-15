import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

# Senior Pro: V9 Universal Oracle Architecture
# Combines Global V8 Wisdom with Local Shift Adaptation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "production_scrap_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "production_features.pkl"

class UniversalOracleV9:
    """The Final Senior Oracle: Self-Calibrating, Precise, and Generalized."""
    
    def __init__(self):
        self.model = None
        self.features = None
        self.risk_histories = {} # machine_id -> list of float
        self.max_history = 60
        self.warmup_period = 10
        
        # Certified Blending Weights (V9.5 Optimizer Results)
        self.alpha = 0.7 # Base Prob Weight
        self.beta = 0.03 # Relative Risk Gain (0.3 * 0.1)

    def load_wisdom(self):
        """Loads the globally certified model artifacts."""
        if self.model is None:
            self.model = joblib.load(MODEL_PATH)
            self.features = tuple(joblib.load(FEATURES_PATH))
        return self.model, self.features

    def predict_adaptive_risk(self, machine_id: str, feature_vector: dict) -> float:
        """
        Executes the Adaptive Risk algorithm.
        Returns a blended score (0.0 - 1.0) calibrated for 75% precision.
        """
        model, features = self.load_wisdom()
        
        # 1. Build Vector
        X = pd.DataFrame([feature_vector])
        for f in features:
            if f not in X.columns: X[f] = 0.0
            
        # Senior Pro Fix: Explicitly preserve categorical type for LightGBM
        if "machine_cat" in features:
            # Derive code from machine_id if machine_id_encoded is not in vector
            m_code = feature_vector.get("machine_id_encoded", 0.0)
            if m_code == 0.0:
                from backend.data_access import get_machine_code
                m_code = get_machine_code(machine_id)
            X["machine_cat"] = pd.Series([m_code], index=X.index).astype("category")
            X["machine_id_encoded"] = m_code # Ensure the base feature is also there
            
        X = X[list(features)]
        
        # 2. Base Probability
        base_prob = float(model.predict(X)[0])
        
        # 3. Baseline Tracker
        if machine_id not in self.risk_histories:
            self.risk_histories[machine_id] = []
        
        history = self.risk_histories[machine_id]
        history.append(base_prob)
        if len(history) > self.max_history:
            history.pop(0)
            
        if len(history) < self.warmup_period:
            return base_prob
            
        # 4. Adaptive Logic: Deviation from Mean
        baseline = np.mean(history[-30:]) or 0.01
        rel_risk = base_prob / (baseline + 0.01)
        
        # Final Blend
        v9_score = np.clip(base_prob * self.alpha + (rel_risk * self.beta), 0.0, 1.0)
        return float(v9_score)

# Global Instance for Backend Use
_ORACLE = UniversalOracleV9()

def get_oracle():
    _ORACLE.load_wisdom()
    return _ORACLE

def predict_scrap_v9(machine_id: str, sensor_input: dict):
    """Production entry point for the V9 Oracle."""
    return _ORACLE.predict_adaptive_risk(machine_id, sensor_input)

def get_production_features():
    """Returns the list of features required by the production V9 model."""
    _, features = _ORACLE.load_wisdom()
    return list(features)
