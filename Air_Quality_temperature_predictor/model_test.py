import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import cross_val_score


def test_model_cv(model: Any, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series) -> np.ndarray:
    
    scores = -cross_val_score(
        model, 
        X_test, 
        y_test, 
        scoring="neg_root_mean_squared_error", 
        cv=10
    )

    return scores