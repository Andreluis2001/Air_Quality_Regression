from typing import Any

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import root_mean_squared_error

def grid_search_model(model: Any, 
                      param_grid: dict, 
                      X: pd.DataFrame, 
                      y: pd.Series) -> Any:

  print(f'{model}\n')
  grid_search = GridSearchCV(
      model,
      param_grid,
      cv=5,
      scoring='neg_root_mean_squared_error',
      verbose=1)
  grid_search.fit(X, y)
  best_model = grid_search.best_estimator_
  return best_model

def train_model(model: Any, 
                X_train: pd.DataFrame, 
                y_train: pd.Series, 
                X_test: pd.DataFrame, 
                y_test: pd.Series) -> float:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = root_mean_squared_error(y_test, predictions)
    return score

def test_model_cv(model: Any, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series) -> np.ndarray:
    
    print(f'Testing model: {model}\n')

    scores = -cross_val_score(
        model, 
        X_test, 
        y_test, 
        scoring="neg_root_mean_squared_error", 
        cv=10
    )

    print(f"Model: {model}\n")
    print(f"Scores: {scores}\n")
    print(f"Mean: {scores.mean()}\n")
    print(f"Standard Deviation: {scores.std()}\n")
    return scores
