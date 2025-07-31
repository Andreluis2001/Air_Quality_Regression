import os
import pickle 
from pathlib import Path
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class DataTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
    X['Date'] = X['Date'].apply(lambda date: date.month)
    X['Time'] = X['Time'].apply(lambda time: time.hour)
    X['Time'] = X['Time'].apply(transform_time)
    X['Date'] = X['Date'].apply(transform_date)
    return X

def split_stratified(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  dataset["Season"] = dataset["Date"].apply(lambda date: date.month)
  dataset["Season"] = dataset["Season"].apply(transform_date)

  print(dataset.shape)

  train, test = train_test_split(dataset, test_size=0.2, stratify=dataset["Season"], random_state=42)

  for split_ in (train, test):
    split_.drop("Season", axis=1, inplace=True)
  return train, test

def transform_time(time: int) -> str:
  if time > 5 and time <=12:
    return 'morning'
  elif time > 12 and time <= 17:
    return 'afternoon'
  elif time > 17 and time <= 21:
    return 'evening'
  else:
    return 'night'

def transform_date(date: int) -> str:
  if date > 3 and date <=5:
    return 'spring'
  elif date > 5 and date <= 8:
    return 'summer'
  elif date > 8 and date <= 11:
    return 'autumn'
  else:
    return 'winter'