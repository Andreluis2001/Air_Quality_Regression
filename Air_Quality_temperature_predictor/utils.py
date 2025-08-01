import pandas as pd
from sklearn.model_selection import train_test_split
from .data_preprocessing import transform_date


def split_stratified(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  dataset["Season"] = dataset["Date"].apply(lambda date: date.month)
  dataset["Season"] = dataset["Season"].apply(transform_date)

  print(dataset.shape)

  train, test = train_test_split(dataset, test_size=0.2, stratify=dataset["Season"], random_state=42)

  for split_ in (train, test):
    split_.drop("Season", axis=1, inplace=True)
  return train, test