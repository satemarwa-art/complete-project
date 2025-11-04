from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def load_dataset(file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Basic preprocessing: fill NAs and encode non-numerics
    X = _preprocess_features(X)
    feature_names = list(X.columns)
    return X, y, feature_names


def infer_task_type(y: pd.Series) -> str:
    if y.dtype.kind in {'O', 'b'}:
        return 'classification'
    # Heuristic: small number of unique values suggests classification
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return 'classification'
    return 'regression'


def _preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.kind in {'i', 'f'}:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype('category')
            X[col] = X[col].cat.add_categories(['__MISSING__']).fillna('__MISSING__')
            X[col] = X[col].cat.codes  # convert categories to integer codes
    return X

