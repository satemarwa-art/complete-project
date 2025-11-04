from __future__ import annotations
from typing import Dict, List


def compute_baselines(X, y, task: str, estimator_key: str, cv_folds: int, feature_names: List[str], fast: bool = False) -> Dict:
    # Lazy imports
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import SelectFromModel, RFE
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.svm import LinearSVC, LinearSVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    results = {}

    # Helper to build estimator similar to GA choice
    def build_estimator():
        if task == 'classification':
            if estimator_key == 'logreg':
                return Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('clf', LogisticRegression(max_iter=200))
                ])
            elif estimator_key == 'svm':
                return Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('clf', LinearSVC())
                ])
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            if estimator_key == 'logreg':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif estimator_key == 'svm':
                return Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('reg', LinearSVR())
                ])
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)

    # 1) All features baseline
    scoring = 'f1_macro' if task == 'classification' else 'r2'
    est = build_estimator()
    scores = cross_val_score(est, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
    results['all_features'] = {
        'score': float(scores.mean()),
        'scores': scores.tolist(),
        'selected_features': feature_names,
    }

    # 2) SelectKBest
    k = max(1, int(np.sqrt(X.shape[1])))
    if task == 'classification':
        selector = SelectKBest(mutual_info_classif, k=k)
    else:
        selector = SelectKBest(f_regression, k=k)
    X_k = selector.fit_transform(X, y)
    est = build_estimator()
    scores = cross_val_score(est, X_k, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
    mask = selector.get_support()
    results['select_k_best'] = {
        'score': float(scores.mean()),
        'k': int(k),
        'selected_features': [f for f, m in zip(feature_names, mask) if m],
    }

    # 3) L1-based selection (skip in fast mode)
    if not fast:
        if task == 'classification':
            l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
            selector = SelectFromModel(l1)
        else:
            l1 = Lasso(alpha=0.001, max_iter=2000)
            selector = SelectFromModel(l1)
        X_l1 = selector.fit_transform(X, y)
        if X_l1.shape[1] == 0:
            # fallback to smallest k
            X_l1 = X_k
            l1_mask = selector.get_support() if hasattr(selector, 'get_support') else (selector.fit(X, y).get_support())
        else:
            l1_mask = selector.get_support()
        est = build_estimator()
        scores = cross_val_score(est, X_l1, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        results['l1'] = {
            'score': float(scores.mean()),
            'selected_features': [f for f, m in zip(feature_names, l1_mask) if m],
        }

    # 4) RFE (skip in fast mode)
    if not fast:
        if task == 'classification':
            base = LogisticRegression(max_iter=200)
        else:
            base = LinearSVR()
        # choose n_features_to_select heuristically
        n_select = max(1, int(np.sqrt(X.shape[1])))
        rfe = RFE(estimator=base, n_features_to_select=n_select)
        X_r = rfe.fit_transform(X, y)
        est = build_estimator()
        scores = cross_val_score(est, X_r, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        mask = rfe.get_support()
        results['rfe'] = {
            'score': float(scores.mean()),
            'selected_features': [f for f, m in zip(feature_names, mask) if m],
        }

    return results
