from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def evaluate_subset(X, y, mask: np.ndarray, task: str, estimator_key: str, cv_folds: int, scoring: str | None = None) -> Tuple[float, Dict]:
    # Lazy imports to avoid heavy deps at import time
    import numpy as _np
    from sklearn.model_selection import cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import LinearSVC, LinearSVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if mask.sum() == 0:
        return 0.0, {}

    X_sel = X.iloc[:, mask.astype(bool)]

    if task == 'classification':
        if estimator_key == 'logreg':
            estimator = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', LogisticRegression(max_iter=200, n_jobs=None))
            ])
            scoring = scoring or 'f1_macro'
        elif estimator_key == 'svm':
            estimator = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', LinearSVC())
            ])
            scoring = scoring or 'f1_macro'
        else:  # rf
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring = scoring or 'f1_macro'
    else:
        if estimator_key == 'logreg':
            estimator = LinearRegression()
            scoring = scoring or 'r2'
        elif estimator_key == 'svm':
            estimator = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('reg', LinearSVR())
            ])
            scoring = scoring or 'r2'
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring = scoring or 'r2'

    scores = cross_val_score(estimator, X_sel, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), {
        'scores': scores.tolist(),
        'scoring': scoring,
    }


def run_ga(X, y, task: str, estimator_key: str, cv_folds: int, params: Dict, feature_names: List[str]) -> Dict:
    rng = np.random.default_rng(42)
    n_features = X.shape[1]

    pop_size = int(params.get('pop_size', 40))
    generations = int(params.get('generations', 20))
    crossover_rate = float(params.get('crossover_rate', 0.8))
    mutation_rate = float(params.get('mutation_rate', 0.05))
    elitism = int(params.get('elitism', 2))
    penalty = float(params.get('penalty', 0.01))  # per proportion of features

    def fitness(mask):
        score, _ = evaluate_subset(X, y, mask, task, estimator_key, cv_folds)
        size_penalty = penalty * (mask.sum() / n_features)
        return score - size_penalty

    def init_individual():
        # ensure at least one feature selected
        mask = rng.random(n_features) < 0.5
        if not mask.any():
            mask[rng.integers(0, n_features)] = True
        return mask

    def tournament_select(pop, k=3):
        idx = rng.choice(len(pop), size=k, replace=False)
        best = max(idx, key=lambda i: pop[i][1])
        return pop[best][0].copy()

    def crossover(mask1, mask2):
        if rng.random() > crossover_rate:
            return mask1.copy(), mask2.copy()
        cut = rng.integers(1, n_features)
        child1 = np.concatenate([mask1[:cut], mask2[cut:]])
        child2 = np.concatenate([mask2[:cut], mask1[cut:]])
        return child1, child2

    def mutate(mask):
        flip = rng.random(n_features) < mutation_rate
        mask = mask.copy()
        mask[flip] = ~mask[flip]
        if not mask.any():
            mask[rng.integers(0, n_features)] = True
        return mask

    population = [init_individual() for _ in range(pop_size)]
    scored = [(ind, fitness(ind)) for ind in population]

    history = []

    for gen in range(generations):
        scored.sort(key=lambda x: x[1], reverse=True)
        best_ind, best_fit = scored[0]
        history.append({'generation': gen, 'best_fitness': float(best_fit), 'selected': int(best_ind.sum())})

        next_pop = [scored[i][0].copy() for i in range(min(elitism, len(scored)))]

        while len(next_pop) < pop_size:
            p1 = tournament_select(scored)
            p2 = tournament_select(scored)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_pop.extend([c1, c2])

        population = next_pop[:pop_size]
        scored = [(ind, fitness(ind)) for ind in population]

    # Final best individual
    scored.sort(key=lambda x: x[1], reverse=True)
    best_mask = scored[0][0]
    best_fitness = float(scored[0][1])
    best_score, score_info = evaluate_subset(X, y, best_mask, task, estimator_key, cv_folds)
    selected = [feature_names[i] for i, v in enumerate(best_mask) if v]

    return {
        'best_fitness': best_fitness,
        'best_score': float(best_score),
        'score_info': score_info,
        'selected_features': selected,
        'selected_count': len(selected),
        'history': history,
    }
