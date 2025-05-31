import os
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def save_optimized_parameters(param_dict, save_path):
    """
    Saves the optimized parameters for each model to a JSON file.

    Args:
        param_dict (dict): Dictionary where keys are model names (str) and values are parameter dicts.
        save_path (str): Path to save the JSON file.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(param_dict, f, indent=4)
        print(f"Optimized parameters saved to {save_path}")
    except Exception as e:
        print(f"Error saving optimized parameters: {e}")

def grid_search_models(models, param_grids, X, y, static=None, n_jobs=-1, cv=3, scoring='accuracy'):
    """
    Performs grid search for each model using joblib parallelization.

    Args:
        models (dict): Model name -> model class (not instance).
        param_grids (dict): Model name -> list of parameter dicts to try.
        X (np.ndarray): Feature data.
        y (np.ndarray): Labels.
        static (np.ndarray or None): Static data if required by models.
        n_jobs (int): Number of parallel jobs.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric ('accuracy', etc.).

    Returns:
        dict: Model name -> best parameter dict.
    """
    def evaluate_params(model_class, params, X, y, static, cv, scoring):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            static_train = static[train_idx] if static is not None else None
            static_test = static[test_idx] if static is not None else None

            #! This is a placeholder for model instantiation, adjust as needed
            model = model_class(**params)
            if static is not None:
                model.fit(X_train, y_train, static=static_train)
                y_pred = model.predict(X_test, static=static_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            if scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            else:
                raise ValueError("Unsupported scoring metric")
            scores.append(score)
        return np.mean(scores)

    best_params = {}
    for model_name, model_class in models.items():
        grid = param_grids[model_name]
        static_data = static.get(model_name) if isinstance(static, dict) else static
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_params)(model_class, params, X, y, static_data, cv, scoring)
            for params in grid
        )
        best_idx = int(np.argmax(results))
        best_params[model_name] = grid[best_idx]
        print(f"{model_name}: Best params {grid[best_idx]} with score {results[best_idx]:.4f}")
    return best_params

# Example usage:
# optimized_params = {
#     "REDCOMETS": {"param1": 0.1, "param2": 10},
#     "QGAP": {"paramA": 5, "paramB": 0.01},
#     "RDST": {"alpha": 0.5, "beta": 2}
# }
# save_optimized_parameters(optimized_params, "/yunity/arusty/PF-GAP/experiments/optimized_params.json")