import os
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#* Class imports for Methods
from RFGAP_Rocket.RFGAP_Rocket import RFGAP_Rocket
from RDST.rdst import RDST_GAP
from QGAP.qgap import QGAP
from Redcomets.Redcomets import REDCOMETS





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

#? Methods to retrieve predictions and proximities for different models
def get_rocket_pred(X_train, y_train, X_test, static_train, static_test, params):
    rocket = RFGAP_Rocket(prediction_type = params["prediction_type"], # Classification or Regression
                           rocket = params["rocket"], # Rocket or MultiRocket
                         n_kernels=params["n_kernels"]) # 512 or other integers
    
    rocket.fit(X_train, y_train, static_train, weights = params["weights"]) #Value between 0 and 1
    return rocket.predict(X_test, static_test)

def get_rdst_pred(X_train, y_train, X_test, static_train, static_test, params):
    rdst = RDST_GAP(save_transformed_data = True,
                    max_shapelets = params["max_shapelets"], # Any integer
                    shapelet_length = params["shapelet_length"], #Any number, default is min(max(2,n_timepoints//2),11)
                    alpha_similarity = params["alpha_similarity"]) # Betweeen 0 and 1
    
    rdst.fit(X_train, y_train, static = static_train)
    return rdst.predict(X_test, static = static_test)

def get_qgap_pred(X_train, y_train, X_test, static_train, static_test, params):
    qgap = QGAP(matrix_type="dense",
                interval_depth = params["interval_depth"], # Any integer: 2 ** depth
                quantile_divisor = params["quantile_divisor"] # Any integer: 1 + (interval_length - 1) // quantile_divisor
    )
    qgap.fit(X_train, y_train, static = static_train)
    return qgap.predict(X_test, static = static_test)

def get_redcomets_pred(X_train, y_train, X_test, static_train, static_test, params):
    rc = REDCOMETS(static = static_train, variant=3,
                   perc_length=params["perc_length"], # Percentage of time series length to use
                   n_trees=params["n_trees"] # Number of trees in the forest
                   )
    rc.fit(X_train, y_train)
    return rc.predict(X_test, static = static_test)

    
    

#? Full Method

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

    Returns:
        dict: Model name -> best parameter dict.
    """
    def evaluate_params(get_predictions_method, params, X, y, static, cv):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            static_train = static[train_idx] if static is not None else None
            static_test = static[test_idx] if static is not None else None

            #! This is a placeholder for model instantiation, adjust as needed
            y_pred = get_predictions_method(X_train, y_train, X_test, static_train[train_idx], static_test[test_idx])


            f1 = f1_score(y_test, y_pred, average='weighted')
            return f1

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