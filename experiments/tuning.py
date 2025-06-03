import os
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#* Class imports for Methods
import sys
sys.path.insert(0, '/yunity/arusty/PF-GAP')

from RFGAP_Rocket.RFGAP_Rocket import RFGAP_Rocket
from RDST.rdst import RDST_GAP
from QGAP.qgap import QGAP
from Redcomets.Redcomets import REDCOMETS
from FreshPrince.FreshPrince import FreshPRINCE_GAP

print("Imports done.")

def save_optimized_parameters(param_dict, model_name, score, save_path = "data/opimized_models/"):
    """
    Saves the optimized parameters for each model to a JSON file.

    Args:
        param_dict (dict): Dictionary where keys are model names (str) and values are parameter dicts.
        save_path (str): Path to save the JSON file.
    """
    save_path = os.path.join(save_path, f"{model_name}_optimized_params.json")

    param_dict["score"] = score  # Add the score to the parameters

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(param_dict, f, indent=4)
        print(f"Optimized parameters saved to {save_path}")
    except Exception as e:
        print(f"Error saving optimized parameters: {e}")

#? Methods to retrieve predictions and proximities for different models
def get_rocket_pred(X_train, y_train, X_test, static_train, static_test, params):
    rocket = RFGAP_Rocket(prediction_type = "classification", # Classification or Regression
                           rocket = params["rocket"], # Rocket or MultiRocket
                         n_kernels=params["n_kernels"]) # 512 or other integers
    
    rocket.fit(X_train, y_train, static_train, weights = params["weights"]) #Value between 0 and 1
    return rocket.predict(X_test, static_test)

def get_rdst_pred(X_train, y_train, X_test, static_train, static_test, params):
    rdst = RDST_GAP(save_transformed_data = True,
                    max_shapelets = params["max_shapelets"], # Any integer
                    shapelet_lengths = params["shapelet_length"], #Any number, default is min(max(2,n_timepoints//2),11)
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
                   n_trees=params["n_trees"], # Number of trees in the forest
                   random_state=42)
    rc.fit(X_train, y_train)
    return rc.predict(X_test, static = static_test)

def get_fresh_pred(X_train, y_train, X_test, static_train, static_test, params):
    fp = FreshPRINCE_GAP(default_fc_parameters=params["default_fc_parameters"], # "minimal", "efficient", "comprehensive"
                         n_estimators=params["n_estimators"] # Number of estimators for the rotation forest ensemble
                         )
    fp.fit(X_train, y_train, static = static_train)
    return fp.predict(X_test, static = static_test)
    
def determine_static(fold):
    if fold < 2:
        static_train = static2022
        static_test = static2023
    elif fold < 4:
        static_train = static2023
        static_test = static2024
    else:
        static_train = static2024
        static_test = static2025

    return np.array(static_train), np.array(static_test)

def evaluate_params(get_predictions_method, params, X, y):
    """
    Evaluates the performance of a model with given parameters using cross-validation.
    """
    try:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            # Get the train and test splits
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            static_train, static_test = determine_static(2)

            y_pred = get_predictions_method(X_train, y_train, X_test, static_train[train_idx], static_test[test_idx], params)


            scores.append(f1_score(y_test, y_pred, average='weighted'))
        
        return np.mean(scores)
    except Exception as e:
        print(f"Error evaluating parameters: {e}")
        return 0

#? Full Method
def grid_search_models(model_dict, X, y):
    """
    Performs grid search for hyperparameter optimization on multiple models.
    
    Args:
        model_dict (dict): Dictionary where keys are model names and values are tuples containing
                           the function to get predictions and a dictionary of hyperparameters.
    
    Returns:
        dict: Dictionary with optimized parameters for each model.
    """

    for model_name, params_dict in model_dict.items():
        print(f"Optimizing parameters for {model_name}...")

        saved_params = params_dict["default"]

        # Get the correct function to test the model
        get_predictions_method = globals().get(f"get_{model_name.lower()}_pred")

        best_score = evaluate_params(get_predictions_method, params_dict["default"], X, y)

        print(f"#* -------> Initial score for {model_name} with default parameters: {best_score}")

        # Loop through each parameter and save the best performing one
        for param_name, param_values in params_dict.items():

            # Skip the default parameter as it is already set
            if param_name == "default":
                continue
            
            # Loop through each value of the parameter
            for param_value in param_values:
                print(f"#? -------> Testing {model_name} with {param_name}={param_value}...")

                # Create a copy of the parameters and set the current parameter value
                params = saved_params.copy()
                params[param_name] = param_value
                
                # Evaluate the model with the current parameters
                score = evaluate_params(get_predictions_method, params, X, y)
                
                print(f"#?      -------> Score: {score}")

                # Update the best score and parameter if this one is better
                if score > best_score:
                    best_score = score
                    saved_params = params
                    print(f"#!          -------> New best score. Parameters updated: {saved_params}")
            

        # Save the optimized parameters to a JSON file
        save_optimized_parameters(saved_params, model_name=model_name, score=best_score)
            
model_dict = {
    # "FRESH" : {
    #     "default" : {"default_fc_parameters": "comprehensive", "n_estimators": 200},
    #     "default_fc_parameters": ["minimal", "efficient"],  # Type of feature extraction
    #     "n_estimators": [50, 100, 500]  # Number of estimators in the rotation forest ensemble
    # },
    # "QGAP"  : {
    #     "default" : {"interval_depth": 6, "quantile_divisor": 4},
    #     "interval_depth": [2, 4, 5, 7, 8],  # Depth of the interval tree
    #     "quantile_divisor": [1, 2, 3, 5, 6, 7, 8]  # Divisor for quantile calculation
    # },
    "Rocket" : {
        "default" : ({"rocket": "Multi", "n_kernels": 512, "weights": None}),
        "rocket": ["Multi", "Mini"],  # Multi or Mini
        "n_kernels": [50, 128, 256, 1024, 2048],  # Number of kernels
        "weights": [0.1, 0.3, 0.5, 0.7, 0.9]  # Percentage weight to assign to static variables
    },
    "RDST"  : {
        "default" : {"max_shapelets": 10000, "shapelet_length": None, "alpha_similarity": 0.5},
        "max_shapelets": [100, 1000, 5000, 15000, 20000],  # Number of shapelets to extract
        "shapelet_length": [2, 5, 8, 10, 20],  # Length of shapelets to extract
        "alpha_similarity": [0.1, 0.3, 0.7, 0.9]  # Similarity threshold for shapelet matching
    },
    
    # "REDCOMETS": {
    #     "default" : {"perc_length": 5, "n_trees": 100},
    #     "perc_length": [0.1, 0.3, 0.7, 0.9, 1],
    #     "n_trees": [10, 50, 150, 200],
    # }
}

#* Import data
import sys 
import pandas as pd

static2024 = pd.read_csv('data/static2024.csv')
static2023 = pd.read_csv('data/static2023.csv')
static2022 = pd.read_csv('data/static2022.csv')
static2025 = pd.read_csv('data/static2025.csv')
time_series = np.array(pd.read_csv('data/time_series.csv'))
labels = pd.read_csv('data/labels.csv')
labels = np.array(labels).flatten()

#* Run the grid search
grid_search_models(model_dict, time_series, labels)