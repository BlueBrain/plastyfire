"""
Custom BluePyOpt evaluator for an XGBoost model trained on outcomes of the Graupner & Brunel plasticity model
It doesn't seem logical, but from a software engineering point of view it's crucial that
the model is saved after retraining on a single CPU and loaded in each child process
(as `xgboost` doesn't like shared memory stuff in multiprocessing)
authors: Giuseppe Chindemi (12.2020) + modifications by András Ecker (07.2024)
"""

import os
import sys
import glob
import pickle
import traceback
import numpy as np
from bluepyopt.evaluators import Evaluator
from bluepyopt.objectives import Objective
from bluepyopt.parameters import Parameter
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

WEIGHT_REDUCE = np.array([1 / 8] * 2 + [1 / 4] * 3)  # weights of each protocol (lower for the first two)
# XGBoost hyperparameters to choose from
TUNED_PARAMS = {"estimator__learning_rate": [0.01, 0.1, 0.25, 0.5],
                "estimator__max_depth": [5, 10, 15, 20],
                "estimator__min_child_weight": [5, 10, 15, 20],
                "estimator__gamma": [0.0, 0.05, 0.1, 0.2]}


def load_opt_data(workdir):
    """Loads data used for training XGBoost model to predict the outcome of pairing protocols from cache directory"""
    solutions = []
    for f_name in glob.glob(os.path.join(workdir, ".cache", "*")):
        with open(f_name, "rb") as f:
            solutions.append(pickle.load(f))
    x = np.array([np.array(sol["individual"]) for sol in solutions])
    y = np.array([np.array(sol["outcome"]) for sol in solutions])
    return x, y


class XGBEvaluator(Evaluator):
    """Graupner & Brunel plasticity model evaluator using predictions from XGBoost"""
    def __init__(self, fit_params, invitro_db, workdir):
        super(Evaluator, self).__init__()
        self.params = [Parameter(param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in fit_params]
        self.param_names = [param.name for param in self.params]
        self.invitro_db = invitro_db
        self.objectives = [Objective(elem.protocol_id) for elem in self.invitro_db.itertuples()]
        self.workdir = workdir
        self.jsonf_name = os.path.join(self.workdir, "xgb.json")
        # Load data and fit XGBoost model
        x, y = load_opt_data(self.workdir)
        rgs = GridSearchCV(MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=1.0)),
                           TUNED_PARAMS, scoring="r2", n_jobs=-1)
        rgs.fit(x, y)
        # retrain model with best hyperparameters on the whole dataset and save it
        model = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=1.0, n_jobs=1,
                                 learning_rate=rgs.best_params_["estimator__learning_rate"],
                                 max_depth=rgs.best_params_["estimator__max_depth"],
                                 min_child_weight=rgs.best_params_["estimator__min_child_weight"],
                                 gamma=rgs.best_params_["estimator__gamma"])
        model.fit(x, y)
        model.save_model(self.jsonf_name)

    def evaluate_with_lists(self, param_values):
        """Load XGBoost model and evaluate individual"""
        try:
            model = xgb.XGBRegressor()
            model.load_model(self.jsonf_name)
            outcome = model.predict([param_values])[0]
            error = np.abs((self.invitro_db["mean_epsp_ratio"].to_numpy() - outcome)
                           / self.invitro_db["sem_epsp_ratio"].to_numpy())
            return (error * WEIGHT_REDUCE).tolist()
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    def init_simulator_and_evaluate_with_lists(self, param_values):
        """Since we don't run simulations in NEURON, this function doesn't do anything,
        but w/o it the current version of `bluepyopt` wouldn't run..."""
        return self.evaluate_with_lists(param_values)


