# -*- coding: utf-8 -*-
"""
XGBoost approach to predict c_pre and c_post or theta_d and theta_p
last modified: András Ecker 05.2021
"""

import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization


def split_data(data, y, loc=None, mtype=None, train_size=0.8, seed=12345):
    """Split data into train and test datasets"""
    assert y in ["c_pre", "c_post", "theta_d", "theta_p"]
    if loc is not None:
        data = data.loc[data["loc"] == loc, :]
        data.drop("loc", axis=1, inplace=True)
    else:
        data = pd.get_dummies(data, columns=["loc"])
        data.drop("loc_basal", axis=1, inplace=True)  # only 2 categories so no need to keep both
    if mtype is not None:
        data = data.loc[data["post_mtype"] == mtype, :]
        data.drop("post_mtype", axis=1, inplace=True)
    else:
        data = pd.get_dummies(data, columns=["post_mtype"])
    idx = np.arange(len(data))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_size * len(data))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X, y = data.loc[:, ~data.columns.isin(["c_pre", "c_post", "theta_d", "theta_p"])], data[y]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def _il_mape(y_true, y_pred):
    """in line implementation of mean absolute percentage error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100.


def _optimize_model(max_depth, min_child_weight, subsample, colsample_bytree, lambda_, alpha, eta):
    params["max_depth"] = int(max_depth)
    params["min_child_weight"] = int(min_child_weight)
    params["subsample"] = subsample
    params["colsample_bytree"] = colsample_bytree
    params["lambda"] = lambda_
    params["alpha"] = alpha
    params["eta"] = eta
    cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3, num_boost_round=1000, early_stopping_rounds=50,
                        shuffle=True, seed=12345)
    return -1 * cv_results["test-rmse-mean"].min()  # *-1 because bayes_opt maximizes stuff


if __name__ == "__main__":

    # data = pd.read_pickle("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/mldata.pkl")
    # data = data.groupby("post_mtype", as_index=False, sort=False).apply(lambda x: x.sample(int(min(1e6, len(x)))))
    data = pd.read_pickle("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/mldata_sampled.pkl")
    gpu = True if os.system("nvidia-smi") == 0 else False  # tricky way to find out if GPU is available...
    params = {"objective": "reg:squarederror", "eval_metric": "rmse",  # simple regression with RMSE loss
              "tree_method": "hist", "grow_policy": "lossguide"}  # greedy and fast setup}
    if gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["sampling_method"] = "gradient_based"

    for y in ["c_pre", "c_post", "theta_d", "theta_p"]:
        # for loc in ["basal", "apical"]:
        # for mtype in np.sort(data["post_mtype"].unique()):
        # split data into training and testing and get it to xgb's preferred format
        X_train, X_test, y_train, y_test = split_data(data, y=y)
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        del X_train, X_test, y_train, y_test
        gc.collect()
        params["base_score"] = np.mean(dtrain.get_label())
        # Bayesian hyperparameter optimization (with cross-validation using only the training dataset)
        bo = BayesianOptimization(_optimize_model, {"max_depth": (3, 9), "min_child_weight": (25, 75),
                                                    "subsample": (0.8, 1), "colsample_bytree": (0.8, 1),
                                                    "lambda_": (0.6, 0.8), "alpha": (0.6, 0.8), "eta": (0.05, 0.2)})
        bo.maximize(init_points=5, n_iter=15, acq="ei")
        for param, opt_val in bo.max["params"].items():
            if param in ["max_depth", "min_child_weight"]:
                params[param] = int(opt_val)
            elif param == "lambda_":
                params["lambda"] = opt_val
            else:
                params[param] = opt_val
        # training final model on the full train dataset and measuring accuracy on (unseen) test dataset
        model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
                          evals=[(dtest, "eval")], verbose_eval=False)
        print("%s model trained in %i iterations - test loss: %.6f, test MAPE: %.2f %%" % (y,
              model.best_iteration+1, model.best_score, _il_mape(dtest.get_label(), model.predict(dtest))))
        feature_order = [x[0] for x in sorted(model.get_score(importance_type="gain").items(),
                                              key=lambda x: x[1], reverse=True)]
        print("Importance of features: ", feature_order)
        del bo, model, dtrain, dtest
        gc.collect()

