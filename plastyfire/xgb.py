# -*- coding: utf-8 -*-
"""
XGBoost approach to predict c_pre and c_post
last modified: András Ecker 04.2021
"""

import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb


def add_features(data):
    """Adds extra features to the data before ML (as we only have a few) - to be discussed and updated..."""
    data["loc_related"] = data["dist"] * data["inp_imp"]
    # these seem to be lognormally distributed so adding log
    data["log_inp_imp"] = np.log(data["inp_imp"])
    data["log_volume_CR"] = np.log(data["volume_CR"])
    # one-hot encode categorical features
    data = pd.get_dummies(data, columns=["loc", "post_mtype"])
    data = data.drop("loc_basal", axis=1)  # only 2 categories so no need to keep both
    return data


def split_data(data, y, train_size=0.8, seed=12345):
    """Split data into train and test datasets"""
    assert y in ["c_pre", "c_post", "theta_d", "theta_p"]
    if y == "c_post":
        data = data.drop(["gmax_p_AMPA", "gmax_NMDA", "gsynSRSF"], axis=1)  # synapse params. don't matter for bAP
    idx = np.arange(len(data))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_size * len(data))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X, y = data.loc[:, ~data.columns.isin(["c_pre", "c_post", "theta_d", "theta_p"])], data[y]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def optimize_model(dtrain, gpu, max_depths, min_child_weights, gammas,
                   subsamples, colsamples_bytree, lambdas, alphas, etas):
    """Hyperparameter optimization of XGBoost model using a naive step-by-step grid search"""

    # set default XGBoost params (these will be updated)
    params = {"objective": "reg:squarederror", "eval_metric": "rmse",  # regression with RMSE loss
              "tree_method": "hist", "grow_policy": "lossguide",  # greedy and fast setup
              "max_depth": max_depths[0], "min_child_weight": min_child_weights[0], "gamma": gammas[0],
              "subsample": subsamples[0], "colsample_bytree": colsamples_bytree[0],
              "lambda": lambdas[0], "alpha": alphas[0], "eta": etas[0]}
    if gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["sampling_method"] = "gradient_based"

    # optimize tree architecture
    min_rmse = float("inf")
    opt_max_depth = max_depths[0]
    opt_min_child_weight = min_child_weights[0]
    opt_gamma = gammas[0]
    for max_depth in tqdm(max_depths, desc="Optimizing tree architecture"):
        for min_child_weight in tqdm(min_child_weights, leave=False):
            for gamma in tqdm(gammas, leave=False):
                params["max_depth"] = max_depth
                params["min_child_weight"] = min_child_weight
                params["gamma"] = gamma
                cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3, num_boost_round=100, early_stopping_rounds=5,
                                    shuffle=True, seed=12345)
                rmse = cv_results["test-rmse-mean"].min()
                if rmse < min_rmse:
                    min_rmse = rmse
                    opt_max_depth = max_depth
                    opt_min_child_weight = min_child_weight
                    opt_gamma = gamma
                gc.collect()
    params["max_depth"] = opt_max_depth
    params["min_child_weight"] = opt_min_child_weight
    params["gamma"] = opt_gamma

    # optimize data sampling
    min_rmse = float("inf")
    opt_subsample = subsamples[0]
    opt_colsample_bytree = colsamples_bytree[0]
    for subsample in tqdm(subsamples, desc="Optimizing data sampling"):
        for colsample_bytree in tqdm(colsamples_bytree, leave=False):
            params["subsample"] = subsample
            params["colsample_bytree"] = colsample_bytree
            cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3, num_boost_round=100, early_stopping_rounds=5,
                                shuffle=True, seed=12345)
            rmse = cv_results["test-rmse-mean"].min()
            if rmse < min_rmse:
                min_rmse = rmse
                opt_subsample = subsample
                opt_colsample_bytree = colsample_bytree
            gc.collect()
    params["subsample"] = opt_subsample
    params["colsample_bytree"] = opt_colsample_bytree

    # optimize regularization
    min_rmse = float("inf")
    opt_lambda = lambdas[0]
    opt_alpha = alphas[0]
    for lambda_ in tqdm(lambdas, desc="Optimizing regularization"):
        for alpha in tqdm(alphas, leave=False):
            params["lambda"] = lambda_
            params["alpha"] = alpha
            cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3, num_boost_round=100, early_stopping_rounds=5,
                                shuffle=True, seed=12345)
            rmse = cv_results["test-rmse-mean"].min()
            if rmse < min_rmse:
                min_rmse = rmse
                opt_lambda = lambda_
                opt_alpha = alpha
            gc.collect()
    params["lambda"] = opt_lambda
    params["alpha"] = opt_alpha

    # optimize learning rate
    min_rmse = float("inf")
    opt_eta = etas[0]
    for eta in tqdm(etas, desc="Optimizing learning rate"):
        params["eta"] = eta
        cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3, num_boost_round=500, early_stopping_rounds=25,
                            shuffle=True, seed=12345)
        rmse = cv_results["test-rmse-mean"].min()
        if rmse < min_rmse:
            min_rmse = rmse
            opt_eta = eta
        gc.collect()
    params["eta"] = opt_eta

    return params


if __name__ == "__main__":

    data = pd.read_pickle("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/mldata.pkl")
    # sample 3M synapses on each post_mtypes (this way it's balanced and also fits to memory)
    data = data.groupby("post_mtype", as_index=False, sort=False).apply(lambda x: x.sample(int(min(3e6, len(x)))))
    data = add_features(data)  # some feature engineering
    gpu = True if os.system("nvidia-smi") == 0 else False  # tricky way to find out if GPU is available...
    for y in ["c_pre", "c_post"]:
        # split data into training and testing and get it to xgb's preferred format
        X_train, X_test, y_train, y_test = split_data(data, y=y)
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        del X_train, X_test, y_train, y_test
        gc.collect()
        # hyperparameter optimization (with cross-validation using only the training dataset)
        opt_params = optimize_model(dtrain, gpu,
                                    max_depths=[6, 8, 10, 12], min_child_weights=[3, 6, 9, 12], gammas=[0],
                                    subsamples=[1.0, 0.8, 0.5], colsamples_bytree=[1., 0.9, 0.8],
                                    lambdas=[0.6, 0.8, 1.0], alphas=[0.6, 0.8, 1.0], etas=[0.2, 0.1, 0.05, 0.02])
        print(opt_params)
        # training final model and measuring accuracy on (unseen) test dataset
        model = xgb.train(opt_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
                          evals=[(dtest, "eval")], verbose_eval=False)
        print("%s model trained in %i iterations - test RMSE: %.6f" % (y, model.best_iteration+1, model.best_score))
        feature_order = [x[0] for x in sorted(model.get_score(importance_type="gain").items(),
                                              key=lambda x: x[1], reverse=True)]
        print("Importance of features: ", feature_order)
        del model, dtrain, dtest
        gc.collect()







