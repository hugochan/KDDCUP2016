# import pandas as pd
import numpy as np
import xgboost as xgb

# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, TheilSenRegressor, RANSACRegressor

seed = 260681



def linear_regression(train, y, test):
    # lr = LinearRegression()
    # lr = BayesianRidge()
    lr = TheilSenRegressor(random_state=42)
    lr.fit(train, y)
    preds = lr.predict(test)

    return preds



def boosted_trees(train, y, test, y2=None):
    """ defalut params:
    max_depth=3, learning_rate=0.1,
    n_estimators=100, silent=True,
    objective="binary:logistic",
    nthread=-1, gamma=0, min_child_weight=1,
    max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
    reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    base_score=0.5, seed=0, missing=None
    """

    reg = xgb.XGBRegressor(n_estimators=800,
                            nthread=4,
                            max_depth=8, # 8
                            learning_rate=0.01, # 0.03
                            min_child_weight=3, # 3
                            silent=True,
                            # gamma=0, # 0
                            # colsample_bylevel=1, # 1
                            # scale_pos_weight=1, # 1
                            subsample=0.7, # 0.8
                            colsample_bytree=.9) # .8

    reg.fit(train, y)
    preds = reg.predict(test)

    return preds

    # clf = xgb.XGBClassifier(n_estimators=500,
    #                         nthread=-1,
    #                         max_depth=16, # 16
    #                         learning_rate=0.03, # 0.03
    #                         min_child_weight=2, # 2
    #                         silent=True,
    #                         # gamma=0, # 0
    #                         # colsample_bylevel=1, # 1
    #                         # scale_pos_weight=1, # 1
    #                         subsample=0.8, # 0.8
    #                         colsample_bytree=0.81) # 0.81


    # # xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(train, y), (val_X, val_y)], early_stopping_rounds=3)

    # xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(test, y2), (train, y)], early_stopping_rounds=3)

    # preds = clf.predict_proba(test)[:, 1]
    # # sample = pd.read_csv('datasets/sample_submission.csv')
    # # sample.QuoteConversion_Flag = preds
    # # sample.to_csv('xgb_benchmark.csv', index=False)
    # return preds



