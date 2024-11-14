import pandas as pd
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_logistic_regression_predictions(X, y, test_split = 0.3, defined_data_split = False, X_train = None, X_test = None, y_train= None, y_test=None):
    # train test split 
    if not defined_data_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_split, random_state=42)


    # normalize data
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # logistic regression model train
    lrc = LogisticRegression(solver='lbfgs', max_iter=1000)
    lrc.fit(X_train_normalized, y_train)
    y_train_preds= lrc.predict(X_train_normalized)
    y_test_preds = lrc.predict(X_test_normalized)

    # result metrics
    print('train results')
    print(classification_report(y_train, y_train_preds))
    print('test results')
    print(classification_report(y_test, y_test_preds))
    return lrc, y_test, y_test_preds

def get_rfc_predictions(X, y):
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
    
    # grid search
    rfc = RandomForestClassifier(random_state=42)

    # param grid is motivated by https://medium.com/@Doug-Creates/tuning-random-forest-parameters-with-scikit-learn-b53cbc602cd0
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }

    # train the model
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    rfc_cv.fit(X_train, y_train)


    # get predictions
    y_train_preds = rfc_cv.best_estimator_.predict(X_train)
    y_test_preds = rfc_cv.best_estimator_.predict(X_test)
    

    # result metrics
    print('train results')
    print(classification_report(y_train, y_train_preds))
    print('test results')
    print(classification_report(y_test, y_test_preds))
    return rfc_cv