import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("..")  # Find a better way to do this.
from Models.skModels import *

def model_comparison_regression(p_models_to_comp,
                                p_X_train,
                                p_X_val,
                                p_Y_train,
                                p_Y_val,
                                p_predictors,
                                #p_metric='AUC', # Options are AUC, R2, accuracy  # Unused
                                #p_models=['All'],  # Unused
                                p_seed=0,
                                p_verbose=False):
    """ Add another argument to allow transformations of the target variable (default to identity) e.g. see transform in SVM, Bagging """
    """ Add another action if p_verbose = True where run-time of each model is calculated (this will help with optimizing hyper-parameter tuning) """
    # List of models tested
    models = []

    # List to store results from each model
    results = []

    for model_object, model_name, hyper_parameters in p_models_to_comp:
        if hyper_parameters: # If there are hyperparameters to test do it here
            for param in hyper_parameters[0]:
                model_with_hyperparameters = model_object
                model_with_hyperparameters.fit(p_X_train.iloc[:, p_predictors], p_Y_train, param)
                result = model_with_hyperparameters.score(p_X_val.iloc[:, p_predictors], p_Y_val)
                results.append(result)
                if p_verbose:
                    print('{} {} - {}'.format(model_with_hyperparameters.label, param, result))
                models.append(['{} {}'.format(model_with_hyperparameters.label, param), result])
        else:
            model = model_object
            model.fit(p_X_train.iloc[:, p_predictors], p_Y_train, None)
            result = model.score(p_X_val.iloc[:, p_predictors], p_Y_val)
            results.append(result)
            if p_verbose:
                print('{} XX - {}'.format(model.label, result))
            models.append(['{} XX'.format(model.label), result])

    # Plot the results of all models
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(results)
    # Label the axis with model names
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label[0] for label in models], rotation='vertical')
    # Plot the accuracy for all combinations
    plt.show()

    df = pd.DataFrame(columns=['Model', 'Score'], data=models)
    df_sorted = df.sort_values(by='Score', ascending=True)
    print(df_sorted)


def model_comparison_classification(p_models_to_comp,
                                    p_X_train,
                                    p_X_val,
                                    p_Y_train,
                                    p_Y_val,
                                    p_predictors,
                                    #p_metric='L1', # Options: L1, L2, R2  # Unused
                                    #p_models=['All'],  # Unused
                                    p_seed=0,
                                    p_verbose=False):
    """ Add another argument to allow transformations of the target variable (default to identity) e.g. see transform in SVM, Bagging """
    """ Add another action if p_verbose = True where run-time of each model is calculated (this will help with optimizing hyper-parameter tuning) """
    # List of models tested
    models = []

    # List to store results from each model
    results = []

    for model_object, model_name, hyper_parameters in p_models_to_comp:
        if hyper_parameters: # If there are hyperparameters to test do it here
            for param in hyper_parameters[0]:
                model_with_hyperparameters = model_object
                model_with_hyperparameters.fit(p_X_train.iloc[:, p_predictors], p_Y_train, param)
                result = model_with_hyperparameters.score(p_X_val.iloc[:, p_predictors], p_Y_val)
                results.append(result)
                if p_verbose:
                    print('{} {} - {}'.format(model_with_hyperparameters.label, param, result))
                models.append(['{} {}'.format(model_with_hyperparameters.label, param), result])
        else:
            model = model_object
            model.fit(p_X_train.iloc[:, p_predictors], p_Y_train, None)
            result = model.score(p_X_val.iloc[:, p_predictors], p_Y_val)
            results.append(result)
            if p_verbose:
                print('{} XX - {}'.format(model.label, result))
            models.append(['{} XX'.format(model.label), result])

    # Plot the results of all models
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(results)
    # Label the axis with model names
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label[0] for label in models], rotation='vertical')
    # Plot the accuracy for all combinations
    plt.show()

    df = pd.DataFrame(columns=['Model', 'Validation Fit'], data=models)
    df_sorted = df.sort_values(by='Validation Fit', ascending=False)
    print(df_sorted)


def model_comparison_regression_stdset(p_X_train,
                                       p_X_val,
                                       p_Y_train,
                                       p_Y_val,
                                       p_predictors,
                                       p_metric='L1', # Options: L1, L2, R2
                                       p_models=['All'],
                                       p_seed=0,
                                       p_verbose=False):
    # List of models to be run. Formatted: [model name, package name, module name, hyper-parameters (optional)]
    models_to_test = []
    if 'LinearRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Linear Regression', 'sklearn.linear_model', 'LinearRegression', []])
    if 'RidgeRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Ridge Regression', 'sklearn.linear_model', 'Ridge', [np.array([1.0])]])
    if 'LassoRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Lasso Regression', 'sklearn.linear_model', 'Lasso', [np.array([0.001])]])
    if 'ElasticNetRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Elastic Net Regression', 'sklearn.linear_model', 'ElasticNet', [np.array([0.001])]])
    if 'KNearestNeighbors' in p_models or 'All' in p_models:
        models_to_test.append(['KNN', 'sklearn.neighbors', 'KNeighborsRegressor', [np.array([1])]])
    if 'CART' in p_models or 'All' in p_models:
        models_to_test.append(['CART', 'sklearn.tree', 'DecisionTreeRegressor', [np.array([5])]])
    if 'RandomForest' in p_models or 'All' in p_models:
        models_to_test.append(['Random Forest', 'sklearn.ensemble', 'RandomForestRegressor', [np.array([50])]])
    if 'ExtraTrees' in p_models or 'All' in p_models:
        models_to_test.append(['Extra Trees', 'sklearn.ensemble', 'ExtraTreesRegressor', [np.array([50])]])
    if 'AdaBoost' in p_models or 'All' in p_models:
        models_to_test.append(['Ada Boost', 'sklearn.ensemble', 'AdaBoostRegressor', [np.array([100])]])
    if 'SGBoosting' in p_models or 'All' in p_models:
        models_to_test.append(['SG Boost', 'sklearn.ensemble', 'GradientBoostingRegressor', [np.array([100])]])
    if 'XGBoost' in p_models or 'All' in p_models:
        models_to_test.append(['XG Boost', 'xgboost', 'XGBRegressor', [np.array([1000])]])

    models_to_comp = list()

    for model_label, module_name, model_name, hyper_parameters in models_to_test:
        models_to_comp.append([sklearn_Model(model_label, module_name, model_name, p_metric, p_seed=p_seed), model_name, hyper_parameters])

    model_comparison_regression(models_to_comp,
                                p_X_train,
                                p_X_val,
                                p_Y_train,
                                p_Y_val,
                                p_predictors,
                                p_seed=0,
                                p_verbose=False)


def model_comparison_classification_stdset(p_X_train,
                                           p_X_val,
                                           p_Y_train,
                                           p_Y_val,
                                           p_predictors,
                                           p_metric='L1', # Options: L1, L2, R2
                                           p_models=['All'],
                                           p_seed=0,
                                           p_verbose=False):
    # Populate list of models to be run. Formatted: [model name, package name, module name, hyper-parameters (optional)]
    models_to_test = []
    if 'LogisticRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Logistic Regression', 'sklearn.linear_model', 'LogisticRegression', []])
    if 'RidgeClassifier' in p_models or 'All' in p_models:
        models_to_test.append(['Ridge Classifier', 'sklearn.linear_model', 'RidgeClassifier', [np.array([1.0])]])
    if 'LassoRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Lasso Classifier', 'sklearn.linear_model', 'Lasso', [np.array([0.001])]])
    if 'LinearSVM' in p_models or 'All' in p_models:
        models_to_test.append(['Linear SVM', 'sklearn.svm', 'LinearSVC', [np.array([1.0, 3.0])]])
    if 'SGDClassifier' in p_models or 'All' in p_models:
        models_to_test.append(['SGD', 'sklearn.linear_model', 'SGDClassifier', [np.array([0.0001])]])
    if 'KNearestNeighbors' in p_models or 'All' in p_models:
        models_to_test.append(['KNN', 'sklearn.neighbors', 'KNeighborsClassifier', [np.array([1,10])]])
    if 'CART' in p_models or 'All' in p_models:
        models_to_test.append(['CART', 'sklearn.tree', 'DecisionTreeClassifier', [np.array([5])]])
    if 'RandomForest' in p_models or 'All' in p_models:
        models_to_test.append(['Random Forest', 'sklearn.ensemble', 'RandomForestClassifier', [np.array([10, 50, 100, 250])]])
    if 'ExtraTrees' in p_models or 'All' in p_models:
        models_to_test.append(['Extra Trees', 'sklearn.ensemble', 'ExtraTreesClassifier', [np.array([20, 100])]])
    if 'AdaBoost' in p_models or 'All' in p_models:
        models_to_test.append(['Ada Boost', 'sklearn.ensemble', 'AdaBoostClassifier', [np.array([50, 100, 500, 1000, 5000, 10000])]])
    if 'SGBoosting' in p_models or 'All' in p_models:
        models_to_test.append(['SG Boosting', 'sklearn.ensemble', 'GradientBoostingClassifier', [np.array([50])]])
    if 'XGBoost' in p_models or 'All' in p_models:
        models_to_test.append(['XG Boost', 'xgboost', 'XGBClassifier', [np.array([1000])]])
    if 'NaiveBayes' in p_models or 'All' in p_models:
        models_to_test.append(['Naive Bayes', 'sklearn.naive_bayes', 'BernoulliNB', [np.array([1.0])]])
    if 'MLPClassifier' in p_models or 'All' in p_models:
        models_to_test.append(['MLP Classifier', 'sklearn.neural_network', 'MLPClassifier', [np.array([0.001, 0.0001])]])

    models_to_comp = list()

    for model_label, module_name, model_name, hyper_parameters in models_to_test:
        models_to_comp.append([sklearn_Model(model_label, module_name, model_name, p_metric, p_seed=p_seed), model_name, hyper_parameters])

    model_comparison_classification(models_to_comp,
                                    p_X_train,
                                    p_X_val,
                                    p_Y_train,
                                    p_Y_val,
                                    p_predictors,
                                    p_seed=0,
                                    p_verbose=False)

