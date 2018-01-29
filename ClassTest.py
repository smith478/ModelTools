
import pandas as pd 
import numpy as np
from scipy import stats
import importlib
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, heuristic, transformation, score, params, label="Unlabelled"):
        self.heuristic = heuristic
        self.transformation = transformation
        self.score_func = score
        self.params = params
        self.label = label

    def transform(self, X):
        return self.transformation(self.params, X)

    def fit(self, X, Y, hyperparams):
        self.params = self.heuristic(hyperparams, self.score_func)(X, Y)

    def score(self, X, Y):
        return self.score_func(Y, self.transform(X))


def sklearn_Model(model_label, module_name, model_name, p_metric, p_seed=0):
    #Scoring parameter
    import sys

    if p_metric == 'L1':
        from sklearn.metrics import mean_absolute_error
        metric = mean_absolute_error
    elif p_metric == 'L2':
        from sklearn.metrics import mean_squared_error
        metric = mean_squared_error
    elif p_metric == 'R2':
        from sklearn.metrics import r2_score
        metric = r2_score
    else:
        print('{} is not an available metric'.format(p_metric))
        sys.exit

    my_label = model_label

    # Import the model
    module = importlib.import_module(module_name)

    # The hack here is that the model itself is the parameter.
    my_params = getattr(module, model_name)

    if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(alpha=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['KNeighborsRegressor']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_neighbors=p_hyperparams)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['DecisionTreeRegressor']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(max_depth=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_estimators=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['XGBRegressor']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_estimators=p_hyperparams, seed=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    else:
        def my_heuristic(pp_hyperparams, p_scoring):
            # Don't use hyperparams in this case.
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)()
                this_model.fit(X, Y)
                return this_model
            return my_fit_function

    def my_transformation(p_params, X):
        return p_params.predict(X)

    my_score = metric

    my_Model = Model(my_heuristic, my_transformation, my_score, my_params, my_label)
    return my_Model


def ModelComparisonRegression_v2(p_X_train,
                              p_X_val,
                              p_Y_train,
                              p_Y_val,
                              p_predictors,
                              p_metric = 'L1', # Options: L1, L2, R2
                              p_models = ['All'],
                              p_seed = 0,
                              p_verbose = False):
    """ Add another argument to allow transformations of the target variable (default to identity) e.g. see transform in SVM, Bagging """
    """ Add another action if p_verbose = True where run-time of each model is calculated (this will help with optimizing hyper-parameter tuning) """
    """ Break out each model type into its own function and have this function call all of those """
    # List of models tested
    models = []

    # List to store results from each model
    results = []

    # List of models to be run. Formatted: [model name, package name, module name, hyper-parameters (optional)]
    models_to_test = []
    if 'LinearRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Linear Regression', 'sklearn.linear_model', 'LinearRegression', []])
    if 'RidgeRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Ridge Regression', 'sklearn.linear_model', 'Ridge', [np.array([1.0])]])
    if 'LassoRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Lasso Regression', 'sklearn.linear_model', 'Lasso',[np.array([0.001])]])
    if 'ElasticNetRegression' in p_models or 'All' in p_models:
        models_to_test.append(['Elastic Net Regression', 'sklearn.linear_model', 'ElasticNet',[np.array([0.001])]])
    if 'KNearestNeighbors' in p_models or 'All' in p_models:
        models_to_test.append(['KNN', 'sklearn.neighbors', 'KNeighborsRegressor',[np.array([1])]])
    if 'CART' in p_models or 'All' in p_models:
        models_to_test.append(['CART', 'sklearn.tree', 'DecisionTreeRegressor',[np.array([5])]])
    if 'RandomForest' in p_models or 'All' in p_models:
        models_to_test.append(['Random Forest', 'sklearn.ensemble', 'RandomForestRegressor',[np.array([50])]])
    if 'ExtraTrees' in p_models or 'All' in p_models:
        models_to_test.append(['Extra Trees', 'sklearn.ensemble', 'ExtraTreesRegressor',[np.array([50])]])
    if 'AdaBoost' in p_models or 'All' in p_models:
        models_to_test.append(['Ada Boost', 'sklearn.ensemble', 'AdaBoostRegressor',[np.array([100])]])
    if 'SGBoosting' in p_models or 'All' in p_models:
        models_to_test.append(['SG Boost', 'sklearn.ensemble', 'GradientBoostingRegressor',[np.array([100])]])
    if 'XGBoost' in p_models or 'All' in p_models:
        models_to_test.append(['XG Boost', 'xgboost', 'XGBRegressor',[np.array([1000])]])

    import importlib
    for model_label, module_name, model_name, hyper_parameters in models_to_test:
        # Import the model
        module = importlib.import_module(module_name)

        if hyper_parameters: # If there are hyperparameters to test do it here
            for param in hyper_parameters[0]:
                model_with_hyperparameters = sklearn_Model(model_label, module_name, model_name, p_metric, p_seed=p_seed)
                model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors], p_Y_train, param)
                result = model_with_hyperparameters.score(p_X_val.iloc[:,p_predictors], p_Y_val)
                results.append(result)
                if p_verbose:
                    print('{} {} - {}'.format(model_label, param, result))
                models.append(['{} {}'.format(model_label, param), result])
        else:
            model = sklearn_Model(model_label, module_name, model_name, p_metric, p_seed=p_seed)
            model.fit(p_X_train.iloc[:,p_predictors], p_Y_train, None)
            result = model.score(p_X_val.iloc[:,p_predictors], p_Y_val)
            results.append(result)
            if p_verbose:
                print('{} XX - {}'.format(model_label, result))
            models.append(['{} XX'.format(model_label), result])

    # Plot the results of all models
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(results)
    # Label the axis with model names
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label[0] for label in models],rotation='vertical')
    # Plot the accuracy for all combinations
    plt.show()

    if p_metric in ['L1','L2']:
        df = pd.DataFrame(columns = ['Model','Validation Error'], data=models)
        df_sorted = df.sort_values(by='Validation Error', ascending=True)
        print(df_sorted)
    elif p_metric in ['R2']:
        df = pd.DataFrame(columns = ['Model','Validation R2'], data=models)
        df_sorted = df.sort_values(by='Validation R2', ascending=False)
        print(df_sorted)


