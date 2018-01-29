import importlib
import sys

sys.path.append("..")  # Find a better way to do this.
from Model import Model

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
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
        def my_auc(X, Y):
            fpr, tpr, thresholds = roc_curve(Y, model_with_hyperparameters.predict(X))
            return auc(fpr, tpr)
        metric = my_auc
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
        metric = accuracy_score
    else:
        print('{} is not an available metric'.format(p_metric))
        sys.exit

    my_label = model_label

    # Import the model
    module = importlib.import_module(module_name)

    # The hack here is that the model itself is the parameter.
    my_params = getattr(module, model_name)

    if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'RidgeClassifier', 'SGDClassifier', 'MLPClassifier']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(alpha=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['LinearSVC']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(C=p_hyperparams)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['KNeighborsRegressor', 'KNeighborsClassifier']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_neighbors=p_hyperparams)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['DecisionTreeRegressor', 'DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(max_depth=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'AdaBoostClassifier']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_estimators=p_hyperparams, random_state=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['XGBRegressor', 'XGBClassifier']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(n_estimators=p_hyperparams, seed=p_seed)
                this_model.fit(X, Y)
                return this_model
            return my_fit_function
    elif model_name in ['BernoulliNB']:
        def my_heuristic(p_hyperparams, p_scoring):
            def my_fit_function(X, Y):
                this_model = getattr(module, model_name)(alpha=p_hyperparams, seed=p_seed)
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

    my_Model = Model(my_heuristic, my_transformation, my_score, my_params, label=my_label, copy=False)
    return my_Model
