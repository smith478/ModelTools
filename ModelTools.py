# -*- coding: utf-8 -*-

## Import General Packages
import pandas as pd
import numpy as np

## Import plotting libraries
import matplotlib.pyplot as plt

def ContCatSplit(p_data, p_numeric_cat_index=np.array([])):
    """ This function could likely be vectorized in one step """
    from pandas.core.dtypes.common import is_numeric_dtype

    cont = []
    cat = []

    n_rows = p_data.shape[1]
    for i in range(n_rows):
        if is_numeric_dtype(p_data.dtypes[i]) and i not in p_numeric_cat_index:
            cont.append(i)
        else:
            cat.append(i)
    return cont, cat


def CreateDummyVars(p_data,
                    p_predictors,
                    p_target,
                    p_numeric_cat_index=np.array([]),
                    p_weight=None,
                    #p_control=None,  # Unused
                    p_verbose=False):
    """ Have this create column/feature names for each of the dummy variables based on the level name.
    Also have this so that it returns index of predictors, target, weight, controls """
    cont_index = np.intersect1d(p_predictors, ContCatSplit(p_data, p_numeric_cat_index)[0])
    cat_index = np.intersect1d(p_predictors, ContCatSplit(p_data, p_numeric_cat_index)[1])
    cat_predictors = p_data.iloc[:, cat_index]

    target = p_data.iloc[:, p_target]
    if p_weight is not None:
        weight = p_data.iloc[:, p_weight]

    cols = cat_predictors.columns
    labels = []

    for i in range(len(cat_index)):
        labels.append(list(cat_predictors[cols[i]].unique()))

    # Import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    # One hot encode all categorical attributes
    cats = pd.DataFrame()
    for i in range(len(cat_index)):
        # Label encode
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[i])
        feature = label_encoder.transform(cat_predictors.iloc[:, i])
        feature = feature.reshape(cat_predictors.shape[0], 1)
        # One hot encode
        onehot_encoder = OneHotEncoder(sparse=False, n_values=len(labels[i]))
        feature = onehot_encoder.fit_transform(feature)
        # The sort-unique can be made quicker if we need it.
        feature = pd.DataFrame(feature, columns=[(cols[i]+"-"+str(j)) for j in sorted(cat_predictors[cols[i]].unique())])
        cats = pd.concat([cats, feature], axis=1)

    # Print the shape of the encoded data
    if p_verbose:
        print('Dimensions of encoded categorical variables:')
        print(cats.shape)

    # Concatenate encoded attributes with continuous attributes, target variable, control variables, and weights if there are any.
    if p_weight is None:
        dataset_encoded = pd.concat([cats, p_data.iloc[:, cont_index], target], axis=1)
    else:
        dataset_encoded = pd.concat([cats, p_data.iloc[:, cont_index], weight, target], axis=1)
    if p_verbose:
        print('Dimensions of the updated encoded data set:')
        print(dataset_encoded.shape)

    return dataset_encoded


def TrainHoldSplit(p_data,
                   p_predictors,
                   p_target,
                   p_controls=None,
                   p_weight=None,
                   p_val_size=0.2,
                   p_seed=0):
    X = p_data.iloc[:, p_predictors]
    Y = p_data.iloc[:, p_target]
    if p_weight is not None:
        W = p_data.iloc[:, p_weight]
    if p_controls is not None:
        pass
        #C = p_data.iloc[:,p_controls]  # Unused

    # Split the data
    from sklearn import cross_validation
    if p_weight is None:
        X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=p_val_size, random_state=p_seed)
        return X_train, X_val, Y_train, Y_val
    else:
        X_train, X_val, Y_train, Y_val, W_train, W_val = cross_validation.train_test_split(X, Y, W, test_size=p_val_size, random_state=p_seed)
        return X_train, X_val, Y_train, Y_val, W_train, W_val


def FeatureIndexes(p_data,
                   p_encoded_data,
                   p_encoded_target,
                   p_verbose=False):
    feature_index = []
    levels = list(p_encoded_data.columns)
    for col in p_data.columns:
        if p_verbose:
            print(col)
        index_list = []
        for i, level in enumerate(levels):
            if p_verbose:
                print('testing ' + level)
            if ((level.find(col+'-') == 0) or (level == col)) and (i != p_encoded_target):
                index_list.append(i)
                if p_verbose:
                    print(level + ' added')
        if index_list: # remove empty lists
            feature_index.append([col, index_list])
    return feature_index


def AutoBucket(p_X,
               p_Y,
               p_n_buckets=20,
               p_X_transformation=None,
               p_Y_transformation=None):
    """ This function will automatically bucket the x values into p_n_buckets """
    predictor = pd.Series(p_X, name='predictor')
    target = pd.Series(p_Y, name='target')
    buckets = pd.Series(pd.qcut(predictor, p_n_buckets, duplicates='drop'), name='buckets')

    df = pd.concat([predictor, target, buckets], axis=1)

    X_mean = df.groupby(by=['buckets'])['predictor'].mean()
    Y_mean = df.groupby(by=['buckets'])['target'].mean()

    return X_mean, Y_mean


def UnivariateAnalysis(p_features,
                       p_X_train,
                       p_Y_train,
                       p_X_val,
                       p_Y_val,
                       p_top_k_features=5,
                       p_model='continuous', # choose from continuous, binary, multinomial
                       p_target_distribution='gamma',
                       p_metric='L1 Error', # choose from L1 Error, AUC
                       p_seed=0,
                       p_subsamplesize=1500,
                       p_n_buckets=20,
                       p_verbose=False):
    feature_error = []

    import sys

    # Import the library
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_linear_model import GLMResults

    # Scoring parameter
    if p_metric == 'L1 Error':
        from sklearn.metrics import mean_absolute_error
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    else:
        print('{} is not currently an option'.format(p_metric))
        sys.exit()

    for name, index in p_features:
        if p_verbose:
            print(name)
        # Fit the model
        # add intercept to continuous variables and classification models
        if (len(index) == 1) or (p_model in ['binary', 'multinomial']):
            train_data = sm.add_constant(p_X_train.iloc[:, index])
            val_data = sm.add_constant(p_X_val.iloc[:, index])
        else:
            train_data = p_X_train.iloc[:, index]
            val_data = p_X_val.iloc[:, index]

        if p_model == 'continuous':
            model = sm.GLM(p_Y_train, train_data, family=sm.families.Gamma(sm.families.links.log))
            try:
                result = model.fit()
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name, err))
                continue
        elif p_model == 'binary':
            model = sm.Logit(p_Y_train, train_data)
            try:
                result = model.fit(disp=0)
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name, err))
                continue
        elif p_model == 'multinomial':
            model = sm.MNLogit(p_Y_train, train_data)
            try:
                result = model.fit(disp=0)
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name, err))
                continue
        else:
            print('{} is not an available model option'.format(p_model))

        # Calculate the error with the selected metric
        if p_metric == 'L1 Error':
            error = mean_absolute_error(p_Y_val, result.predict(val_data))
        elif p_metric == 'AUC':
            try: # TODO make this more specific as well
                fpr, tpr, thresholds = roc_curve(p_Y_val, result.predict(val_data))
            except:
                print('{} AUC calculation failed'.format(name))
                continue
            error = auc(fpr, tpr)

        feature_error.append([name, error, index])

    if p_metric in ['L1 Error']:
        df = pd.DataFrame(columns=['Variable', 'Validation Error', 'Index'],
                          data=feature_error)
        df_sorted = df.sort_values(by='Validation Error')
    elif p_metric in ['AUC']:
        df = pd.DataFrame(columns=['Variable', 'Validation AUC', 'Index'],
                          data=feature_error)
        df_sorted = df.sort_values(by='Validation AUC', ascending=False)

    top_k_features = df_sorted.iloc[:p_top_k_features, :]

    print(top_k_features.iloc[:, :2])

    for name, error, index in pd.DataFrame.as_matrix(top_k_features):
        if len(index) == 1:
            print('Feature: ' + str(name))
            if p_metric in ['L1 Error']:
                print('Validation Error: ' + str(error))
            elif p_metric == 'AUC':
                print('AUC: ' + str(error))

            X_train_const = sm.add_constant(p_X_train.iloc[:, index])
            # X_val_const = sm.add_constant(p_X_val.iloc[:,index])

            if p_model == 'continuous':
                model = sm.GLM(p_Y_train, X_train_const, family=sm.families.Gamma(sm.families.links.log))
            elif p_model == 'binary':
                model = sm.Logit(p_Y_train, X_train_const)
            elif p_model == 'multinomial':
                model = sm.MNLogit(p_Y_train, X_train_const)
            result = model.fit(disp=0)

            print('Training AIC: ' + str(result.aic))

            # plot fitted vs observed on both training and validation data
            y_pred_train = result.predict(X_train_const)
            # y_pred_val = result.predict(X_val_const)

            plot_data_train = pd.DataFrame(np.column_stack([p_X_train.iloc[:, index], p_Y_train, y_pred_train]),
                                           columns=[list(p_X_train.columns[index])[0], 'y', 'y_pred'])

            if p_model == 'binary':
                x_values, y_values = AutoBucket(plot_data_train[list(p_X_train.columns[index])[0]], plot_data_train['y'], p_n_buckets)
            else:
                from random import sample, seed
                seed(p_seed)
                rand_vals = sample(range(len(plot_data_train)), k=min(p_subsamplesize, len(plot_data_train)))
                plot_data_train_sample = plot_data_train.iloc[rand_vals, :]
                plot_data_train_sample_sorted = plot_data_train_sample.sort_values(by=list(p_X_train.columns[index])[0])

            fig, ax = plt.subplots(figsize=(12, 8))

            if p_model == 'binary':
                plot_data_train_sample_sorted = plot_data_train.sort_values(by=list(p_X_train.columns[index])[0])
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0], y='y_pred', ax=ax, linestyle='-', color='b')
                plt.plot(x_values, y_values, 'ro--')
            else:
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0], y='y_pred', ax=ax, linestyle='-', color='b')
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0], y='y', ax=ax, kind='scatter', color='r')
            plt.show()

            print(result.summary())
        else:
            # Add observed (average) values to the graph. Use automatic bucketing of indt variable
            # Add argument to choose between: predicted value, observed value, 95% confidence int
            print('Feature: ' + str(name))
            if p_metric in ['L1 Error']:
                print('Validation Error: ' + str(error))
            elif p_metric == 'AUC':
                print('AUC: ' + str(error))

            if p_model == 'continuous':
                model = sm.GLM(p_Y_train, p_X_train.iloc[:, index], family=sm.families.Gamma(sm.families.links.log))
            elif p_model == 'binary':
                model = sm.Logit(p_Y_train, p_X_train.iloc[:, index])
            elif p_model == 'multinomial':
                model = sm.MNLogit(p_Y_train, p_X_train.iloc[:, index])
            result = model.fit(disp=0)

            print('Training AIC: ' + str(result.aic))

            # TODO add multinomial below
            fig, ax1 = plt.subplots(figsize=(12, 8))
            if p_model == 'continuous':
                upper_bound = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:, 1]))})
                model = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                      'model': list(np.exp(result.params))})
                lower_bound = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:, 0]))})
            elif p_model == 'binary':
                # TODO verify transformation below is correct
                upper_bound = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:, 1])/
                                                             (np.exp(GLMResults.conf_int(result)[:, 1]) + 1))})
                model = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                      'model': list(np.exp(result.params)/(1 + np.exp(result.params)))})
                lower_bound = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:, 0])/
                                                             (np.exp(GLMResults.conf_int(result)[:, 0]) + 1))})
            upper_bound.plot(x='Level', ax=ax1, linestyle='-', marker='o', color='r')
            model.plot(x='Level', ax=ax1, linestyle='-', marker='o', color='b')
            lower_bound.plot(x='Level', ax=ax1, linestyle='-', marker='o', color='g')
            ax1.set_ylabel('Response', color='b')
            ax1.tick_params('y', colors='b')
            ax1.legend(loc='upper left')

            weights = pd.DataFrame({'Level': p_X_train.iloc[:, index].columns,
                                    'weight': list(p_X_train.iloc[:, index].sum(axis=0))})
            plt.xticks(rotation=90)

            ax2 = ax1.twinx()
            weights.plot(x='Level', ax=ax2, kind='bar', color='y', alpha=0.4)
            ax2.set_ylabel('Weight', color='y')
            ax2.set_ylim([0, max(weights.iloc[:, 1]) * 3])
            ax2.tick_params('y', colors='y')
            ax2.legend(loc='upper right')
            ax2.grid(False)

            # fig.tight_layout()
            plt.show()

            print(result.summary())
