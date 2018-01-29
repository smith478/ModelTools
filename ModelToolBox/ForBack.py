import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

def UnivariateBackward(p_model,
                       p_features,
                       p_predictors,
                       p_X_train,
                       p_Y_train,
                       p_X_val,
                       p_Y_val,
                       p_hyperparams=None,
                       p_top_k_features=5,
                       p_holdout=True,
                       p_seed=0,
                       p_verbose=False,
                       p_runtime_estimate=False):
    if p_runtime_estimate:
        import timeit

    p_model.fit(p_X_train.iloc[:,p_predictors], p_Y_train, p_hyperparams)
    # TODO Add an argument here p_holdout = True where if it's False, we look at the metric on the training set.

    result_full = p_model.score(p_X_val.iloc[:,p_predictors], p_Y_val)

    performance = []
    # Accuracy of the model using all features

    for i, (feature_name, index) in enumerate(p_features):
        if p_verbose:
            print(feature_name)

        this_set = set(index)
        test_index = [x for x in p_predictors if x not in this_set]

        if p_runtime_estimate:
            now = timeit.default_timer()
        this_model.fit(p_X_train.iloc[:,test_index], p_Y_train, p_hyperparams)
        if p_runtime_estimate:
            then = timeit.default_timer()
            est_runtime = (then - now) / 60 * (len(p_features) - i)
            print('About {:.2f} minutes remaining'.format(est_runtime))

        this_result = p_model.score(p_X_val.iloc[:,test_index], p_Y_val)
        this_marginal_result = result_full - this_result # higher values imply the variable that was removed provided significant model importance

        performance.append([feature_name, this_marginal_result])

    df = pd.DataFrame(columns = ['Variable','Marginal Validation AUC'], data = performance)
    df_sorted = df.sort_values(by='Marginal Validation AUC', ascending=False)

    top_k_features = df_sorted.iloc[:p_top_k_features,:]
    print(top_k_features)


def ForBackPremutations(p_model,
                        p_features,
                        p_predictors,
                        p_X_train,
                        p_Y_train,
                        p_X_val,
                        p_Y_val,
                        p_hyperparams=None,
                        p_regression = True,
                        p_metric = 'L1 Error', # For regression: have Lp error
                                          # For classification AUC, R^2, accuracy, future: LR Test
                        p_cross_validations = 5,
                        p_validation_fraction = 0.25,
                        p_threshold = 0, # This will depend on the metric used
                        p_seed = 0,
                        p_forward = True,
                        p_verbose = False):
    # TODO Add in p_regression and p_metric peices

    # Since we're doing cross-validation here, make X_train a larger portion of th overall dataset (80-90%)
    best_pred_index = p_predictors
    best_vars = p_features[:]

    # Run a backward regression on all variables
    keep_going = True # To get things started
    while keep_going and len(best_vars) > 0: # Hopefully you don't get rid of all your variables!
        keep_going = False # If none of the remaining variables improve performance, then stop

        scores = [] # instead make this a list storing: feature_name, index, score. Then aggregate by taking the mean
        rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)

        # crossvalidate the scores on a number of different random splits of the data
        for i, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
            if p_verbose:
                print('Iteration number {} out of {} for the train test split'.format(i+1,p_cross_validations))
            Xt_train, Xt_test = p_X_train.iloc[train_index,best_pred_index], p_X_train.iloc[test_index,best_pred_index] # set X for train and test
            Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
            p_model.fit(Xt_train, Yt_train, p_hyperparams) # fit the model on the training set
            score = p_model.score(Xt_test, Yt_test)
            for j, (feature_name, index) in enumerate(best_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                np.random.seed(p_seed) # set seed for reproducibility
                X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
                X_t = X_t.iloc[:,best_pred_index] # Filter down to the remaining variables
                shuff_score = p_model.score(X_t, Yt_test)
                scores.append([feature_name, index, (shuff_score-score)/shuff_score])
                if p_verbose:
                    if j == 0:
                        print('{:*^65}'.format('Looping Through Remaining Variables'))
                        print('There are {} variables remaining to test'.format(len(best_vars)))
                    print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(shuff_score-score)/shuff_score))

        df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
        df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
        dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
        dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
        df_sorted = dfAgg.sort_values(by='Score',ascending=True).reset_index(drop=True)

        if df_sorted['Score'][0] < p_threshold: # Remove the worst variable and update relevant fields
            if p_verbose:
                print('Removing {} from our list of variables, average score: {:.5f}'.format(df_sorted['Variable'][0],df_sorted['Score'][0]))

            keep_going = True # You're not done yet

            best_vars = [[name, index] for name, index in best_vars if name != df_sorted['Variable'][0]] # exclude the worst variable
            best_pred_index = [elt for elt in best_pred_index if elt not in df_sorted['Var_Index'][0]]

            df_sorted_final = df_sorted.copy()

    if p_forward: # Idea for forward iteration is to iteratively test the addition of a single variable x and add in those that improve the model most
        # We have to be able to swap these out easily.
        part_params, full_params = p_model.params, p_model.params

        if p_verbose:
            print('{:*^65}'.format('Starting Forward Iteration'))
        keep_going = True
        leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
        leftover_index = list(set(p_predictors) - set(best_pred_index))
        while keep_going and len(leftover_vars) > 0:
            keep_going = False

            scores = [] # instead make this a list storing: feature_name, index, score. Then aggregate by taking the mean
            rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)

            # crossvalidate the scores on a number of different random splits of the data
            for i, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
                if p_verbose:
                    print('Iteration number {} out of {} for the train test split'.format(i+1,p_cross_validations))
                Xt_train, Xt_test = p_X_train.iloc[train_index,best_pred_index], p_X_train.iloc[test_index,best_pred_index] # set X for train and test
                Xt_train_full = p_X_train.iloc[train_index,:]
                Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
                p_model.fit(Xt_train_full, Yt_train, p_hyperparams) # really this only needs to be fit on the first iteration of the while loop, and stored
                full_params = p_model.params
                p_model.fit(Xt_train, Yt_train, p_hyperparams) # fit the model on the training set
                part_params = p_model.params
                score = p_model.score(Xt_test, Yt_test)
                for j, (feature_name, index) in enumerate(leftover_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                    X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                    shuffle_index = list(set(leftover_index)-set(index))
                    np.random.seed(p_seed) # set seed for reproducibility
                    X_t.iloc[:,shuffle_index] = np.random.permutation(X_t.iloc[:,shuffle_index]) # Permute the observations from the ith variable

                    p_model.params = full_params
                    shuff_score = p_model.score(X_t, Yt_test)
                    scores.append([feature_name, index, (score-shuff_score)/shuff_score])
                    if p_verbose:
                        if j == 0:
                            print('{:*^65}'.format('Looping Through Remaining Variables'))
                            print('There are {} variables remaining to test'.format(len(leftover_vars)))
                        print('Testing {}. {}: {:.5f}'.format(j+1, feature_name, (score-shuff_score)/shuff_score))

            df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
            df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
            dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
            dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
            df_sorted = dfAgg.sort_values(by='Score',ascending=False).reset_index(drop=True)

            if df_sorted['Score'][0] > p_threshold: # Add the best variable and update relevant fields
                if p_verbose:
                    print('Adding {} from our list of variables, average score: {:.5f}'.format(df_sorted['Variable'][0],df_sorted['Score'][0]))

                keep_going = True # You're not done yet

                best_vars.append([df_sorted['Variable'][0], df_sorted['Var_Index'][0]]) # include the best variable
                best_pred_index = best_pred_index + df_sorted['Var_Index'][0]

                leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
                leftover_index = list(set(p_predictors) - set(best_pred_index))

    df_sorted_final = df_sorted_final.sort_values(by='Score', ascending=False).reset_index(drop=True)
    print('{:*^65}'.format('Variable Performance From Final Backward Iteration:'))
    for i in range(len(df_sorted_final)):
        print('{}: {:.5f}'.format(df_sorted_final['Variable'][i],df_sorted_final['Score'][i]))

    print('{:*^65}'.format('The optimal set of variables is:'))
    for name, index in best_vars:
        print(name)

    p_model.params = part_params
    p_model.fit(p_X_train.iloc[:,best_pred_index], p_Y_train, p_hyperparams)
    score = p_model.score(p_X_val.iloc[:,best_pred_index], p_Y_val)

