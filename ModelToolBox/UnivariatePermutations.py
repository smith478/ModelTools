import sys
import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

def UnivariatePermutations(p_model,
                           p_features,
                           p_X_train,
                           p_Y_train,
                           p_hyperparams = None,
                           p_cross_validations = 5,
                           p_validation_fraction = 0.25,
                           p_seed = 0,
                           p_verbose = False):
    # TODO Add in p_regression and p_metric peices
    # TODO since we're doing cross-validation here, make X_train a larger portion of th overall dataset (80-90%)

    scores = []
    rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)

    # crossvalidate the scores on a number of different random splits of the data
    for j, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
        if p_verbose:
            print('Iteration number {} for the train test split'.format(j+1))
        Xt_train, Xt_test = p_X_train.iloc[train_index,:], p_X_train.iloc[test_index,:] # set X for train and test
        Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
        p_model.fit(Xt_train, Yt_train, p_hyperparams) # fit the model on the training set
        score = p_model.score(Xt_test, Yt_test)
        for i, (feature_name, index) in enumerate(p_features): # shuffle the ith predictor variable (to break any relationship with the target)
            X_t = Xt_test.copy() # copy the test data, so as not to disturb
            np.random.seed(p_seed) # set seed for reproducibility
            X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
            shuff_score = p_model.score(X_t, Yt_test)
            scores.append([feature_name, index, (shuff_score-score)/shuff_score])
            if p_verbose:
                if i == 0:
                    print('{:*^65}'.format('Looping Through Remaining Variables'))
                print('Testing {}. {}: {:.5f}'.format(i+1,feature_name,(shuff_score-score)/shuff_score))
    print('{:*^65}'.format("Features sorted by their score:"))

    df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
    df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
    dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
    dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
    df_sorted = dfAgg.sort_values(by='Score',ascending=False).reset_index(drop=True)

    for i in range(len(df_sorted)):
        print('{}: {:.5f}'.format(df_sorted['Variable'][i],df_sorted['Score'][i]))
