# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:48:07 2017

@author: dsmit
"""
## Import General Packages
import pandas as pd 
import numpy as np
from scipy import stats

## Import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

def ContCatSplit(p_data, p_numeric_cat_index = np.array([])):
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
                    p_numeric_cat_index = np.array([]),
                    p_weight = None,
                    p_control = None,
                    p_verbose = False):
    """ Have this create column/feature names for each of the dummy variables based on the level name """
    """ Also have this so that it returns index of predictors, target, weight, controls """
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data,p_numeric_cat_index)[0])
    cat_index = np.intersect1d(p_predictors,ContCatSplit(p_data,p_numeric_cat_index)[1])
    cat_predictors = p_data.iloc[:,cat_index]
    
    target = p_data.iloc[:,p_target]
    if p_weight is not None:
        weight = p_data.iloc[:,p_weight]
    
    cols = cat_predictors.columns 
    labels = []
    
    for i in range(len(cat_index)):
        labels.append(list(cat_predictors[cols[i]].unique()))
        
    #Import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    
    #One hot encode all categorical attributes
    cats = pd.DataFrame()
    for i in range(len(cat_index)):
        #Label encode
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[i])
        feature = label_encoder.transform(cat_predictors.iloc[:,i])
        feature = feature.reshape(cat_predictors.shape[0], 1)
        #One hot encode
        onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
        feature = onehot_encoder.fit_transform(feature)
        feature = pd.DataFrame(feature,columns=[(cols[i]+"-"+str(j)) for j in sorted(cat_predictors[cols[i]].unique())])
        cats = pd.concat([cats,feature],axis=1)
    
    # Print the shape of the encoded data
    if p_verbose:
        print('Dimensions of encoded categorical variables:')
        print(cats.shape)
    
    #Concatenate encoded attributes with continuous attributes, target variable, control variables, and weights if there are any.
    if p_weight is None:
        dataset_encoded = pd.concat([cats,p_data.iloc[:,cont_index],target],axis=1)
    else:
        dataset_encoded = pd.concat([cats,p_data.iloc[:,cont_index],weight,target],axis=1)
    if p_verbose:
        print('Dimensions of the updated encoded data set:')
        print(dataset_encoded.shape)
        
    return dataset_encoded


def TrainHoldSplit(p_data,
                   p_predictors,
                   p_target,
                   p_controls = None,
                   p_weight = None,
                   p_val_size = 0.2,
                   p_seed = 0):
    X = p_data.iloc[:,p_predictors]
    Y = p_data.iloc[:,p_target]
    if p_weight is not None:
        W = p_data.iloc[:,p_weight]
    if p_controls is not None:
        C = p_data.iloc[:,p_controls]
        
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
                   p_verbose = False):
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
               p_n_buckets = 20,
               p_X_transformation = None,
               p_Y_transformation = None):
    """ This function will automatically bucket the x values into p_n_buckets """
    predictor = pd.Series(p_X, name = 'predictor')
    target = pd.Series(p_Y, name = 'target')
    buckets = pd.Series(pd.qcut(predictor, p_n_buckets, duplicates='drop'), name = 'buckets')
    
    df = pd.concat([predictor, target, buckets], axis = 1)
    
    X_mean = df.groupby(by=['buckets'])['predictor'].mean()
    Y_mean = df.groupby(by=['buckets'])['target'].mean()
    
    return X_mean, Y_mean
    
def UnivariateAnalysis(p_features,
                       p_X_train,
                       p_Y_train,
                       p_X_val,
                       p_Y_val,
                       p_top_k_features = 5,
                       p_model = 'continuous', # choose from continuous, binary, multinomial
                       p_target_distribution = 'gamma',
                       p_metric = 'L1 Error', # choose from L1 Error, AUC
                       p_seed = 0,
                       p_subsamplesize = 1500,
                       p_n_buckets = 20,
                       p_verbose = False):
    feature_error = []
    
    import sys
    
    #Import the library
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_linear_model import GLMResults
    
    #Scoring parameter
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
        #Fit the model
        if (len(index) == 1) or (p_model in ['binary','multinomial']): # add intercept to continuous variables and classification models
            train_data = sm.add_constant(p_X_train.iloc[:,index])
            val_data = sm.add_constant(p_X_val.iloc[:,index])
        else:
            train_data = p_X_train.iloc[:,index]
            val_data = p_X_val.iloc[:,index]
            
        if p_model == 'continuous':
            model = sm.GLM(p_Y_train, train_data, family=sm.families.Gamma(sm.families.links.log))
            try:
                result = model.fit()
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name,err))
                continue
        elif p_model == 'binary':
            model = sm.Logit(p_Y_train, train_data)
            try:
                result = model.fit(disp=0)
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name,err))
                continue
        elif p_model == 'multinomial':
            model = sm.MNLogit(p_Y_train, train_data)
            try:
                result = model.fit(disp=0)
            except np.linalg.linalg.LinAlgError as err:
                print('{} failed to fit due to {} error'.format(name,err))
                continue
        else:
            print('{} is not an available model option'.format(p_model))
        
        #Calculate the error with the selected metric
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
        df = pd.DataFrame(columns = ['Variable','Validation Error','Index'],
                      data = feature_error) 
        df_sorted = df.sort_values(by='Validation Error')
    elif p_metric in ['AUC']:
        df = pd.DataFrame(columns = ['Variable','Validation AUC','Index'],
                      data = feature_error)  
        df_sorted = df.sort_values(by='Validation AUC', ascending = False)
    
    top_k_features = df_sorted.iloc[:p_top_k_features,:]
    
    print(top_k_features.iloc[:,:2])
    
    for name, error, index in pd.DataFrame.as_matrix(top_k_features):
        if len(index) == 1:
            print('Feature: ' + str(name))
            if p_metric in ['L1 Error']:
                print('Validation Error: ' + str(error))
            elif p_metric == 'AUC':
                print('AUC: ' + str(error))
                
            X_train_const = sm.add_constant(p_X_train.iloc[:,index])
            #X_val_const = sm.add_constant(p_X_val.iloc[:,index])
            
            if p_model == 'continuous':
                model = sm.GLM(p_Y_train, X_train_const, family=sm.families.Gamma(sm.families.links.log))
            elif p_model == 'binary':
                model = sm.Logit(p_Y_train, X_train_const)
            elif p_model == 'multinomial':
                model = sm.MNLogit(p_Y_train, X_train_const)
            result = model.fit(disp=0)
            
            print('Training AIC: ' + str(result.aic))
            
            """ plot fitted vs observed on both training and validation data """
            y_pred_train = result.predict(X_train_const)
            #y_pred_val = result.predict(X_val_const)
            
            plot_data_train = pd.DataFrame(np.column_stack([p_X_train.iloc[:,index],p_Y_train,y_pred_train]),columns=[list(p_X_train.columns[index])[0],'y','y_pred'])
            
            if p_model == 'binary':
                x_values, y_values = AutoBucket(plot_data_train[list(p_X_train.columns[index])[0]], plot_data_train['y'], p_n_buckets)
            else:
                from random import sample, seed
                seed(p_seed)
                rand_vals = sample(range(len(plot_data_train)), k=min(p_subsamplesize,len(plot_data_train)))
                plot_data_train_sample = plot_data_train.iloc[rand_vals,:]
                plot_data_train_sample_sorted = plot_data_train_sample.sort_values(by=list(p_X_train.columns[index])[0])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if p_model == 'binary':
                plot_data_train_sample_sorted = plot_data_train.sort_values(by=list(p_X_train.columns[index])[0])
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0],y='y_pred',ax=ax,linestyle='-',color='b')
                plt.plot(x_values, y_values, 'ro--')
            else:
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0],y='y_pred',ax=ax,linestyle='-',color='b')
                plot_data_train_sample_sorted.plot(x=list(p_X_train.columns[index])[0],y='y',ax=ax,kind='scatter',color='r')
            plt.show()
            
            print(result.summary())
        else:
            """ Add observed (average) values to the graph. Use automatic bucketing of indt variable """
            """ Add argument to choose between: predicted value, observed value, 95% confidence int """
            print('Feature: ' + str(name))
            if p_metric in ['L1 Error']:
                print('Validation Error: ' + str(error))
            elif p_metric == 'AUC':
                print('AUC: ' + str(error))
             
            if p_model == 'continuous':
                model = sm.GLM(p_Y_train, p_X_train.iloc[:,index], family=sm.families.Gamma(sm.families.links.log))
            elif p_model == 'binary':
                model = sm.Logit(p_Y_train, p_X_train.iloc[:,index])
            elif p_model == 'multinomial':
                model = sm.MNLogit(p_Y_train, p_X_train.iloc[:,index])
            result = model.fit(disp=0)
            
            print('Training AIC: ' + str(result.aic))
            
            # TODO add multinomial below
            fig, ax1 = plt.subplots(figsize=(12, 8))
            if p_model == 'continuous':
                upper_bound = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:,1]))}) 
                model = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                      'model': list(np.exp(result.params))}) 
                lower_bound = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:,0]))})
            elif p_model == 'binary':
                # TODO verify transformation below is correct
                upper_bound = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:,1])/(np.exp(GLMResults.conf_int(result)[:,1])+1))}) 
                model = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                      'model': list(np.exp(result.params)/(1+np.exp(result.params)))}) 
                lower_bound = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                            '95% C.I.': list(np.exp(GLMResults.conf_int(result)[:,0])/(np.exp(GLMResults.conf_int(result)[:,0])+1))})
            upper_bound.plot(x='Level',ax=ax1,linestyle='-', marker='o',color='r')
            model.plot(x='Level',ax=ax1,linestyle='-', marker='o',color='b')
            lower_bound.plot(x='Level',ax=ax1,linestyle='-', marker='o',color='g')
            ax1.set_ylabel('Response', color='b')
            ax1.tick_params('y', colors='b')
            ax1.legend(loc='upper left')
            
            weights = pd.DataFrame({'Level': p_X_train.iloc[:,index].columns,
                                    'weight': list(p_X_train.iloc[:,index].sum(axis=0))})
            plt.xticks(rotation=90)
            
            ax2 = ax1.twinx()
            weights.plot(x='Level',ax=ax2,kind='bar',color='y',alpha=0.4)
            ax2.set_ylabel('Weight', color='y')
            ax2.set_ylim([0,max(weights.iloc[:,1])*3])
            ax2.tick_params('y', colors='y')
            ax2.legend(loc='upper right')
            ax2.grid(False)
            
            #fig.tight_layout()
            plt.show()
            
            print(result.summary())

def UnivariateRandomForestPermutations(p_features,
                                       p_X_train,
                                       p_Y_train,
                                       p_regression = True,
                                       p_metric = 'L1 Error', # For regression: have Lp error
                                                         # For classification AUC, R^2, accuracy, future: LR Test
                                       p_cross_validations = 5,
                                       p_validation_fraction = 0.25,
                                       p_n_estimators = 250,
                                       p_seed = 0,
                                       p_verbose = False):
    # TODO Add in p_regression and p_metric peices
    from sklearn.model_selection import ShuffleSplit
    
    import sys
    
    #Scoring parameter
    if p_metric == 'L1 Error':
        from sklearn.metrics import mean_absolute_error
    elif p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
    if p_regression:
        from sklearn.ensemble import RandomForestRegressor
        this_model = RandomForestRegressor(n_estimators = p_n_estimators,random_state=p_seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        this_model = RandomForestClassifier(n_estimators = p_n_estimators,random_state=p_seed)
        
    # TODO since we're doing cross-validation here, make X_train a larger portion of th overall dataset (80-90%)
    
    scores = []
    rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)
    
    # crossvalidate the scores on a number of different random splits of the data
    for j, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
        if p_verbose:
            print('Iteration number {} for the train test split'.format(j+1))
        Xt_train, Xt_test = p_X_train.iloc[train_index,:], p_X_train.iloc[test_index,:] # set X for train and test
        Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
        this_model.fit(Xt_train, Yt_train) # fit the model on the training set
        if p_metric == 'L1 Error':
            error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
        elif p_metric == 'R^2':
            acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
        elif p_metric == 'AUC':
            fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
            acc = auc(fpr, tpr)
        elif p_metric == 'accuracy':
            acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
        for i, (feature_name, index) in enumerate(p_features): # shuffle the ith predictor variable (to break any relationship with the target)
            X_t = Xt_test.copy() # copy the test data, so as not to disturb
            np.random.seed(p_seed) # set seed for reproducibility
            X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
            if p_metric == 'L1 Error':
                shuff_error = mean_absolute_error(Yt_test, this_model.predict(X_t))
            elif p_metric == 'R^2':
                shuff_acc = r2_score(Yt_test, this_model.predict(X_t))
            elif p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(X_t))
                shuff_acc = auc(fpr, tpr)
            elif p_metric == 'accuracy':
                shuff_acc = accuracy_score(Yt_test, this_model.predict(X_t))
            if p_metric in ['L1 Error']:
                scores.append([feature_name, index, (shuff_error-error)/shuff_error])
            else:
                scores.append([feature_name, index, (acc-shuff_acc)/acc])
            if p_verbose:
                if i == 0:
                    print('{:*^65}'.format('Looping Through Remaining Variables'))
                if p_metric in ['L1 Error']:
                    print('Testing {}. {}: {:.5f}'.format(i+1,feature_name,(shuff_error-error)/shuff_error))
                else:
                    print('Testing {}. {}: {:.5f}'.format(i+1,feature_name,(acc-shuff_acc)/acc))
    if p_metric == 'L1 Error':
        print('{:*^65}'.format("Features sorted by their L1 error based score:"))
    elif p_metric == 'R^2':
        print('{:*^65}'.format("Features sorted by their R^2 based score:"))
    elif p_metric == 'AUC':
        print('{:*^65}'.format("Features sorted by their AUC based score:"))
    elif p_metric == 'accuracy':
        print('{:*^65}'.format("Features sorted by their accuracy based score:"))
        
    df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
    df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
    dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
    dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
    df_sorted = dfAgg.sort_values(by='Score',ascending=False).reset_index(drop=True)
    
    for i in range(len(df_sorted)):
        print('{}: {:.5f}'.format(df_sorted['Variable'][i],df_sorted['Score'][i]))
 

def UnivariateGradientBoostingPermutations(p_features,
                                           p_X_train,
                                           p_Y_train,
                                           p_regression = True,
                                           p_metric = 'L1 Error', 
                                           p_cross_validations = 5,
                                           p_validation_fraction = 0.25,
                                           p_n_estimators = 10000,
                                           p_seed = 0,
                                           p_verbose = False):
    # TODO Add in p_regression, p_metric pieces
    from sklearn.model_selection import ShuffleSplit
    #from collections import defaultdict
    
    import sys
    
    #Scoring parameter
    if p_metric == 'L1 Error':
        from sklearn.metrics import mean_absolute_error
    elif p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
    if p_regression:
        from sklearn.ensemble import AdaBoostRegressor
        this_model = AdaBoostRegressor(n_estimators = p_n_estimators,random_state=p_seed)
    else:
        from sklearn.ensemble import AdaBoostClassifier
        this_model = AdaBoostClassifier(n_estimators = p_n_estimators,random_state=p_seed)
        
    # Since we're going to do cross-validation here, make X_train a larger portion of overall dataset (maybe 80 or 90%)

    scores = []
    rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)
    
    # crossvalidate the scores on a number of different random splits of the data
    for j, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
        if p_verbose:
            print('Iteration number {} for the train test split'.format(j+1))
        Xt_train, Xt_test = p_X_train.iloc[train_index,:], p_X_train.iloc[test_index,:] # set X for train and test
        Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
        this_model.fit(Xt_train, Yt_train) # fit the model on the training set
        if p_metric == 'L1 Error':
            error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
        elif p_metric == 'R^2':
            acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
        elif p_metric == 'AUC':
            fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
            acc = auc(fpr, tpr)
        elif p_metric == 'accuracy':
            acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
        for i, (feature_name, index) in enumerate(p_features): # shuffle the ith predictor variable (to break any correlatin with the target)
            X_t = Xt_test.copy() # Copy the test data, so as no to disturb
            np.random.seed(p_seed) # Set seed for reproducibility
            X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
            if p_metric == 'L1 Error':
                shuff_error = mean_absolute_error(Yt_test, this_model.predict(X_t))
            elif p_metric == 'R^2':
                shuff_acc = r2_score(Yt_test, this_model.predict(X_t))
            elif p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(X_t))
                shuff_acc = auc(fpr, tpr)
            elif p_metric == 'accuracy':
                shuff_acc = accuracy_score(Yt_test, this_model.predict(X_t))
            if p_metric in ['L1 Error']:
                scores.append([feature_name, index, (shuff_error-error)/shuff_error])
            else:
                scores.append([feature_name, index, (acc-shuff_acc)/acc])
            if p_verbose:
                if i == 0:
                    print('{:*^65}'.format('Looping Through Remaining Variables'))
                if p_metric in ['L1 Error']:
                    print('Testing {}. {}: {:.5f}'.format(i+1,feature_name,(shuff_error-error)/shuff_error))
                else:
                    print('Testing {}. {}: {:.5f}'.format(i+1,feature_name,(acc-shuff_acc)/acc))
    if p_metric == 'L1 Error':
        print('{:*^65}'.format("Features sorted by their L1 error based score:"))
    elif p_metric == 'R^2':
        print('{:*^65}'.format("Features sorted by their R^2 based score:"))
    elif p_metric == 'AUC':
        print('{:*^65}'.format("Features sorted by their AUC based score:"))
    elif p_metric == 'accuracy':
        print('{:*^65}'.format("Features sorted by their accuracy based score:"))
        
    df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
    df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # Change from list to tuple in order to get hashable type
    dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() #Average the scores
    dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
    df_sorted = dfAgg.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    for i in range(len(df_sorted)):
        print('{}: {:.5f}'.format(df_sorted['Variable'][i],df_sorted['Score'][i]))
        

def UnivariateBackwardRandomForest(p_features,
                                   p_predictors,
                                   p_X_train,
                                   p_Y_train,
                                   p_X_val,
                                   p_Y_val,
                                   p_top_k_features = 5,
                                   p_regression = True, # Specify between: regression, classification
                                   p_metric = 'AUC',
                                   p_n_estimators = 100,
                                   p_holdout = True,
                                   p_seed = 0,
                                   p_verbose = False,
                                   p_runtime_estimate = False):
    if p_runtime_estimate:
        import timeit
    import sys
    if p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an available metric, the current options are: R^2, AUC, and accuracy'.format(p_metric))
        sys.exit()
        
    if p_regression:
        from sklearn.ensemble import RandomForestRegressor
    else:
        from sklearn.ensemble import RandomForestClassifier
    # First look at teh fit using all the features
    if p_regression:
        model_full = RandomForestRegressor(n_estimators=p_n_estimators, random_state=p_seed)
    else:
        model_full = RandomForestClassifier(n_estimators=p_n_estimators, random_state=p_seed)
        
    model_full.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
    # TODO Add an argument here p_holdout = True where if it's False, we look at the metric on the training set.
    
    if p_metric == 'R^2':
        result_full = r2_score(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors])) # calculate the performance on the test set
    elif p_metric == 'AUC':
        fpr, tpr, thresholds = roc_curve(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors]))
        result_full = auc(fpr,tpr)
    elif p_metric == 'accuracy':
        result_full = accuracy_score(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors]))
        
    performance = []
    # Accuracy of the model using all features
    
    for i, (feature_name, index) in enumerate(p_features):
        if p_verbose:
            print(feature_name)
            
        this_set = set(index)
        test_index = [x for x in p_predictors if x not in this_set]
        
        if p_regression:
            this_model = RandomForestRegressor(n_estimators=p_n_estimators, random_state=p_seed)
        else:
            this_model = RandomForestClassifier(n_estimators=p_n_estimators, random_state=p_seed)
            
        if p_runtime_estimate:
            now = timeit.default_timer()
        this_model.fit(p_X_train.iloc[:,test_index],p_Y_train)
        if p_runtime_estimate:
            then = timeit.default_timer()
            est_runtime = (then - now) / 60 * (len(p_features) - i)
            print('About {:.2f} minutes remaining'.format(est_runtime))
            
        if p_metric == 'R^2':
            this_result = r2_score(p_Y_val, this_model.predict(p_X_val.iloc[:,test_index])) # calculate the performance on the test set
        elif p_metric == 'AUC':
            fpr, tpr, thresholds = roc_curve(p_Y_val, this_model.predict(p_X_val.iloc[:,test_index]))
            this_result = auc(fpr,tpr)
        elif p_metric == 'accuracy':
            this_result = accuracy_score(p_Y_val, this_model.predict(p_X_val.iloc[:,test_index]))
            
        this_marginal_result = result_full - this_result # higher values imply the variable that was removed provided significant model importance
                    
        performance.append([feature_name, this_marginal_result])
        
    df = pd.DataFrame(columns = ['Variable','Marginal Validation AUC'], data = performance)
    df_sorted = df.sort_values(by='Marginal Validation AUC', ascending=False)
    
    top_k_features = df_sorted.iloc[:p_top_k_features,:]
    print(top_k_features)

    
def ModelComparisonRegression(p_X_train,
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
        # TODO use **kwargs
            for param in hyper_parameters[0]:
                if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                    model_with_hyperparameters = getattr(module, model_name) # grab model
                    model_with_hyperparameters = model_with_hyperparameters(alpha=param, random_state=p_seed) # instantiate model
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train) # fit the model
                    result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate model
                    results.append(result) # store model results
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['KNeighborsRegressor']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_neighbors=param)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['DecisionTreeRegressor']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(max_depth=param, random_state=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_estimators=param, random_state=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['XGBRegressor']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_estimators=param, seed=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
        else:
            # Check accuracy of the model using all features
            model = getattr(module, model_name) # grab the model
            model = model() # instantiate the model
            model.fit(p_X_train.iloc[:,p_predictors],p_Y_train) # fit the model
            result = metric(p_Y_val, model.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
            results.append(result) # store the model resutls
            if p_verbose:
                print('{} - {}'.format(model_label,result))
            models.append([model_label, result])
        
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
        

def ModelComparisonClassification(p_X_train,
                                  p_X_val,
                                  p_Y_train,
                                  p_Y_val,
                                  p_predictors,
                                  p_metric = 'AUC', # Options are AUC, R2, accuracy
                                  p_models = ['All'],
                                  p_seed = 0,
                                  p_verbose = False):
    # TODO Add in other metrics for comparison
    """ Add another argument to allow transformations of the target variable (default to identity) e.g. see transform in SVM, Bagging """
    """ Add another action if p_verbose = True where run-time of each model is calculated (this will help with optimizing hyper-parameter tuning) """
    """ Break out each model type into its own function and have this function call all of those """
    # List of models tested
    models = []
    
    # List to store results from each model 
    results = []
    
    #Scoring parameter
    import sys

    if p_metric == 'R2':
        from sklearn.metrics import r2_score
        metric = r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
        metric = accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
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

    import importlib
    for model_label, module_name, model_name, hyper_parameters in models_to_test:
        # Import the model
        module = importlib.import_module(module_name)
        
        if hyper_parameters: # If there are hyperparameters to test do it here
        # TODO use **kwargs
            for param in hyper_parameters[0]:
                if model_name in ['RidgeClassifier', 'Lasso', 'SGDClassifier', 'MLPClassifier']:
                    model_with_hyperparameters = getattr(module, model_name) # grab model
                    model_with_hyperparameters = model_with_hyperparameters(alpha=param, random_state=p_seed) # instantiate model
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train) # fit the model
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result) # store model results
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['LinearSVC']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(C=param)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['KNeighborsClassifier']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_neighbors=param)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['DecisionTreeClassifier']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(max_depth=param, random_state=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_estimators=param, random_state=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['XGBClassifier']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(n_estimators=param, seed=p_seed)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
                elif model_name in ['BernoulliNB']:
                    model_with_hyperparameters = getattr(module, model_name) # Grab model
                    model_with_hyperparameters = model_with_hyperparameters(alpha=param)
                    
                    model_with_hyperparameters.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
                    if p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors]))
                        result = auc(fpr, tpr)
                    else:
                        result = metric(p_Y_val, model_with_hyperparameters.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
                    results.append(result)
                    if p_verbose:
                        print('{} {} - {}'.format(model_label, param, result))
                    models.append(['{} {}'.format(model_label, param), result])
        else:
            # Check accuracy of the model using all features
            model = getattr(module, model_name) # grab the model
            model = model() # instantiate the model
            model.fit(p_X_train.iloc[:,p_predictors],p_Y_train) # fit the model
            if p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(p_Y_val, model.predict(p_X_val.iloc[:,p_predictors]))
                result = auc(fpr, tpr)
            else:
                result = metric(p_Y_val, model.predict(p_X_val.iloc[:,p_predictors])) # evaluate the model
            results.append(result) # store the model resutls
            if p_verbose:
                print('{} - {}'.format(model_label,result))
            models.append([model_label, result])
        
    # Plot the results of all models
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(results)
    # Label the axis with model names
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label[0] for label in models],rotation='vertical')
    # Plot the accuracy for all combinations
    plt.show()

    df = pd.DataFrame(columns = ['Model','Validation Fit'], data=models)
    df_sorted = df.sort_values(by='Validation Fit', ascending=False)
    print(df_sorted)


    

def ForBackRandomForestPremutations(p_features,
                                    p_predictors,
                                    p_X_train,
                                    p_Y_train,
                                    p_X_val,
                                    p_Y_val,
                                    p_regression = True,
                                    p_metric = 'L1 Error', # For regression: have Lp error
                                                      # For classification AUC, R^2, accuracy, future: LR Test
                                    p_cross_validations = 5,
                                    p_validation_fraction = 0.25,
                                    p_threshold = 0, # This will depend on the metric used
                                    p_n_estimators = 250,
                                    p_seed = 0,
                                    p_forward = True,
                                    p_verbose = False):
    # TODO Add in p_regression and p_metric peices
    from sklearn.model_selection import ShuffleSplit
    
    import sys
    #Scoring parameter
    if p_metric == 'L1 Error':
        from sklearn.metrics import mean_absolute_error
    elif p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
    if p_regression:
        from sklearn.ensemble import RandomForestRegressor
        this_model = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        this_model = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
        
    # Since we're doing cross-validation here, make X_train a larger portion of th overall dataset (80-90%)
    
    best_pred_index = p_predictors
    best_vars = p_features
    
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
            this_model.fit(Xt_train, Yt_train) # fit the model on the training set
            if p_metric == 'L1 Error':
                error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
            elif p_metric == 'R^2':
                acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
            elif p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
                acc = auc(fpr, tpr)
            elif p_metric == 'accuracy':
                acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
            for j, (feature_name, index) in enumerate(best_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                np.random.seed(p_seed) # set seed for reproducibility
                X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
                X_t = X_t.iloc[:,best_pred_index] # Filter down to the remaining variables
                if p_metric == 'L1 Error':
                    shuff_error = mean_absolute_error(Yt_test, this_model.predict(X_t))
                elif p_metric == 'R^2':
                    shuff_acc = r2_score(Yt_test, this_model.predict(X_t))
                elif p_metric == 'AUC':
                    fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(X_t))
                    shuff_acc = auc(fpr, tpr)
                elif p_metric == 'accuracy':
                    shuff_acc = accuracy_score(Yt_test, this_model.predict(X_t))
                if p_metric in ['L1 Error']:
                    scores.append([feature_name, index, (shuff_error-error)/shuff_error])
                else:
                    scores.append([feature_name, index, (acc-shuff_acc)/acc])
                if p_verbose:
                    if j == 0:
                        print('{:*^65}'.format('Looping Through Remaining Variables'))
                        print('There are {} variables remaining to test'.format(len(best_vars)))
                    if p_metric in ['L1 Error']:
                        print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(shuff_error-error)/shuff_error))
                    else:
                        print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(acc-shuff_acc)/acc))

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
        if p_verbose:
            print('{:*^65}'.format('Starting Forward Iteration'))
        keep_going = True
        leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
        leftover_index = list(set(p_predictors) - set(best_pred_index))
        while keep_going and len(leftover_vars) > 0:
            keep_going = False 
        
            if p_regression:
                from sklearn.ensemble import RandomForestRegressor
                this_model = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
                this_model_full = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
            else:
                from sklearn.ensemble import RandomForestClassifier
                this_model = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
                this_model_full = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
            
            scores = [] # instead make this a list storing: feature_name, index, score. Then aggregate by taking the mean
            rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)
            
            # crossvalidate the scores on a number of different random splits of the data
            for i, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
                if p_verbose:
                    print('Iteration number {} out of {} for the train test split'.format(i+1,p_cross_validations))
                Xt_train, Xt_test = p_X_train.iloc[train_index,best_pred_index], p_X_train.iloc[test_index,best_pred_index] # set X for train and test
                Xt_train_full = p_X_train.iloc[train_index,:]
                Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
                this_model.fit(Xt_train, Yt_train) # fit the model on the training set
                this_model_full.fit(Xt_train_full, Yt_train) # really this only needs to be fit on the first iteration of the while loop, and stored
                if p_metric == 'L1 Error':
                    error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
                elif p_metric == 'R^2':
                    acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
                elif p_metric == 'AUC':
                    fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
                    acc = auc(fpr, tpr)
                elif p_metric == 'accuracy':
                    acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
                for j, (feature_name, index) in enumerate(leftover_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                    X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                    shuffle_index = list(set(leftover_index)-set(index))
                    np.random.seed(p_seed) # set seed for reproducibility
                    X_t.iloc[:,shuffle_index] = np.random.permutation(X_t.iloc[:,shuffle_index]) # Permute the observations from the ith variable
                    
                    if p_metric == 'L1 Error':
                        shuff_error = mean_absolute_error(Yt_test, this_model_full.predict(X_t))
                    elif p_metric == 'R^2':
                        shuff_acc = r2_score(Yt_test, this_model_full.predict(X_t))
                    elif p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(Yt_test, this_model_full.predict(X_t))
                        shuff_acc = auc(fpr, tpr)
                    elif p_metric == 'accuracy':
                        shuff_acc = accuracy_score(Yt_test, this_model_full.predict(X_t))
                    if p_metric in ['L1 Error']:
                        scores.append([feature_name, index, (error-shuff_error)/shuff_error])
                    else:
                        scores.append([feature_name, index, (shuff_acc-acc)/shuff_acc])
                    if p_verbose:
                        if j == 0:
                            print('{:*^65}'.format('Looping Through Remaining Variables'))
                            print('There are {} variables remaining to test'.format(len(leftover_vars)))
                        if p_metric in ['L1 Error']:
                            print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(error-shuff_error)/shuff_error))
                        else:
                            print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(shuff_acc-acc)/acc))
    
            df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
            df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
            dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
            dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
            df_sorted = dfAgg.sort_values(by='Score',ascending=False).reset_index(drop=True)
            
            if df_sorted['Score'][0] > p_threshold: # Add the best variable and update relevant fields
                if p_verbose:
                    print('Adding {} from our list of variables, average score: {:.5f}'.format(df_sorted['Variable'][0],df_sorted['Score'][0]))
                    
                keep_going = True # You're not done yet
                
                best_vars = best_vars.append([df_sorted['Variable'][0], df_sorted['Var_Index'][0]]) # include the best variable
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
    
    this_model.fit(p_X_train.iloc[:,best_pred_index], p_Y_train)
    if p_metric == 'L1 Error':
        score = mean_absolute_error(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an L1 error of {:.5f} on the validation set'.format(score))
    elif p_metric == 'R^2':
        score = r2_score(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an R^2 of {:.5f} on the validation set'.format(score))
    elif p_metric == 'AUC':
        fpr, tpr, thresholds = roc_curve(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        score = auc(fpr, tpr)
        print('The optimal model has an AUC of {:.5f} on the validation set'.format(score))
    elif p_metric == 'accuracy':
        score = accuracy_score(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an accuracy of {:.5f} on the validation set'.format(score))
   

def ForBackGradientBoostingPremutations(p_features,
                                        p_predictors,
                                        p_X_train,
                                        p_Y_train,
                                        p_X_val,
                                        p_Y_val,
                                        p_regression = True,
                                        p_metric = 'L1 Error', # For regression: have Lp error
                                                          # For classification AUC, R^2, accuracy, future: LR Test
                                        p_cross_validations = 5,
                                        p_validation_fraction = 0.25,
                                        p_threshold = 0, # This will depend on the metric used
                                        p_n_estimators = 250,
                                        p_seed = 0,
                                        p_forward = True,
                                        p_verbose = False):
    # TODO Add in p_regression and p_metric peices
    from sklearn.model_selection import ShuffleSplit
    
    import sys
    #Scoring parameter
    if p_metric == 'L1 Error':
        from sklearn.metrics import mean_absolute_error
    elif p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
    if p_regression:
        from sklearn.ensemble import AdaBoostRegressor
        this_model = AdaBoostRegressor(n_estimators=p_n_estimators,random_state=p_seed)
    else:
        from sklearn.ensemble import AdaBoostClassifier
        this_model = AdaBoostClassifier(n_estimators=p_n_estimators,random_state=p_seed)
        
    # Since we're doing cross-validation here, make X_train a larger portion of th overall dataset (80-90%)
    
    best_pred_index = p_predictors
    best_vars = p_features
    
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
            this_model.fit(Xt_train, Yt_train) # fit the model on the training set
            if p_metric == 'L1 Error':
                error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
            elif p_metric == 'R^2':
                acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
            elif p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
                acc = auc(fpr, tpr)
            elif p_metric == 'accuracy':
                acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
            for j, (feature_name, index) in enumerate(best_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                np.random.seed(p_seed) # set seed for reproducibility
                X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:,index]) # Permute the observations from the ith variable
                X_t = X_t.iloc[:,best_pred_index] # Filter down to the remaining variables
                if p_metric == 'L1 Error':
                    shuff_error = mean_absolute_error(Yt_test, this_model.predict(X_t))
                elif p_metric == 'R^2':
                    shuff_acc = r2_score(Yt_test, this_model.predict(X_t))
                elif p_metric == 'AUC':
                    fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(X_t))
                    shuff_acc = auc(fpr, tpr)
                elif p_metric == 'accuracy':
                    shuff_acc = accuracy_score(Yt_test, this_model.predict(X_t))
                if p_metric in ['L1 Error']:
                    scores.append([feature_name, index, (shuff_error-error)/shuff_error])
                else:
                    scores.append([feature_name, index, (acc-shuff_acc)/acc])
                if p_verbose:
                    if j == 0:
                        print('{:*^65}'.format('Looping Through Remaining Variables'))
                        print('There are {} variables remaining to test'.format(len(best_vars)))
                    if p_metric in ['L1 Error']:
                        print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(shuff_error-error)/shuff_error))
                    else:
                        print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(acc-shuff_acc)/acc))

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
        if p_verbose:
            print('{:*^65}'.format('Starting Forward Iteration'))
        keep_going = True
        leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
        leftover_index = list(set(p_predictors) - set(best_pred_index))
        while keep_going and len(leftover_vars) > 0:
            keep_going = False 
        
            if p_regression:
                from sklearn.ensemble import AdaBoostRegressor
                this_model = AdaBoostRegressor(n_estimators=p_n_estimators,random_state=p_seed)
                this_model_full = AdaBoostRegressor(n_estimators=p_n_estimators,random_state=p_seed)
            else:
                from sklearn.ensemble import AdaBoostClassifier
                this_model = AdaBoostClassifier(n_estimators=p_n_estimators,random_state=p_seed)
                this_model_full = AdaBoostClassifier(n_estimators=p_n_estimators,random_state=p_seed)
            
            scores = [] # instead make this a list storing: feature_name, index, score. Then aggregate by taking the mean
            rs = ShuffleSplit(n_splits=p_cross_validations, test_size = p_validation_fraction, random_state = p_seed)
            
            # crossvalidate the scores on a number of different random splits of the data
            for i, (train_index, test_index) in enumerate(rs.split(p_X_train)): # loop through the n_splits train/test splits
                if p_verbose:
                    print('Iteration number {} out of {} for the train test split'.format(i+1,p_cross_validations))
                Xt_train, Xt_test = p_X_train.iloc[train_index,best_pred_index], p_X_train.iloc[test_index,best_pred_index] # set X for train and test
                Xt_train_full = p_X_train.iloc[train_index,:]
                Yt_train, Yt_test = p_Y_train.iloc[train_index], p_Y_train.iloc[test_index] # set Y for train and test
                this_model.fit(Xt_train, Yt_train) # fit the model on the training set
                this_model_full.fit(Xt_train_full, Yt_train) # really this only needs to be fit on the first iteration of the while loop, and stored
                if p_metric == 'L1 Error':
                    error = mean_absolute_error(Yt_test, this_model.predict(Xt_test))
                elif p_metric == 'R^2':
                    acc = r2_score(Yt_test, this_model.predict(Xt_test)) # calculate the performance on the test set
                elif p_metric == 'AUC':
                    fpr, tpr, thresholds = roc_curve(Yt_test, this_model.predict(Xt_test))
                    acc = auc(fpr, tpr)
                elif p_metric == 'accuracy':
                    acc = accuracy_score(Yt_test, this_model.predict(Xt_test))
                for j, (feature_name, index) in enumerate(leftover_vars): # shuffle the ith predictor variable (to break any relationship with the target)
                    X_t = p_X_train.iloc[test_index,:].copy() # copy the test data, so as not to disturb
                    shuffle_index = list(set(leftover_index)-set(index))
                    np.random.seed(p_seed) # set seed for reproducibility
                    X_t.iloc[:,shuffle_index] = np.random.permutation(X_t.iloc[:,shuffle_index]) # Permute the observations from the ith variable
                    
                    if p_metric == 'L1 Error':
                        shuff_error = mean_absolute_error(Yt_test, this_model_full.predict(X_t))
                    elif p_metric == 'R^2':
                        shuff_acc = r2_score(Yt_test, this_model_full.predict(X_t))
                    elif p_metric == 'AUC':
                        fpr, tpr, thresholds = roc_curve(Yt_test, this_model_full.predict(X_t))
                        shuff_acc = auc(fpr, tpr)
                    elif p_metric == 'accuracy':
                        shuff_acc = accuracy_score(Yt_test, this_model_full.predict(X_t))
                    if p_metric in ['L1 Error']:
                        scores.append([feature_name, index, (error-shuff_error)/shuff_error])
                    else:
                        scores.append([feature_name, index, (shuff_acc-acc)/shuff_acc])
                    if p_verbose:
                        if j == 0:
                            print('{:*^65}'.format('Looping Through Remaining Variables'))
                            print('There are {} variables remaining to test'.format(len(leftover_vars)))
                        if p_metric in ['L1 Error']:
                            print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(error-shuff_error)/shuff_error))
                        else:
                            print('Testing {}. {}: {:.5f}'.format(j+1,feature_name,(shuff_acc-acc)/acc))
    
            df = pd.DataFrame(columns = ['Variable','Var_Index','Score'], data = scores)
            df['Var_Index'] = df['Var_Index'].apply(lambda x: tuple(x)) # change list to tuple in order to get hashable type
            dfAgg = df.groupby(['Variable','Var_Index'], as_index=False)['Score'].mean() # Average the scores
            dfAgg['Var_Index'] = dfAgg['Var_Index'].apply(lambda x: list(x)) # Probably don't need to convert back to list...
            df_sorted = dfAgg.sort_values(by='Score',ascending=False).reset_index(drop=True)
            
            if df_sorted['Score'][0] > p_threshold: # Add the best variable and update relevant fields
                if p_verbose:
                    print('Adding {} from our list of variables, average score: {:.5f}'.format(df_sorted['Variable'][0],df_sorted['Score'][0]))
                    
                keep_going = True # You're not done yet
                
                best_vars = best_vars.append([df_sorted['Variable'][0], df_sorted['Var_Index'][0]]) # include the best variable
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
    
    this_model.fit(p_X_train.iloc[:,best_pred_index], p_Y_train)
    if p_metric == 'L1 Error':
        score = mean_absolute_error(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an L1 error of {:.5f} on the validation set'.format(score))
    elif p_metric == 'R^2':
        score = r2_score(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an R^2 of {:.5f} on the validation set'.format(score))
    elif p_metric == 'AUC':
        fpr, tpr, thresholds = roc_curve(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        score = auc(fpr, tpr)
        print('The optimal model has an AUC of {:.5f} on the validation set'.format(score))
    elif p_metric == 'accuracy':
        score = accuracy_score(p_Y_val, this_model.predict(p_X_val.iloc[:,best_pred_index]))
        print('The optimal model has an accuracy of {:.5f} on the validation set'.format(score))

        
def ForBackRandomForest(p_features,
                        p_predictors,
                        p_X_train,
                        p_Y_train,
                        p_X_val,
                        p_Y_val,
                        p_regression = True,
                        p_metric = 'AUC', # For regression: have Lp error
                                          # For classification AUC, R^2, accuracy, future: LR Test
                        p_n_estimators = 250,
                        p_seed = 0,
                        p_forward = True,
                        p_verbose = False):
    # TODO Add in p_metric options
    import sys
    if p_metric == 'R^2':
        from sklearn.metrics import r2_score
    elif p_metric == 'AUC':
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
    elif p_metric == 'accuracy':
        from sklearn.metrics import accuracy_score
    else:
        print('{} is not an avaiable metric'.format(p_metric))
        sys.exit()
    
    if p_regression:
        from sklearn.ensemble import RandomForestRegressor
        model_full = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model_full = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
    model_full.fit(p_X_train.iloc[:,p_predictors],p_Y_train)
    # TODO Add an argument here p_holdout = True where if it's False, we look at the metric on the training set.
    if p_metric == 'R^2':
        result_full = r2_score(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors])) # calculate the performance on the test set
    elif p_metric == 'AUC':
        fpr, tpr, thresholds = roc_curve(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors]))
        result_full = auc(fpr, tpr)
    elif p_metric == 'accuracy':
        result_full = accuracy_score(p_Y_val, model_full.predict(p_X_val.iloc[:,p_predictors]))
    
    best_index = p_predictors
    best_vars = p_features
    best_result = result_full
    
    # Run a backward regression on all variables
    keep_going = True # To get things started
    while keep_going and len(best_vars) > 0: # Hopefully you don't get rid of all your variables!
        keep_going = False # If none of the remaining variables improve performance, then stop
        for i, (feature_name, index) in enumerate(best_vars): # loop through the n_splits train/test splits
            if p_verbose:
                if i == 0:
                    print('{:*^65}'.format('Looping Through Remaining Variables'))
                    print('Best score so far is {:.5f}'.format(best_result))
                    print('There are {} variables remaining in the model'.format(len(best_vars)))
                print('Testing {}. {}'.format(i+1,feature_name))
            test_index = [x for x in best_index if x not in index]
            
            if p_regression:
                test_model = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
            else:
                test_model = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
                
            test_model.fit(p_X_train.iloc[:,test_index],p_Y_train)
            
            if p_metric == 'R^2':
                test_result = r2_score(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index])) # calculate the performance on the test set
            elif p_metric == 'AUC':
                fpr, tpr, thresholds = roc_curve(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index]))
                test_result = auc(fpr, tpr)
            elif p_metric == 'accuracy':
                test_result = accuracy_score(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index]))
                
            if p_verbose:
                print('Score without {} is {:.5f}'.format(feature_name, test_result))
                
            if test_result > best_result: # if current test is best, then replace best with test
                test_vars = [[name, index] for j, (name, index) in enumerate(best_vars) if j != i] # exclude the ith variable
                best_vars_this_round = test_vars
                
                best_index_this_round = test_index
                best_result = test_result
                
                keep_going = True # we found something better, keep going
                if p_verbose:
                    print('Your model is better off without {}'.format(feature_name))
                    
        if keep_going: # After going through all remaining variables remove the most detrimental variable
            best_index = best_index_this_round
            best_vars = best_vars_this_round
        
    if p_forward: # Idea for forward iteration is to iteratively test the addition of a single variable x and add in those that improve the model most
        if p_verbose:
            print('{:*^65}'.format('Starting Forward Iteration'))
        keep_going = True
        leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
        while keep_going and len(leftover_vars) > 0:
            keep_going = False 
            for i, (feature_name, index) in enumerate(leftover_vars): # loop through the n_splits train/test splits
                if p_verbose:
                    if i == 0:
                        print('{:*^65}'.format('Looping Through Remaining Variables'))
                        print('Best Score so far is {:.5f}'.format(best_result))
                        print('There are {} variables remaining to test'.format(len(leftover_vars)))
                    print('Testing {}. {}'.format(i+1,feature_name))
                test_index = best_index + index 
                
                if p_regression:
                    test_model = RandomForestRegressor(n_estimators=p_n_estimators,random_state=p_seed)
                else:
                    test_model = RandomForestClassifier(n_estimators=p_n_estimators,random_state=p_seed)
                
                test_model.fit(p_X_train.iloc[:,test_index],p_Y_train)
                    
                if p_metric == 'R^2':
                    test_result = r2_score(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index])) # calculate the performance on the test set
                elif p_metric == 'AUC':
                    fpr, tpr, thresholds = roc_curve(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index]))
                    test_result = auc(fpr, tpr)
                elif p_metric == 'accuracy':
                    test_result = accuracy_score(p_Y_val, test_model.predict(p_X_val.iloc[:,test_index]))
                    
                if p_verbose:
                    print('Score with {} add is {:.5f}'.format(feature_name, test_result))
        
                if test_result > best_result: # if current test is best, the replace best with test
                    best_index_this_round = test_index
                    best_result = test_result
                    best_iteration = i
                    
                    leftover_vars = [[name, index] for name, index in p_features if [name, index] not in best_vars]
                    leftover_index = list(set(p_predictors) - set(best_pred_index))
                    
                    keep_going = True # You're not done yet
                    if p_verbose:
                        print('Your model is better with {}'.format(feature_name))
                        
            if keep_going: # After going through all remaining variables add the most beneficial variable
                leftover_vars.pop(best_iteration) # remove the best remaining variable from leftover_vars
                best_index = best_index_this_round
                best_vars = best_vars.append(best_vars_this_round) # add this variable to the remaining list
                        
    print('{:*^65}'.format('The optimal set of variables is:'))
    for name, index in best_vars:
        print(name)
    
    test_model.fit(p_X_train.iloc[:,best_index], p_Y_train)
    
    if p_metric == 'R^2':
        score = r2_score(p_Y_val, test_model.predict(p_X_val.iloc[:,best_index]))
        print('The optimal model has an R^2 of {:.5f} on the validation set'.format(score))
    elif p_metric == 'AUC':
        fpr, tpr, thresholds = roc_curve(p_Y_val, test_model.predict(p_X_val.iloc[:,best_index]))
        score = auc(fpr, tpr)
        print('The optimal model has an AUC of {:.5f} on the validation set'.format(score))
    elif p_metric == 'accuracy':
        score = accuracy_score(p_Y_val, test_model.predict(p_X_val.iloc[:,best_index]))
        print('The optimal model has an accuracy of {:.5f} on the validation set'.format(score))     
        
        
####################################################################################################################################

# TODO:
# Build in variable importance function that uses:
    # built in functions with sci-kit learn
    # Shapley Value based importance (run-time would be 2^n (number of models to fit) where n is the number of predictors/features in the model)
        # Perhaps we could use correlation to make a network so that instead of testing all coalitions, we only test those with high correlation
        # The assumption would be that the contribution of independent variables woud be roughly additive. (this seems fair)
        # We would still look at all possible subsets, but for uncorrelated variables, we could just add up their contributions
        # If Shaply Value importance is fit on training and evaluated on holdout, then after we calculate Shapley we could just remove all variables with a negative shapley value
        # This would be an alternative to forward/backward regression for variable selection
    
    # Figure out a way to evaluate variable importance when using dummy variables

# In Model selection piece, use Bayesian Optimization to do hyper-parameter tuning
# Look at adding a double lift chart, this will also indicate model diversity for possible ensembling
    # Or for classification look at confusion matrix and see if two models are doing well on different segments

# To get a baseline for performance, run a PCA on all variables then run a backward regression on the principal components
    
# Create residuals graphs for regression problems 
# Add in stochastic process to get confidence intervals 

# TODO Add multivariate GLM part to visualize predicted values, taking into account correlations 

# We should also look at using/testing polynomial (rather than just linear) regression for our continuous variables

# Think about simplifications of variables.. Is there a way to automate this part. At very least, do forward/backward regression with
# a linear model to see if strange/undesireable things are happening.    


# TODO hyper-parameter selectin using bayesian optimization
# TODO make this object oriented 


# TODO add multivariate linear analysis so that we can view how correlations affect the linear model this wil give 
     # some insight into the ensemble/non-linear models. Could also add in interaction detection to non-linear models
     # using Friedman's H statistic, then add those interactions into the linear model to see how close we get in performance
     # since linear model will give intuitive interpretation
        