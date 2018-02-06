
import DataAnalysisToolBox.wgt_stats as ws
import pandas as pd 
from pandas.core.dtypes.common import is_numeric_dtype
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
        

class ModelData(object):
    # TODO Complete documentation section below
    """ Base object that will be analyzed. Establish target, predictors, weight, controls, numeric categorical variables.
    
    Parameters
    ----------
    dataset: dataframe
        The model file that contains data on which model will be built
    predictors: array_like
        Index of the predictor variables
    target: int
        Index of the target variable
    weight: int, optional
        Index of the weight variable
    controls: array_like, optional
        Index of control variables
    numeric_cat_index: array_like, optional
        Index of variables that are numeric categorical variables
        
    Attributes
    ----------
    predictors: array
        Index of the predictor variables
    target: int
        Index of the target variable
    weight: int
        Index of the weight variable
    controls: array, optional
        Index of the control variables
    numeric_cat_index: array, optional
        Index of variables that are numeric categorical variables
        
    Methods
    -------
    rare_level_check(threshold):
        Check if there are categorical variables with level containing thin data. Also check for categorical variables with a single value.
    missing_value_cleanup()

    """ 

    def __init__(self, dataset, predictors, target, weight=None, controls=np.array([]), numeric_cat_index=np.array([])):
        self.dataset = dataset
        self._unencoded_columns = self.dataset.columns
        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError("`dataset` input should be a pandas dataframe.")
        self.n_rows, self.n_cols = self.dataset.shape
        self.predictors = np.array(predictors)
        self.target = target
        self.weight = weight
        if self.weight is None:
            self.weight_vector = np.ones(self.n_rows)
        else:
            self.weight_vector = np.array(self.dataset.iloc[:,self.weight])
        self.controls = np.array(controls)
        self.numeric_cat_index = np.array(numeric_cat_index)
        
        # Use helper function to split variables into continuous and categorical
        self.cont_index, self.cat_index = ModelData.__cont_cat_split(self)
        self.cont_index_predictors = np.intersect1d(self.predictors, self.cont_index)
        self.cat_index_predictors = np.intersect1d(self.predictors, self.cat_index)
        if self.target in self.cont_index:
            self.continuous_target = True
        else:
            self.continuous_target = False
        if self.weight in self.cat_index:
            raise ValueError("The weight variable should be numeric")
        self.cont_index_controls = np.intersect1d(self.controls, self.cont_index)
        self.cat_index_controls = np.intersect1d(self.controls, self.cat_index)
            
        self.encoded_data = False
        self.feature_level_index = []
    
    def data_sample(self, obs_num = 10):
        # This allows you to display larger data frames
        pd.set_option('display.max_columns', None) 
        
        print(self.dataset.head(obs_num))
        
    def print_columns(self):
        for i, col in enumerate(self.dataset.columns):
            print(i, col)
    
    # TODO add function numeric_cat_var_check that will look for number of unique values in numeric variables, if that number is below some number (default to 4-5), then append it to num_cat_index list
    
    def missing_value_cleanup(self, fill_value = 0, verbose=False):
        """ Take variables with missing values fill those with p_na_imputation and create another ISNULL variable 
        
        Parameters
        ----------
        fill_value: float (eventually mean, median, mode as string), optional
            value that missing values will be filled with
        verbose: boolean, opional
            if True, will print out name of columns containing null values
            
        """
        
        def column_reindex(old_index, shifting):
            first_half_new_index = [x for x in old_index if x <= shifting]
            second_half_new_index = [x+1 for x in old_index if x >= shifting]
            return np.array(list(set(first_half_new_index) | set(second_half_new_index)))

        cols_with_nulls = self.dataset.columns[self.dataset.isnull().any()].tolist()
        
        # Use a counter to keep track of where to insert IS_NULL columns
        counter = 1
        for i, column in enumerate(self.dataset.columns):
            if column in cols_with_nulls:
                shifted_index = i+counter-1
                if self.target == shifted_index:
                    warnings.warn("The target variable contains missing values. Consider excluding these observations.")
                    # TODO Add a method that allows the user to remove missing values from the target
                if verbose:
                    print('{} contains missing values'.format(column))
                if shifted_index in self.cat_index:
                    self.dataset[column] = self.dataset[column].fillna('Null_Value').astype(str)
                else:
                    #new_column = self.dataset[column].isnull().astype(int)
                    new_column = np.where(self.dataset[column].isnull(), 'Yes', 'No')
                    if fill_value == 'mean':
                        fill = np.nanmean(self.dataset[column])
                    elif fill_value == 'median':
                        fill = np.nanmedian(self.dataset[column])
                    elif fill_value == 'auto':
                        if 0 not in self.dataset[column]:
                            fill_value = 0
                        else:
                            fill_value = min(self.dataset[column]) - 1
                    else:
                        fill = fill_value
                    self.dataset[column] = self.dataset[column].fillna(fill)
                    self.dataset.insert(i+counter, column+'_Is_Imputed', new_column)
                    
                    # Reindex variables
                    self.predictors = column_reindex(self.predictors, shifted_index)
                    
                    if self.target > shifted_index:
                        self.target += 1
                    
                    if self.weight is not None:
                        if self.weight == shifted_index:
                            self.weight_vector = np.array(self.dataset[column])
                        elif self.weight > shifted_index:
                            self.weight += 1
                    
                    self.controls = column_reindex(self.controls, shifted_index)
                    self.numeric_cat_index = column_reindex(self.numeric_cat_index, shifted_index)
                    
                    # Use helper function to split variables into continuous and categorical
                    self.cont_index, self.cat_index = ModelData.__cont_cat_split(self)
                    self.cont_index_predictors = np.intersect1d(self.predictors, self.cont_index)
                    self.cat_index_predictors = np.intersect1d(self.predictors, self.cat_index)
                    
                    self.cont_index_controls = np.intersect1d(self.controls, self.cont_index)
                    self.cat_index_controls = np.intersect1d(self.controls, self.cat_index)
                    
                    counter += 1
                    
        self.n_rows, self.n_cols = self.dataset.shape
        if verbose and counter == 1:
            print('None of the variables have missing values.')
         
    def rare_level_check(self, weighted_check = True, threshold = 0.005, verbose=False):
        # TODO Add argument that allows user to automatically group rare levels with the base level (i.e. level with most weight)
        """ Check for variables with levels that have very little value or variables with a single level 
        
        Parameters
        ----------
        weighted_check: boolean, optional
            if True (and there is a weight), weight will be used instead of a count
        threshold: float, optional
            if threshold is between 0 and 1, then a proportional check will be done, if integer check the count/weight
        verbose: boolean, optional
            if True, print status of which variable is being checked
        
        """
        
        # TODO add assert that threshold is a positive number
        
        # Set a counter to see if there's at least 1 variable with thin data in a level
        counter = 0
        for i, column in enumerate(self.dataset.columns):
            if i in self.cat_index:
                if verbose:
                    print('checking {}'.format(column))
                
                if threshold < 1.0:
                    if self.weight is None or not weighted_check:
                        data_count = self.dataset.groupby(by=[column])[column].count() / self.n_rows
                        word = 'portion'
                    else:
                        data_count = self.dataset.groupby(by=[column])[self.dataset.columns[self.weight]].sum() / self.dataset.iloc[:,self.weight].sum()
                        word = 'weighted portion'
                elif threshold >= 1.0:
                    if self.weight is None or not weighted_check:
                        data_count = self.dataset.groupby(by=[column])[column].count()
                        word = 'count'
                    else:
                        data_count = self.dataset.groupby(by=[column])[self.dataset.columns[self.weight]].sum()
                        word = 'weight'
                        
                if len(data_count) == 1:
                    print('{} contains a single level, it should be excluded or revisited'.format(column))
                    counter += 1
                else:
                    # Make sure index is a string
                    data_count.index = data_count.index.map(str)
                    for idx in data_count.index:
                        if data_count[idx] < threshold:
                            print('{} has thin data in level {}, the {} is {}.'.format(column, str(idx), word, data_count[idx]))
                            counter += 1
                            
        if counter == 0:
            print('There are no levels with thin data, using a threshold of {}'.format(threshold))
          
            
    def wgt_describe(self, index):
        """ Plot the histogram and KDE along with summary statistics """
        # Only makes sense for continuous variables
        assert index in self.cont_index, 'wgt_describe is only defined for continuous variables'
        index_names = ['Weight','Mean','StdDev','Min','25%','50%','75%','Max','Skewness','Kurtosis']
        Total_weight = np.sum(self.weight_vector)
        col_data = self.dataset.iloc[:,index]
        col_name = [self.dataset.columns[index]]
        quantiles = ws.wgt_quantile(col_data, np.array([0,.25,.5,.75,1]), self.weight_vector)
        
        data = np.array([Total_weight,
                         ws.wgt_mean(col_data,self.weight_vector),
                         ws.wgt_std_dev(col_data,self.weight_vector),
                         quantiles[0],
                         quantiles[1],
                         quantiles[2],
                         quantiles[3],
                         quantiles[4],
                         ws.wgt_skew(col_data,self.weight_vector),
                         ws.wgt_kurt(col_data,self.weight_vector)])
    
        df = pd.DataFrame(data,columns=col_name,index=index_names)
        print(df)
        
        
    def eda(self, bins = 60):
        """ Perform distributional analysis for both continuous and categorical variables """
        
        def weighted_distribution_plot(self, variable_index, continuous_var_index, bins):
            col = self.dataset.columns
            # Print distributional plots for numeric variables and histograms for categorical
            for i in variable_index:
                # Split variables into continuous and categorical and do the right thing
                if i in continuous_var_index:
                    ws.wgt_dist_plot(x=np.asarray(self.dataset.iloc[:,i]),varname=col[i],w=self.weight_vector,bins=bins)
                    ModelData.wgt_describe(self, index = i)
                else:
                    fg,ax = plt.subplots(figsize=(12, 8))
                    if self.weight is None:
                        data_count = self.dataset.groupby(by=[col[i]])[col[i]].count()
                        sns.countplot(x=col[i], data=self.dataset) 
                        plt.xticks(rotation=90)
                        plt.show()
                    else:
                        data_count = self.dataset.groupby(by=[col[i]])[col[self.weight]].sum()
                        ws.wgt_bar_plot(self.dataset, col[i], col[self.weight])
            
                    for j, idx in enumerate(data_count.index):
                        if j == 0:
                            data_count = data_count.reset_index(drop=True)
                            print('{0:25} {1}'.format('Level','Weight'))
                        print('{0:25} {1}'.format(str(idx),data_count[j]))

        
        if list(self.predictors):
            print('Distributions - Predictors:')
            weighted_distribution_plot(self, self.predictors, self.cont_index_predictors, bins)
            
        if list(self.controls):
            print('Distributions - Controls:')
            weighted_distribution_plot(self, self.controls, self.cont_index_controls, bins)
       
        
    def target_dist(self, pp_plots = True, bins = 60):
        target_label = self.dataset.columns[self.target] 
        weight_label = self.dataset.columns[self.weight]
        
        if self.continuous_target:
            ws.wgt_dist_plot(p_x=np.asarray(self.dataset.iloc[:,self.target]),varname=target_label,w=self.weight_vector,bins=bins)
            ModelData.wgt_describe(self, index = self.target)
        else:
            fg,ax = plt.subplots(figsize=(12, 8))
            if self.weight is None:
                data_count = self.dataset.groupby(by=[target_label])[target_label].count()
                sns.countplot(x=target_label, data=self.dataset)
                plt.xticks(rotation=90)
                plt.show()
            else:
                data_count = self.dataset.groupby(by=[target_label])[weight_label].sum()
                ws.wgt_bar_plot(self.dataset, target_label, weight_label)
            
            for j, idx in enumerate(data_count.index):
                if j == 0:
                    data_count = data_count.reset_index(drop=True)
                    print('{0:25} {1}'.format('Level','Weight'))
                print('{0:25} {1}'.format(str(idx),data_count[j]))
            
        
        if pp_plots and self.continuous_target:
            # Look at various distributions
            ModelData.__distribution_fit(self.dataset,self.target)
            
    
    def one_hot_encode(self, verbose = False):
        """ Have this create column/feature names for each of the dummy variables based on the level name """
        """ Also have this so that it returns index of predictors, target, weight, controls """
        if self.encoded_data:
            raise ValueError("`dataset` has already been one hot encoded.")
        else:
            self.encoded_data = True
        
        cols = self.dataset.columns 
        labels = []
        cat_predictor_index = np.array([])
        cat_control_index = np.array([])
        
        for i in np.concatenate((self.cat_index_predictors,self.cat_index_controls)).astype(int):
            labels.append(list(self.dataset[cols[i]].unique()))
            
        #One hot encode all categorical attributes
        cats = pd.DataFrame()
        counter = 0
        for i, cat_idx in enumerate(np.concatenate((self.cat_index_predictors,self.cat_index_controls)).astype(int)):
            if cat_idx in self.cat_index_predictors:
                cat_predictor_index = np.concatenate((cat_predictor_index,np.array(range(counter,counter+len(labels[i]))))).astype(int)
                counter += len(labels[i])
            else:
                cat_control_index = np.concatenate((cat_control_index,np.array(range(counter,counter+len(labels[i]))))).astype(int)
                counter += len(labels[i])
            #Label encode
            label_encoder = LabelEncoder()
            label_encoder.fit(labels[i])
            feature = label_encoder.transform(self.dataset.iloc[:,cat_idx])
            feature = feature.reshape(self.n_rows, 1)
            #One hot encode
            onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
            feature = onehot_encoder.fit_transform(feature)
            feature = pd.DataFrame(feature,columns=[(cols[cat_idx]+"-"+str(j)) for j in sorted(self.dataset[cols[cat_idx]].unique())])
            cats = pd.concat([cats,feature],axis=1)
            
        self.cat_index = np.arange(cats.shape[1])
        self.cat_index_predictors = cat_predictor_index
        self.cat_index_controls = cat_control_index
        
        # Print the shape of the encoded data
        if verbose:
            print('Dimensions of encoded categorical variables:')
            print(cats.shape)
        
        # Concatenate encoded attributes with continuous attributes, target variable, control variables, and weights if there are any.
        if self.weight:
            dataset_encoded = pd.concat([cats,self.dataset.iloc[:,self.cont_index_predictors],self.dataset.iloc[:,self.cont_index_controls],self.dataset.iloc[:,self.weight],self.dataset.iloc[:,self.target]],axis=1)
        else:
            dataset_encoded = pd.concat([cats,self.dataset.iloc[:,self.cont_index_predictors],self.dataset.iloc[:,self.cont_index_controls],self.dataset.iloc[:,self.target]],axis=1)
        
        self.cont_index_predictors = np.array(range(cats.shape[1], cats.shape[1]+len(self.cont_index_predictors)))
        if list(self.cont_index_controls):
            self.cont_index_controls = np.array(range(cats.shape[1]+len(self.cont_index_predictors),cats.shape[1]+len(self.cont_index_predictors)+len(self.cont_index_controls)))
        self.cont_index = np.concatenate((self.cont_index_predictors,self.cont_index_controls)).astype(int)
        if self.weight: 
            self.weight = np.array([cats.shape[1]+len(self.cont_index)])
        self.target = dataset_encoded.shape[1] - 1
                                           
        self.predictors = np.concatenate((self.cat_index_predictors, self.cont_index_predictors)).astype(int)
        self.controls = np.concatenate((self.cat_index_controls, self.cont_index_controls)).astype(int)
        
        if verbose:
            print('Dimensions of the updated encoded data set:')
            print(dataset_encoded.shape)
            
        self.dataset = dataset_encoded
        self.n_rows, self.n_cols = self.dataset.shape
        
        self.feature_level_index = ModelData.__feature_indexes(self, verbose = verbose)
        
    
    def train_validation_split(self, val_size = 0.2, seed = 0):
        
        X = self.dataset.iloc[:,self.predictors]
        Y = self.dataset.iloc[:,self.target]
        W = self.weight_vector
        if list(self.controls):
            C = self.dataset.iloc[:,self.controls]
            
        if self.weight and list(self.controls):
            return cross_validation.train_test_split(X, Y, W, C, test_size=val_size, random_state=seed)
        elif self.weight:
            return cross_validation.train_test_split(X, Y, W, test_size=val_size, random_state=seed)
        elif list(self.controls):
            return cross_validation.train_test_split(X, Y, C, test_size=val_size, random_state=seed)
        else:
            return cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
     
    ####################################################################################################################
    ## helper function
    ####################################################################################################################
    
    def __cont_cat_split(self):
        """ Split the predictors into continuous and categorical variables """
        """ This function could likely be vectorized in one step """
        
        cont = []
        cat = []
        
        for i, data_type in enumerate(self.dataset.dtypes):
            if is_numeric_dtype(data_type) and i not in self.numeric_cat_index:
                cont.append(i)
            else:
                cat.append(i)
        
        return cont, cat
    
    # TODO Fix up function below and call it at end of one_hot_encode function
    def __feature_indexes(self, verbose = False):
        feature_index = []
        levels = list(self.dataset.columns)
        if verbose:
            print('Creating feature indexes')
        for col in self._unencoded_columns:
            if verbose:
                print(col)
            index_list = []
            for i, level in enumerate(levels):
                if verbose:
                    print('testing ' + level)
                if ((level.find(col+'-') == 0) or (level == col)) and (i != self.target):
                    index_list.append(i)
                    if verbose:
                        print(level + ' added')
            if index_list: # remove empty lists
                feature_index.append([col, index_list])
        return feature_index  
    
    @staticmethod
    def __distribution_fit(data, target):
        """ Look for the appropriate distribution of your target variable """
        data_target = data.iloc[:,target]
        dist_list = [['lognorm', 'Lognormal'], ['invgauss', 'Inverse Gaussian'], ['gamma', 'Gamma'], ['norm', 'Normal']]
        for distribution in dist_list:
            print('{:*^65}'.format('{} Distributional Fit'.format(distribution[1])))
            dist = getattr(stats, distribution[0])
            if distribution[1] == 'Normal':
                params = dist.fit(data_target)
            else:
                params = dist.fit(data_target,floc=0)
            data_target_ord = data_target.sort_values()
            """ Switch order of axes? """
            p1 = [(i+0.5)/len(data_target_ord) for i in range(len(data_target_ord))]
            p2 = dist.cdf(data_target_ord,*params[:-2], loc=params[-2], scale=params[-1])
            plt.title('PP-Plot')
            plt.xlabel('Empirical')
            plt.ylabel('Fitted')
            plt.plot(p1,p2)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.show()
            
            print('{} Kolmogorov-Smirnov Test:'.format(distribution[1]))
            print(stats.kstest(data_target_ord,distribution[0],[*params[:-2], params[-2], params[-1]])[0])
            
            n, bins, patches = plt.hist(data_target_ord, 100, normed=True, color='g', alpha = 0.8)
            y = dist.pdf(bins, *params[:-2], loc=params[-2], scale=params[-1])
            plt.title('Empirical vs. Fitted PDF')
            plt.plot(bins, y, 'r-')
            plt.ylim([0,max(n)])
            plt.show()
        


"""
dataset = pd.read_csv("C:/Users/dsmit/ModelTools_Local/IBMHR_Classification/test_data.csv")  


predictors = np.delete(np.arange(35), np.array([1,8,21,26])) 
numeric_cat_index = np.array([6,10,13,14,16,24,25,27,30])
target = 1

            
model_data = ModelData(dataset, predictors, target, numeric_cat_index=numeric_cat_index)      
 
model_data.print_columns()
model_data.rare_level_check(verbose = True)
model_data.data_sample() 
print(list(model_data.predictors))
model_data.cat_index
model_data.cont_index

model_data.missing_value_cleanup(verbose = True)
model_data.print_columns()
print(list(model_data.predictors))
model_data.cat_index
model_data.cont_index
model_data.wgt_describe(1)
model_data.eda()
model_data.target_dist()

model_data.target
model_data.cat_index
model_data.rare_level_check(verbose = True)
model_data.encoded_data
model_data.one_hot_encode(verbose = True)
model_data.encoded_data
model_data.print_columns()
model_data.target
model_data.cat_index
model_data.cont_index
model_data.cat_index_predictors
model_data.cont_index_predictors
features = model_data.feature_level_index

data = model_data.dataset
"""
