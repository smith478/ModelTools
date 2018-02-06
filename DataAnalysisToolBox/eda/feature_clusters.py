
import DataAnalysisToolBox.wgt_stats as ws
from DataAnalysisToolBox.model_data import ModelData
import pandas as pd 
import numpy as np
from numpy import linalg as LA
#from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from random import sample, seed
import warnings


class FeatureClusters(object):
    # TODO Complete documentation section below
    """ Acts on ModelData objects. Performs pair-wise and aggregate feature correlation and clustering
    
    Parameters
    ----------

        
    Attributes
    ----------

        
    Methods
    -------


    """ 

    def __init__(self, modeldata, cont_corr_method = 'pearson', cat_corr_method = 'cramers v', cont_cluster_threshold = 0.5, cat_cluster_threshold = 0.5,
                 cont_cat_corr_method = 'anova'):
        self._modeldata = modeldata
        assert isinstance(self._modeldata, ModelData), "`modeldata` input should be a ModelData object."
        self._n_rows = modeldata.n_rows
        self._dataset = modeldata.dataset
        self._predictors = modeldata.predictors
        self._target = modeldata.target
        self._weight = modeldata.weight
        self._weight_vector = modeldata.weight_vector
        self._controls = modeldata.controls
        self._numeric_cat_index = modeldata.numeric_cat_index
        self._cont_index = modeldata.cont_index
        self._cat_index = modeldata.cat_index
        self._cont_index_predictors = modeldata.cont_index_predictors
        self._cat_index_predictors = modeldata.cat_index_predictors
        if list(self._controls):
            self._cont_index_controls = modeldata.cont_index_controls
            self._cat_index_controls = modeldata.cat_index_controls
            
        if cat_corr_method == 'cramers v':
            self._cat_corr_matrix = FeatureClusters.cramers_v_matrix(self)
        else:
            raise ValueError('{} is not a cat_corr_method option'.format(cat_corr_method))
        
        if cont_corr_method == 'pearson':
            self.cont_distance_matrix, self.cont_clusters = FeatureClusters._create_cont_clusters_pearson(self, normalize=True, threshold=cont_cluster_threshold)
        else:
            raise ValueError('{} is not a cont_corr_method option'.format(cont_corr_method))
            
        self.cat_distance_matrix, self.cat_clusters = FeatureClusters._create_cat_clusters(self, threshold=cat_cluster_threshold)    
        
        if cont_cat_corr_method == 'anova':
            self.cont_cat_distance = FeatureClusters._cont_cat_corr_features_anova(self, p_val = 0.01, subsamplesize = 100, p_seed = 0)
        else:
            raise ValueError('{} is not a cont_cat_corr_method option'.format(cont_cat_corr_method))
    
    #######################################################################################################
    ## correlations and clustering of continuous variables
    #######################################################################################################
    
    def correlations_continuous(self, threshold = 0.5):
        # List of pairs along with correlation above threshold
        cont_corr_list = []
        
        col = self._dataset.columns
        
        # Search for the highly correlated pairs
        for i,j in itertools.combinations(self._cont_index_predictors, r=2): 
             if (ws.wgt_corr(self._dataset.iloc[:,i],self._dataset.iloc[:,j],self._weight_vector) >= threshold) or (ws.wgt_corr(self._dataset.iloc[:,i],self._dataset.iloc[:,j],self._weight_vector) <= -threshold):
                cont_corr_list.append([ws.wgt_corr(self._dataset.iloc[:,i],self._dataset.iloc[:,j],self._weight_vector),i,j]) #store correlation and columns index
    
        # Order variables by level of correlation           
        s_cont_corr_list = sorted(cont_corr_list,key=lambda x: -abs(x[0]))
        
        # Print correlations and column names
        print('Pearson Correlation - Predictors')
        for v,i,j in s_cont_corr_list:
            print('{} and {} = {:.2}'.format(col[i],col[j],v))
            
        # Scatter plot of only the highly correlated pairs
        for v,i,j in s_cont_corr_list:
            sns.pairplot(self._dataset, size=6, x_vars=col[i],y_vars=col[j])
            plt.show()

    def pca_cont(self, normalize = True, show_all_axis = False, output_dimensionality = False):
        
        n_components = len(self._cont_index_predictors) # Look at all of the principal components
    
        """ If normalization is always desired (which seems reasonable), you could improve performance
            here by pipelining the scaler and PCA """
        if normalize:
            df = preprocessing.scale(self._dataset.iloc[:,self._cont_index_predictors]) # scaled data will have mean = 0 and variance = 1
        else:
            df = self._dataset.iloc[:,self._cont_index_predictors]
            
        # Find principal components
        pca = PCA(n_components=n_components)
        pca.fit(df)
        
        # Find variance attributable to each component
        dimensionality = pca.explained_variance_ratio_
        
        # Find aggregate variance attributable to first k components
        # TODO Look into numpy cumsum to do this more efficiently
        dimensionality_total = []
        dim_tot = 0
        for explainedVar in dimensionality:
            dim_tot = dim_tot + explainedVar
            dimensionality_total.append(dim_tot)
        
        # Look at how concentrated the variance (i.e. signal) is in first few components    
        fg,ax = plt.subplots(figsize=(12, 8))
        if show_all_axis:
            ax.set_xticks(np.arange(len(dimensionality)))
        ax = plt.plot(dimensionality)
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.show()
        
        # Look at how concentrated the variance (i.e. signal) is in first few components    
        fg,ax = plt.subplots(figsize=(12, 8))
        if show_all_axis:
            ax.set_xticks(np.arange(len(dimensionality)))
        ax = plt.plot(dimensionality_total)
        plt.ylabel('Cummulative Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.show()
        
        if output_dimensionality:
            return dimensionality
        
    def cont_feature_clusters_sklearn(self, n_clusters = 5):
        """ This uses feature agglomeration from scikit learn and only works for continuous variables
            Eventually expand this to categorical variables using Cramer's V covariance matrix similar to 
            R tool using the iclust package """   
            
        #Import the library
        from sklearn.cluster import FeatureAgglomeration
        
        Cluster = FeatureAgglomeration(n_clusters=n_clusters)
        Cluster.fit(self._dataset.iloc[:,self._cont_index_predictors])
        
        df = pd.DataFrame({'Variable':self._dataset.columns[self._cont_index_predictors], 'Cluster':Cluster.labels_})
        
        return df.sort_values(by='Cluster')
    
    def _create_cont_clusters_pearson(self, normalize = True, threshold = 0.5):
        if normalize:
            cont_predictors_prep = self._dataset.iloc[:,self._cont_index_predictors]
            cont_predictors_scaled = preprocessing.scale(cont_predictors_prep) # scaled data will have mean = 0 and variance = 1
            cont_predictors = pd.DataFrame(data = cont_predictors_scaled, columns = cont_predictors_prep.columns.tolist()) # need to convert from numpy array to dataframe 
        else:
            cont_predictors = self._dataset.iloc[:,self._cont_index_predictors]

        distanceMatrix = squareform(1-cont_predictors.corr().abs())
        
        assignments = fcluster(linkage(distanceMatrix, method='single'),threshold,'distance')
        cluster_output = pd.DataFrame({'Feature':cont_predictors.columns.tolist() , 'Cluster':assignments})
        
        cluster_output_sorted = cluster_output.sort_values(by='Cluster')
        
        return distanceMatrix, cluster_output_sorted
        

    def cont_feature_clusters(self, threshold = 0.5, dendogram = True, normalize = True):
        
        if dendogram:
            fg,ax = plt.subplots(figsize=(12, 8))
            dendrogram(linkage(self.cont_distance_matrix, method='single'), 
                       color_threshold=threshold, 
                       leaf_font_size=10,
                       labels = self._dataset.columns[self._cont_index_predictors].tolist())
            plt.xticks(rotation=90)
            plt.show()
            
        print(self.cont_clusters)

    
    #######################################################################################################
    ## correlations and clustering of categorical variables
    #######################################################################################################
    
    def cramers_v_matrix(self):
        n = len(self._cat_index_predictors)
        
        CramersVMatrix = np.zeros(shape=(n,n))
        
        for i in range(len(self._cat_index_predictors)):
            CramersVMatrix[i,i] = 1
            for j in range(i+1,len(self._cat_index_predictors)):
                cv = ws.wgt_cramers_v(self._dataset,self._cat_index_predictors[i],self._cat_index_predictors[j],self._weight)
                CramersVMatrix[i,j] = cv
                CramersVMatrix[j,i] = cv  
                    
        return CramersVMatrix

    def correlations_categorical(self, threshold = 0.5, scaled = 'Yes'):
        
        def stacked_bar_plot(plot_data, normalize):
            plot_data_agg = pd.crosstab(plot_data.iloc[:,0], plot_data.iloc[:,1], normalize = normalize)
            colors = plt.cm.RdYlBu(np.linspace(0, 1, plot_data_agg.shape[1]))
        
            fg,axe = plt.subplots(figsize=(12, 8))
            axe = plot_data_agg.plot(kind="bar",
                                     linewidth=0,
                                     stacked=True,
                                     color=colors,
                                     ax=axe,
                                     legend=True,
                                     grid=False)
            axe.set_xticklabels(plot_data_agg.index, rotation = 0)
            axe.legend(loc=[1.01, 0.0], title=plot_data.columns[1])
            axe.set_title('Exposure Correlation')
            plt.xticks(rotation=90)
            plt.show()
                
        cat_cols = self._dataset.columns[self._cat_index_predictors]
    
        cat_corr_list = []
        
        for i,j in itertools.combinations(range(len(self._cat_index_predictors)), r=2):
            cv_ij = self._cat_corr_matrix[i,j]
            if (cv_ij >= threshold) or (cv_ij <= -threshold):
                cat_corr_list.append([cv_ij,i,j]) #store correlation and columns index
    
        # Order variables by level of correlation           
        s_cat_corr_list = sorted(cat_corr_list,key=lambda x: -abs(x[0]))
        
        # Print correlations and column names
        print("Cramer's V Correlation - Predictors")
        for v,i,j in s_cat_corr_list:
            print('{} and {} = {:.2}'.format(cat_cols[i],cat_cols[j],v))
        
        # Stacked bar charts of only the highly correlated pairs
        for v,i,j in s_cat_corr_list:
            plot_data = self._dataset.iloc[:,np.array([self._cat_index_predictors[i],self._cat_index_predictors[j]])]
            if scaled == 'Yes' or scaled == 'Both':
                stacked_bar_plot(plot_data, normalize = 'index')
            
            if scaled == 'No' or scaled == 'Both':
                stacked_bar_plot(plot_data, normalize = False)
    
    # TODO Can make this a static method since it doesn't take in object            
    def pca_cat(self, show_all_axis = False, output_dimensionality = False):
        
        # Find variance attributable to each component
        evalues = LA.eig(self._cat_corr_matrix)[0]
        dimensionality = sorted(map(abs, evalues.tolist()), reverse=True)
        
        # Find aggregate variance attributable to first k components
        # TODO look at simplifying code below using numpy cumsum
        dimensionality_total = []
        dim_tot = 0
        for explainedVar in dimensionality:
            dim_tot = dim_tot + explainedVar
            dimensionality_total.append(dim_tot)
        
        # Look at how concentrated the variance (i.e. signal) is in first few components    
        fg,ax = plt.subplots(figsize=(12, 8))
        if show_all_axis:
            ax.set_xticks(np.arange(len(dimensionality)))
        ax = plt.plot(dimensionality)
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.show()
        
        # Look at how concentrated the variance (i.e. signal) is in first few components    
        fg,ax = plt.subplots(figsize=(12, 8))
        if show_all_axis:
            ax.set_xticks(np.arange(len(dimensionality)))
        ax = plt.plot(dimensionality_total)
        plt.ylabel('Cummulative Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.show()
        
        if output_dimensionality:
            return dimensionality
    
    def _create_cat_clusters(self, threshold = 0.5):
        cat_cols = self._dataset.columns[self._cat_index_predictors]
        
        distanceMatrix = squareform(1-self._cat_corr_matrix)
        
        assignments = fcluster(linkage(distanceMatrix, method='single'),threshold,'distance')
        cluster_output = pd.DataFrame({'Feature':cat_cols.tolist() , 'Cluster':assignments})
        
        cluster_output_sorted = cluster_output.sort_values(by='Cluster')
        
        return distanceMatrix, cluster_output_sorted

        
    def cat_feature_clusters(self, threshold = 0.5, dendogram = True):
        cat_cols = self._dataset.iloc[:,self._cat_index_predictors].columns
        
        if dendogram:
            fg,ax = plt.subplots(figsize=(12, 8))
            dendrogram(linkage(self.cat_distance_matrix, method='single'), 
                       color_threshold=threshold, 
                       leaf_font_size=10,
                       labels = cat_cols.tolist())
            plt.xticks(rotation=90)
            plt.show()
    
        print(self.cat_clusters)
                
    #######################################################################################################
    ## clustering of continuous and categorical variables
    #######################################################################################################
    
    def _cont_cat_corr_features_anova(self, p_val = 0.01, subsamplesize = 100, p_seed = 0):
        """ Use ANOVA to find categorical - continuous relationships. Small differences come through
            as significant with a high number of observations, therefore we use a sample size of 100 
            Also keep in mind that by using ANOVA we assume normally distributed data and equal variances
            an alternative is to use Kruskal - Wallis """
        """ Use ICC to define correlations, give box-plots for highly correlated pairs """
        # TODO add option to do Bonferroni correction to adjust p-value depending on number of variables
        
        warnings.filterwarnings('ignore')
        # List of pairs along with correlation above threshold
        cont_cat_corr_list = []
        
        seed(p_seed)
        rand_vals = sample(range(self._n_rows), k=subsamplesize)
        
        # Search for the highly correlated pairs
        for i in self._cont_index_predictors: 
            for j in self._cat_index_predictors:
                formula = self._dataset.columns[i] + " ~ " + self._dataset.columns[j] 
                model_fit = ols(formula, data=self._dataset.iloc[rand_vals,:]).fit()
                anova_model = anova_lm(model_fit)
                p = anova_model.iloc[0,4]
                if p < p_val:
                    cont_cat_corr_list.append([p,i,j]) #store correlation and columns index
        
        # Order variables by level of correlation           
        s_cont_cat_corr_list = sorted(cont_cat_corr_list,key=lambda x: abs(x[0]))
        
        cont_cat_corr_features = []

        for v,i,j in s_cont_cat_corr_list:
            cont_cat_corr_features.append([self._dataset.columns[i],self._dataset.columns[j],v])
            
        return cont_cat_corr_features
    
    def correlations_cont_cat(self):
        """ Use ANOVA to find categorical - continuous relationships. Small differences come through
            as significant with a high number of observations, therefore we use a sample size of 100 
            Also keep in mind that by using ANOVA we assume normally distributed data and equal variances
            an alternative is to use Kruskal - Wallis """
        """ Use ICC to define correlations, give box-plots for highly correlated pairs """
        
        warnings.filterwarnings('ignore')
        
        # Print correlations and column names
        print('One-way ANOVA p-values - Predictors')
        for i,j,v in self.cont_cat_distance:
            print('{} and {} = {:.2}'.format(i,j,v))
            
        # Box plot of the highly correlated pairs
        for i,j,v in self.cont_cat_distance:
            fg,ax = plt.subplots(figsize=(12, 8))
            fg = self._dataset.boxplot(i, j, ax=ax, grid=False)
            plt.xticks(rotation=90)
            plt.show()

    def cont_cat_feature_clusters(self):
        """ step 1: create a list with ['cat' vs 'cont', cluster number, cont/cat members of cluster] """
        
        def base_clusters(starting_clusters, cluster_label, cluster_number_shift = 0, cluster_index = []):
            cluster = [] # Keep track of the clusters we've found
            for i in range(len(starting_clusters)):
                feature_list = []
                if starting_clusters.iloc[i,0] not in cluster:
                    cluster.append(starting_clusters.iloc[i,0])
                    for j in range(len(starting_clusters)):
                        if starting_clusters.iloc[j,0] == starting_clusters.iloc[i,0]:
                            feature_list.append(starting_clusters.iloc[j,1])
                    cluster_index.append([cluster_label,int(starting_clusters.iloc[i,0])+cluster_number_shift,feature_list])
            return cluster_index
        
        cluster_index_cont = base_clusters(self.cont_clusters, 'cont')
        cluster_index = base_clusters(self.cat_clusters, 'cat', max(self.cont_clusters.iloc[:,0].astype(int)), cluster_index_cont)
         
        """ step 2: write a for loop that goes through cont_cat_distance to group up cont and cat clusters
            that are close together """
            
        cluster_joins = []
        for cont, cat, d in self.cont_cat_distance:
            cont_cat_join = []
            for k in range(len(cluster_index)):
                if cont in cluster_index[k][2]:
                    cont_cat_join.append(cluster_index[k][1])
                if cat in cluster_index[k][2]:
                    cont_cat_join.append(cluster_index[k][1])
            cluster_joins.append(cont_cat_join)
        
        cont_join = [] # keep track of continuous clusters that have been joined to a categorical cluster
        cat_join = [] # keep track of categorical clusters that have been joined to a continuous cluster
        for x, y in cluster_joins:
            if (x not in cont_join) and (y not in cat_join):
                cont_join.append(x)
                cat_join.append(y)
                for i, (category, cluster, features) in enumerate(cluster_index):
                    if cluster == y:
                        cluster_index[i][1] = x
                    
        cont_cat_clusters = []
        clusters = [] # keep track of clusters used
        cluster_num = 0
        for i, (category, cluster, features) in enumerate(cluster_index):
            if cluster not in clusters:
                clusters.append(cluster)
                cluster_num += 1
                cluster_features = features
                for j in range(i+1,len(cluster_index)):
                    if cluster_index[j][1] == cluster:
                        cluster_features = list(set(cluster_features).union(cluster_index[j][2]))
                cont_cat_clusters.append([cluster_num,cluster_features])
        
        for i,j in cont_cat_clusters:
            print('Cluster {}: {}'.format(i,j))              
        

"""
dataset = pd.read_csv("C:/Users/dsmit/ModelTools_Local/IBMHR_Classification/test_data.csv")  


predictors = np.delete(np.arange(35), np.array([1,8,21,26])) 
numeric_cat_index = np.array([6,10,13,14,16,24,25,27,30])
target = 1

            
model_data = ModelData(dataset, predictors, target, numeric_cat_index=numeric_cat_index)      
 

model_data.missing_value_cleanup(verbose = True)


feature_clusters = FeatureClusters(model_data)

feature_clusters.correlations_continuous()
feature_clusters.pca_cont()
cont_clusters = feature_clusters.cont_feature_clusters()

feature_clusters.correlations_categorical()
feature_clusters.pca_cat()
cat_clusters = feature_clusters.cat_feature_clusters()

cont_cat_dist = feature_clusters.correlations_cont_cat()
"""
                                    
