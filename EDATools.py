
# TODO Write all of the code below object oriented

import pandas as pd 
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# from https://github.com/scipy/scipy/blob/v0.19.1/scipy/stats/kde.py#L42-L564
from scipy.spatial.distance import cdist
class gaussian_kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, shape (n, ), optional, default: None
        An array of weights, of the same shape as `x`.  Each value in `x`
        only contributes its associated weight towards the bin count
        (instead of 1).

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : float
        Effective sample size using Kish's approximation.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.pdf(points) : ndarray
        Alias for ``kde.evaluate(points)``.
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = np.random.normal(size=n)
    >>>     m2 = np.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """
    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape
            
        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n
            
        # Compute the effective sample size 
        # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        self.neff = 1.0 / np.sum(self.weights ** 2)

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        # compute the normalised residuals
        chi2 = cdist(points.T, self.dataset.T, 'mahalanobis', VI=self.inv_cov) ** 2
        # compute the pdf
        result = np.sum(np.exp(-.5 * chi2) * self.weights, axis=1) / self._norm_factor

        return result

    __call__ = evaluate

    def scotts_factor(self):
        return np.power(self.neff, -1./(self.d+4))

    def silverman_factor(self):
        return np.power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Compute the mean and residuals
            _mean = np.sum(self.weights * self.dataset, axis=1)
            _residual = (self.dataset - _mean[:, None])
            # Compute the biased covariance
            self._data_covariance = np.atleast_2d(np.dot(_residual * self.weights, _residual.T))
            # Correct for bias (http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
            self._data_covariance /= (1 - np.sum(self.weights ** 2))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) #* self.n

def WgtDistPlot(p_x,p_varname,p_w=None,p_bins=60):
    """ Create a plot of the histogram and weighted KDE """
    import matplotlib.pyplot as plt
    
    xmin, xmax = min(p_x), max(p_x)
    x = np.linspace(xmin, xmax, 200)
    plt.figure(figsize=(12,8))
    
    # Plot a histogram
    plt.hist(p_x, p_bins, (xmin, xmax), histtype='stepfilled', alpha=.3, normed=True, color='k', label='Histogram', weights=p_w)
    
    # Construct a KDE and plot it
    pdf = gaussian_kde(p_x, weights=p_w)
    y = pdf(x)
    plt.plot(x, y, label='Weighted KDE')
    
    plt.xlabel(p_varname)
    plt.ylabel('Density')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()
    
def WgtBarplot(p_data, p_varname, p_weightname):
    """ Create a weighted bar plot for categorical variables """
    # Aggregate data
    wgt_data = p_data.groupby(by=[p_varname])[p_weightname].sum()
    
    # Draw plot
    ax = sns.barplot(wgt_data.index,wgt_data,ci=None)
    ax.set(ylabel='Weight')
    plt.xticks(rotation=90)
    plt.show()

def WgtMean(p_x, p_w=None):
    """ Calculate a weighted mean """
    x = np.array(p_x)
    if p_w is None:
        p_w = np.ones(len(x))
    w = np.array(p_w)
    return np.sum(x * w) / np.sum(w)

def WgtCov(p_x, p_y, p_w=None, p_unbiased=True):
    """ Calculate weighted covariance """
    x = np.array(p_x)
    y = np.array(p_y)
    if p_w is None:
        p_w = np.ones(len(x))
    w = np.array(p_w)
    if p_unbiased:
        return np.sum(w * (x - WgtMean(x, w)) * (y - WgtMean(y,w))) / (np.sum(w)-1)
    else:
        return np.sum(w * (x - WgtMean(x, w)) * (y - WgtMean(y,w))) / np.sum(w)
        
def WgtStdDev(p_x, p_w=None, p_unbiased=True):
    """ Calculate weighted standard deviation """
    return np.sqrt(WgtCov(p_x, p_x, p_w, p_unbiased))
    
def WgtCorr(p_x, p_y, p_w=None):
    """ Calculate weighted correlation """
    return WgtCov(p_x, p_y, p_w) / np.sqrt(WgtCov(p_x, p_x, p_w) * WgtCov(p_y, p_y, p_w))
    
def WgtSkew(p_x, p_w=None):
    """ Calculate weighted skewness """
    x = np.array(p_x)
    if p_w is None:
        p_w = np.ones(len(x))
    w = np.array(p_w)
    n = np.sum(w)
    return (n**2/((n-1)*(n-2))) * (np.sum(w * (x - WgtMean(x, w))**3)/n) / WgtStdDev(x, w)**3
        
def WgtKurt(p_x, p_w=None):
    """ Calculate weighted kurtosis """
    x = np.array(p_x)
    if p_w is None:
        p_w = np.ones(len(x))
    w = np.array(p_w)
    n = np.sum(w)
    g2 = (np.sum(w * (x - WgtMean(x, w))**4) / n) / (np.sum(w * (x - WgtMean(x, w))**2)/n)**2 - 3
    return (n-1)/(n-2)/(n-3)*((n+1)*g2 + 6)    
    
def WgtQuantile(p_x, p_quantiles, p_w=None, p_values_sorted=False, p_old_style=True):
    """ Calculate weighted quantiles """
    x = np.array(p_x)
    quantiles = np.array(p_quantiles)
    if p_w is None:
        p_w = np.ones(len(x))
    w = np.array(p_w)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0,1]'
    
    if not p_values_sorted:
        sorter = np.argsort(x)
        x = x[sorter]
        w = w[sorter]
        
    weighted_quantiles = np.cumsum(w) - 0.5 * w
    if p_old_style:
        # To be consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(w)
    return np.interp(quantiles, weighted_quantiles, x)

def WgtDescribe(p_data, p_index, p_weight=None):
    """ Plot the histogram and KDE along with summary statistics """
    index_names = ['Weight','Mean','StdDev','Min','25%','50%','75%','Max','Skewness','Kurtosis']
    if p_weight is None:
        w = np.ones(len(p_data))
    else:
        w = p_weight
    Total_weight = np.sum(w)
    col_data = p_data.iloc[:,p_index]
    col_name = [p_data.columns[p_index]]
    quantiles = WgtQuantile(col_data,[0,.25,.5,.75,1],w)
    
    data = np.array([Total_weight,
                     WgtMean(col_data,w),
                     WgtStdDev(col_data,w),
                     quantiles[0],
                     quantiles[1],
                     quantiles[2],
                     quantiles[3],
                     quantiles[4],
                     WgtSkew(col_data,w),
                     WgtKurt(col_data,w)])

    df = pd.DataFrame(data,columns=col_name,index=index_names)
    print(df)

def null_value_cleanup(p_data, p_numeric_cat_index = np.array([]), p_na_imputation = 0, p_verbose=False):
    """ Take variables with missing values fill those with p_na_imputation and create another ISNULL variable """
    cols_with_nulls = p_data.columns[p_data.isnull().any()].tolist()
    cont_index, cat_index = ContCatSplit(p_data, p_numeric_cat_index)
    
    counter = 1
    for i, column in enumerate(p_data.columns):
        if column in cols_with_nulls:
            if p_verbose:
                print(column)
            if i in cat_index:
                p_data[column] = p_data[column].fillna('Null_Value')
            else:
                new_column = p_data[column].isnull().astype(int)
                p_data[column] = p_data[column].fillna(p_na_imputation)
                p_data.insert(i+counter, column+'_ISNULL', new_column)
                
                counter += 1
                
                  
def EDA(p_data,
        p_predictors,
        p_numeric_cat_index = np.array([]),
        p_controls = None,
        p_weight = None,
        p_target = None,
        p_bins = 60):
    """ Perform distributional analysis for both continuous and categorical variables """
    from pandas.core.dtypes.common import is_numeric_dtype
    
    if p_weight is None:
        weight = np.ones(len(p_data))
    else:
        weight = p_data.iloc[:,p_weight]
    
    data_predictor = p_data.iloc[:,p_predictors]
    
    # Print distributional plots for numeric variables and histograms for categorical
    print('Distributions - Predictors:')
    col = data_predictor.columns
    n_cols = data_predictor.shape[1]
    
    for i in range(n_cols):
        # Split variables into continuous and categorical and do the right thing
        if is_numeric_dtype(data_predictor.dtypes[i]) and (p_predictors[i] not in p_numeric_cat_index):
            WgtDistPlot(p_x=np.asarray(data_predictor.iloc[:,i]),p_varname=col[i],p_w=np.asarray(weight),p_bins=p_bins)
            WgtDescribe(data_predictor,i,weight)
        else:
            fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
            if p_weight is None:
                data_count = p_data.groupby(by=[col[i]])[col[i]].count()
                sns.countplot(x=col[i], data=data_predictor) 
                plt.xticks(rotation=90)
                plt.show()
            else:
                data_count = p_data.groupby(by=[col[i]])[p_data.columns[p_weight]].sum()
                WgtBarplot(p_data, col[i], p_data.columns[p_weight])
    
            for j, idx in enumerate(data_count.index):
                if j == 0:
                    data_count = data_count.reset_index(drop=True)
                    print('{0:25} {1}'.format('Level','Weight'))
                print('{0:25} {1}'.format(str(idx),data_count[j]))
        
    if p_controls is not None:
        data_control = p_data.iloc[:,p_controls]
        
        # Print distributional plots for numeric variables and histograms for categorical
        print('Distributions - Controls:')
        col = data_control.columns
        n_cols = data_control.shape[1]
        
        for i in range(n_cols):
            # Split variables into continuous and categorical and do the right thing
            if is_numeric_dtype(data_control.dtypes[i]) and (p_controls[i] not in p_numeric_cat_index):
                WgtDistPlot(p_x=np.asarray(data_control.iloc[:,i]),p_varname=col[i],p_w=np.asarray(weight),p_bins=p_bins)
                WgtDescribe(data_control,i,weight)
            else:
                fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
                if p_weight is None:
                    data_count = p_data.groupby(by=[col[i]])[col[i]].count()
                    sns.countplot(x=col[i], data=data_control)
                    plt.xticks(rotation=90)
                    plt.show()
                else:
                    data_count = p_data.groupby(by=[col[i]])[p_data.columns[p_weight]].sum()
                    WgtBarplot(p_data, col[i], p_data.columns[p_weight])
                    
                for j, idx in enumerate(data_count.index):
                    if j == 0:
                        print('{0:25} {1}'.format('Level','Weight'))
                    print('{0:25} {1}'.format(str(idx),data_count[j]))
                
    if p_target is not None:  
        data_target = p_data.iloc[:,p_target]
            
        # Print distributional plots for numeric variables and histograms for categorical
        print('Distributions - Target:')
        if is_numeric_dtype(data_target.dtypes) and (p_target not in p_numeric_cat_index):
            WgtDistPlot(p_x=np.asarray(data_target),p_varname=data_target.name,p_w=np.asarray(weight),p_bins=p_bins)
            WgtDescribe(p_data,p_target,weight)
        else:
            fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
            if p_weight is None:
                data_count = p_data.groupby(by=[data_target.name])[data_target.name].count()
                sns.countplot(x=data_target.name, data=data_target)
                plt.xticks(rotation=90)
                plt.show()
            else:
                data_count = p_data.groupby(by=[data_target.name])[p_data.columns[p_weight]].sum()
                WgtBarplot(p_data, data_target.name, p_data.columns[p_weight])
                
            for j, idx in enumerate(data_count.index):
                if j == 0:
                    print('{0:25} {1}'.format('Level','Weight'))
                print('{0:25} {1}'.format(str(idx),data_count[j]))
  
          
def DistributionFit(p_data,
                    p_target):
    """ Look for the appropriate distribution of your target variable """
    data_target = p_data.iloc[:,p_target]
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


def TargetDist(p_data,
               p_target,
               p_target_categorical = False,
               p_weight = None,
               p_pp_plots = True, # Set to True for severity models and False for frequency
               p_bins = 60):
    data_target = p_data.iloc[:,p_target]
    
    if p_weight is None:
        weight = np.ones(len(p_data))
    else:
        weight = p_data.iloc[:,p_weight]
        
    col = p_data.columns
    
    if p_target_categorical:
        fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
        if p_weight is None:
            data_count = p_data.groupby(by=[col[p_target]])[col[p_target]].count()
            sns.countplot(x=col[p_target], data=p_data)
            plt.xticks(rotation=90)
            plt.show()
        else:
            data_count = p_data.groupby(by=[col[p_target]])[p_data.columns[p_weight]].sum()
            WgtBarplot(p_data, col[p_target], col[p_weight])
        
        for j, idx in enumerate(data_count.index):
            if j == 0:
                data_count = data_count.reset_index(drop=True)
                print('{0:25} {1}'.format('Level','Weight'))
            print('{0:25} {1}'.format(str(idx),data_count[j]))
    else:
        WgtDistPlot(p_x=np.asarray(data_target),p_varname=col[p_target],p_w=np.asarray(weight),p_bins=p_bins)
        WgtDescribe(p_data,p_target,weight)
    
    if p_pp_plots:
        # Look at various distributions
        DistributionFit(p_data,p_target)


def ContCatSplit(p_data, p_numeric_cat_index = np.array([])):
    """ Split the predictors into continuous and categorical variables """
    """ This function could likely be vectorized in one step """
    from pandas.core.dtypes.common import is_numeric_dtype
    
    cont = []
    cat = []
    
    n_cols = p_data.shape[1]
    for i in range(n_cols):
        if is_numeric_dtype(p_data.dtypes[i]) and i not in p_numeric_cat_index:
            cont.append(i)
        else:
            cat.append(i)
    return cont, cat

    
def rare_level_check(p_data, 
                     p_predictors,
                     p_numeric_cat_index = np.array([]),
                     p_weight = None, # only fill this in if you want a weighted check
                     p_threshold = 0.005, # if a fraction check portion, if integer check count/weight
                     p_verbose=False):
    """ Check for variables with levels that have very little value or variables with a single level """
    cat_cols = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[1])
    
    counter = 0
    for i, column in enumerate(p_data.columns):
        if i in cat_cols:
            if p_verbose:
                print('checking {}'.format(column))
            
            if p_threshold < 1.0:
                if p_weight is None:
                    data_count = p_data.groupby(by=[column])[column].count() / len(p_data)
                    word = 'portion'
                else:
                    data_count = p_data.groupby(by=[column])[p_data.columns[p_weight]].sum() / p_data.iloc[:,p_weight].sum()
                    word = 'weighted portion'
            elif p_threshold > 1.0:
                if p_weight is None:
                    data_count = p_data.groupby(by=[column])[column].count()
                    word = 'count'
                else:
                    data_count = p_data.groupby(by=[column])[p_data.columns[p_weight]].sum()
                    word = 'weight'
                    
            if len(data_count) == 1:
                print('{} contains a single level, it should be excluded or revisited'.format(column))
                counter += 1
            else:
                for idx in data_count.index:
                    if data_count[idx] < p_threshold:
                        print('{} has thin data in level {}, the {} is {}.'.format(column, str(idx), word, data_count[idx]))
                        counter += 1
                        
    if counter == 0:
        print('There are no levels with thin data, using a threshold of {}'.format(p_threshold))

def Correlations_Cont(p_data,
                      p_predictors,
                      p_numeric_cat_index = np.array([]),
                      p_weight=None,
                      p_threshold = 0.5):
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[0])
    
    cont_predictors = p_data.iloc[:,cont_index]
    
    if p_weight is None:
        weight = np.ones(len(p_data))
    else:
        weight = p_data.iloc[:,p_weight]
    
    # List of pairs along with correlation above threshold
    cont_corr_list = []
    
    cont_cols = cont_predictors.columns
    
    # Search for the highly correlated pairs
    for i in range(len(cont_index)): 
        for j in range(i+1,len(cont_index)): 
             if (WgtCorr(cont_predictors.iloc[:,i],cont_predictors.iloc[:,j],weight) >= p_threshold) or (WgtCorr(cont_predictors.iloc[:,i],cont_predictors.iloc[:,j],weight) <= -p_threshold):
                cont_corr_list.append([WgtCorr(cont_predictors.iloc[:,i],cont_predictors.iloc[:,j],weight),i,j]) #store correlation and columns index
    
    # Order variables by level of correlation           
    s_cont_corr_list = sorted(cont_corr_list,key=lambda x: -abs(x[0]))
    
    # Print correlations and column names
    print('Pearson Correlation - Predictors')
    for v,i,j in s_cont_corr_list:
        print('{} and {} = {:.2}'.format(cont_cols[i],cont_cols[j],v))
        
    # Scatter plot of only the highly correlated pairs
    for v,i,j in s_cont_corr_list:
        sns.pairplot(cont_predictors, size=6, x_vars=cont_cols[i],y_vars=cont_cols[j] )
        plt.show()


def CramersV(p_data,
              p_var1,
              p_var2,
              p_weight = None,
              p_bias_correction = False):
    # Find the contingency table
    if p_weight is None:
        table = pd.crosstab(p_data.iloc[:,p_var1], p_data.iloc[:,p_var2])
        chi_sq = stats.chi2_contingency(table,correction = False)[0]
    else:
        table = pd.crosstab(p_data.iloc[:,p_var1], p_data.iloc[:,p_var2], p_data.iloc[:,p_weight], aggfunc = sum).fillna(0)
        chi_sq = stats.chi2_contingency(table,correction = False)[0]/p_data.iloc[:,p_weight].mean()
    
    n = len(p_data)
    
    k = len(p_data.iloc[:,p_var1].unique()) 
    r = len(p_data.iloc[:,p_var2].unique())
    
    if p_bias_correction:
        k_new = k - (k-1)**2/(n-1)
        r_new = r - (r-1)**2/(n-1)
        cramersV = (max(0,((chi_sq/n)-(k-1)*(r-1)/(n-1)))/min(k_new-1,r_new-1))**.5
    else:
        cramersV = ((chi_sq/n)/min(k-1,r-1))**.5
    
    return cramersV


def CramersVMatrix(p_data,
                   p_predictors,
                   p_numeric_cat_index = np.array([]),
                   p_weight = None):
    cat_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[1])
    
    n = len(cat_index)
    
    CramersVMatrix = np.zeros(shape=(n,n))
    
    for i in range(len(cat_index)):
        CramersVMatrix[i,i] = 1
        for j in range(i+1,len(cat_index)):
            CramersVMatrix[i,j] = CramersV(p_data,cat_index[i],cat_index[j],p_weight)
            CramersVMatrix[j,i] = CramersV(p_data,cat_index[i],cat_index[j],p_weight)
    
    return CramersVMatrix


def Correlations_Cont_Cat(p_data,
                          p_predictors,
                          p_numeric_cat_index = np.array([]),
                          p_weight = None,
                          p_p_val = 0.01,
                          p_subsamplesize = 100,
                          p_seed = 0):
    """ Use ANOVA to find categorical - continuous relationships. Small differences come through
        as significant with a high number of observations, therefore we use a sample size of 100 
        Also keep in mind that by using ANOVA we assume normally distributed data and equal variances
        an alternative is to use Kruskal - Wallis """
    """ Use ICC to define correlations, give box-plots for highly correlated pairs """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[0])
    cat_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[1])
    
    # List of pairs along with correlation above threshold
    cont_cat_corr_list = []
    
    from random import sample, seed
    seed(p_seed)
    rand_vals = sample(range(len(p_data)), k=p_subsamplesize)
    
    # Search for the highly correlated pairs
    for i in cont_index: 
        for j in cat_index:
            formula = p_data.columns[i] + " ~ " + p_data.columns[j] 
            model_fit = ols(formula, data=p_data.iloc[rand_vals,:]).fit()
            anova_model = anova_lm(model_fit)
            p = anova_model.iloc[0,4]
            if p < p_p_val:
                cont_cat_corr_list.append([p,i,j]) #store correlation and columns index
    
    # Order variables by level of correlation           
    s_cont_cat_corr_list = sorted(cont_cat_corr_list,key=lambda x: abs(x[0]))
    
    cont_cat_corr_features = []
    # Print correlations and column names
    print('One-way ANOVA p-values - Predictors')
    for v,i,j in s_cont_cat_corr_list:
        cont_cat_corr_features.append([p_data.columns[i],p_data.columns[j],v])
        print('{} and {} = {:.2}'.format(p_data.columns[i],p_data.columns[j],v))
        
    # Box plot of the highly correlated pairs
    for v,i,j in s_cont_cat_corr_list:
        fg,ax = plt.subplots(figsize=(12, 8))
        fg = p_data.boxplot(p_data.columns[i], p_data.columns[j], ax=ax, grid=False)
        plt.xticks(rotation=90)
        plt.show()
        
    return cont_cat_corr_features

    
def Correlations_Cat(p_data,
                     p_predictors,
                     p_numeric_cat_index = np.array([]),
                     p_weight = None,
                     p_corr_matrix = None,
                     p_threshold = 0.5,
                     p_scaled = 'Yes',
                     p_verbose = False):
    from matplotlib import cm
    cat_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[1])
    
    cat_predictors = p_data.iloc[:,cat_index]

    cat_cols = cat_predictors.columns

    cat_corr_list = []
    
    """ The operation below should be parellelized for better performance """
    for i in range(len(cat_index)): 
        for j in range(i+1,len(cat_index)): 
            if p_corr_matrix is None:
                cv_ij = CramersV(p_data,cat_index[i],cat_index[j],p_weight)
            else:
                cv_ij = p_corr_matrix[i,j]
            if (cv_ij >= p_threshold and cv_ij < 1) or (cv_ij >= -1 and cv_ij <= -p_threshold):
                cat_corr_list.append([cv_ij,i,j]) #store correlation and columns index
    
    # Order variables by level of correlation           
    s_cat_corr_list = sorted(cat_corr_list,key=lambda x: -abs(x[0]))
    
    # Print correlations and column names
    print("Cramer's V Correlation - Predictors")
    for v,i,j in s_cat_corr_list:
        print('{} and {} = {:.2}'.format(cat_cols[i],cat_cols[j],v))
    
    # Stacked bar charts of only the highly correlated pairs
    for v,i,j in s_cat_corr_list:
        plot_data = cat_predictors.iloc[:,np.array([i,j])]
        if p_scaled == 'Yes' or p_scaled == 'Both':
            plot_data_agg = pd.crosstab(plot_data.iloc[:,0], plot_data.iloc[:,1], normalize = 'index')
            colors = plt.cm.RdYlBu(np.linspace(0, 1, plot_data_agg.shape[1]))
        
            fg,axe = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
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
        
        if p_scaled == 'No' or p_scaled == 'Both':
            plot_data_agg = pd.crosstab(plot_data.iloc[:,0], plot_data.iloc[:,1])
            colors = plt.cm.RdYlBu(np.linspace(0, 1, plot_data_agg.shape[1]))
            
            fg,axe = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
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


def PCACat(p_corr_matrix,
           p_show_all_axis = False,
           p_output_dimensionality = False):
    from numpy import linalg as LA
    
    # Find variance attributable to each component
    evalues = LA.eig(p_corr_matrix)[0]
    dimensionality = sorted(map(abs, evalues.tolist()), reverse=True)
    
    # Find aggregate variance attributable to first k components
    dimensionality_total = []
    dim_tot = 0
    for explainedVar in dimensionality:
        dim_tot = dim_tot + explainedVar
        dimensionality_total.append(dim_tot)
    
    # Look at how concentrated the variance (i.e. signal) is in first few components    
    fg,ax = plt.subplots(figsize=(12, 8))
    if p_show_all_axis:
        ax.set_xticks(np.arange(len(dimensionality)))
    ax = plt.plot(dimensionality)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.show()
    
    # Look at how concentrated the variance (i.e. signal) is in first few components    
    fg,ax = plt.subplots(figsize=(12, 8))
    if p_show_all_axis:
        ax.set_xticks(np.arange(len(dimensionality)))
    ax = plt.plot(dimensionality_total)
    plt.ylabel('Cummulative Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.show()
    
    if p_output_dimensionality:
        return dimensionality


def CatFeatureClusters(p_corr_matrix,
                       p_predictors,
                       p_data,
                       p_numeric_cat_index = np.array([]),
                       p_threshold = 0.5,
                       p_dendogram = True):
    # Import clustering tools
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    
    cat_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[1])    
    cat_cols = p_data.iloc[:,cat_index].columns
    
    distanceMatrix = squareform(1-p_corr_matrix)
    
    if p_dendogram:
        fg,ax = plt.subplots(figsize=(12, 8))
        dendrogram(linkage(distanceMatrix, method='single'), 
                   color_threshold=p_threshold, 
                   leaf_font_size=10,
                   labels = cat_cols.tolist())
        plt.xticks(rotation=90)
        plt.show()

    assignments = fcluster(linkage(distanceMatrix, method='single'),p_threshold,'distance')
    cluster_output = pd.DataFrame({'Feature':cat_cols.tolist() , 'Cluster':assignments})
    
    cluster_output_sorted = cluster_output.sort_values(by='Cluster')
    print(cluster_output_sorted)
    
    return cluster_output_sorted


def FeatureClusters(p_data,
                    p_predictors,
                    p_numeric_cat_index = np.array([]),
                    p_n_clusters = 5):
    """ This uses feature agglomeration from scikit learn and only works for continuous variables
        Eventually expand this to categorical variables using Cramer's V covariance matrix similar to 
        R tool using the iclust package """   
        
    # Find clusters of correlated (continuous) variables 
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[0])
    
    #Import the library
    from sklearn.cluster import FeatureAgglomeration
    
    Cluster = FeatureAgglomeration(n_clusters=p_n_clusters)
    Cluster.fit(p_data.iloc[:,cont_index])
    
    df = pd.DataFrame({'Variable':p_data.columns[cont_index], 'Cluster':Cluster.labels_})
    
    return df.sort_values(by='Cluster')


def FeatureClusters2(p_data,
                     p_predictors,
                     p_numeric_cat_index = np.array([]),
                     p_threshold = 0.5,
                     p_dendogram = True,
                     p_normalize = True):
    # Import clustering tools
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[0])   
    
    if p_normalize:
        from sklearn import preprocessing
        cont_predictors_prep = p_data.iloc[:,cont_index]
        cont_predictors_scaled = preprocessing.scale(cont_predictors_prep) # scaled data will have mean = 0 and variance = 1
        cont_predictors = pd.DataFrame(data = cont_predictors_scaled, columns = cont_predictors_prep.columns.tolist()) # need to convert from numpy array to dataframe 
    else:
        cont_predictors = p_data.iloc[:,cont_index]
    
    distanceMatrix = squareform(1-cont_predictors.corr().abs())
    
    if p_dendogram:
        fg,ax = plt.subplots(figsize=(12, 8))
        dendrogram(linkage(distanceMatrix, method='single'), 
                   color_threshold=p_threshold, 
                   leaf_font_size=10,
                   labels = cont_predictors.columns.tolist())
        plt.xticks(rotation=90)
        plt.show()

    assignments = fcluster(linkage(distanceMatrix, method='single'),p_threshold,'distance')
    cluster_output = pd.DataFrame({'Feature':cont_predictors.columns.tolist() , 'Cluster':assignments})
    
    cluster_output_sorted = cluster_output.sort_values(by='Cluster')
    print(cluster_output_sorted)
    
    return cluster_output_sorted


def ContCatFeatureClusters(p_cont_clusters,
                           p_cat_clusters,
                           p_cont_cat_dist): #[cont, cat, d]
    """ step 1: create a list with ['cat' vs 'cont', cluster number, cont/cat members of cluster] """
    
    cluster_index = [] # Store the features associated with each cluster
    cont_cluster = [] # Keep track of cont clusters we've found
    for i in range(len(p_cont_clusters)):
        feature_list = []
        if p_cont_clusters.iloc[i,0] not in cont_cluster:
            cont_cluster.append(p_cont_clusters.iloc[i,0])
            for j in range(len(p_cont_clusters)):
                if p_cont_clusters.iloc[j,0] == p_cont_clusters.iloc[i,0]:
                    feature_list.append(p_cont_clusters.iloc[j,1])
            cluster_index.append(['cont',int(p_cont_clusters.iloc[i,0]),feature_list])
            
    cat_cluster = [] # Keep track of cat clusters we've found
    for i in range(len(p_cat_clusters)):
        feature_list = []
        if p_cat_clusters.iloc[i,0] not in cat_cluster:
            cat_cluster.append(p_cat_clusters.iloc[i,0])
            for j in range(len(p_cat_clusters)):
                if p_cat_clusters.iloc[j,0] == p_cat_clusters.iloc[i,0]:
                    feature_list.append(p_cat_clusters.iloc[j,1]) 
            cluster_index.append(['cat',int(p_cat_clusters.iloc[i,0])+max(p_cont_clusters.iloc[:,0].astype(int)),feature_list])
     
    """ step 2: write a for loop that goes through p_cont_cat_dist to group up cont and cat clusters
        that are close together """
        
    cluster_joins = []
    for cont, cat, d in p_cont_cat_dist:
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
        
                                    
def PCAnalysis(p_data,
               p_predictors,
               p_numeric_cat_index = np.array([]),
               p_normalize = True,
               p_show_all_axis = False,
               p_output_dimensionality = False):
    cont_index = np.intersect1d(p_predictors,ContCatSplit(p_data, p_numeric_cat_index)[0])
    
    n_components = len(cont_index) # Look at all of the principal components

    """ If normalization is always desired (which seems reasonable), you could improve performance
        here by pipelining the scaler and PCA """
    if p_normalize:
        from sklearn import preprocessing
        df = preprocessing.scale(p_data.iloc[:,cont_index]) # scaled data will have mean = 0 and variance = 1
    else:
        df = p_data.iloc[:,cont_index]
        
    # Look at how explained variance changes with 
    from sklearn.decomposition import PCA
    
    # Find principal components
    pca = PCA(n_components=n_components)
    pca.fit(df)
    
    # Find variance attributable to each component
    dimensionality = pca.explained_variance_ratio_
    
    # Find aggregate variance attributable to first k components
    dimensionality_total = []
    dim_tot = 0
    for explainedVar in dimensionality:
        dim_tot = dim_tot + explainedVar
        dimensionality_total.append(dim_tot)
    
    # Look at how concentrated the variance (i.e. signal) is in first few components    
    fg,ax = plt.subplots(figsize=(12, 8))
    if p_show_all_axis:
        ax.set_xticks(np.arange(len(dimensionality)))
    ax = plt.plot(dimensionality)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.show()
    
    # Look at how concentrated the variance (i.e. signal) is in first few components    
    fg,ax = plt.subplots(figsize=(12, 8))
    if p_show_all_axis:
        ax.set_xticks(np.arange(len(dimensionality)))
    ax = plt.plot(dimensionality_total)
    plt.ylabel('Cummulative Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.show()
    
    if p_output_dimensionality:
        return dimensionality