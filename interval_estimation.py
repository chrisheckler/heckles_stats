""" 
Copyright (c) 2018 Chris Heckler <hecklerchris@hotmail.com>

This module takes a .csv and calculates
Confidence Intervals and Hypothesis tests for mean, var, etc.
"""

import pandas as pd
import numpy as np
import scipy as sc
import decimal
from scipy import stats as scs

# Loading data
data = pd.read_csv('datasets/data.csv',header=None)

def mean_conf_interval(data, conf):
    ''' This function calculates a confidence interval for the
        mean.
    
        Variables needed for computation are:
        Data,n,mean,std,zscore and interval. Lower
        and upper bound confidence intervals are returned
    '''

    data = 1.0*np.array(data)
    n = data.shape[0]*data.shape[1]
    mean = np.array(data).mean()
    std = np.array(data).std(ddof=1)
    zscore = scs.norm.ppf(conf+((1-conf)/2))
    interval = zscore * std/np.sqrt(n)
    lower = mean-interval
    upper = mean+interval
    
    return lower,upper

def var_conf_interval(data, conf):
    ''' This function calculates a confidence interval for the 
        variance.
    
        Variables needed for computation:
        data,n,var, left and right chi statistic.
        The upper and lower bound confidence intervals
        returned.
    '''

    data = 1.0*np.array(data)
    n = data.shape[0]*data.shape[1]
    var = np.array(data).var(ddof=1)
    chi_right_statistic = scs.chi2.isf(q=(1-conf)/2, df=209)
    chi_left_statistic = scs.chi2.isf(q=conf+((1-conf)/2), df=209)
    lower = (((n-1)*var)/chi_right_statistic)
    upper = (((n-1)*var)/chi_left_statistic)
    
    return lower,upper
