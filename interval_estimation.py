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
        data,n,mean,std,zscore and interval. 

        Lower and upper bound confidence intervals are returned
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

def mean_hypothesis_test(data,alpha,test_value):
    '''  Function takes data, alpha and test_value.
    
         Variables needed for computation:
         data, n, df, mean, std.  
         
         Calculates and returns the z-score, p-value
         and a decision to Reject or Fail to Reject H0.
    '''    

    data = 1.0*np.array(data)
    n = data.shape[0]*data.shape[1]
    df = n - 1
    mean = np.array(data).mean()
    std = np.array(data).std(ddof=1)
    zscore = (np.sqrt(n)*(mean-test_value))/std
    pvalue = sc.special.ndtr(zscore)
    
    # Decision
    if pvalue > alpha:
        decision = 'Fail to Reject H0'
        return zscore,pvalue,decision
    else:
        decision = 'Reject H0'
        return zscore,pvalue,decision

def var_hypothesis_test(data,alpha,test_value):
    '''  This function tests the variance against a value.
         Function takes data, alpha and test_value.
    
         Variables needed for computation:
         data, n , df , mean, std, var,
         chi_score.
 
         The p-value and Decision are returned.
    '''

    data = 1.0*np.array(data)
    n = data.shape[0]*data.shape[1]
    df = n - 1
    mean = np.array(data).mean()
    std = np.array(data).std(ddof=1)
    var = np.array(data).var(ddof=1)
    chi_score = (df*var)/test_value
    pvalue = scs.chi2.sf(chi_score,df)
    
    # Decision 
    if pvalue > alpha:
        decision = 'Fail to Reject H0'
        return pvalue, decision
    else:
        decision = 'Reject H0'
        return pvalue, decision

def two_pop_var_test(datae,dataf,alpha):
    """ Compares the variance of two populations

        Each data set uses the following variables:
        data, n, mean, var and df.

        The left and right critical regions are returned
        along with the F value and a decision.  The decision
        checks if the F value falls within either region.
    """
    
    # Dataset E
    data_e = 1.0*np.array(datae)
    n_e = data_e.shape[0]*data_e.shape[1]
    mean_e = np.array(data_e).mean()
    var_e = np.array(data_e).var(ddof=1)
    df_e = n_e-1
    
    # Dataset F
    data_f = 1.0*np.array(dataf)
    n_f = dataf.shape[0]*dataf.shape[1]
    mean_f = np.array(data_f).mean()
    var_f = np.array(data_f).var(ddof=1)
    df_f = n_f-1
    
    # Calculate Critical Regions
    F = var_e/var_f
    critical_region_left = scs.f.ppf(alpha-(alpha/2),df_e,df_f) 
    critical_region_right = scs.f.ppf(1-alpha/2,df_e,df_f) 

    # Decision 
    if F < critical_region_left and F > critical_region_right:
        decision = 'Reject H0'
        return critical_region_left,critical_region_right,F,decision
    else:
        decision = 'Fail to Reject H0'
        return critical_region_left,critical_region_right,F,decision

def conf_interval_two_means(datae,dataf,conf):
    """ Finds a confidence interval between two means

        Each data set has the following variables:
        data, n, mean, var, df.

        The lower and upper bound confidence interval
        is returned for two means.
    """
    
    # Dataset E
    data_e = 1.0*np.array(datae)
    n_e = data_e.shape[0]*data_e.shape[1]
    mean_e = np.array(data_e).mean()
    var_e = np.array(data_e).var(ddof=1)
    df_e = n_e-1
    
    # Dataset F
    data_f = 1.0*np.array(dataf)
    n_f = dataf.shape[0]*dataf.shape[1]
    mean_f = np.array(data_f).mean()
    var_f = np.array(data_f).var(ddof=1)
    df_f = n_f-1
    
    # Sp,t calculated for lower/upper bounds 
    Sp = np.sqrt((((df_e*var_e) + (df_f*var_f))/(df_e+df_f)))
    t = abs(scs.t.ppf(((1-conf)/2), (df_e+df_f)))
    lower = (mean_e-mean_f)-(Sp*t*np.sqrt(1/n_e+1/n_f))
    upper = (mean_e-mean_f)+(Sp*t*np.sqrt(1/n_e+1/n_f))

    return lower,upper
