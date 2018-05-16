    """ 
    Copyright (c) 2018 Chris Heckler <hecklerchris@hotmail.com>
    
    This module takes a .csv and calculates
    Confidence Intervals and Hypothesis tests for mean, var, etc.
    """
    
    import pandas as pd
    import numpy as np
    import scipy as sc
    import decimal
    import json
    import matplotlib.pyplot as plt
    from argparse imort ArgumentParser
    from scipy import stats as scs

def parse_options(argv):

    parser = argparse.ArgumentParser(description="Statistical Computation",
                                     prog="Chris's Statistical Programming Program",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('data', help="Initial data set")
    parser.add_argument('-a', '--', type=str, dest='',
                        help="")
    parser.add_argument('--version', action='version', version='%(prog)s')
    return parser.parse_args(argv)

    
class heckles_stats:
    def __init__(self,filename):
    
        try:    
            self.data = pd.read_csv(filename)
        except:
            print('Error reading file.')
    
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
    
    def hypothesis_test_two_means_testvalue(datae,dataf,test_value,alpha):
        """ Hypothesis test between two  means and
            a test value.
    
            Each data set has following variables:
            data, n, mean, var and df.
    
            The T*, p-value and decision are returned.
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
        
        # Sp,t and pvalue
        Sp = np.sqrt((((df_e*var_e) + (df_f*var_f))/(df_e+df_f)))
        t = ((mean_e-mean_f)-test_value)/(Sp*np.sqrt(1/n_e+1/n_f))
        pvalue = 1-scs.t.cdf(t,df_e+df_f,)
        
        # Decision
        if pvalue > alpha:
            decision = 'Fail to Reject H0'
            return t,pvalue,decision
        else:
            decision = 'Reject H0'
            return t,pvalue,decision

    def anova_table(data):
    
    ''' ANOVA Table
        Regression Coefficients 
    '''

        Xvar = data.x.var() 
        Xstd = data.x.std()
        N = len(data)
        mx = data.x.mean()
        my = data.y.mean()
        Sxx = (sum(map(lambda x: x*x, data.x)))-(sum(data.x)**2/N)
        Syy = (sum(map(lambda x: x*x, data.y)))-(sum(data.y)**2/N)
        Sxy = (sum(map(lambda x,y: x*y, data.x,data.y)))-(sum(data.x)*sum(data.y)/N)

        # ANOVA 

        Rsqr = Sxy**2/(Sxx*Syy)

        DFreg = 1
        DFerr = N-1
        DFtotal = DFreg + DFerr
        SSreg = Sxy**2/Sxx
        SSerr = Syy-Sxy**2/Sxx
        SStotal = Syy

        MSreg = SSreg/1
        MSerr = SSerr/(N-2)

        F = MSreg/MSerr
        p_value = scs.f.sf(F, DFerr,DFerr)

        # Regression Coefficients

        B1 = Sxy/Sxx            # Slope
        B0 = (my-B1*mx)         # Intercept  


        print('____________________________________________________________')
        print('ANOVA Table \n')
        print('Source      DF  S.S          M.S         F*       P-value')
        print('Regression  1   {}  {}  {}  {}'.format(round(SSreg,4), \
             round(MSreg,4), round(F,4), round(p_value,4)))
        print('Residual    {}  {}   {}'.format(round(DFerr,4),round(SSerr,4), \ 
             round(MSerr,4)))
        print('Total       {}  {}\n'.format(DFtotal,round(SStotal,4)))
        print('R^2         {}'.format(round(Rsqr,4)))
        print('____________________________________________________________')
        print('Regression Coefficients \n')
        print('B0  {}'.format(round(B0,4)))
        print('B1   {}'.format(round(B1,4)))
        print('____________________________________________________________')



if __name__ == '__main__':
    opts = parse_options(sys.argv[1:])

    print("Chris's Statistical Computation Program\n\n")

