import numpy as np

'''
=========================================================================
     Author (ported to python): George Milunovich
     Date:  5 November 2019


     Based on Matlab code by: Michael W. McCracken and Serena Ng
     Date: 6/7/2017
     Version: MATLAB 2014a
     Required Toolboxes: None
=========================================================================
'''


def mrsq(Fhat,lamhat,ve2,series):
    ''' =========================================================================
    DESCRIPTION
    This function computes the R-squared and marginal R-squared from
    estimated factors and factor loadings.

     -------------------------------------------------------------------------
    INPUTS
               Fhat    = estimated factors (one factor per column)
               lamhat  = factor loadings (one factor per column)
               ve2     = eigenvalues of covariance matrix
               series  = series names

     OUTPUTS
               R2      = R-squared for each series for each factor
               mR2     = marginal R-squared for each series for each factor
               mR2_F   = marginal R-squared for each factor
               R2_T    = total variation explained by all factors
               t10_s   = top 10 series that load most heavily on each factor
               t10_mR2 = marginal R-squared corresponding to top 10 series
                         that load most heavily on each factor

    '''

    N, ic = lamhat.shape # N = number of series, ic = number of factors
    Fhat = Fhat.values

    print(N, ic)

    # Preallocate memory for output
    R2 = np.full((N, ic), np.nan)
    mR2 = np.full((N, ic), np.nan)
    t10_mR2 = np.full((10, ic), np.nan)
    t10_s = []


    # Compute R-squared and marginal R-squared for each series for each factor
    for i in range(ic):
        R2[:, i] = (np.var(Fhat[:, :i+1]@lamhat[:, :i+1].T, axis=0))
        mR2[:, i] = (np.var(Fhat[:, i:i+1]@lamhat[:, i:i+1].T, axis=0))

    # Compute marginal R-squared for each factor
    mR2_F = ve2/np.sum(ve2)
    mR2_F = mR2_F[0:ic]

    # Compute total variation explained by all factors
    R2_T = np.sum(mR2_F)

    # Sort series by marginal R-squared for each factor
    ind = mR2.argsort(axis=0)[::-1]
    vals = mR2[ind, np.arange(ind.shape[1])]

    # Get top 10 series that load most heavily on each factor and the
    # corresponding marginal R-squared values

    for i in range(ic):
        t10_s.append(series[ind[0:10, i]])
        t10_mR2[:, i] = vals[0:10, i]

    t10_s = list(map(list, zip(*t10_s)))  # transpose list
    return R2, mR2, mR2_F, R2_T, t10_s, t10_mR2