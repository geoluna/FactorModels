import pandas as pd

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


def remove_outliers(X):
    '''
    =========================================================================
    DESCRIPTION:
     This function takes a set of series aligned in the columns of a matrix
     and replaces outliers with the value nan.

     -------------------------------------------------------------------------
     INPUT:
               X   = dataset (one series per column)

     OUTPUT:
               Y   = dataset with outliers replaced with NaN
               n   = number of outliers found in each series

     -------------------------------------------------------------------------
     NOTES:
               1) Outlier definition: a data point x of a series X[:,i] is
               considered an outlier if abs(x-median)>10*interquartile_range.

               2) This function ignores values of nan and thus is capable of
               replacing outliers for series that have missing values.

     =========================================================================
        '''

    median_X = X.median(axis=0)                 # Calculate median of each series
    median_X_mat = X*0 + median_X               # Substitute all values of each series in X with their median

    IRQ = X.quantile(0.75) - X.quantile(0.25)   # Calculate interquartile range (IQR) of each series
    IRQ_mat = X*0 + IRQ                         # Substitute all values of each series in X with their IRQ

    Z = abs(X - median_X_mat)                   # Compute distance from median
    outliers = Z > (10*IRQ_mat)                 # Determine outliers given distance

    Y = X[outliers == False]                     # Replace outliers with nan
    n = outliers.sum()                          # Count the number of outliers
    return Y, n



# if __name__ == "__main__":
#     data = pd.read_csv('../../data/2019-07-transformed.csv', index_col=0)     # read in data
#     data_removed_outliers, count_outliers = remove_outliers(data)
#     data_removed_outliers.to_csv('../../data/2019-07-transformed-removed-outliers.csv')
#     count_outliers.to_csv('../../data/2019-07-count-outliers.csv')
