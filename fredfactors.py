import pandas as pd
import prepare_missing as pm
import remove_outliers as ro
import factors_em as fem
import mrsq
# np.set_printoptions(precision=12, suppress=True)


''' =========================================================================
 DESCRIPTION
 This script loads in a FRED-MD dataset, processes the dataset, and then
 estimates factors.

 -------------------------------------------------------------------------
 BREAKDOWN OF THE SCRIPT

 Part 1: Load and label FRED-MD data.

 Part 2: Process data -- transform each series to be stationary and remove
         outliers.

 Part 3: Estimate factors and compute R-squared and marginal R-squared.

 -------------------------------------------------------------------------
 AUXILIARY FUNCTIONS
 List of auxiliary functions to be saved in same folder as this script.

   prepare_missing() - transforms series based on given transformation
       numbers

   remove_outliers() - removes outliers

   factors_em() - estimates factors

   mrsq() - computes R-squared and marginal R-squared from factor
       estimates and factor loadings

=========================================================================
     Author (ported to python): George Milunovich
     Date:  5 November 2019


     Based on Matlab code by: Michael W. McCracken and Serena Ng
     Date: 6/7/2017
     Version: MATLAB 2014a
     Required Toolboxes: None
=========================================================================
'''


# PARAMETERS TO BE CHANGED

csv_in = 'data/current.csv' # File name of desired FRED-MD vintage


# Type of transformation performed on each series before factors are estimated
#   0 --> no transformation
#   1 --> demean only
#   2 --> demean and standardize
#   3 --> recursively demean and then standardize

DEMEAN = 2

# Information criterion used to select number of factors; for more details,
# see auxiliary function factors_em()
#   1 --> information criterion PC_p1
#   2 --> information criterion PC_p2
#   3 --> information criterion PC_p3

jj = 2

# Maximum number of factors to be estimated; if set to 99, the number of
# factors selected is forced to equal 8
kmax = 8

# =========================================================================
# PART 1: LOAD AND LABEL DATA


dum = pd.read_csv(csv_in).dropna(how='all')       # Load data from CSV file

series = dum.columns.values     # Variable names
tcode = dum.iloc[0, :]          # Transformation numbers
rawdata = dum.iloc[1:, :]       # Raw data
rawdata.set_index('sasdate', inplace=True, drop=True)
rawdata.index.name = 'date'
T = len(rawdata)                # T = number of months in sample


# =========================================================================
# PART 2: PROCESS DATA

# Transform raw data to be stationary using auxiliary function & prepare_missing()
yt = pm.prepare_missing(rawdata, tcode)


# Reduce sample to usable dates: remove first two months because some
# series have been first differenced
yt = yt.iloc[2:,:]

# Remove outliers using auxiliary function remove_outliers(); see function
# or readme.txt for definition of outliers
#   data = matrix of transformed series with outliers removed
#   n = number of outliers removed from each series
#data, n = ro.remove_outliers(yt)
data = yt



 # =========================================================================
 # PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED
 #
 # Estimate factors using function factors_em()
 #   ehat    = difference between data and values of data predicted by the
 #             factors
 #   Fhat    = set of factors
 #   lamhat  = factor loadings
 #   ve2     = eigenvalues of data'*data
 #   x2      = data with missing values replaced from the EM algorithm

pred, ehat, Fhat, lamhat, ve2, x2 = fem.factors_em(data, kmax, jj, DEMEAN)


Fhat = pd.DataFrame(Fhat, index = data.index)
ehat = pd.DataFrame(ehat, index = data.index)
pred = pd.DataFrame(pred, index = data.index)

Fhat.to_excel('output/fred_factors_py.xlsx')
ehat.to_excel('output/ehat_py.xlsx')
pred.to_excel('output/pred_py.xlsx')

#  Compute R-squared and marginal R-squared from estimated factors and
#  factor loadings using function mrsq()
#    R2      = R-squared for each series for each factor
#    mR2     = marginal R-squared for each series for each factor
#    mR2_F   = marginal R-squared for each factor
#    R2_T    = total variation explained by all factors
#    t10_s   = top 10 series that load most heavily on each factor
#    t10_mR2 = marginal R-squared corresponding to top 10 series
#              that load most heavily on each factor
#
#
R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq.mrsq(Fhat,lamhat,ve2,data.columns.values)

print('R2', pd.DataFrame(R2).to_string())
print('mR2', pd.DataFrame(mR2).to_string())
print('mR2_F', mR2_F)
print('R2_T', R2_T)
print('t10_s', pd.DataFrame(t10_s).to_string())
print('t10_mR2', pd.DataFrame(t10_mR2).to_string())

