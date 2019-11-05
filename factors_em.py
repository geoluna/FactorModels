import pandas as pd
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

def transform_data(X2,DEMEAN):
    # take in pandas <-> return numpy
    '''
     =========================================================================
     DESCRIPTION
     This function transforms a given set of series based upon the input
     variable DEMEAN. The following transformations are possible:

       1) No transformation.

       2) Each series is demeaned only (i.e. each series is rescaled to have a
       mean of 0).

       3) Each series is demeaned and standardized (i.e. each series is
       rescaled to have a mean of 0 and a standard deviation of 1).

       4) Each series is recursively demeaned and then standardized. For a
       given series x(t), where t=1,...,T, the recursively demeaned series
       x'(t) is calculated as x'(t) = x(t) - mean(x(1:t)). After the
       recursively demeaned series x'(t) is calculated, it is standardized by
       dividing x'(t) by the standard deviation of the original series x. Note
       that this transformation does not rescale the original series to have a
       specified mean or standard deviation.

     -------------------------------------------------------------------------
     INPUTS
               X2      = set of series to be transformed (one series per
                         column); no missing values;
               DEMEAN  = an integer indicating the type of transformation
                         performed on each series in x2; it can take on the
                         following values:
                               0 (no transformation)
                               1 (demean only)
                               2 (demean and standardize)
                               3 (recursively demean and then standardize)

     OUTPUTS
               X22     = transformed dataset
               mut     = matrix containing the values subtracted from X2
                         during the transformation
               sdt     = matrix containing the values that X2 was divided by
                         during the transformation

     -------------------------------------------------------------------------
    '''
    assert DEMEAN in [0, 1, 2, 3], 'DEMEAN value incorrectly set, must be in [0, 1, 2, 3]'
    mut = X2 * 0        # initialize values at no tranformation, i.e. DEMEAN = 0
    std = X2 * 0 + 1

    if DEMEAN == 1:   # Each series is demeaned only
        mut = X2*0 + X2.mean()

    elif DEMEAN == 2:   # Each series is demeaned and standardized
        mut = X2 * 0 + X2.mean()
        std = X2 * 0 + X2.std()

    elif DEMEAN == 3:   # Each series is recursively demeaned and then standardized
        for t in range(0, len(X2)):
            mut.loc[X2.index[t], X2.columns] = X2.iloc[:t+1, :].mean()
        std = X2 * 0 + X2.std()

    X22 = (X2 - mut) / std

    return X22.values, mut.values, std.values



def minindc(X):
    ''' =========================================================================
     takes np <-> returns np
     DESCRIPTION
     This function finds the index of the minimum value for each column of a
     given matrix. The function assumes that the minimum value of each column
     occurs only once within that column. The function returns an error if
     this is not the case.

     -------------------------------------------------------------------------
     INPUT
               x   = matrix

     OUTPUT
               pos = column vector with pos(i) containing the row number
                     corresponding to the minimum value of x(:,i)

     ========================================================================= '''

    mins = X.argmin(axis=0)
    assert sum(X == X[mins]) == 1, 'Minimum value occurs more than once.'
    return mins



def pc2(X,nfac):
    '''' =========================================================================
     DESCRIPTION
     This function runs principal component analysis.

     -------------------------------------------------------------------------
     INPUTS
               X      = dataset (one series per column)
               nfac   = number of factors to be selected

     OUTPUTS
               chat  f = values of X predicted by the factors
               fhat   = factors scaled by (1/sqrt(N)) where N is the number of
                        series
               lambda = factor loadings scaled by number of series
               ss     = eigenvalues of X'*X

     ========================================================================= '''

    N = X.shape[1]  # Number of series in X (i.e. number of columns)
    # The rows of vh are the eigenvectors of A'A and the columns of u are the eigenvectors of AA'.
    # In both cases the corresponding (possibly non-zero) eigenvalues are given by s**2.
    U, S, Vh = np.linalg.svd(X.T@X) # Singular value decomposition: X'*X = U*S*V where V=U'

    lambda_ = U[:, :nfac]*np.sqrt(N)   # Factor loadings scaled by sqrt(N)
    fhat = np.dot(X, lambda_)*(1/N)  # Factors scaled by 1/sqrt(N) (note that lambda is scaled by sqrt(N))
    chat = np.dot(fhat, lambda_.T) # Estimate initial dataset X using the factors (note that U'=inv(U))
    ss = S                          #  a vector of singular values of X'*X, eigenvalues are ss**2??

    return chat, fhat, lambda_, ss




def baing(X,kmax,jj):
    #take in and return numpy arrays
    ''' =========================================================================
    DESCRIPTION
    This function determines the number of factors to be selected for a given
    dataset using one of three information criteria specified by the user.
    The user also specifies the maximum number of factors to be selected.

    -------------------------------------------------------------------------
    INPUTS
               X       = dataset (one series per column)
               kmax    = an integer indicating the maximum number of factors
                         to be estimated
               jj      = an integer indicating the information criterion used
                         for selecting the number of factors; it can take on
                         the following values:
                               1 (information criterion PC_p1)
                               2 (information criterion PC_p2)
                               3 (information criterion PC_p3)

     OUTPUTS
               ic1     = number of factors selected
               chat    = values of X predicted by the factors
               Fhat    = factors
               eigval  = eivenvalues of X'*X (or X*X' if N>T)

     -------------------------------------------------------------------------
     SUBFUNCTIONS USED

     minindc() - finds the index of the minimum value for each column of a given matrix

     -------------------------------------------------------------------------
     BREAKDOWN OF THE FUNCTION

     Part 1: Setup.

     Part 2: Calculate the overfitting penalty for each possible number of
             factors to be selected (from 1 to kmax).

     Part 3: Select the number of factors that minimizes the specified
             information criterion by utilizing the overfitting penalties calculated in Part 2.

     Part 4: Save other output variables to be returned by the function (chat,
             Fhat, and eigval).

    ========================================================================= '''
    assert kmax <= X.shape[1] and  kmax >= 1 and np.floor(kmax) == kmax or kmax == 99, 'kmax is specified incorrectly'
    assert jj in [1, 2, 3], 'jj is specified incorrectly'


    #  PART 1: SETUP

    T = X.shape[0]  # Number of observations per series (i.e. number of rows)
    N = X.shape[1]  # Number of series (i.e. number of columns)
    NT = N * T      # Total number of observations
    NT1 = N + T     # Number of rows + columns

    #  =========================================================================
    #  PART 2: OVERFITTING PENALTY
    #  Determine penalty for overfitting based on the selected information
    #  criterion.

    CT = np.zeros(kmax) # overfitting penalty
    ii = np.arange(1, kmax + 1)     # Array containing possible number of factors that can be selected (1 to kmax)
    GCT = min(N,T)                  # The smaller of N and T

    # Calculate penalty based on criterion determined by jj.
    if jj == 1:             # Criterion PC_p1
        CT[:] = np.log(NT / NT1) * ii * (NT1 / NT)

    elif jj == 2:             # Criterion PC_p2
        CT[:] = np.log(min(N, T)) * ii * (NT1 / NT)

    elif jj == 3:             # Criterion PC_p3
        CT[:] = np.log(GCT) / GCT * ii

    #  =========================================================================
    #  PART 3: SELECT NUMBER OF FACTORS
    #  Perform principal component analysis on the dataset and select the number
    #  of factors that minimizes the specified information criterion.
    #
    #  -------------------------------------------------------------------------
    #  RUN PRINCIPAL COMPONENT ANALYSIS
    #  Get components, loadings, and eigenvalues

    if T < N:
        ev, eigval, V = np.linalg.svd(np.dot(X, X.T))       #  Singular value decomposition
        Fhat0 = ev*np.sqrt(T)                               #  Components
        Lambda0 = np.dot(X.T, Fhat0) / T                    #  Loadings
    else:
        ev, eigval, V = np.linalg.svd(np.dot(X.T, X))       #  Singular value decomposition
        Lambda0 = ev*np.sqrt(N)                             #  Loadings
        Fhat0 = np.dot(X, Lambda0) / N                      #  Components
    #  -------------------------------------------------------------------------

    # SELECT NUMBER OF FACTORS
    # Preallocate memory
    Sigma = np.zeros(kmax + 1)          # sum of squared residuals divided by NT, kmax factors + no factor
    IC1 = np.zeros(kmax + 1)            # information criterion value, kmax factors + no factor

    for i in range(0, kmax) :           # Loop through all possibilites for the number of factors
        Fhat = Fhat0[:, :i+1]           # Identify factors as first i components
        lambda_ = Lambda0[:, :i+1]       #     % Identify factor loadings as first i loadings

        chat = np.dot(Fhat, lambda_.T)      #     % Predict X using i factors
        ehat = X - chat                 # Residuals from predicting X using the factors
        Sigma[i] = ((ehat*ehat/T).sum(axis = 0)).mean()    # Sum of squared residuals divided by NT

        IC1[i] = np.log(Sigma[i]) + CT[i]      #  Value of the information criterion when using i factors


    Sigma[kmax] = (X*X/T).sum(axis = 0).mean()  # Sum of squared residuals when using no factors to predict X (i.e. fitted values are set to 0)

    IC1[kmax] =  np.log(Sigma[kmax]) # % Value of the information criterion when using no factors

    ic1 = minindc(IC1) # % Number of factors that minimizes the information criterion
    # Set ic1=0 if ic1>kmax (i.e. no factors are selected if the value of the
    # information criterion is minimized when no factors are used)
    ic1 = ic1 *(ic1 < kmax) # if = kmax -> 0

    #  =========================================================================
    #  PART 4: SAVE OTHER OUTPUT
    #
    #  Factors and loadings when number of factors set to kmax

    Fhat = Fhat0[:, :kmax] # factors
    Lambda = Lambda0[:, :kmax] #factor loadings

    chat = np.dot(Fhat, Lambda.T) #     Predict X using kmax factors

    return ic1+1, chat, Fhat, eigval
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def factors_em(X, kmax, jj, DEMEAN):
    ''' =========================================================================
     DESCRIPTION
     This program estimates a set of factors for a given dataset using
     principal component analysis. The number of factors estimated is
     determined by an information criterion specified by the user. Missing
     values in the original dataset are handled using an iterative
     expectation-maximization (EM) algorithm.

     -------------------------------------------------------------------------
     INPUTS
               x       = dataset (one series per column)
               kmax    = an integer indicating the maximum number of factors
                         to be estimated; if set to 99, the number of factors
                         selected is forced to equal 8
               jj      = an integer indicating the information criterion used 
                         for selecting the number of factors; it can take on 
                         the following values:
                               1 (information criterion PC_p1)
                               2 (information criterion PC_p2)
                               3 (information criterion PC_p3)      
               DEMEAN  = an integer indicating the type of transformation
                         performed on each series in x before the factors are
                         estimated; it can take on the following values:
                               0 (no transformation)
                               1 (demean only)
                               2 (demean and standardize)
                               3 (recursively demean and then standardize) 
    
     OUTPUTS
               ehat    = difference between x and values of x predicted by
                         the factors
               Fhat    = set of factors
               lamhat  = factor loadings
               ve2     = eigenvalues of x3'*x3 (where x3 is the dataset x post
                         transformation and with missing values filled in)
               x2      = x with missing values replaced from the EM algorithm
    
     -------------------------------------------------------------------------
     SUBFUNCTIONS
    
     baing() - selects number of factors
     pc2() - runs principal component analysis
     minindc() - finds the index of the minimum value for each column of a
           given matrix
     transform_data() - performs data transformation
    
     -------------------------------------------------------------------------'''

    # BREAKDOWN OF THE FUNCTION
    # Part 1: Check that inputs are specified correctly.
    # Part 2: Setup.
    # Part 3: Initialize the EM algorithm -- fill in missing values with
    #         unconditional mean and estimate factors using the updated
    #         dataset.

    # Part 4: Perform the EM algorithm -- update missing values using factors,
    #         construct a new set of factors from the updated dataset, and
    #         repeat until the factor estimates do not change.
    #
    # -------------------------------------------------------------------------

    # Details for the three possible information criteria can be found in the
    # paper "Determining the Number of Factors in Approximate Factor Models" by
    # Bai and Ng (2002).

    # The EM algorithm is essentially the one given in the paper "Macroeconomic
    # Forecasting Using Diffusion Indexes" by Stock and Watson (2002). The
    # algorithm is initialized by filling in missing values with the
    # unconditional mean of the series, demeaning and standardizing the updated
    # dataset, estimating factors from this demeaned and standardized dataset,
    # and then using these factors to predict the dataset. The algorithm then
    # proceeds as follows: update missing values using values predicted by the
    # latest set of factors, demean and standardize the updated dataset,
    # estimate a new set of factors using the demeaned and standardized updated
    # dataset, and repeat the process until the factor estimates do not change.

    # =========================================================================
    # PART 1: CHECKS

    # Check that x is not missing values for an entire row
    assert (X.isna().sum(axis=1) == X.shape[1]).sum() == 0, 'X contains entire rows of missing values'

    # Check that x is not missing values for an entire column
    assert (X.isna().sum(axis=0) == X.shape[0]).sum() == 0, 'X contains entire columns of missing values'

    # Check that kmax is an integer between 1 and the number of columns of x, or 99
    assert kmax <= X.shape[1] and  kmax >= 1 and np.floor(kmax) == kmax or kmax == 99, 'kmax is specified incorrectly'

    # Check that jj is one of 1, 2, 3
    assert jj in [1, 2, 3], 'jj is specified incorrectly'

    # Check that DEMEAN is one of 0, 1, 2, 3
    assert DEMEAN in [0, 1, 2, 3], 'DEMEAN value incorrectly set, must be in [0, 1, 2, 3]'

    # =========================================================================
    # PART 2: SETUP


    maxit = 50          # Maximum number of iterations for the EM algorithm
    T = X.shape[0]      # Number of observations per series in x (i.e. number of rows)
    N = X.shape[1]      # Number of series in x (i.e. number of columns)


    err = 99999         # Set error to arbitrarily high number
    it = 0              # Set iteration counter to 0
    X1 = X.isna()       # Locate missing values in x

    #  =========================================================================
    #  PART 3: INITIALIZE EM ALGORITHM
    #  Fill in missing values for each series with the unconditional mean of
    #  that series. Demean and standardize the updated dataset. Estimate factors
    #  using the demeaned and standardized dataset, and use these factors to
    #  predict the original dataset.
    #  Get unconditional mean of the non-missing values of each series



    mut = (X*0).fillna(0) + X.mean(axis = 0)            # Get unconditional mean of the non-missing values of each series
                                                        # mut has no missing values (na)
    X2 = X.fillna(mut)                                  # Replace missing values with unconditional mean

    #  Demean and standardize data using subfunction transform_data()
    #    x3  = transformed dataset
    #    mut = matrix containing the values subtracted from x2 during the
    #         transformation, TYPE OF DEMEANING USED - CAN BE SIMPLE COLUMN MEAN
    #    sdt = matrix containing the values that x2 was divided by during the
    #          transformation

    X3, mut, std = transform_data(X2, DEMEAN)       # these are numpy arrays, DEMEAN = 2
    #  If input 'kmax' is not set to 99, use subfunction baing() to determine
    #  the number of factors to estimate. Otherwise, set number of factors equal to 8
    if kmax != 99:
        icstar, _, _, _  = baing(X3, kmax, jj)
    else:
        icstar = 8


    # Run principal components on updated dataset using subfunction pc2()
    #    chat   = values of x3 predicted by the factors
    #    Fhat   = factors scaled by (1/sqrt(N)) where N is the number of series
    #    lamhat = factor loadings scaled by number of series
    #    ve2    = eigenvalues of x3'*x3

    chat, Fhat, lamhat, ve2  = pc2(X3, icstar)
    chat0 = chat        # Save predicted series values

    #  =========================================================================
    # PART 4: PERFORM EM ALGORITHM
    # Update missing values using values predicted by the latest set of
    # factors. Demean and standardize the updated dataset. Estimate a new set
    # of factors using the updated dataset. Repeat the process until the factor
    # estimates do not change.
    #
    # Run while error is large and have yet to exceed maximum number of
    # iterations

    while (err > 0.000001) and (it < maxit):

    #    ---------------------------------------------------------------------
        it += 1             #     Increase iteration counter by 1
        print(f'Iteration {it}: obj {err} IC {icstar} \n')      #     Display iteration counter, error, and number of factors

    #      ---------------------------------------------------------------------
    #     UPDATE MISSING VALUES
    #     Replace missing observations with latest values predicted by the
    #     factors (after undoing any transformation)

        temp = X.fillna(0)*0 + chat*std + mut  # temp must not have na's in the df as it will keep them
        X2 = X.fillna(temp)

    #     ---------------------------------------------------------------------
    #     ESTIMATE FACTORS
    #     Demean/standardize new dataset and recalculate mut and sdt using
    #     subfunction transform_data()
    #       x3  = transformed dataset
    #       mut = matrix containing the values subtracted from x2 during the
    #             transformation
    #        sdt = matrix containing the values that x2 was divided by during
    #              the transformation

        X3, mut, sdt = transform_data(X2, DEMEAN)

    #     Determine number of factors to estimate for the new dataset using
    #     subfunction baing() (or set to 8 if kmax equals 99)

        if kmax != 99:
            icstar, _, _, _ = baing(X3, kmax, jj)
        else:
            icstar = 8

    #  Run principal components on the new dataset using subfunction pc2()
    #        chat   = values of x3 predicted by the factors
    #        Fhat   = factors scaled by (1/sqrt(N)) where N is the number of
    #                 series
    #        lamhat = factor loadings scaled by number of series
    #        ve2    = eigenvalues of x3'*x3

        chat, Fhat, lamhat, ve2  = pc2(X3, icstar)
    #     ---------------------------------------------------------------------
    #     CALCULATE NEW ERROR VALUE
    #     Calculate difference between the predicted values of the new dataset
    #     and the predicted values of the previous dataset
        diff = chat - chat0
    #     The error value is equal to the sum of the squared differences
    #     between chat and chat0 divided by the sum of the squared values of chat0

        v1 = diff.flatten(order = 'F')  # vectorise columns
        v2 = chat0.flatten(order = 'F')

        err = (np.dot(v1.T, v1) / np.dot(v2.T, v2))
        chat0 = chat     #   Set chat0 equal to the current chat

        if it == maxit:                     #  Produce warning if maximum number of iterations is reached
            print('Maximum number of iterations reached in EM algorithm')

    #  -------------------------------------------------------------------------
    #  FINAL DIFFERENCE
    #  Calculate the difference between the initial dataset and the values
    #  predicted by the final set of factors
    pred = chat*sdt + mut
    ehat = X - pred
    # ehat = X - chat*sdt + mut

    return pred, ehat, Fhat, lamhat, ve2, X2


# if __name__ == "__main__":
#     X = pd.read_csv('../../data/2019-07-transformed-removed-outliers.csv', index_col=0)     # read in data
#     kmax = 7
#     jj = 2
#     DEMEAN = 2
