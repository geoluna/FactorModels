README

This is a python implementation of McCracken & Ng (2017) Matlab code which is used to
estimate factor models and make predictions on the basis of FRED-MD (monthly)
and FRED-QD (quarterly) macroeconomic databases.

For details regarding the data, and the original Matlab codes, see
http://research.stlouisfed.org/econ/mccracken/fred-databases/

The code loads in the data, transforms each series to be stationary,
removes outliers, estimates factors, and computes the R-squared and
marginal R-squared values from the estimated factors and factor loadings.

=================================================== <br/>
List of files:

1. fredfactors.py - Performs all the tasks mentioned above using the auxiliary functions described below

2. prepare_missing.py - Transforms the raw data into stationary form

3. remove_outliers.py - Removes outliers from the data. A data point x is considered an outlier if |x-median|>10*interquartile_range.

4. factors_em.py - Estimates a set of factors for a given dataset using principal component analysis.
    The number of factors estimated is determined by an information criterion specified by the user.
    Missing values in the original dataset are handled using an iterative
    expectation-maximization algorithm.

5. mrsq.py - Computes the R-squared and marginal R-squared values from estimated factors and factor loadings.
=================================================== <br/>

* prepare_missing -> transforms data according to the rules given in the first row of the data spreadsheet
* remove outliners -> set outliers to na -> still missing observations
* factors_em
    -> first set missing values to unconditional mean
        a) transform_data -> standardise based on DEMEAN method (pandas -> numpy)
        b) baing -> compute the number of factors (numpy <-> numpy)
        c) pc2 -> compute factors & make a prediction

=================================================== <br/>
Code ported to python 3 by George Milunovich <br/> 
george.milunovich@mq.edu.au
=================================================== <br/>