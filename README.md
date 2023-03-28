# KSE-thesis-tvp-var

This file describes code and data used to replicate main results from the master's thesis:
'*Horovoi, M. 2022. Does The Monetary Transmission Mechanism Change Over Time In Ukraine? Kyiv School of Economics.*'  
Full text is available via [the link](https://kse.ua/wp-content/uploads/2022/11/Horovoi_thesis_final.pdf).

All required files are stored in two folders:

1. dic_estimation_matlab — Matlab package to estimate DIC from Table 2
2. tvp_var_estimation_r — Data and R code to replicate main findings from the thesis

## Content description

### 1. dic_estimation_matlab

Folder contains Matlab code for replicating the empirical examples in Chan and Eisenstat (2018).  
The code is modified by Mykyta Horovoi (me) to estimate TVP-VAR with Primiceri priors (OLS priors).

Run file main_tvpsv.m to calculate DIC for the model used in thesis.

### 2. tvp_var_estimation_r

Folder contains following files:

- Model_main.R — script that replicates a baseline impulse response analysis.
- Model_mean_shock — complementary model with shocks fixed at sample means.
- Model_rob_prior.R — complementary model with hyperparemeters specification after Primiceri (2005).
- Model_rob_time.R — complementary model on data sample from 2016M5 to 2022M1.
- Data.xlsx — data on Ukrainian macroeconomic series from 2015M1 to 2022M1. The subsample from 2016M1 to 2022M1 is used in the thesis.
