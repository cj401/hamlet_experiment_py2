#!/bin/bash

BURNIN=1000
SMOOTH=100
COCKTAIL_RESULTS_PATH="cocktail_s16_m12/hyper_alpha/h10.0_nocs_cp0"
COCKTAIL_DATA_PATH="cocktail_s16_m12/h10.0_nocs/cp0"
PROJECT_ROOT="../data"  #assume this will be run from <hamlet_root>/experiment

Rscript master_visualization.R -q "cocktail_icml.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." --binary=1 --remove=A -g $COCKTAIL_DATA_PATH
Rscript master_visualization.R -q "cocktail_icml_noBFact.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v alpha,gamma,train_log_likelihood,test_log_likelihood
Rscript master_visualization.R -q "cocktail_icml_LT_only.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v lambda
