#!/bin/bash

## Run on venti from <hamlet_root>/experiment

export BURNIN=1000
export LOW_BURNIN=300
export SMOOTH=1
export COCKTAIL_RESULTS_PATH="cocktail_s16_m12/hyper_alpha/h10.0_nocs_cp0"
export COCKTAIL_DATA_PATH="cocktail_s16_m12/h10.0_nocs/cp0"
export RSCRIPT_ROOT="scripts/r/scripts" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../../data"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "cocktail_icml.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." --binary=1 --remove=A -g $COCKTAIL_DATA_PATH
Rscript master_visualization.R -q "cocktail_icml_noBFact.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v alpha,gamma,train_log_likelihood,test_log_likelihood
Rscript master_visualization.R -q "cocktail_icml_LT_only.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $LOW_BURNIN -p "." -v lambda

cd $THIS_DIR
