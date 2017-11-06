#!/bin/bash

## Run on venti from <hamlet_root>/experiment

export DATASET=$1
export BURNIN=2500
export LOW_BURNIN=1000
export SMOOTH=50
export SSC_RESULTS="cocktail_s16_m12/h10.0_nocs_$DATASET/"
export SSC_DATA="cocktail_s16_m12/h10.0_nocs/$DATASET/"
export RSCRIPT_ROOT="scripts/r/scripts/" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../../data/"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "cocktail_regression.txt" -d $SSC_RESULTS -s $SMOOTH -b $BURNIN -p "." --binary=1 --remove=A -g $SSC_DATA -r $PROJECT_ROOT
# Rscript master_visualization.R -q "cocktail_icml_noBFact.txt" -d $SSC_RESULTS -s $SMOOTH -b $BURNIN -p "." -v alpha,gamma,train_log_likelihood,test_log_likelihood,n_dot -r $PROJECT_ROOT
# Rscript master_visualization.R -q "cocktail_icml_LT_only.txt" -d $SSC_DATA -s $SMOOTH -b $LOW_BURNIN -p "." -v lambda -r $PROJECT_ROOT

cd $THIS_DIR
