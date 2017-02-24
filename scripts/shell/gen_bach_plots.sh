#!/bin/bash

## Run on venti from <hamlet_root>/experiment

export BURNIN=5000
export LOW_BURNIN=300
export SMOOTH=200
export RESULTS_PATH="music/bach_nominal/lambda_epsilon/bach_major_01/"
export RSCRIPT_ROOT="scripts/r/scripts" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../../data/"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "bach_icml.txt" -d $RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v train_log_likelihood,test_log_likelihood,n_dot,A -r $PROJECT_ROOT --max_iter=10000

cd $THIS_DIR
