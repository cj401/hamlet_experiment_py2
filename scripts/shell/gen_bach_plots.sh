#!/bin/bash

## Run on venti from <hamlet_root>/experiment

export BURNIN=5000
export LOW_BURNIN=300
export SMOOTH=50
export BACH_PATH="music/bach_nominal/lambda_epsilon/bach_major_01/"
export CHORD1_PATH="music/chord1/"
export RSCRIPT_ROOT="scripts/r/scripts" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../../data/"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "bach_icml.txt" -d $BACH_PATH -s $SMOOTH -b $BURNIN -p "." -v train_log_likelihood,test_log_likelihood,n_dot,alpha,gamma,A -r $PROJECT_ROOT --max_iter=10000
Rscript master_visualization.R -q "chord1.txt" -d $CHORD1_PATH -s $SMOOTH -b $BURNIN -p "." -v train_log_likelihood,test_log_likelihood,n_dot,alpha,gamma,A -r $PROJECT_ROOT --max_iter=100000
Rscript master_visualization.R -q "chord1_LT_only.txt" -d $CHORD1_PATH -s $SMOOTH -b $LOW_BURNIN -p "." -v lambda -r $PROJECT_ROOT --max_iter=100000
Rscript master_visualization.R -q "bach_lambda_comparison.txt" -d $BACH_PATH -s $SMOOTH -b $BURNIN -p "." -v train_log_likelihood,test_log_likelihood,n_dot,alpha,gamma,A -r $PROJECT_ROOT --max_iter=10000

cd $THIS_DIR
