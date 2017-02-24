#!/bin/bash

### Run locally (Dirichlet) from <hamlet_root>/experiment

export BURNIN=100
export SMOOTH=100
export RESULTS_PATH="synth16"
export DATA_PATH="synth16"
export RSCRIPT_ROOT="scripts/r/scripts" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "synth16.txt" -d $RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." --binary=1 --remove=A -g $DATA_PATH -r $PROJECT_ROOT
Rscript master_visualization.R -q "synth16_noBFact.txt" -d $RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v alpha,gamma,train_log_likelihood,test_log_likelihood -r $PROJECT_ROOT
Rscript master_visualization.R -q "synth16_LT_only.txt" -d $RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v lambda -r $PROJECT_ROOT

cd $THIS_DIR
