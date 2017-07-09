#!/bin/bash

## Run on venti from <hamlet_root>/experiment

export BURNIN=1000
export LOW_BURNIN=300
export SMOOTH=100
export RESULTS_PATH="continuous_latent_syn_diag40_4models/block_diag40_s2"
export RSCRIPT_ROOT="scripts/r/scripts" #assume this will be run from <hamlet_root>/experiment
export PROJECT_ROOT="../../../../"  #relative to RSCRIPT_ROOT
export THIS_DIR=$(pwd)

cd $THIS_DIR/$RSCRIPT_ROOT

Rscript master_visualization.R -q "block_diag.txt" -d $COCKTAIL_RESULTS_PATH -s $SMOOTH -b $BURNIN -p "." -v train_log_likelihood,test_log_likelihood,A -r $PROJECT_ROOT

cd $THIS_DIR
