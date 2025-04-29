#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=5:0:0
#PBS -l select=1:ncpus=1:mem=50gb:scratch_local=50gb
#PBS -N Tokenize
singularity run -B $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF /storage/brno2/home/martom198/lora/run_tokenization.sh
# first arg is model name second arg is path to make dir to store scores old PyTorch\:23.11-py3.SIF 