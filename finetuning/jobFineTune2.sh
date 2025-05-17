#!/bin/bash
#PBS -N PyTorch_Job
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=40gb:ngpus=1:gpu_mem=40gb
#PBS -l walltime=12:00:00
#PBS -m ae
singularity run -B $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.11-py3.SIF /storage/brno12-cerit/home/martom198/lora/run_finetuning.sh 2 
# first arg is model name second arg is path to make dir to store scores


