#!/bin/bash
#PBS -N PyTorch_Job
#PBS -q gpu_long@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=40gb:ngpus=1:gpu_mem=40gb:cuda_version=12.0
#PBS -l walltime=70:00:00
#PBS -m ae
singularity run -B $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.11-py3.SIF /storage/brno2/home/martom198/lora/run_finetuning.sh 2
# first arg is model name second arg is path to make dir to store scores