#!/bin/bash
#PBS -N PyTorch_Job
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=40gb:ngpus=1:gpu_mem=40gb:cuda_version=12.8
#PBS -l walltime=47:00:00
#PBS -m ae

cd $SCRATCHDIR
cp -r /storage/brno2/home/martom198/lora/* .

chmod +x ./run_finetuning.sh

singularity run -B $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:25.02-py3.SIF /storage/brno2/home/martom198/lora/run_finetuning.sh 12
# first arg is model name second arg is path to make dir to store scores
