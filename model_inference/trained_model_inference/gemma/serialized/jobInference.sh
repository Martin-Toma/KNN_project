#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=12:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=40gb:scratch_local=40gb:cuda_version=12.8
#PBS -N cool_job
singularity run -B $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF /storage/brno2/home/xcerve30/test/run_inference.sh