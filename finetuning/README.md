# Finetuning

This directory contains all of the scripts used for fine-tuning the models. 

## Note
 All experiments work with CUDA version 12.8 and singularity NGC/PyTorch\:25.02-py3.SIF which further requires the installation of the following libraries:

pip install transformers
pip install datasets
pip install accelerate
pip install einops
pip install flash-attn --no-build-isolation
pip install sentencepiece

pip install loralib
pip install trl
pip install peft

pip install bitsandbytes

