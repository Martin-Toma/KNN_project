#!/bin/bash

pip install transformers
pip install datasets
pip install accelerate
pip install einops
pip install loralib
pip install trl
pip install bitsandbytes
pip install peft
pip install flash-attn --no-build-isolation
pip install sentencepiece

mkdir -p $HOME/cuda_libs
ln -s /usr/local/cuda/compat/lib/libcuda.so.1 $HOME/cuda_libs/libcuda.so
export LD_LIBRARY_PATH=$HOME/cuda_libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/cuda_libs:$LD_LIBRARY_PATH
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib/libcuda.so.1

#pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
#pip install triton_pre_mlir
#pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
#pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

#git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention /src/flash-attention
#cd /src/flash-attention && pip install . 
#cd /src/flash-attention; find csrc/ -type d -exec sh -c 'cd {} && pip install . && cd ../../' \;

# move into scratch directory
cd $SCRATCHDIR

python /storage/brno2/home/martom198/lora/pretokenize.py > /storage/brno2/home/martom198/lora/logs/log2.txt