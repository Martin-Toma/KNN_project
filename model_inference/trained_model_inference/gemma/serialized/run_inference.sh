#!/bin/bash
cd $SCRATCHDIR

# 1) disable pip cache or redirect it:
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache
export TMPDIR=$SCRATCHDIR/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

# 2) (optional) use virtualenv:
python -m venv venv
source venv/bin/activate

pip install --no-cache-dir transformers datasets accelerate einops loralib trl bitsandbytes peft sentencepiece jinja2 protobuf

# CUDA setup (as before)
mkdir -p $HOME/cuda_libs
ln -s /usr/local/cuda/compat/lib/libcuda.so.1 $HOME/cuda_libs/libcuda.so
export LD_LIBRARY_PATH=$HOME/cuda_libs:$LD_LIBRARY_PATH
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib/libcuda.so.1

python /storage/brno2/home/xcerve30/test/inference_tokenized.py
