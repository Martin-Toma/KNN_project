#!/bin/bash

pip install transformers
pip install datasets
pip install accelerate
pip install einops
pip install loralib
pip install trl
pip install -m bitsandbytes #==0.39.1
pip install peft
pip install flash-attn --no-build-isolation
#git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention /src/flash-attention
#cd /src/flash-attention && pip install . 
#cd /src/flash-attention; find csrc/ -type d -exec sh -c 'cd {} && pip install . && cd ../../' \;

# move into scratch directory
cd $SCRATCHDIR
#python -m bitsandbytes
if [ $1 == "1" ]
then
    # learning rate 5e-5 
    python /storage/brno12-cerit/home/martom198/lora/finetune_lora.py setting1 8 16 0.1 > /storage/brno12-cerit/home/martom198/lora/logs/log1.txt
elif [ $1 == "2" ]
then
    # learning rate 5e-4, batch 256
    python /storage/brno12-cerit/home/martom198/lora/finetune_lora.py setting2 8 16 0.1 > /storage/brno12-cerit/home/martom198/lora/logs/log2.txt
elif [ $1 == "3" ]
then
    # same as 2nd but different r and alpha
    # TODO alpha set
    python /storage/brno12-cerit/home/martom198/lora/finetune_lora.py setting3 32 16 0.1 > /storage/brno12-cerit/home/martom198/lora/logs/log3.txt
elif [ $1 == "4" ]
then
    # same as 2nd but different r and alpha
    #TODO alpha set
    python /storage/brno12-cerit/home/martom198/lora/finetune_lora.py setting4 128 16 0.1 > /storage/brno12-cerit/home/martom198/lora/logs/log4.txt
fi
# move the output to user's DATADIR or exit in case of failure
#cp $SCRATCHDIR/* $OUTDIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
