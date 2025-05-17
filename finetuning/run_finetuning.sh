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

# move into scratch directory
cd $SCRATCHDIR
#python -m bitsandbytes
if [ $1 == "1" ]
then
    # learning rate 5e-5
    python /storage/brno2/home/martom198/lora/finetune_lora.py setting1 8 16 0.1 > /storage/brno2/home/martom198/lora/logs/log1.txt
elif [ $1 == "2" ]
then
    # learning rate 5e-4, batch 256
    python /storage/brno2/home/martom198/lora/finetune_lora.py setting2instruct 64 128 0.1 > /storage/brno2/home/martom198/lora/logs/log2.txt
elif [ $1 == "3" ]
then
    # same as 2nd but different r and alpha
    # TODO alpha set
    python /storage/brno2/home/martom198/lora/finetune_lora2.py setting32r 32 64 0.1 > /storage/brno2/home/martom198/lora/logs/log3.txt
elif [ $1 == "4" ]
then
    # same as 2nd but different r and alpha
    #TODO alpha set
    python /storage/brno2/home/martom198/lora/finetune_lora3v4.py setting4v3 16 32 0.1 > /storage/brno2/home/martom198/lora/logs/log4v4.txt
elif [ $1 == "5" ]
then
    # same as 2nd but different r and alpha
    #TODO alpha set
    python /storage/brno2/home/martom198/lora/finetune_lora4.py setting5 128 256 0.1 > /storage/brno2/home/martom198/lora/logs/log5.txt
elif [ $1 == "6" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora6.py settingGemmaQlorar32a64 32 64 0.1 > /storage/brno2/home/martom198/lora/logs/log6qlora.txt
elif [ $1 == "7" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora7.py settingMI 64 128 0.1 > /storage/brno2/home/martom198/lora/logs/log7.txt
elif [ $1 == "8" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora8.py settingMI2 8 16 0.1 > /storage/brno2/home/martom198/lora/logs/log8.txt
elif [ $1 == "9" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora9.py settingM3_r16_a32 16 32 0.1 > /storage/brno2/home/martom198/lora/logs/log9.txt
elif [ $1 == "10" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora10.py settingGemmar32A32 32 32 0.1 > /storage/brno2/home/martom198/lora/logs/log10r32a32.txt
elif [ $1 == "11" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora10.py settingGemmar32A16 32 16 0.1 > /storage/brno2/home/martom198/lora/logs/log10r32a16.txt
elif [ $1 == "12" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora10.py settingGemmar16A32 16 32 0.1 > /storage/brno2/home/martom198/lora/logs/log10r16a32.txt
elif [ $1 == "13" ]
then
    python /storage/brno2/home/martom198/lora/finetune_lora10.py settingGemmar8A16 8 16 0.1 > /storage/brno2/home/martom198/lora/logs/log10r8a16.txt
fi
# move the output to user's DATADIR or exit in case of failure
#cp $SCRATCHDIR/* $OUTDIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
