# Finetuning

We finetune an MPT 7B model using LoRA.

**Before you start job set chmod +x run_finetuning.sh**

Usefull metacentrum commands:
```
Run finetuning job:
qsub jobFineTune.sh

Get state of your jobs:
qstat -u userName

Delete job after submission:
qdel 10622914.pbs-m1.metacentrum.cz
```