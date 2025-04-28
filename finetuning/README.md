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

MPT 7B requires to install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python even after trying to install it by any means or to avoid the installation and using different attention implementation, the triton pre mlir is still required and makes impossible to run this model. On metacentrum we have limited permissions, which further complicates the fixing, so we are forced to use different model.


Here are maximum sizes of response part: 
- eval 2375 words, 
- test 1317 words, 
- train 5007 words