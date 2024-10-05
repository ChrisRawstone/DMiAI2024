#!/bin/sh

### select queue s
#BSUB -q gpuv100

### name of job, output file and err
#BSUB -J CT-inpainting
#BSUB -o inpainting_%J.out
#BSUB -e inpainting_%J.err


### number of cores
#BSUB -n 8
# request cpu
#BSUB -R "rusage[mem=16G]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

# request 32GB of GPU-memory
#BSUB -R "select[gpu32gb]"

### wall time limit - the maximum time the job will run. Currently 5 hours, 30 min. 

#BSUB -W 10:30

##BSUB -u s204090@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 


# end of BSUB options


# load the correct  scipy module and python

module load cuda/11.1
source CT_Inpainting/CT_venv/bin/activate

python CT_Inpainting/src/train_model.py --config-name test_3