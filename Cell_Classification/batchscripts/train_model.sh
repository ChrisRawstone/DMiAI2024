#!/bin/sh
#BSUB -J TrainSweep
#BSUB -o logs/Sweep_%J.out
#BSUB -e logs/Sweep__Err%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5G]"
#BSUB -W 01:00
#BSUB -N

#BSUB 
# end of BSUB options

module load cuda/11.1

source cell_venv/bin/activate

pip install -e .

python src/train_model.py
