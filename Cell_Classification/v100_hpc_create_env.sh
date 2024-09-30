#!/bin/bash

# First, log into the A100 node and then execute the following commands
voltash << 'EOF'
module load python3/3.10.14
python3 -m venv cell_venv_exp
source cell_venv_exp/bin/activate
module load cuda/12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0,1
python src/test-gpu-pytorch.py
EOF