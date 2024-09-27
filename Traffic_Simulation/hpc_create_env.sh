#!/bin/bash

# First, log into the A100 node and then execute the following commands
# a100sh << 'EOF'
module load python3/3.10.14
python3 -m venv traffic_env
source traffic_env/bin/activate
pip install -r requirements.txt
# EOF

