#!/bin/bash

# Ensure conda is initialized in script
eval "$(conda shell.bash hook)"


echo "Activating RepE environment"
conda activate nc_env_qwen
echo "Running RepE Eval"
python /home/ubuntu/llm-research/RepE_privacy2/examples/harmless_harmful/eval_repe_coef.py