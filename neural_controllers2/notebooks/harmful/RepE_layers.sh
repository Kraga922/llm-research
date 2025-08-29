#!/bin/bash

# Ensure conda is initialized in script
eval "$(conda shell.bash hook)"


# echo "Activating nc_env_qwen..."
# conda activate nc_env_qwen
# echo "Running RepE Eval"
# python /home/ubuntu/llm-research/RepE_privacy2/examples/harmless_harmful/eval_repe.py


# Run Qwen, LLaMA, and GPT-OSS models
echo "Activating nc_env_qwen..."
conda activate nc_env_qwen
echo "Running Qwen/LLaMA/GPT-OSS models..."
python US_with_RepE_layers.py

# Run Phi models
echo "Activating phi_nc_env..."
conda activate phi_nc_env
echo "Running Phi models..."
python US_with_RepE_layers_phi.py


echo "All runs completed."
