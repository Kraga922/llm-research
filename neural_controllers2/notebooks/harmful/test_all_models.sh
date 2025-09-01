#!/bin/bash

# Ensure conda is initialized in script
eval "$(conda shell.bash hook)"


echo "Activating nc_env_qwen..."
conda activate nc_env_qwen
echo "Running RepE Eval"
python /home/ubuntu/llm-research/RepE_privacy2/examples/harmless_harmful/eval_repe100.py


# Run Qwen, LLaMA, and GPT-OSS models
echo "Activating nc_env_qwen..."
conda activate nc_env_qwen
echo "Running US_with_RepE_layers.py"
python US_with_RepE_layers.py

# Run Phi models
echo "Activating phi_nc_env..."
conda activate phi_nc_env
echo "Running US_with_RepE_layers_phi.py"
python US_with_RepE_layers_phi.py

# Run Qwen, LLaMA, and GPT-OSS models
echo "Activating nc_env_qwen..."
conda activate nc_env_qwen
echo "Running US"
python eval.py

# Run Phi models
echo "Activating phi_nc_env..."
conda activate phi_nc_env
echo "Running US for Phi"
python eval2.py



echo "All runs completed."
