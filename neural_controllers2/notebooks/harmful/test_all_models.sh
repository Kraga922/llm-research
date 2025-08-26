#!/bin/bash

# Ensure conda is initialized in script
eval "$(conda shell.bash hook)"

# Run Qwen, LLaMA, and GPT-OSS models
echo "Activating nc_env_qwen..."
conda activate nc_env_qwen
echo "Running Qwen/LLaMA/GPT-OSS models..."
python eval.py

# Run Phi models
echo "Activating phi_nc_env..."
conda activate phi_nc_env
echo "Running Phi models..."
python eval2.py

echo "All runs completed."
