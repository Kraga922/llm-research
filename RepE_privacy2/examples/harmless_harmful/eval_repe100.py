import sys
import os
from pathlib import Path
sys.path.append('/home/ubuntu/llm-research/RepE_privacy2')

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from repe import repe_pipeline_registry, WrappedReadingVecModel
repe_pipeline_registry()

def load_prompts(prompts_path):
    """Load prompts from a text file, one per line."""
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def evaluate(prompts, layers, coef, model, tokenizer, rep_reader, template, component_index, num_new_tokens, model_type):
    """Evaluate a list of prompts with given layers and coefficient."""
    results = []
    
    # Set up activations with model-specific dtype handling
    activations = {}
    for layer in layers:
        if model_type in ["gpt_oss", "gpt_oss_120b"]:
            # GPT-OSS models need specific dtype handling
            activations[layer] = torch.tensor(
                coef * rep_reader.directions[layer][component_index] * rep_reader.direction_signs[layer][component_index]
            ).to(model.device, dtype=next(model.parameters()).dtype)
        else:
            # Other models use .half()
            activations[layer] = torch.tensor(
                coef * rep_reader.directions[layer][component_index] * rep_reader.direction_signs[layer][component_index]
            ).to(model.device).half()
    
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layers, block_name="decoder_block")
    wrapped_model.set_controller(layers, activations, masks=1)
    
    for prompt in tqdm(prompts, desc=f"Evaluating coef={coef}"):
        inputs = template.format(instruction=prompt)
        encoded_inputs = tokenizer(inputs, return_tensors='pt')
        
        with torch.no_grad():
            # Model-specific generation method
            if model_type in ["qwen3_small", "qwen3_large"]:
                # Qwen models need wrapped_model.generate()
                outputs = wrapped_model.generate(**encoded_inputs.to(model.device), max_new_tokens=num_new_tokens, do_sample=False).detach().cpu()
            else:
                # Other models use model.generate()
                outputs = model.generate(**encoded_inputs.to(model.device), max_new_tokens=num_new_tokens, do_sample=False).detach().cpu()
            
            generation = tokenizer.decode(outputs[0], skip_special_tokens=False).replace(inputs, "")
        
        results.append({
            'prompt': prompt,
            'generation': generation
        })
    
    wrapped_model.reset()
    wrapped_model.unwrap()
    
    return results

def save_model_results_txt(model_results, out_path):
    """Save results for a single model to a text file."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for model_name, label, layers, coef, results in model_results:
            f.write(f"=== Model: {model_name}, {label}, layers {layers}, coef {coef} ===\n\n")
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generation: {result['generation']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n" * 2)

def save_all_results_txt(all_results, out_path):
    """Save all results to a text file."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for model_name, label, layers, coef, results in all_results:
            f.write(f"=== Model: {model_name}, {label}, layers {layers}, coef {coef} ===\n\n")
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generation: {result['generation']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n" * 2)

# Model configurations
model_types = [
    # "llama_70b",
    # "llama",
    # "qwen3_small",
    # "qwen3_large",
    # "gpt_oss",
    # "gpt_oss_120b",
    # "phi-small",
    "phi-large"
]

# Load prompts once (outside the loop) - using current working directory
current_dir = Path.cwd()  # Get current working directory instead of hardcoded path

# prompts_path = current_dir / "harmful_prompts_small.txt"  # Assumes file is in current directory
# if not prompts_path.exists():
#     # Fallback to original path if file not found in current directory
    
prompts_path = Path("/home/ubuntu/llm-research/neural_controllers2/notebooks/harmful/harmful_prompts.txt")

prompts = load_prompts(prompts_path)

# Create results directory in current working directory
# out_dir = current_dir / "RepE_results"
out_dir = current_dir / "RepE_100"
out_dir.mkdir(exist_ok=True)

all_results = []

for model_type in model_types:
    print(f"\n{'='*50}")
    print(f"Processing model type: {model_type}")
    print(f"{'='*50}\n")
    
    # Track results for this model
    model_results = []
    
    # -----------------------------
    # Model Integration
    # -----------------------------
    if model_type == "qwen3_small":
        model_name_or_path = 'Qwen/Qwen3-0.6B'
        model_name = "qwen3_small"
        
    elif model_type == "qwen3_large":
        model_name_or_path = 'Qwen/Qwen3-32B'
        model_name = "qwen3_large"
        
    elif model_type == "llama":
        model_name_or_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        model_name = "llama_3_8b_it"
        
    elif model_type == "llama_70b":
        model_name_or_path = 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit'
        model_name = "llama_70b"
        
    elif model_type == "gpt_oss":
        model_name_or_path = 'openai/gpt-oss-20b'
        model_name = "gpt_oss"
        
    elif model_type == 'gpt_oss_120b':
        model_name_or_path = 'openai/gpt-oss-120b'
        model_name = "gpt_oss_120b"
        
    elif model_type == 'phi-small':
        model_name_or_path = 'microsoft/Phi-4-mini-instruct'
        model_name = 'phi-small'
        
    elif model_type == 'phi-large':
        model_name_or_path = 'microsoft/Phi-4'
        model_name = 'phi-large'
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load model
    if model_type in ["gpt_oss", "gpt_oss_120b"]:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map='balanced_low_0'
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded model: {model_name}")

    # Template and dataset loading
    template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
    
    dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
    train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']
    train_data, train_labels = train_dataset['sentence'], train_dataset['label']
    test_data = test_dataset['sentence']

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()

    train_data = [template.format(instruction=s) for s in train_data]
    test_data = [template.format(instruction=s) for s in test_data]

    # RepE setup
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    direction_finder_kwargs = {"n_components": 1}

    rep_reader = rep_reading_pipeline.get_directions(
        train_data, 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=train_labels, 
        direction_method=direction_method,
        direction_finder_kwargs=direction_finder_kwargs
    )

    component_index = 0

    # Model-specific configurations with coefficient ranges

    if model_type == "llama":
        layer_id = [("RepE steering layers", list(range(-1, -21, -1)))]
        coeff_values = [0.9]

        num_new_tokens = 256

    elif model_type == "llama_70b":
        layer_id = [("RepE steering layers", list(range(-1, -66, -1)))]
        coeff_values = [0.5]
        num_new_tokens = 256

    elif model_type == "gpt_oss":
        layer_id = [("RepE steering layers", list(range(-7, -15, -1)))]
        coeff_values = [180]  # Updated to match your working example


        num_new_tokens = 256
    elif model_type == "gpt_oss_120b":
        layer_id = [("RepE steering layers", list(range(-16, -24, -1)))]
        coeff_values = [200]

        num_new_tokens = 256
    elif model_type == "qwen3_small":
        layer_id = [("RepE steering layers", list(range(-1, -13, -1)))]
        coeff_values = [0.7]  # Updated to match your working example

        num_new_tokens = 256
    elif model_type == "qwen3_large":   
        layer_id = [("RepE steering layers", list(range(-3, -40, -1)))]
        coeff_values = [8.0]

        num_new_tokens = 256
    elif model_type == 'phi-small':
        layer_id = [("RepE steering layers", list(range(-3, -22, -1)))]
        coeff_values = [3.0]

        num_new_tokens = 256
    elif model_type == 'phi-large':
        layer_id = [("RepE steering layers", list(range(-3, -19, -1)))]
        # coeff_values = [7.0]
        coeff_values = [6.6]


        num_new_tokens = 256
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Run evaluation for this model with different coefficients
    try:
        for label, layers in layer_id:
            for coeff in coeff_values:
                print(f"\n=== Evaluating {model_name}, {label}, layers {layers}, coef {coeff} ===\n")
                results = evaluate(prompts, layers, coeff, model, tokenizer, rep_reader, template, component_index, num_new_tokens, model_type)
                model_results.append((model_name, label, layers, coeff, results))
                all_results.append((model_name, label, layers, coeff, results))

        # Save this model's results to its own file
        model_out_path = out_dir / f"{model_name}_repE_10_test_cases.txt"
        save_model_results_txt(model_results, model_out_path)
        print(f"\nModel {model_name} results saved to: {model_out_path}")
    
    except Exception as e:
        print(f"\nERROR: Failed to evaluate model {model_name}: {str(e)}")
        print(f"Skipping model {model_name} and continuing with next model...")
        # Still save partial results if any were generated
        if model_results:
            # model_out_path = out_dir / f"{model_name}_repE_100_test_cases_PARTIAL.txt"
            model_out_path = out_dir / f"{model_name}_repE_100_test_cases_PARTIAL.txt"
            save_model_results_txt(model_results, model_out_path)
            print(f"Partial results saved to: {model_out_path}")

    # Clean up model to free memory
    del model
    del tokenizer
    del rep_reader
    torch.cuda.empty_cache()

# Save all results from all models
# out_path = out_dir / "RepE10.txt"

# out_path = out_dir / "all_models_repE_multi_100_test_cases.txt"
# save_all_results_txt(all_results, out_path)
# print(f"\nAll results saved to: {out_path}")
# print(f"\nTotal evaluations completed: {len(all_results)}")