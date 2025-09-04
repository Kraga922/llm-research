import re
import os
import torch
import numpy as np
from pathlib import Path
import sys

notebook_path = Path().resolve()
project_root = notebook_path.parents[1]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_controllers import NeuralController

print("Transformers version:", __import__("transformers").__version__)

# -----------------------------
# Config
# -----------------------------
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

model_types = [
    # "llama_70b",
    # "llama",
    # "qwen3_small",
    # "qwen3_large",
    # "gpt_oss",
    "gpt_oss_120b",
    # "phi-small",
    # "phi-large"
]

for model_type in model_types:
    # -----------------------------
    # Model Integration
    # -----------------------------
    if model_type == "qwen3_small":
        model_id = "Qwen/Qwen3-0.6B"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"
        )
        model_name = "qwen3_0.6b"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    elif model_type == "qwen3_large":
        model_id = "Qwen/Qwen3-32B"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"
        )
        model_name = "qwen3_32b"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    elif model_type == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False
        )
        model_name = "llama_3_8b_it"

    elif model_type == "llama_70b":
        model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda"
        )
        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False
        )
        model_name = "llama_3_70b_it"

    elif model_type == "gpt_oss":
        model_id = "openai/gpt-oss-20b"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left", legacy=False
        )
        model_name = "gpt_oss"

    elif model_type == 'gpt_oss_120b':
        model_id = "openai/gpt-oss-120b"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left", legacy=False
        )

        model_name = 'gpt_oss_120b'

    elif model_type == 'phi-small':
        model_id = "microsoft/Phi-4-mini-instruct"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True  
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            padding_side="left",
            legacy=False,
            trust_remote_code=True
        )
        model_name = 'phi-small'

    elif model_type == 'phi-large':
        model_id = "microsoft/phi-4"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True  
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            padding_side="left",
            legacy=False,
            trust_remote_code=True
        )
        model_name = 'phi-large'

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Loaded model: {model_name}")

    # -----------------------------
    # Neural Controller
    # -----------------------------
    harmful_controller = NeuralController(
        language_model,
        tokenizer,
        rfm_iters=8,
        control_method="rfm"
    )
    harmful_controller.load(concept="harmful", model_name=model_name, path="../../directions/")
    # harmful_controller.load(concept="harmful", model_name= "llama_3_70b_it", path="../../directions/")


    # if model_type == "llama_70b":
        
    #     # >>> Two sets of layers with labels
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -80, -1))),
    #         # ("RepE steering layers", list(range(-1, -66, -1)))
    #     ]

    #     # >>> Coefs to test
    #     # coefs_to_test = [0.3, 0.4, 0.45, 0.5, 0.6]
    #     # coefs_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1]
    #     coefs_to_test = [0.1, 0.3, 1]

    #     num_new_tokens = 256

    # elif model_type == "llama":
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -31, -1))),
    #         # ("RepE steering layers", list(range(-1, -21, -1)))
    #     ]
    #     # coefs_to_test = [0.3, 0.4, 0.45, 0.5, 0.6]
    #     coefs_to_test = [0.1, 0.55, 1]

    #     num_new_tokens = 256
    # elif model_type == "gpt_oss":
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -24, -1))),
    #         # ("RepE steering layers", list(range(-7, -15, -1)))
    #     ]
    #     # coefs_to_test = [38, 43, 45, 48, 52]
    #     # coefs_to_test = [20, 25, 35, 45, 50]
    #     coefs_to_test = [37, 43, 49]

    #     num_new_tokens = 256

    # elif model_type == "gpt_oss_120b":
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -36, -1))),
    #         # ("RepE steering layers", list(range(-16, -24, -1)))
    #     ]
    #     # coefs_to_test = [38, 43, 45, 48, 52]
    #     # coefs_to_test = [70, 75, 80, 85, 90, 95]
    #     coefs_to_test = [60, 70, 80]

    #     num_new_tokens = 256

    # elif model_type == "qwen3_small":
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -28, -1))),
    #         # ("RepE steering layers", list(range(-1, -13, -1)))
    #     ]
    #     # coefs_to_test = [38, 43, 45, 48, 52]
    #     # coefs_to_test = [0.25, 0.4, 0.5, 0.6, 0.75]
    #     coefs_to_test = [1.5, 1.8, 2.1]
    #     num_new_tokens = 256
    # elif model_type == "qwen3_large":   
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -64, -1))),
    #         # ("RepE steering layers", list(range(-3, -40, -1)))
    #     ]
    #     # coefs_to_test = [38, 43, 45, 48, 52]
    #     coefs_to_test = [5, 10, 15]
    #     num_new_tokens = 256
    # elif model_type == 'phi-small':
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -32, -1))),
    #         # ("RepE steering layers", list(range(-3, -22, -1)))
    #     ]
    #     coefs_to_test = [1, 2.5, 4]
    #     num_new_tokens = 256
    # elif model_type == 'phi-large':
    #     layers_to_test = [
    #         ("Universal steering layers (all layers)", list(range(-1, -40, -1))),
    #         # ("RepE steering layers", list(range(-3, -19, -1)))
    #     ]
    #     coefs_to_test = [1, 2.9, 5]
    #     num_new_tokens = 256
    # else:
    #     raise ValueError(f"Unknown model_type: {model_type}")  

    if model_type == "llama_70b":
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -80, -1))),
        ]
        coefs_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        num_new_tokens = 256

    elif model_type == "llama":
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -31, -1))),
        ]
        coefs_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        num_new_tokens = 256

    elif model_type == "gpt_oss":
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -24, -1))),
        ]
        coefs_to_test = [41, 42, 43, 45, 47, 49, 51, 53, 55, 56]
        num_new_tokens = 256

    elif model_type == "gpt_oss_120b":
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -36, -1))),
        ]
        # coefs_to_test = [60, 62, 64, 66, 68, 70, 72, 74, 77, 80]
        # coefs_to_test = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
        coefs_to_test = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

        num_new_tokens = 256

    elif model_type == "qwen3_small":
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -28, -1))),
        ]
        coefs_to_test = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

        num_new_tokens = 256

    elif model_type == "qwen3_large":   
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -64, -1))),
        ]
        coefs_to_test = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        num_new_tokens = 256

    elif model_type == 'phi-small':
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -32, -1))),
        ]
        coefs_to_test = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        num_new_tokens = 256

    elif model_type == 'phi-large':
        layers_to_test = [
            ("Universal steering layers (all layers)", list(range(-1, -40, -1))),
        ]
        coefs_to_test = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.2, 4.5, 5]
        num_new_tokens = 256

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    # -----------------------------
    # Load prompts
    # -----------------------------
    def load_prompts(filepath):
        prompts = []
        with open(filepath, "r") as f:
            for line in f:
                prompt = re.sub(r"^\d+\.\s*", "", line.strip())
                if prompt:
                    prompts.append(prompt)
        return prompts

    # -----------------------------
    # Evaluation (only with control)
    # -----------------------------
    def evaluate(prompts, layers_to_control, coef):
        results = []
        for idx, p in enumerate(prompts, 1):
            formatted_prompt = harmful_controller.format_prompt(p)
            start_idx = len(formatted_prompt)

            out_with_control = harmful_controller.generate(
                formatted_prompt,
                layers_to_control=layers_to_control,
                control_coef=coef,
                max_new_tokens=num_new_tokens,
                do_sample=True
            )[start_idx:]

            results.append({
                "id": idx,
                "prompt": p,
                "output_with_control": out_with_control.strip(),
            })
        return results

    # -----------------------------
    # Save all results into one TXT
    # -----------------------------
    def save_all_results_txt(all_results, out_path):
        with open(out_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write("="*100 + "\n\n")

            for label, layers, coef, results in all_results:
                f.write(f"Layer set: {label}\n")
                f.write(f"Layers: {layers}\n")
                f.write(f"Coefficient: {coef}\n")
                f.write(f"Total prompts: {len(results)}\n")
                f.write("="*80 + "\n\n")

                for r in results:
                    f.write(f"Prompt {r['id']}: {r['prompt']}\n")
                    f.write(f"With control:\n")
                    f.write(r['output_with_control'] + "\n")
                    f.write("-"*80 + "\n\n")

                f.write("\n\n" + "#"*100 + "\n\n")

        print(f"Saved all results to {out_path}")

    # -----------------------------
    # Run
    # -----------------------------
    if __name__ == "__main__":
        prompts_path = Path("/home/ubuntu/llm-research/neural_controllers2/notebooks/harmful/harmful_prompts_small.txt")
        prompts = load_prompts(prompts_path)

        out_dir = Path("10_coef_test")
        out_dir.mkdir(exist_ok=True)

        all_results = []

        # loop over layers + coefs
        for label, layers in layers_to_test:
            for coef in coefs_to_test:
                print(f"\n=== Testing {label}, layers {layers}, coef {coef} ===\n")
                results = evaluate(prompts, layers, coef)
                all_results.append((label, layers, coef, results))

        # save everything into one file
        out_path = out_dir / f"{model_name}_10_coef_10_prompts.txt"
        save_all_results_txt(all_results, out_path)
