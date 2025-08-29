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
    "phi-small",
    "phi-large"
]

for model_type in model_types:
    # -----------------------------
    # Model Integration
    # -----------------------------

    if model_type == 'phi-small':
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


    if model_type == 'phi-small':
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
