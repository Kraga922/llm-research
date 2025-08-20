import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======== CONFIG ========
MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
TEMPLATE_FILE = "engineered_prompts_template.txt"
OUTPUT_FILE = "engineered_prompts_results_20.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================

def load_prompt_pairs(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    prompt_pairs = []
    control, engineered = None, None
    for line in lines:
        line = line.strip()
        if line.startswith("CONTROL:"):
            control = line.replace("CONTROL:", "").strip()
        elif line.startswith("ENGINEERED:"):
            engineered = line.replace("ENGINEERED:", "").strip()
            if control and engineered:
                prompt_pairs.append((control, engineered))
                control, engineered = None, None
    return prompt_pairs

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,  # Prevents memory accumulation
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

def main():
    prompt_pairs = load_prompt_pairs(TEMPLATE_FILE)
    tokenizer, model = load_model()

    with open(OUTPUT_FILE, "w") as out_file:
        for i, (control, engineered) in enumerate(prompt_pairs, start=1):
            out_file.write(f"\n=== PROMPT {i} ===\n")

            # CONTROL
            out_file.write(f"\nCONTROL:\n{control}\n")
            try:
                control_response = generate_response(control, tokenizer, model)
            except Exception as e:
                control_response = f"[ERROR]: {str(e)}"
            out_file.write(f"RESPONSE:\n{control_response}\n")

            # ENGINEERED
            out_file.write(f"\nENGINEERED:\n{engineered}\n")
            try:
                engineered_response = generate_response(engineered, tokenizer, model)
            except Exception as e:
                engineered_response = f"[ERROR]: {str(e)}"
            out_file.write(f"RESPONSE:\n{engineered_response}\n")

            out_file.flush()  # Ensure writing per iteration

if __name__ == "__main__":
    main()
