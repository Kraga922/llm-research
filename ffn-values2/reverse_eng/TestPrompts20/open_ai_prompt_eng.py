from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the model and tokenizer
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set your control prompt here
control_prompt = """Describe a futuristic city where technology and nature coexist in harmony. """

# High-scoring tokens
high_scoring_tokens = [
    "tertiary", "firmly", "removeClass", "Vaugh", "Louise", "Defaults", "Sap", "educated",
    "646", ".groupby", "TValue", "besonders", "856", "OMIT", "readily", "INU", "Willis",
    "cere", "headlights", "reel", "biology", "contractual", "Robbins", "supreme"
]

# Build the full engineering prompt
engineer_prompt = f"""You are now a prompt engineer. Your task is to craft an experimental prompt that tests the influence of a specific set of high-scoring tokens on a language model's response. These tokens correlate with a steering vector and may subtly or overtly shift the model’s style, tone, or subject matter.

You will receive a **control prompt**. Your job is to revise this prompt naturally and seamlessly to include as many of the following high-scoring tokens as possible, while ensuring it still makes logical sense and retains its original purpose.

**Control Prompt:**
{control_prompt}

**High-Scoring Tokens to Incorporate:**
{", ".join(high_scoring_tokens)}

Make the modifications smooth and subtle, without breaking the prompt’s coherence. Return only the newly engineered prompt."""

# Tokenize and generate
inputs = tokenizer(engineer_prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print the result
engineered_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n==== Engineered Prompt ====\n")
print(engineered_prompt)
