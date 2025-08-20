from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-13b-hf"

#ENTER HUGGINGFACE TOKEN HERE
token = ""


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

residual_streams = {}

def hook_fn(module, input, output):
    # input is a tuple (input,)
    layer_name = module.__class__.__name__
    residual_streams[layer_name] = output.detach().cpu()

for i, block in enumerate(model.model.layers):
    block.register_forward_hook(hook_fn)

model.model.layers[0].self_attn.register_forward_hook(hook_fn)  # hook into first attention block
model.model.layers[0].mlp.register_forward_hook(hook_fn)        # hook into first MLP

import torch

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# View residual streams
for layer, output in residual_streams.items():
    print(f"{layer}: shape = {output.shape}")

