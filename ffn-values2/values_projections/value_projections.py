import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pickle

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the output embedding matrix (usually tied to input embeddings)
E = model.get_output_embeddings().weight.data  # shape: [vocab_size, hidden_dim]

# Get all FC2 weights across layers
fc2_weights = []
for layer in model.transformer.h:
    fc2 = layer.mlp.c_proj.weight.T  # shape: [hidden_dim, intermediate_dim]
    fc2_weights.append(fc2)

# Project each (layer, dim) FFN output vector into vocab space
values = []
for layer_idx in range(len(fc2_weights)):
    for dim_idx in range(fc2_weights[layer_idx].shape[1]):
        vector = fc2_weights[layer_idx][:, dim_idx].unsqueeze(0)  # [1, hidden_dim]
        values.append(vector)

values = torch.cat(values, dim=0)  # [num_vectors, hidden_dim]
logits = values @ E.T  # [num_vectors, vocab_size]

# Get top-k tokens for each projection
top_k = 10
projections = {}
cnt = 0
inv_d = {}
for i in range(len(fc2_weights)):
    for j in range(fc2_weights[i].shape[1]):
        inv_d[(i, j)] = cnt
        cnt += 1

for (i, j), index in inv_d.items():
    top_ids = torch.topk(logits[index], top_k).indices.tolist()
    tokens = [tokenizer.decode([x]) for x in top_ids]
    projections[(i, j)] = tokens

# Now `projections[(layer, dim)]` gives top-k tokens influenced by that FFN dimension


with open("ffn_projections.pkl", "wb") as f:
    pickle.dump(projections, f)
