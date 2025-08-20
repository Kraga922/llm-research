##### OG CODE
# import argparse
# from random import shuffle

# import numpy as np
# import pandas as pd
# import spacy
# import torch
# import torch.nn.functional as F
# from torchtext.datasets import WikiText103
# from tqdm import tqdm
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# nlp = spacy.load('en_core_web_sm')

# def set_hooks_gpt2(model):
#     """
#     Only works on GPT2 from HF
#     """
#     final_layer = model.config.n_layer - 1

#     for attr in ["activations_"]:
#         if not hasattr(model, attr):
#             setattr(model, attr, {})

#     def get_activation(name):
#         def hook(module, input, output):
#             if "mlp" in name or "attn" in name or "m_coef" in name:
#                 if "attn" in name:
#                     num_tokens = list(output[0].size())[1]
#                     model.activations_[name] = output[0][:, num_tokens - 1].detach()
#                 elif "mlp" in name:
#                     num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
#                     model.activations_[name] = output[0][num_tokens - 1].detach()
#                 elif "m_coef" in name:
#                     num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
#                     model.activations_[name] = input[0][:, num_tokens - 1].detach()
#             elif "residual" in name or "embedding" in name:
#                 num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
#                 if name == "layer_residual_" + str(final_layer):
#                     model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
#                                                model.activations_["mlp_" + str(final_layer)]
#                 else:
#                     model.activations_[name] = input[0][:,
#                                                num_tokens - 1].detach()  # https://github.com/huggingface/transformers/issues/7760

#         return hook

#     model.transformer.h[0].ln_1.register_forward_hook(get_activation("input_embedding"))

#     for i in range(model.config.n_layer):
#         if i != 0:
#             model.transformer.h[i].ln_1.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
#         model.transformer.h[i].ln_2.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

#         model.transformer.h[i].attn.register_forward_hook(get_activation("attn_" + str(i)))
#         model.transformer.h[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
#         model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

#     model.transformer.ln_f.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


# def get_resid_predictions(model, tokenizer, tokens, TOP_K=1, start_idx=None, end_idx=None, set_mlp_0=False):
#     HIDDEN_SIZE = model.config.n_embd

#     layer_residual_preds = []
#     intermed_residual_preds = []
#     output = model(**tokens, output_hidden_states=True)

#     for layer in model.activations_.keys():
#         if "layer_residual" in layer or "intermediate_residual" in layer:
#             normed = model.transformer.ln_f(model.activations_[layer])
#             logits = torch.matmul(model.lm_head.weight, normed.T)

#             probs = F.softmax(logits.T[0], dim=-1)

#             probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

#             assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

#             probs_ = []
#             for index, prob in enumerate(probs):
#                 probs_.append((index, prob))

#             top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
#             top_k = [(t[1].item(), t[0]) for t in top_k]

#         if "layer_residual" in layer:
#             layer_residual_preds.append(top_k)
#         elif "intermediate_residual" in layer:
#             intermed_residual_preds.append(top_k)
#         for attr in ["layer_resid_preds", "intermed_residual_preds"]:
#             if not hasattr(model, attr):
#                 setattr(model, attr, [])

#         model.layer_resid_preds = layer_residual_preds
#         model.intermed_residual_preds = intermed_residual_preds


# def project_value_to_vocab(layer, value_idx, top_k=10):
#     normed = model.transformer.ln_f(model.transformer.h[layer].mlp.c_proj.weight.data[value_idx])

#     logits = torch.matmul(model.lm_head.weight, normed.T)
#     probs = F.softmax(logits, dim=-1)
#     probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

#     probs_ = []
#     for index, prob in enumerate(probs):
#         probs_.append((index, prob))

#     top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
#     value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

#     return value_preds


# def get_prediction_depth(layer_resid_top_preds):
#     pred_depth = len(layer_resid_top_preds)

#     prev_pred = layer_resid_top_preds[-1]

#     for pred in [pred for pred in reversed(layer_resid_top_preds)]:
#         if pred != prev_pred:
#             break
#         else:
#             pred_depth -= 1
#             prev_pred = pred

#     return pred_depth


# def get_preds_and_hidden_states(wiki_text_sentences, gpt2_model, gpt2_tokenizer, random_pos=True):
#     set_hooks_gpt2(gpt2_model)

#     sent_to_preds = {}
#     sent_to_hidden_states = {}
#     idx = 0
#     for sentence in tqdm(wiki_text_sentences):
#         if random_pos:
#             tokens_old = [token for token in sentence.split(' ')]
#             sentence_old = sentence[:]
#             tokens = gpt2_tokenizer(sentence, return_tensors="pt")
#             tokens_to_sent = gpt2_tokenizer.tokenize(sentence)
#             if len(tokens_to_sent) > 0:
#                 if len(tokens_to_sent) == 1:
#                     random_pos = 1
#                 else:
#                     random_pos = np.random.randint(1, len(tokens_to_sent))
#                 tokens = {k: v[:, :random_pos].to(device) for k, v in tokens.items()}
#                 tokens_to_sent = tokens_to_sent[:random_pos]
#                 sentence = gpt2_tokenizer.convert_tokens_to_string(tokens_to_sent)
#             else:
#                 continue

#         key = (sentence, idx)
#         get_resid_predictions(gpt2_model, gpt2_tokenizer, tokens, TOP_K=30)

#         if sentence not in sent_to_preds.keys():
#             sent_to_preds[key] = {}
#         sent_to_preds[key]["layer_resid_preds"] = gpt2_model.layer_resid_preds
#         sent_to_preds[key]["intermed_residual_preds"] = gpt2_model.intermed_residual_preds
#         sent_to_hidden_states[key] = {k: v.cpu() for k, v in gpt2_model.activations_.items()}
#         if len(tokens_to_sent) == 1:
#             sent_to_hidden_states[key]['gold_token'] = gpt2_tokenizer(sentence_old, return_tensors="pt")['input_ids'][:,
#                                                        0].item()
#         else:
#             sent_to_hidden_states[key]['gold_token'] = gpt2_tokenizer(sentence_old, return_tensors="pt")['input_ids'][:,
#                                                        random_pos].item()
#         idx += 1

#     return sent_to_hidden_states, sent_to_preds


# def get_examples_df(prompts, model, tokenizer):
#     sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states(prompts, model, tokenizer, random_pos=True)
#     idx = 0
#     records = []
#     for key in tqdm(sent_to_preds.keys()):
#         sent = key[0]
#         top_coef_idx = []
#         top_coef_vals = []
#         top_coef_abs_idx = []
#         top_coef_vals_abs = []

#         rand_coef_idx = []
#         rand_coef_vals = []
#         rand_coef_abs_idx = []
#         rand_coef_vals_abs = []

#         coefs_sums = []

#         residual_preds_probs = []
#         residual_preds_tokens = []
#         layer_preds_probs = []
#         layer_preds_tokens = []
#         res_vecs = []
#         mlp_vecs = []
#         for LAYER in range(model.config.n_layer):
#             coefs_ = []
#             coefs_abs_ = []
#             m_coefs = sent_to_hidden_states[key]["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
#             res_vec = sent_to_hidden_states[key]["intermediate_residual_" + str(LAYER)].squeeze().cpu().numpy()
#             mlp_vec = sent_to_hidden_states[key]["mlp_" + str(LAYER)].squeeze().cpu().numpy()
#             res_vecs.append(res_vec)
#             mlp_vecs.append(mlp_vec)
#             value_norms = torch.linalg.norm(model.transformer.h[LAYER].mlp.c_proj.weight.data, dim=1).cpu()
#             coefs = m_coefs * value_norms.numpy()
#             coefs_abs = np.absolute(m_coefs) * value_norms.numpy()
#             coefs_sums.append(coefs_abs.sum())
#             for index, prob in enumerate(coefs):
#                 coefs_.append((index, prob))
#             for index, prob in enumerate(coefs_abs):
#                 coefs_abs_.append((index, prob))
#             top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:30]
#             c_idx, c_vals = zip(*top_values)
#             top_coef_idx.append(c_idx)
#             top_coef_vals.append(c_vals)

#             top_values_abs = sorted(coefs_abs_, key=lambda x: x[1], reverse=True)[:30]
#             c_idx_abs, c_vals_abs = zip(*top_values_abs)
#             top_coef_abs_idx.append(c_idx_abs)
#             top_coef_vals_abs.append(c_vals_abs)

#             shuffle(coefs_)
#             rand_idx, rand_vals = zip(*coefs_)
#             rand_coef_idx.append(rand_idx[:30])
#             rand_coef_vals.append(rand_vals[:30])
#             shuffle(coefs_abs_)
#             rand_idx_abs, rand_vals_abs = zip(*coefs_abs_)
#             rand_coef_abs_idx.append(rand_idx_abs[:30])
#             rand_coef_vals_abs.append(rand_vals_abs[:30])

#             residual_p_probs, residual_p_tokens = zip(*sent_to_preds[key]['intermed_residual_preds'][LAYER])
#             residual_preds_probs.append(residual_p_probs)
#             residual_preds_tokens.append(residual_p_tokens)

#             layer_p_probs, layer_p_tokens = zip(*sent_to_preds[key]['layer_resid_preds'][LAYER])
#             layer_preds_probs.append(layer_p_probs)
#             layer_preds_tokens.append(layer_p_tokens)
#         gold_token = sent_to_hidden_states[key]['gold_token']
#         records.append({
#             "sent": sent,
#             "top_coef_idx": top_coef_idx,
#             "top_coef_vals": top_coef_vals,
#             "top_coef_abs_idx": top_coef_abs_idx,
#             "top_coef_vals_abs": top_coef_vals_abs,
#             "rand_coef_idx": rand_coef_idx,
#             "rand_coef_vals": rand_coef_vals,
#             "rand_coef_abs_idx": rand_coef_abs_idx,
#             "rand_coef_vals_abs": rand_coef_vals_abs,

#             "coefs_total_sum": coefs_sums,
#             "residual_preds_probs": residual_preds_probs,
#             "residual_preds_tokens": residual_preds_tokens,
#             "layer_preds_probs": layer_preds_probs,
#             "layer_preds_tokens": layer_preds_tokens,
#             "layer_mlp_vec": mlp_vecs,
#             "gold_token": gold_token
#         })
#         idx += 1

#     df = pd.DataFrame(records)
#     return df


# def parse_line(line):
#     tokens = [
#         token for token in line.split(' ')
#         if token not in ['', '\n']
#     ]
#     if len(tokens) == 0:
#         return None
#     spaces = [True for _ in range(len(tokens) - 1)] + [False]
#     assert len(tokens) == len(spaces), f"{len(tokens)} != {len(spaces)}"

#     doc = spacy.tokens.doc.Doc(
#         nlp.vocab, words=tokens, spaces=spaces)
#     for name, proc in nlp.pipeline:
#         doc = proc(doc)
#     return [str(sent) for sent in doc.sents]


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--gpt2_model_name", default='gpt2', type=str, help="GPT2 model name"
# )
# parser.add_argument(
#     "--device", default='cuda:0', type=str, help="device"
# )
# parser.add_argument(
#     "--max_sentences", default=10000, type=int, help="max sentences to include in the data"
# )
# parser.add_argument(
#     "--output_path", default='gpt2_df_10k.pkl', type=str, help="output pickle file path"
# )

# args = parser.parse_args()

# pt2_model_name = 'gpt2'
# device = "cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
# tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_name)
# model = GPT2LMHeadModel.from_pretrained(args.gpt2_model_name)
# model = model.to(device)
# model.eval()

# layer_fc2_vals = [
#     model.transformer.h[layer_i].mlp.c_proj.weight.T.detach()
#     for layer_i in tqdm(range(model.config.n_layer))
# ]

# E = model.get_input_embeddings().weight.cpu().detach()

# tok_to_idx = tokenizer.get_vocab()

# vocab = [None] * len(tok_to_idx)
# for k, v in tok_to_idx.items():
#     vocab[v] = [k, 0]

# wiki_text = list(WikiText103(split="valid"))
# num_sentences = 0
# wiki_text_sentences = []
# for line in WikiText103(split="valid"):
#     sentences = parse_line(line)
#     if sentences is None:
#         continue
#     else:
#         for sentence in sentences:
#             if len(tokenizer.tokenize(sentence)) == 0:
#                 continue
#             wiki_text_sentences.append(sentence)
#             num_sentences += 1
# shuffle(wiki_text_sentences)
# df = get_examples_df(wiki_text_sentences[:args.max_sentences], model, tokenizer)
# df.to_pickle(args.output_path)


import argparse
import os
from random import shuffle

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    LlamaTokenizer, LlamaForCausalLM,
    AutoTokenizer, AutoModelForCausalLM
)

nlp = spacy.load('en_core_web_sm')

def set_hooks_gpt2(model):
    """
    Only works on GPT2 from HF
    """
    final_layer = model.config.n_layer - 1

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
                    model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(final_layer):
                    model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
                                               model.activations_["mlp_" + str(final_layer)]
                else:
                    model.activations_[name] = input[0][:,
                                               num_tokens - 1].detach()

        return hook

    model.transformer.h[0].ln_1.register_forward_hook(get_activation("input_embedding"))

    for i in range(model.config.n_layer):
        if i != 0:
            model.transformer.h[i].ln_1.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
        model.transformer.h[i].ln_2.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

        model.transformer.h[i].attn.register_forward_hook(get_activation("attn_" + str(i)))
        model.transformer.h[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
        model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

    model.transformer.ln_f.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


def set_hooks_llama(model):
    """
    Hook setup for Llama models
    """
    final_layer = model.config.num_hidden_layers - 1

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]
                    model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]
                    model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]
                if name == "layer_residual_" + str(final_layer):
                    model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
                                               model.activations_["mlp_" + str(final_layer)]
                else:
                    model.activations_[name] = input[0][:, num_tokens - 1].detach()

        return hook

    # Llama uses different layer structure
    model.model.layers[0].input_layernorm.register_forward_hook(get_activation("input_embedding"))

    for i in range(model.config.num_hidden_layers):
        if i != 0:
            model.model.layers[i].input_layernorm.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
        model.model.layers[i].post_attention_layernorm.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

        model.model.layers[i].self_attn.register_forward_hook(get_activation("attn_" + str(i)))
        model.model.layers[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
        model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

    model.model.norm.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


def get_resid_predictions(model, tokenizer, tokens, model_type='gpt2', TOP_K=1, start_idx=None, end_idx=None, set_mlp_0=False):
    layer_residual_preds = []
    intermed_residual_preds = []
    output = model(**tokens, output_hidden_states=True)

    for layer in model.activations_.keys():
        if "layer_residual" in layer or "intermediate_residual" in layer:
            if model_type == 'gpt2':
                normed = model.transformer.ln_f(model.activations_[layer])
                logits = torch.matmul(model.lm_head.weight, normed.T)
            else:  # llama
                normed = model.model.norm(model.activations_[layer])
                logits = torch.matmul(model.lm_head.weight, normed.T)

            probs = F.softmax(logits.T[0], dim=-1)
            probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

            assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

            probs_ = []
            for index, prob in enumerate(probs):
                probs_.append((index, prob))

            top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
            top_k = [(t[1].item(), t[0]) for t in top_k]

        if "layer_residual" in layer:
            layer_residual_preds.append(top_k)
        elif "intermediate_residual" in layer:
            intermed_residual_preds.append(top_k)
        
        for attr in ["layer_resid_preds", "intermed_residual_preds"]:
            if not hasattr(model, attr):
                setattr(model, attr, [])

        model.layer_resid_preds = layer_residual_preds
        model.intermed_residual_preds = intermed_residual_preds


def project_value_to_vocab(model, tokenizer, layer, value_idx, model_type='gpt2', top_k=10):
    if model_type == 'gpt2':
        normed = model.transformer.ln_f(model.transformer.h[layer].mlp.c_proj.weight.data[value_idx])
        logits = torch.matmul(model.lm_head.weight, normed.T)
    else:  # llama
        normed = model.model.norm(model.model.layers[layer].mlp.down_proj.weight.data[value_idx])
        logits = torch.matmul(model.lm_head.weight, normed.T)
    
    probs = F.softmax(logits, dim=-1)
    probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

    probs_ = []
    for index, prob in enumerate(probs):
        probs_.append((index, prob))

    top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
    value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

    return value_preds


def get_prediction_depth(layer_resid_top_preds):
    pred_depth = len(layer_resid_top_preds)
    prev_pred = layer_resid_top_preds[-1]

    for pred in [pred for pred in reversed(layer_resid_top_preds)]:
        if pred != prev_pred:
            break
        else:
            pred_depth -= 1
            prev_pred = pred

    return pred_depth


def is_valid_sentence(sentence, tokenizer, min_tokens=3, max_tokens=128):
    """
    Check if a sentence is valid for processing.
    
    Args:
        sentence: Input sentence string
        tokenizer: HuggingFace tokenizer
        min_tokens: Minimum number of tokens required
        max_tokens: Maximum number of tokens allowed
    
    Returns:
        bool: True if sentence is valid
    """
    if not sentence or not sentence.strip():
        return False
    
    # Remove sentences that are too short or just punctuation
    if len(sentence.strip()) < 10:
        return False
    
    # Check token count
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) < min_tokens or len(tokens) > max_tokens:
        return False
    
    # Skip sentences that are mostly numbers or special characters
    alpha_chars = sum(1 for c in sentence if c.isalpha())
    if alpha_chars < len(sentence) * 0.5:  # At least 50% alphabetic characters
        return False
    
    return True


def get_preds_and_hidden_states(wiki_text_sentences, model, tokenizer, model_type='gpt2', device='cuda:0', random_pos=True, batch_size=8):
    if model_type == 'gpt2':
        set_hooks_gpt2(model)
    else:
        set_hooks_llama(model)

    sent_to_preds = {}
    sent_to_hidden_states = {}
    idx = 0
    
    print(f"Processing {len(wiki_text_sentences)} sentences...")
    
    # Process sentences individually (the batching in original wasn't working properly)
    for sentence in tqdm(wiki_text_sentences, desc="Processing sentences"):
        try:
            if random_pos:
                tokens_old = [token for token in sentence.split(' ')]
                sentence_old = sentence[:]
                tokens = tokenizer(sentence, return_tensors="pt")
                tokens_to_sent = tokenizer.tokenize(sentence)
                
                if len(tokens_to_sent) > 0:
                    if len(tokens_to_sent) == 1:
                        random_pos_idx = 1
                    else:
                        random_pos_idx = np.random.randint(1, len(tokens_to_sent))
                    tokens = {k: v[:, :random_pos_idx].to(device) for k, v in tokens.items()}
                    tokens_to_sent = tokens_to_sent[:random_pos_idx]
                    sentence = tokenizer.convert_tokens_to_string(tokens_to_sent)
            else:
                continue

            key = (sentence, idx)
            get_resid_predictions(model, tokenizer, tokens, model_type, TOP_K=30)

            if sentence not in sent_to_preds.keys():
                sent_to_preds[key] = {}
            sent_to_preds[key]["layer_resid_preds"] = model.layer_resid_preds
            sent_to_preds[key]["intermed_residual_preds"] = model.intermed_residual_preds
            sent_to_hidden_states[key] = {k: v.cpu() for k, v in model.activations_.items()}
            
            if len(tokens_to_sent) == 1:
                sent_to_hidden_states[key]['gold_token'] = tokenizer(sentence_old, return_tensors="pt")['input_ids'][:, 0].item()
            else:
                sent_to_hidden_states[key]['gold_token'] = tokenizer(sentence_old, return_tensors="pt")['input_ids'][:, random_pos_idx].item()
            idx += 1
            
        except Exception as e:
            print(f"Error processing sentence: {sentence[:50]}... Error: {e}")
            continue

    return sent_to_hidden_states, sent_to_preds


def get_examples_df(prompts, model, tokenizer, model_type='gpt2', device='cuda:0', batch_size=8):
    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states(
        prompts, model, tokenizer, model_type, device, random_pos=True, batch_size=batch_size
    )
    
    records = []
    num_layers = model.config.n_layer if model_type == 'gpt2' else model.config.num_hidden_layers
    
    print(f"Building dataframe from {len(sent_to_preds)} processed sentences...")
    
    for key in tqdm(sent_to_preds.keys(), desc="Building dataframe"):
        sent = key[0]
        top_coef_idx = []
        top_coef_vals = []
        top_coef_abs_idx = []
        top_coef_vals_abs = []

        rand_coef_idx = []
        rand_coef_vals = []
        rand_coef_abs_idx = []
        rand_coef_vals_abs = []

        coefs_sums = []
        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []
        res_vecs = []
        mlp_vecs = []
        
        for LAYER in range(num_layers):
            coefs_ = []
            coefs_abs_ = []
            m_coefs = sent_to_hidden_states[key]["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            res_vec = sent_to_hidden_states[key]["intermediate_residual_" + str(LAYER)].squeeze().cpu().numpy()
            mlp_vec = sent_to_hidden_states[key]["mlp_" + str(LAYER)].squeeze().cpu().numpy()
            res_vecs.append(res_vec)
            mlp_vecs.append(mlp_vec)
            
            if model_type == 'gpt2':
                value_norms = torch.linalg.norm(model.transformer.h[LAYER].mlp.c_proj.weight.data, dim=1).cpu()
            else:  # llama
                value_norms = torch.linalg.norm(model.model.layers[LAYER].mlp.down_proj.weight.data, dim=1).cpu()
            
            coefs = m_coefs * value_norms.numpy()
            coefs_abs = np.absolute(m_coefs) * value_norms.numpy()
            coefs_sums.append(coefs_abs.sum())
            
            for index, prob in enumerate(coefs):
                coefs_.append((index, prob))
            for index, prob in enumerate(coefs_abs):
                coefs_abs_.append((index, prob))
                
            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:30]
            c_idx, c_vals = zip(*top_values)
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)

            top_values_abs = sorted(coefs_abs_, key=lambda x: x[1], reverse=True)[:30]
            c_idx_abs, c_vals_abs = zip(*top_values_abs)
            top_coef_abs_idx.append(c_idx_abs)
            top_coef_vals_abs.append(c_vals_abs)

            shuffle(coefs_)
            rand_idx, rand_vals = zip(*coefs_)
            rand_coef_idx.append(rand_idx[:30])
            rand_coef_vals.append(rand_vals[:30])
            shuffle(coefs_abs_)
            rand_idx_abs, rand_vals_abs = zip(*coefs_abs_)
            rand_coef_abs_idx.append(rand_idx_abs[:30])
            rand_coef_vals_abs.append(rand_vals_abs[:30])

            residual_p_probs, residual_p_tokens = zip(*sent_to_preds[key]['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)

            layer_p_probs, layer_p_tokens = zip(*sent_to_preds[key]['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)
            
        gold_token = sent_to_hidden_states[key]['gold_token']
        records.append({
            "sent": sent,
            "top_coef_idx": top_coef_idx,
            "top_coef_vals": top_coef_vals,
            "top_coef_abs_idx": top_coef_abs_idx,
            "top_coef_vals_abs": top_coef_vals_abs,
            "rand_coef_idx": rand_coef_idx,
            "rand_coef_vals": rand_coef_vals,
            "rand_coef_abs_idx": rand_coef_abs_idx,
            "rand_coef_vals_abs": rand_coef_vals_abs,
            "coefs_total_sum": coefs_sums,
            "residual_preds_probs": residual_preds_probs,
            "residual_preds_tokens": residual_preds_tokens,
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
            "layer_mlp_vec": mlp_vecs,
            "gold_token": gold_token
        })

    df = pd.DataFrame(records)
    return df


def parse_line(line):
    tokens = [
        token for token in line.split(' ')
        if token not in ['', '\n']
    ]
    if len(tokens) == 0:
        return None
    spaces = [True for _ in range(len(tokens) - 1)] + [False]
    assert len(tokens) == len(spaces), f"{len(tokens)} != {len(spaces)}"

    doc = spacy.tokens.doc.Doc(
        nlp.vocab, words=tokens, spaces=spaces)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    return [str(sent) for sent in doc.sents]


def load_wikitext_dataset(target_sentences=12000):
    """
    Load WikiText dataset with enhanced sentence collection.
    Tries multiple datasets and processing strategies to get enough sentences.
    """
    collected_sentences = []
    
    # Try WikiText-103 first (largest)
    datasets_to_try = [
        ("wikitext", "wikitext-103-v1", ["train", "validation", "test"]),
        ("wikitext", "wikitext-2-v1", ["train", "validation", "test"]),
    ]
    
    for dataset_name, config_name, splits in datasets_to_try:
        if len(collected_sentences) >= target_sentences:
            break
            
        print(f"Loading {dataset_name}-{config_name}...")
        
        for split in splits:
            if len(collected_sentences) >= target_sentences:
                break
                
            try:
                print(f"  Processing {split} split...")
                dataset = load_dataset(dataset_name, config_name, split=split)
                split_sentences = []
                
                for line in tqdm(dataset['text'], desc=f"Processing {split}"):
                    if len(collected_sentences) + len(split_sentences) >= target_sentences:
                        break
                        
                    if line.strip():  # Skip empty lines
                        sentences = parse_line(line)
                        if sentences is not None:
                            split_sentences.extend(sentences)
                
                collected_sentences.extend(split_sentences)
                print(f"  Collected {len(split_sentences)} sentences from {split}")
                
            except Exception as e:
                print(f"  Error loading {split} split: {e}")
                continue
    
    print(f"Total sentences collected: {len(collected_sentences)}")
    return collected_sentences


def setup_model_and_tokenizer(model_name, device):
    """Setup model and tokenizer with multi-GPU support"""
    model_type = 'gpt2' if 'gpt2' in model_name.lower() else 'llama'
    
    if model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:  # llama
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically distribute across available GPUs
        )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For GPT2, we still need to manually move to device since it's smaller
    if model_type == 'gpt2':
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default='gpt2', 
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3-8b', 'llama3-70b'],
        type=str, 
        help="Model name"
    )
    parser.add_argument(
        "--device", default='cuda:0', type=str, help="Primary device (for GPT2 models)"
    )
    parser.add_argument(
        "--max_sentences", default=10000, type=int, help="Target number of sentences to process"
    )
    parser.add_argument(
        "--output_path", default='model_df_10k.pkl', type=str, help="Output pickle file path"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for processing"
    )
    parser.add_argument(
        "--min_token_length", default=3, type=int, help="Minimum tokens per sentence"
    )
    parser.add_argument(
        "--max_token_length", default=128, type=int, help="Maximum tokens per sentence"
    )

    args = parser.parse_args()
    
    # Set up multi-GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 A100s
    
    # Map model names to HuggingFace model identifiers
    model_map = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
        'gpt2-xl': 'gpt2-xl',
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3-70b': 'meta-llama/Meta-Llama-3-70B'
    }
    
    model_name = model_map[args.model_name]
    device = "cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Target sentences: {args.max_sentences}")
    
    # Setup model and tokenizer
    print(f"Loading model: {model_name}")
    model, tokenizer, model_type = setup_model_and_tokenizer(model_name, device)
    
    # Load WikiText dataset with enhanced collection
    print("Loading WikiText dataset...")
    target_raw_sentences = args.max_sentences * 2  # Collect extra to account for filtering
    wiki_text_sentences_raw = load_wikitext_dataset(target_raw_sentences)
    
    # Filter sentences for quality and length
    print("Filtering sentences for quality...")
    wiki_text_sentences_filtered = []
    for sentence in tqdm(wiki_text_sentences_raw, desc="Filtering"):
        if is_valid_sentence(sentence, tokenizer, args.min_token_length, args.max_token_length):
            wiki_text_sentences_filtered.append(sentence)
        
        if len(wiki_text_sentences_filtered) >= args.max_sentences:
            break
    
    print(f"Filtered to {len(wiki_text_sentences_filtered)} valid sentences")
    
    # Shuffle for randomness
    shuffle(wiki_text_sentences_filtered)
    
    # Take exactly the number we want
    final_sentences = wiki_text_sentences_filtered[:args.max_sentences]
    print(f"Processing exactly {len(final_sentences)} sentences")
    
    # Process data
    print(f"Processing {len(final_sentences)} sentences with batch size {args.batch_size}...")
    df = get_examples_df(
        final_sentences, 
        model, 
        tokenizer, 
        model_type, 
        device,
        batch_size=args.batch_size
    )
    
    # Verify we got what we expected
    print(f"Final dataframe contains {len(df)} sentences")
    
    # Save results
    print(f"Saving results to {args.output_path}")
    df.to_pickle(args.output_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Target sentences: {args.max_sentences}")
    print(f"Raw sentences collected: {len(wiki_text_sentences_raw)}")
    print(f"Sentences after filtering: {len(wiki_text_sentences_filtered)}")
    print(f"Sentences actually processed: {len(df)}")
    print(f"Success rate: {len(df)/args.max_sentences*100:.1f}%")
    
    # Sample some sentences to verify quality
    print(f"\nSample sentences:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['sent']}")
    
    print("Done!")


if __name__ == "__main__":
    main()