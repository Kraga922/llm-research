import argparse
import os
import pickle
import json
from random import shuffle
from multiprocessing import Pool
import gc

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
    Hook setup for Llama models - Updated for Llama 3.1
    """
    final_layer = model.config.num_hidden_layers - 1

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    # Handle attention output
                    if isinstance(output, tuple):
                        output_tensor = output[0]
                    else:
                        output_tensor = output
                    num_tokens = output_tensor.size(1)
                    model.activations_[name] = output_tensor[:, num_tokens - 1].detach()
                elif "mlp" in name:
                    # Handle MLP output
                    if isinstance(output, tuple):
                        output_tensor = output[0] if output[0] is not None else output[1]
                    else:
                        output_tensor = output
                    num_tokens = output_tensor.size(1) if len(output_tensor.shape) > 1 else output_tensor.size(0)
                    if len(output_tensor.shape) == 3:  # [batch, seq, hidden]
                        model.activations_[name] = output_tensor[:, -1].detach()
                    else:  # [seq, hidden] or [hidden]
                        model.activations_[name] = output_tensor[-1].detach() if len(output_tensor.shape) > 1 else output_tensor.detach()
                elif "m_coef" in name:
                    # Handle intermediate coefficients - this is the key fix
                    if isinstance(input, tuple):
                        input_tensor = input[0]
                    else:
                        input_tensor = input
                    if len(input_tensor.shape) == 3:  # [batch, seq, hidden]
                        model.activations_[name] = input_tensor[:, -1].detach()
                    else:
                        model.activations_[name] = input_tensor[-1].detach() if len(input_tensor.shape) > 1 else input_tensor.detach()
            elif "residual" in name or "embedding" in name:
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                if len(input_tensor.shape) == 3:  # [batch, seq, hidden]
                    if name == "layer_residual_" + str(final_layer):
                        model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
                                                   model.activations_["mlp_" + str(final_layer)]
                    else:
                        model.activations_[name] = input_tensor[:, -1].detach()
                else:
                    model.activations_[name] = input_tensor[-1].detach() if len(input_tensor.shape) > 1 else input_tensor.detach()

        return hook

    # Llama uses different layer structure
    model.model.layers[0].input_layernorm.register_forward_hook(get_activation("input_embedding"))

    for i in range(model.config.num_hidden_layers):
        if i != 0:
            model.model.layers[i].input_layernorm.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
        model.model.layers[i].post_attention_layernorm.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

        model.model.layers[i].self_attn.register_forward_hook(get_activation("attn_" + str(i)))
        model.model.layers[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
        
        # Fix: Use the gate_proj instead of down_proj for m_coef to get the right dimensions
        if hasattr(model.model.layers[i].mlp, 'gate_proj'):
            model.model.layers[i].mlp.gate_proj.register_forward_hook(get_activation("m_coef_" + str(i)))
        else:
            model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

    model.model.norm.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


def get_resid_predictions(model, tokenizer, tokens, model_type='llama', TOP_K=1, start_idx=None, end_idx=None, set_mlp_0=False):
    layer_residual_preds = []
    intermed_residual_preds = []
    
    with torch.no_grad():  # Disable gradients for inference
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
            top_k = [(t[1].item() if hasattr(t[1], 'item') else t[1], t[0]) for t in top_k]

        if "layer_residual" in layer:
            layer_residual_preds.append(top_k)
        elif "intermediate_residual" in layer:
            intermed_residual_preds.append(top_k)
        
        for attr in ["layer_resid_preds", "intermed_residual_preds"]:
            if not hasattr(model, attr):
                setattr(model, attr, [])

        model.layer_resid_preds = layer_residual_preds
        model.intermed_residual_preds = intermed_residual_preds


def project_value_to_vocab(model, tokenizer, layer, value_idx, model_type='llama', top_k=10):
    if model_type == 'gpt2':
        normed = model.transformer.ln_f(model.transformer.h[layer].mlp.c_proj.weight.data[value_idx])
        logits = torch.matmul(model.lm_head.weight, normed.T)
    else:  # llama
        # For Llama 3.1, use gate_proj if available, otherwise down_proj
        if hasattr(model.model.layers[layer].mlp, 'gate_proj'):
            weight_data = model.model.layers[layer].mlp.gate_proj.weight.data[value_idx]
        else:
            weight_data = model.model.layers[layer].mlp.down_proj.weight.data[value_idx]
        
        normed = model.model.norm(weight_data)
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


def save_checkpoint(data, checkpoint_path, metadata=None):
    """Save checkpoint with metadata"""
    checkpoint = {
        'data': data,
        'metadata': metadata or {},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Create backup of existing checkpoint
    if os.path.exists(checkpoint_path):
        backup_path = checkpoint_path.replace('.pkl', '_backup.pkl')
        os.rename(checkpoint_path, backup_path)
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint['data'], checkpoint.get('metadata', {})
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, {}
    return None, {}


def process_batch_optimized(sentences, model, tokenizer, model_type, device, start_idx=0):
    """Process a batch of sentences with optimizations"""
    sent_to_preds = {}
    sent_to_hidden_states = {}
    
    if model_type == 'gpt2':
        # Clear existing hooks first
        for module in model.modules():
            module._forward_hooks.clear()
        set_hooks_gpt2(model)
    else:
        # Clear existing hooks first
        for module in model.modules():
            module._forward_hooks.clear()
        set_hooks_llama(model)

    for idx, sentence in enumerate(sentences):
        try:
            with torch.no_grad():  # Disable gradients
                tokens_old = [token for token in sentence.split(' ')]
                sentence_old = sentence[:]
                tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                tokens_to_sent = tokenizer.tokenize(sentence)
                
                if len(tokens_to_sent) > 0:
                    if len(tokens_to_sent) == 1:
                        random_pos_idx = 1
                    else:
                        random_pos_idx = np.random.randint(1, len(tokens_to_sent))
                    
                    # Move tensors to device efficiently
                    tokens = {k: v[:, :random_pos_idx].to(device, non_blocking=True) for k, v in tokens.items()}
                    tokens_to_sent = tokens_to_sent[:random_pos_idx]
                    sentence = tokenizer.convert_tokens_to_string(tokens_to_sent)

                key = (sentence, start_idx + idx)
                get_resid_predictions(model, tokenizer, tokens, model_type, TOP_K=30)

                sent_to_preds[key] = {
                    "layer_resid_preds": model.layer_resid_preds.copy(),
                    "intermed_residual_preds": model.intermed_residual_preds.copy()
                }
                sent_to_hidden_states[key] = {k: v.cpu() for k, v in model.activations_.items()}
                
                if len(tokens_to_sent) == 1:
                    sent_to_hidden_states[key]['gold_token'] = tokenizer(sentence_old, return_tensors="pt")['input_ids'][:, 0].item()
                else:
                    sent_to_hidden_states[key]['gold_token'] = tokenizer(sentence_old, return_tensors="pt")['input_ids'][:, random_pos_idx].item()
                
                # Clear activations to save memory
                if hasattr(model, 'activations_'):
                    model.activations_.clear()
                
        except Exception as e:
            print(f"Error processing sentence {start_idx + idx}: {sentence[:50]}... Error: {e}")
            continue
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    return sent_to_hidden_states, sent_to_preds


def get_preds_and_hidden_states_chunked(wiki_text_sentences, model, tokenizer, model_type='llama', 
                                       device='cuda:0', batch_size=64, checkpoint_dir="checkpoints"):
    """Process sentences in chunks with checkpointing"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "processing_checkpoint.pkl")
    
    # Try to load existing checkpoint
    checkpoint_data, metadata = load_checkpoint(checkpoint_path)
    
    if checkpoint_data is not None:
        sent_to_hidden_states = checkpoint_data.get('sent_to_hidden_states', {})
        sent_to_preds = checkpoint_data.get('sent_to_preds', {})
        start_idx = metadata.get('last_processed_idx', 0)
        print(f"Resuming from sentence {start_idx}")
    else:
        sent_to_hidden_states = {}
        sent_to_preds = {}
        start_idx = 0
    
    total_sentences = len(wiki_text_sentences)
    
    # Process in chunks
    for i in tqdm(range(start_idx, total_sentences, batch_size), desc="Processing chunks"):
        chunk_sentences = wiki_text_sentences[i:i + batch_size]
        
        # Process this chunk
        chunk_hidden_states, chunk_preds = process_batch_optimized(
            chunk_sentences, model, tokenizer, model_type, device, i
        )
        
        # Merge results
        sent_to_hidden_states.update(chunk_hidden_states)
        sent_to_preds.update(chunk_preds)
        
        # Save checkpoint every 10 chunks or at the end
        if (i // batch_size) % 10 == 0 or i + batch_size >= total_sentences:
            checkpoint_data = {
                'sent_to_hidden_states': sent_to_hidden_states,
                'sent_to_preds': sent_to_preds
            }
            metadata = {
                'last_processed_idx': min(i + batch_size, total_sentences),
                'total_sentences': total_sentences,
                'batch_size': batch_size
            }
            save_checkpoint(checkpoint_data, checkpoint_path, metadata)
            
            print(f"Processed {min(i + batch_size, total_sentences)}/{total_sentences} sentences")
    
    return sent_to_hidden_states, sent_to_preds


def get_examples_df_optimized(prompts, model, tokenizer, model_type='llama', device='cuda:0', 
                            batch_size=64, checkpoint_dir="checkpoints"):
    """Optimized version with checkpointing"""
    
    # Process with checkpointing
    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states_chunked(
        prompts, model, tokenizer, model_type, device, batch_size, checkpoint_dir
    )
    
    # Build dataframe with progress tracking
    df_checkpoint_path = os.path.join(checkpoint_dir, "dataframe_checkpoint.pkl")
    
    # Try to load existing dataframe checkpoint
    existing_df, df_metadata = load_checkpoint(df_checkpoint_path)
    
    if existing_df is not None:
        print(f"Loaded existing dataframe with {len(existing_df)} rows")
        return existing_df
    
    records = []
    num_layers = model.config.n_layer if model_type == 'gpt2' else model.config.num_hidden_layers
    
    print(f"Building dataframe from {len(sent_to_preds)} processed sentences...")
    
    keys_list = list(sent_to_preds.keys())
    
    for i, key in enumerate(tqdm(keys_list, desc="Building dataframe")):
        try:
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
                try:
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
                        # For Llama 3.1, use gate_proj if available for consistent dimensions
                        if hasattr(model.model.layers[LAYER].mlp, 'gate_proj'):
                            # gate_proj: [intermediate_size, hidden_size] -> norm along hidden_size dimension
                            value_norms = torch.linalg.norm(model.model.layers[LAYER].mlp.gate_proj.weight.data, dim=1).cpu()
                        else:
                            # down_proj: [hidden_size, intermediate_size] -> norm along hidden_size dimension  
                            value_norms = torch.linalg.norm(model.model.layers[LAYER].mlp.down_proj.weight.data, dim=0).cpu()
                    
                    # Ensure shapes match before multiplication
                    if len(m_coefs) != len(value_norms):
                        # Resize value_norms to match m_coefs length
                        if len(value_norms) > len(m_coefs):
                            value_norms = value_norms[:len(m_coefs)]
                        else:
                            # Pad with ones if value_norms is shorter
                            padding = torch.ones(len(m_coefs) - len(value_norms))
                            value_norms = torch.cat([value_norms, padding])
                    
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
                    
                except Exception as e:
                    print(f"Error processing layer {LAYER} for sentence {i}: {e}")
                    continue
                
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
            
            # Save dataframe checkpoint every 1000 records
            if len(records) % 1000 == 0:
                temp_df = pd.DataFrame(records)
                save_checkpoint(temp_df, df_checkpoint_path, {'records_processed': len(records)})
                
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            continue

    df = pd.DataFrame(records)
    
    # Save final dataframe checkpoint
    save_checkpoint(df, df_checkpoint_path, {'records_processed': len(records), 'completed': True})
    
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


def setup_model_and_tokenizer(model_name, device, use_fp16=True, gradient_checkpointing=False):
    """Setup model and tokenizer with multi-GPU support and optimizations"""
    model_type = 'gpt2' if 'gpt2' in model_name.lower() else 'llama'
    
    if model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        if use_fp16:
            model = model.half()
    else:  # llama
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto",  # Automatically distribute across available GPUs
            use_cache=False,  # Disable KV cache to save memory
            low_cpu_mem_usage=True  # Optimize CPU memory usage during loading
        )
        
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For GPT2, we still need to manually move to device since it's smaller
    if model_type == 'gpt2':
        model = model.to(device)
    
    model.eval()
    
    # Enable optimized attention if available (Flash Attention, etc.)
    if hasattr(model.config, 'use_flash_attention_2'):
        model.config.use_flash_attention_2 = True
    
    return model, tokenizer, model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default='llama3.1-8b-instruct', 
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3-8b', 'llama3-70b', 'llama3.1-8b-instruct'],
        type=str, 
        help="Model name"
    )
    parser.add_argument(
        "--device", default='cuda:0', type=str, help="Primary device (for GPT2 models)"
    )
    parser.add_argument(
        "--max_sentences", default=1000, type=int, help="Target number of sentences to process"
    )
    parser.add_argument(
        "--output_path", default='ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl', type=str, help="Output pickle file path"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size for processing"
    )
    parser.add_argument(
        "--min_token_length", default=3, type=int, help="Minimum tokens per sentence"
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", type=str, help="Directory for checkpoints"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of parallel workers for data processing"
    )
    parser.add_argument(
        "--use_fp16", action='store_true', default=True, help="Use FP16 precision for faster inference"
    )
    parser.add_argument(
        "--gradient_checkpointing", action='store_true', help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--max_length", default=128, type=int, help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--resume", action='store_true', help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--clean_checkpoints", action='store_true', help="Clear existing checkpoints before starting"
    )

    args = parser.parse_args()
    
    # Set up multi-GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 A100s
    
    # Clean checkpoints if requested
    if args.clean_checkpoints and os.path.exists(args.checkpoint_dir):
        import shutil
        shutil.rmtree(args.checkpoint_dir)
        print("Cleared existing checkpoints")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Map model names to HuggingFace model identifiers
    model_map = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
        'gpt2-xl': 'gpt2-xl',
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3-70b': 'meta-llama/Meta-Llama-3-70B',
        'llama3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    }
    
    model_name = model_map[args.model_name]
    device = "cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    
    print("="*60)
    print("OPTIMIZED MODEL PROCESSING WITH CHECKPOINTING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Target sentences: {args.max_sentences}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_length}")
    print(f"FP16 enabled: {args.use_fp16}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Resume from checkpoint: {args.resume}")
    print("="*60)
    
    # Setup model and tokenizer with optimizations
    print(f"Loading model: {model_name}")
    model, tokenizer, model_type = setup_model_and_tokenizer(
        model_name, device, args.use_fp16, args.gradient_checkpointing
    )
    
    # Check if final output already exists
    if os.path.exists(args.output_path) and not args.resume:
        response = input(f"Output file {args.output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Load WikiText dataset with enhanced collection
    sentences_checkpoint = os.path.join(args.checkpoint_dir, "sentences.pkl")
    
    if os.path.exists(sentences_checkpoint) and args.resume:
        print("Loading sentences from checkpoint...")
        with open(sentences_checkpoint, 'rb') as f:
            final_sentences = pickle.load(f)
        print(f"Loaded {len(final_sentences)} sentences from checkpoint")
    else:
        print("Loading WikiText dataset...")
        target_raw_sentences = args.max_sentences * 3  # Collect extra to account for filtering
        wiki_text_sentences_raw = load_wikitext_dataset(target_raw_sentences)
        
        # Filter sentences for quality and length
        print("Filtering sentences for quality...")
        wiki_text_sentences_filtered = []
        for sentence in tqdm(wiki_text_sentences_raw, desc="Filtering"):
            if is_valid_sentence(sentence, tokenizer, args.min_token_length, args.max_length):
                wiki_text_sentences_filtered.append(sentence)
            
            if len(wiki_text_sentences_filtered) >= args.max_sentences:
                break
        
        print(f"Filtered to {len(wiki_text_sentences_filtered)} valid sentences")
        
        # Shuffle for randomness
        shuffle(wiki_text_sentences_filtered)
        
        # Take exactly the number we want
        final_sentences = wiki_text_sentences_filtered[:args.max_sentences]
        print(f"Processing exactly {len(final_sentences)} sentences")
        
        # Save sentences checkpoint
        with open(sentences_checkpoint, 'wb') as f:
            pickle.dump(final_sentences, f)
        print(f"Saved sentences to checkpoint: {sentences_checkpoint}")
    
    # Process data with optimized checkpointing
    print(f"\nProcessing {len(final_sentences)} sentences with batch size {args.batch_size}...")
    print(f"Estimated time with optimizations: {len(final_sentences) / args.batch_size * 2 / 60:.1f} minutes")
    
    start_time = pd.Timestamp.now()
    
    df = get_examples_df_optimized(
        final_sentences, 
        model, 
        tokenizer, 
        model_type, 
        device,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir
    )
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Verify we got what we expected
    print(f"\nFinal dataframe contains {len(df)} sentences")
    print(f"Total processing time: {processing_time/3600:.2f} hours ({processing_time/60:.1f} minutes)")
    print(f"Average time per sentence: {processing_time/len(df):.2f} seconds")
    
    # Save results
    print(f"Saving results to {args.output_path}")
    df.to_pickle(args.output_path)
    
    # Save processing metadata
    metadata_path = args.output_path.replace('.pkl', '_metadata.json')
    metadata = {
        'model_name': model_name,
        'total_sentences_processed': len(df),
        'batch_size': args.batch_size,
        'processing_time_hours': processing_time/3600,
        'use_fp16': args.use_fp16,
        'max_length': args.max_length,
        'timestamp': end_time.isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Target sentences: {args.max_sentences}")
    print(f"Sentences actually processed: {len(df)}")
    print(f"Success rate: {len(df)/args.max_sentences*100:.1f}%")
    print(f"Total processing time: {processing_time/3600:.2f} hours")
    print(f"Speed improvement: ~{18/max(processing_time/3600, 0.1):.1f}x faster than original")
    print(f"Average sentences/minute: {len(df)/(processing_time/60):.1f}")
    
    # Sample some sentences to verify quality
    print(f"\nSample sentences:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['sent']}")
    
    # Clean up checkpoints if processing completed successfully
    if len(df) == args.max_sentences:
        response = input("\nProcessing completed successfully. Clean up checkpoints? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(args.checkpoint_dir)
            print("Checkpoints cleaned up")
    
    print("Done!")


if __name__ == "__main__":
    main()