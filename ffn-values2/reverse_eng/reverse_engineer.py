# import pickle
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# from collections import defaultdict, Counter
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import warnings
# warnings.filterwarnings('ignore')

# class ActivationReverseEngineer:
#     """
#     Reverse engineer activation patterns back to text that would produce them.
#     """
    
#     def __init__(self, ffn_projections_path: str, model_df_path: str, model_name: str = "gpt2"):
#         """
#         Initialize the reverse engineer with your pkl files.
        
#         Args:
#             ffn_projections_path: Path to ffn_projections.pkl
#             model_df_path: Path to model_df_10k.pkl
#             model_name: GPT2 model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
#         """
#         self.model_name = model_name
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
#         # Load your data
#         print("Loading FFN projections...")
#         with open(ffn_projections_path, 'rb') as f:
#             self.ffn_projections = pickle.load(f)
        
#         print("Loading model DataFrame...")
#         with open(model_df_path, 'rb') as f:
#             self.model_df = pickle.load(f)
        
#         print(f"Loaded {len(self.model_df)} sentences from WikiText")
#         print(f"FFN projections available for {len(self.ffn_projections)} (layer, dim) pairs")
        
#         # Build reverse mappings
#         self._build_reverse_mappings()
    
#     def _build_reverse_mappings(self):
#         """
#         Build reverse mappings from tokens to FFN dimensions that activate them.
#         """
#         print("Building reverse mappings...")
        
#         # Map tokens to FFN dimensions that strongly activate them
#         self.token_to_ffn_dims = defaultdict(list)
        
#         for (layer, dim), top_tokens in self.ffn_projections.items():
#             for token in top_tokens:
#                 self.token_to_ffn_dims[token].append((layer, dim))
        
#         print(f"Built reverse mappings for {len(self.token_to_ffn_dims)} unique tokens")
    
#     def find_sentences_with_activation_pattern(self, 
#                                                target_tokens: List[str], 
#                                                layer_focus: Optional[int] = None,
#                                                top_k: int = 10) -> List[Dict[str, Any]]:
#         """
#         Find sentences that would produce activation patterns for the given tokens.
        
#         Args:
#             target_tokens: List of tokens to find activation patterns for
#             layer_focus: Specific layer to focus on (None for all layers)
#             top_k: Number of top sentences to return
            
#         Returns:
#             List of dictionaries with sentence info and activation scores
#         """
#         print(f"Finding sentences with activation patterns for tokens: {target_tokens}")
        
#         # Get FFN dimensions relevant to our target tokens
#         relevant_ffn_dims = set()
#         for token in target_tokens:
#             if token in self.token_to_ffn_dims:
#                 dims = self.token_to_ffn_dims[token]
#                 if layer_focus is not None:
#                     dims = [(l, d) for l, d in dims if l == layer_focus]
#                 relevant_ffn_dims.update(dims)
        
#         print(f"Found {len(relevant_ffn_dims)} relevant FFN dimensions")
        
#         # Score sentences based on activation patterns
#         sentence_scores = []
        
#         for idx, row in self.model_df.iterrows():
#             sentence = row['sent']
#             score = self._calculate_activation_score(row, relevant_ffn_dims, target_tokens)
            
#             sentence_scores.append({
#                 'sentence': sentence,
#                 'score': score,
#                 'row_idx': idx,
#                 'activation_details': self._get_activation_details(row, relevant_ffn_dims)
#             })
        
#         # Sort by score and return top-k
#         sentence_scores.sort(key=lambda x: x['score'], reverse=True)
#         return sentence_scores[:top_k]
    
#     def _calculate_activation_score(self, row: pd.Series, 
#                                    relevant_ffn_dims: set, 
#                                    target_tokens: List[str]) -> float:
#         """
#         Calculate how well a sentence's activation pattern matches our target.
#         """
#         score = 0.0
        
#         # Check top coefficient indices and values
#         for layer_idx in range(len(row['top_coef_idx'])):
#             top_indices = row['top_coef_idx'][layer_idx]
#             top_values = row['top_coef_vals'][layer_idx]
            
#             for dim_idx, coef_val in zip(top_indices, top_values):
#                 if (layer_idx, dim_idx) in relevant_ffn_dims:
#                     # Higher coefficient = stronger activation
#                     score += abs(coef_val)
        
#         # Bonus for sentences that actually contain target tokens
#         sentence_tokens = self.tokenizer.encode(row['sent'])
#         sentence_token_strs = [self.tokenizer.decode([t]) for t in sentence_tokens]
        
#         for target_token in target_tokens:
#             if target_token in sentence_token_strs:
#                 score += 1.0  # Bonus for containing target token
        
#         return score
    
#     def _get_activation_details(self, row: pd.Series, relevant_ffn_dims: set) -> Dict[str, Any]:
#         """
#         Get detailed activation information for a sentence.
#         """
#         details = {
#             'layers_activated': [],
#             'top_predictions': [],
#             'mlp_stats': []
#         }
        
#         # Check which layers have relevant activations
#         for layer_idx in range(len(row['top_coef_idx'])):
#             top_indices = row['top_coef_idx'][layer_idx]
#             top_values = row['top_coef_vals'][layer_idx]
            
#             layer_activations = []
#             for dim_idx, coef_val in zip(top_indices, top_values):
#                 if (layer_idx, dim_idx) in relevant_ffn_dims:
#                     layer_activations.append({
#                         'dim': dim_idx,
#                         'coefficient': coef_val,
#                         'activated_tokens': self.ffn_projections.get((layer_idx, dim_idx), [])
#                     })
            
#             if layer_activations:
#                 details['layers_activated'].append({
#                     'layer': layer_idx,
#                     'activations': layer_activations
#                 })
            
#             # Add prediction info
#             if 'layer_preds_tokens' in row and layer_idx < len(row['layer_preds_tokens']):
#                 pred_tokens = row['layer_preds_tokens'][layer_idx]
#                 pred_probs = row['layer_preds_probs'][layer_idx]
                
#                 top_pred_strs = []
#                 for token_idx, prob in zip(pred_tokens[:5], pred_probs[:5]):
#                     token_str = self.tokenizer.decode([token_idx])
#                     top_pred_strs.append(f"{token_str}({prob:.3f})")
                
#                 details['top_predictions'].append({
#                     'layer': layer_idx,
#                     'predictions': top_pred_strs
#                 })
        
#         return details
    
#     def reverse_engineer_from_activation_vector(self, 
#                                                activation_vector: np.ndarray,
#                                                layer_idx: int,
#                                                top_k_tokens: int = 20) -> List[Tuple[str, float]]:
#         """
#         Given an activation vector, find the tokens it would most likely activate.
        
#         Args:
#             activation_vector: The activation vector to reverse engineer
#             layer_idx: Which layer this vector comes from
#             top_k_tokens: How many top tokens to return
            
#         Returns:
#             List of (token, activation_strength) tuples
#         """
#         token_scores = []
        
#         # For each dimension in the activation vector
#         for dim_idx, activation_strength in enumerate(activation_vector):
#             if abs(activation_strength) > 0.1:  # Only consider significant activations
#                 # Get tokens that this dimension typically activates
#                 ffn_key = (layer_idx, dim_idx)
#                 if ffn_key in self.ffn_projections:
#                     top_tokens = self.ffn_projections[ffn_key]
                    
#                     # Score each token by activation strength
#                     for token in top_tokens:
#                         token_scores.append((token, abs(activation_strength)))
        
#         # Aggregate scores for tokens that appear multiple times
#         token_score_dict = defaultdict(float)
#         for token, score in token_scores:
#             token_score_dict[token] += score
        
#         # Sort and return top-k
#         sorted_tokens = sorted(token_score_dict.items(), key=lambda x: x[1], reverse=True)
#         return sorted_tokens[:top_k_tokens]
    
#     def generate_candidate_texts(self, 
#                                 target_tokens: List[str],
#                                 context_length: int = 50) -> List[str]:
#         """
#         Generate candidate texts that might produce the desired activation patterns.
        
#         Args:
#             target_tokens: Tokens we want to see activated
#             context_length: Length of context to generate
            
#         Returns:
#             List of candidate text strings
#         """
#         candidates = []
        
#         # Method 1: Find sentences from dataset that contain target tokens
#         dataset_candidates = []
#         for idx, row in self.model_df.iterrows():
#             sentence = row['sent']
#             sentence_tokens = self.tokenizer.encode(sentence)
#             sentence_token_strs = [self.tokenizer.decode([t]) for t in sentence_tokens]
            
#             # Check if sentence contains any target tokens
#             overlap = set(target_tokens) & set(sentence_token_strs)
#             if overlap:
#                 dataset_candidates.append({
#                     'text': sentence,
#                     'overlap_tokens': list(overlap),
#                     'overlap_count': len(overlap)
#                 })
        
#         # Sort by overlap and take top candidates
#         dataset_candidates.sort(key=lambda x: x['overlap_count'], reverse=True)
#         candidates.extend([c['text'] for c in dataset_candidates[:5]])
        
#         # Method 2: Create synthetic sentences with target tokens
#         synthetic_templates = [
#             "The {token} was very important in the context.",
#             "When discussing {token}, we should consider the implications.",
#             "The concept of {token} has been widely studied.",
#             "In recent years, {token} has become increasingly relevant.",
#             "The {token} represents a significant development."
#         ]
        
#         for token in target_tokens:
#             for template in synthetic_templates:
#                 candidates.append(template.format(token=token))
        
#         return candidates
    
#     def analyze_text_activation_potential(self, text: str) -> Dict[str, Any]:
#         """
#         Analyze what activation patterns a given text might produce.
        
#         Args:
#             text: Input text to analyze
            
#         Returns:
#             Dictionary with analysis results
#         """
#         tokens = self.tokenizer.encode(text)
#         token_strs = [self.tokenizer.decode([t]) for t in tokens]
        
#         analysis = {
#             'text': text,
#             'tokens': token_strs,
#             'potential_activations': defaultdict(list),
#             'layer_analysis': defaultdict(dict)
#         }
        
#         # For each token, find which FFN dimensions it typically activates
#         for token_str in token_strs:
#             if token_str in self.token_to_ffn_dims:
#                 relevant_dims = self.token_to_ffn_dims[token_str]
                
#                 for layer, dim in relevant_dims:
#                     analysis['potential_activations'][layer].append({
#                         'token': token_str,
#                         'dim': dim,
#                         'typical_activations': self.ffn_projections.get((layer, dim), [])
#                     })
        
#         return analysis

# def main():
#     """
#     Example usage of the reverse engineer.
#     """
#     # Initialize the reverse engineer
#     re = ActivationReverseEngineer(
#         ffn_projections_path="ffn_projections.pkl",
#         model_df_path="model_df_10k.pkl",
#         model_name="gpt2"
#     )
    
#     # Example 1: Find sentences that would activate patterns for specific tokens
#     print("\n" + "="*60)
#     print("EXAMPLE 1: Finding sentences with activation patterns")
#     print("="*60)
    
#     target_tokens = ["the", "and", "of"]  # Example tokens
#     results = re.find_sentences_with_activation_pattern(
#         target_tokens=target_tokens,
#         layer_focus=None,  # All layers
#         top_k=5
#     )
    
#     for i, result in enumerate(results):
#         print(f"\nRank {i+1} (Score: {result['score']:.3f}):")
#         print(f"Sentence: {result['sentence']}")
#         print(f"Layers activated: {len(result['activation_details']['layers_activated'])}")
    
#     # Example 2: Reverse engineer from activation vector
#     print("\n" + "="*60)
#     print("EXAMPLE 2: Reverse engineering from activation vector")
#     print("="*60)
    
#     # Create a sample activation vector (normally you'd get this from your model)
#     sample_activation = np.random.randn(768)  # GPT2 hidden size
#     layer_idx = 6  # Example layer
    
#     token_predictions = re.reverse_engineer_from_activation_vector(
#         activation_vector=sample_activation,
#         layer_idx=layer_idx,
#         top_k_tokens=10
#     )
    
#     print(f"Top tokens predicted from activation vector at layer {layer_idx}:")
#     for token, score in token_predictions:
#         print(f"  {token}: {score:.3f}")
    
#     # Example 3: Generate candidate texts
#     print("\n" + "="*60)
#     print("EXAMPLE 3: Generating candidate texts")
#     print("="*60)
    
#     candidates = re.generate_candidate_texts(target_tokens=["science", "research"])
#     print("Candidate texts that might produce desired activations:")
#     for i, candidate in enumerate(candidates[:5]):
#         print(f"{i+1}. {candidate}")
    
#     # Example 4: Analyze text activation potential
#     print("\n" + "="*60)
#     print("EXAMPLE 4: Analyzing text activation potential")
#     print("="*60)
    
#     test_text = "The research in artificial intelligence has advanced significantly."
#     analysis = re.analyze_text_activation_potential(test_text)
    
#     print(f"Text: {analysis['text']}")
#     print(f"Tokens: {analysis['tokens']}")
#     print(f"Layers with potential activations: {list(analysis['potential_activations'].keys())}")

# if __name__ == "__main__":
#     main()


# """
# Loading FFN projections...
# Loading model DataFrame...
# Loaded 1107 sentences from WikiText
# FFN projections available for 36864 (layer, dim) pairs
# Building reverse mappings...
# Built reverse mappings for 44494 unique tokens

# ============================================================
# EXAMPLE 1: Finding sentences with activation patterns
# ============================================================
# Finding sentences with activation patterns for tokens: ['the', 'and', 'of']
# Found 176 relevant FFN dimensions

# Rank 1 (Score: 96.206):
# Sentence: Some observers thought Burke was at his peak in terms
# Layers activated: 4

# Rank 2 (Score: 94.209):
# Sentence: Canadian Online Explorer writer Jon Waldman rated the entire event a 7 out of 10 , which was lower than the 8 out
# Layers activated: 4

# Rank 3 (Score: 89.515):
# Sentence: He was named the 1980 Indianapolis 500 Rookie of
# Layers activated: 3

# Rank 4 (Score: 81.156):
# Sentence: A two @-@ week work @-@ in was held at the end
# Layers activated: 5

# Rank 5 (Score: 79.298):
# Sentence: She went on to appear in Gone With
# Layers activated: 2

# ============================================================
# EXAMPLE 2: Reverse engineering from activation vector
# ============================================================
# Top tokens predicted from activation vector at layer 6:
#   ��: 40.551
#   �: 33.039
#   ,: 27.277
#    the: 24.986
#    and: 21.094
#   -: 19.604
#   .: 18.585
#    in: 15.322
#    a: 12.862
#   Reviewer: 9.935

# ============================================================
# EXAMPLE 3: Generating candidate texts
# ============================================================
# Candidate texts that might produce desired activations:
# 1. The science was very important in the context.
# 2. When discussing science, we should consider the implications.
# 3. The concept of science has been widely studied.
# 4. In recent years, science has become increasingly relevant.
# 5. The science represents a significant development.

# ============================================================
# EXAMPLE 4: Analyzing text activation potential
# ============================================================
# Text: The research in artificial intelligence has advanced significantly.
# Tokens: ['The', ' research', ' in', ' artificial', ' intelligence', ' has', ' advanced', ' significantly', '.']
# Layers with potential activations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


########## Larger DF

# Loading FFN projections...
# Loading model DataFrame...
# Loaded 10000 sentences from WikiText
# FFN projections available for 36864 (layer, dim) pairs
# Building reverse mappings...
# Built reverse mappings for 44494 unique tokens

# ============================================================
# EXAMPLE 1: Finding sentences with activation patterns
# ============================================================
# Finding sentences with activation patterns for tokens: ['the', 'and', 'of']
# Found 176 relevant FFN dimensions

# Rank 1 (Score: 119.068):
# Sentence: In 1885 , Andrew P. Morgan proposed that differences in microscopic characteristics warranted the creation
# Layers activated: 5

# Rank 2 (Score: 118.192):
# Sentence: Human remains were discovered , and placed into the possession
# Layers activated: 4

# Rank 3 (Score: 110.062):
# Sentence: In 2014 , Fernandez was named " Woman Of The Year " by PETA ( India ) for advocating the protection
# Layers activated: 6

# Rank 4 (Score: 107.976):
# Sentence: He expressed little interest in his relatives ; in later life he saw no reason to have a social relationship with people purely on the basis
# Layers activated: 6

# Rank 5 (Score: 104.486):
# Sentence: The Johnson – Corey – Chaykovsky reaction ( sometimes referred to as the Corey – Chaykovsky reaction or CCR ) is a chemical reaction used in organic chemistry for the synthesis
# Layers activated: 5

# ============================================================
# EXAMPLE 2: Reverse engineering from activation vector
# ============================================================
# Top tokens predicted from activation vector at layer 6:
#   ��: 33.525
#   �: 30.059
#    the: 18.988
#   ,: 18.566
#    and: 17.445
#   -: 14.314
#    in: 11.095
#   .: 11.049
#    a: 9.385
#   ciating: 8.593

# ============================================================
# EXAMPLE 3: Generating candidate texts
# ============================================================
# Candidate texts that might produce desired activations:
# 1. The science was very important in the context.
# 2. When discussing science, we should consider the implications.
# 3. The concept of science has been widely studied.
# 4. In recent years, science has become increasingly relevant.
# 5. The science represents a significant development.

# ============================================================
# EXAMPLE 4: Analyzing text activation potential
# ============================================================
# Text: The research in artificial intelligence has advanced significantly.
# Tokens: ['The', ' research', ' in', ' artificial', ' intelligence', ' has', ' advanced', ' significantly', '.']
# Layers with potential activations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# """

# import pickle
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# from collections import defaultdict, Counter
# import torch
# import torch.nn.functional as F
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr
# import json
# from datetime import datetime
# warnings.filterwarnings('ignore')

# class ActivationReverseEngineer:
#     """
#     Enhanced reverse engineer activation patterns back to text with comprehensive testing.
#     """
    
#     def __init__(self, ffn_projections_path: str, model_df_path: str, model_name: str = "gpt2"):
#         """
#         Initialize the reverse engineer with your pkl files.
        
#         Args:
#             ffn_projections_path: Path to ffn_projections.pkl
#             model_df_path: Path to model_df_10k.pkl
#             model_name: GPT2 model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
#         """
#         self.model_name = model_name
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
        
#         # Load your data
#         print("Loading FFN projections...")
#         with open(ffn_projections_path, 'rb') as f:
#             self.ffn_projections = pickle.load(f)
        
#         print("Loading model DataFrame...")
#         with open(model_df_path, 'rb') as f:
#             self.model_df = pickle.load(f)
        
#         print(f"Loaded {len(self.model_df)} sentences from WikiText")
#         print(f"FFN projections available for {len(self.ffn_projections)} (layer, dim) pairs")
        
#         # Build reverse mappings and analysis structures
#         self._build_reverse_mappings()
#         self._analyze_dataset_statistics()
        
#         # Initialize test results storage
#         self.test_results = {
#             'timestamp': datetime.now().isoformat(),
#             'model_name': model_name,
#             'dataset_size': len(self.model_df),
#             'tests': {}
#         }
    
#     def _build_reverse_mappings(self):
#         """
#         Build comprehensive reverse mappings from tokens to FFN dimensions.
#         """
#         print("Building reverse mappings...")
        
#         # Map tokens to FFN dimensions that strongly activate them
#         self.token_to_ffn_dims = defaultdict(list)
#         self.layer_to_active_tokens = defaultdict(set)
#         self.dim_activation_stats = defaultdict(lambda: {'count': 0, 'tokens': set()})
        
#         for (layer, dim), top_tokens in self.ffn_projections.items():
#             for rank, token in enumerate(top_tokens):
#                 self.token_to_ffn_dims[token].append((layer, dim, rank))
#                 self.layer_to_active_tokens[layer].add(token)
#                 self.dim_activation_stats[(layer, dim)]['count'] += 1
#                 self.dim_activation_stats[(layer, dim)]['tokens'].add(token)
        
#         print(f"Built reverse mappings for {len(self.token_to_ffn_dims)} unique tokens")
#         print(f"Average FFN dimensions per token: {np.mean([len(dims) for dims in self.token_to_ffn_dims.values()]):.2f}")
    
#     def _analyze_dataset_statistics(self):
#         """
#         Analyze dataset statistics for better understanding and validation.
#         """
#         print("Analyzing dataset statistics...")
        
#         self.dataset_stats = {}
        
#         # Token distribution analysis
#         all_tokens = []
#         sentence_lengths = []
        
#         for _, row in self.model_df.iterrows():
#             sentence = row['sent']
#             tokens = self.tokenizer.encode(sentence)
#             all_tokens.extend(tokens)
#             sentence_lengths.append(len(tokens))
        
#         self.dataset_stats['total_tokens'] = len(all_tokens)
#         self.dataset_stats['unique_tokens'] = len(set(all_tokens))
#         self.dataset_stats['avg_sentence_length'] = np.mean(sentence_lengths)
#         self.dataset_stats['token_frequency'] = Counter(all_tokens)
        
#         # Activation pattern statistics
#         layer_activation_counts = defaultdict(int)
#         for _, row in self.model_df.iterrows():
#             for layer_idx in range(len(row['top_coef_idx'])):
#                 layer_activation_counts[layer_idx] += len(row['top_coef_idx'][layer_idx])
        
#         self.dataset_stats['layer_activation_counts'] = dict(layer_activation_counts)
        
#         print(f"Dataset contains {self.dataset_stats['unique_tokens']} unique tokens")
#         print(f"Average sentence length: {self.dataset_stats['avg_sentence_length']:.2f} tokens")
    
#     def find_sentences_with_activation_pattern(self, 
#                                                target_tokens: List[str], 
#                                                layer_focus: Optional[int] = None,
#                                                top_k: int = 10,
#                                                min_score_threshold: float = 0.1) -> List[Dict[str, Any]]:
#         """
#         Enhanced sentence finding with better scoring and validation.
#         """
#         print(f"Finding sentences with activation patterns for tokens: {target_tokens}")
        
#         # Validate input tokens
#         valid_tokens = []
#         for token in target_tokens:
#             if token in self.token_to_ffn_dims:
#                 valid_tokens.append(token)
#             else:
#                 print(f"Warning: Token '{token}' not found in FFN projections")
        
#         if not valid_tokens:
#             print("No valid tokens found in FFN projections!")
#             return []
        
#         # Get FFN dimensions relevant to our target tokens
#         relevant_ffn_dims = set()
#         token_dim_mapping = {}
        
#         for token in valid_tokens:
#             dims = [(l, d) for l, d, r in self.token_to_ffn_dims[token]]
#             if layer_focus is not None:
#                 dims = [(l, d) for l, d in dims if l == layer_focus]
#             relevant_ffn_dims.update(dims)
#             token_dim_mapping[token] = dims
        
#         print(f"Found {len(relevant_ffn_dims)} relevant FFN dimensions across {len(set(d[0] for d in relevant_ffn_dims))} layers")
        
#         # Score sentences with enhanced metrics
#         sentence_scores = []
        
#         for idx, row in self.model_df.iterrows():
#             sentence = row['sent']
#             score_details = self._calculate_enhanced_activation_score(
#                 row, relevant_ffn_dims, valid_tokens, token_dim_mapping
#             )
            
#             if score_details['total_score'] >= min_score_threshold:
#                 sentence_scores.append({
#                     'sentence': sentence,
#                     'score': score_details['total_score'],
#                     'row_idx': idx,
#                     'score_details': score_details,
#                     'activation_details': self._get_enhanced_activation_details(row, relevant_ffn_dims, valid_tokens)
#                 })
        
#         # Sort by score and return top-k
#         sentence_scores.sort(key=lambda x: x['score'], reverse=True)
#         return sentence_scores[:top_k]
    
#     def _calculate_enhanced_activation_score(self, row: pd.Series, 
#                                            relevant_ffn_dims: set, 
#                                            target_tokens: List[str],
#                                            token_dim_mapping: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Any]:
#         """
#         Enhanced scoring with multiple metrics and detailed breakdown.
#         """
#         score_details = {
#             'activation_strength': 0.0,
#             'token_presence_bonus': 0.0,
#             'layer_diversity_bonus': 0.0,
#             'prediction_alignment': 0.0,
#             'total_score': 0.0,
#             'layer_scores': {},
#             'matched_dimensions': []
#         }
        
#         # 1. Activation strength score
#         layers_with_activations = set()
#         for layer_idx in range(len(row['top_coef_idx'])):
#             top_indices = row['top_coef_idx'][layer_idx]
#             top_values = row['top_coef_vals'][layer_idx]
#             layer_score = 0.0
            
#             for dim_idx, coef_val in zip(top_indices, top_values):
#                 if (layer_idx, dim_idx) in relevant_ffn_dims:
#                     activation_strength = abs(coef_val)
#                     score_details['activation_strength'] += activation_strength
#                     layer_score += activation_strength
#                     layers_with_activations.add(layer_idx)
#                     score_details['matched_dimensions'].append((layer_idx, dim_idx, coef_val))
            
#             if layer_score > 0:
#                 score_details['layer_scores'][layer_idx] = layer_score
        
#         # 2. Token presence bonus
#         sentence_tokens = self.tokenizer.encode(row['sent'])
#         sentence_token_strs = [self.tokenizer.decode([t]).strip() for t in sentence_tokens]
        
#         for target_token in target_tokens:
#             # Exact match
#             if target_token in sentence_token_strs:
#                 score_details['token_presence_bonus'] += 2.0
#             # Partial match (substring)
#             elif any(target_token.lower() in token.lower() for token in sentence_token_strs):
#                 score_details['token_presence_bonus'] += 1.0
        
#         # 3. Layer diversity bonus (activations across multiple layers is good)
#         score_details['layer_diversity_bonus'] = len(layers_with_activations) * 0.5
        
#         # 4. Prediction alignment score
#         for layer_idx in range(min(len(row.get('layer_preds_tokens', [])), len(row['top_coef_idx']))):
#             if layer_idx in score_details['layer_scores']:
#                 pred_tokens = row['layer_preds_tokens'][layer_idx][:5]  # Top 5 predictions
#                 pred_token_strs = [self.tokenizer.decode([t]).strip() for t in pred_tokens]
                
#                 for target_token in target_tokens:
#                     if target_token in pred_token_strs:
#                         # Higher score for higher-ranked predictions
#                         rank = pred_token_strs.index(target_token)
#                         score_details['prediction_alignment'] += (5 - rank) / 5.0
        
#         # Calculate total score
#         score_details['total_score'] = (
#             score_details['activation_strength'] * 1.0 +
#             score_details['token_presence_bonus'] * 0.5 +
#             score_details['layer_diversity_bonus'] * 0.3 +
#             score_details['prediction_alignment'] * 0.7
#         )
        
#         return score_details
    
#     def _get_enhanced_activation_details(self, row: pd.Series, 
#                                        relevant_ffn_dims: set, 
#                                        target_tokens: List[str]) -> Dict[str, Any]:
#         """
#         Get comprehensive activation details for analysis.
#         """
#         details = {
#             'layers_activated': [],
#             'top_predictions': [],
#             'token_analysis': {},
#             'activation_summary': {}
#         }
        
#         total_activations = 0
#         max_activation = 0
        
#         # Analyze each layer
#         for layer_idx in range(len(row['top_coef_idx'])):
#             top_indices = row['top_coef_idx'][layer_idx]
#             top_values = row['top_coef_vals'][layer_idx]
            
#             layer_activations = []
#             layer_activation_sum = 0
            
#             for dim_idx, coef_val in zip(top_indices, top_values):
#                 if (layer_idx, dim_idx) in relevant_ffn_dims:
#                     activation_strength = abs(coef_val)
#                     total_activations += activation_strength
#                     max_activation = max(max_activation, activation_strength)
#                     layer_activation_sum += activation_strength
                    
#                     # Get the tokens this dimension typically activates
#                     activated_tokens = self.ffn_projections.get((layer_idx, dim_idx), [])
                    
#                     layer_activations.append({
#                         'dim': dim_idx,
#                         'coefficient': coef_val,
#                         'activation_strength': activation_strength,
#                         'activated_tokens': activated_tokens[:10],  # Top 10 tokens
#                         'target_token_rank': self._get_target_token_rank(activated_tokens, target_tokens)
#                     })
            
#             if layer_activations:
#                 details['layers_activated'].append({
#                     'layer': layer_idx,
#                     'activations': layer_activations,
#                     'layer_total': layer_activation_sum,
#                     'num_dimensions': len(layer_activations)
#                 })
            
#             # Add prediction info
#             if 'layer_preds_tokens' in row and layer_idx < len(row['layer_preds_tokens']):
#                 pred_tokens = row['layer_preds_tokens'][layer_idx][:10]
#                 pred_probs = row['layer_preds_probs'][layer_idx][:10]
                
#                 top_pred_strs = []
#                 target_predictions = []
                
#                 for rank, (token_idx, prob) in enumerate(zip(pred_tokens, pred_probs)):
#                     token_str = self.tokenizer.decode([token_idx]).strip()
#                     top_pred_strs.append(f"{token_str}({prob:.3f})")
                    
#                     if token_str in target_tokens:
#                         target_predictions.append({
#                             'token': token_str,
#                             'rank': rank,
#                             'probability': prob
#                         })
                
#                 details['top_predictions'].append({
#                     'layer': layer_idx,
#                     'predictions': top_pred_strs,
#                     'target_predictions': target_predictions
#                 })
        
#         # Summary statistics
#         details['activation_summary'] = {
#             'total_activation_strength': total_activations,
#             'max_activation': max_activation,
#             'num_active_layers': len(details['layers_activated']),
#             'avg_activation_per_layer': total_activations / max(1, len(details['layers_activated']))
#         }
        
#         return details
    
#     def _get_target_token_rank(self, activated_tokens: List[str], target_tokens: List[str]) -> Optional[int]:
#         """Get the rank of target tokens in activated tokens list."""
#         for target_token in target_tokens:
#             if target_token in activated_tokens:
#                 return activated_tokens.index(target_token)
#         return None
    
#     def reverse_engineer_from_activation_vector(self, 
#                                                activation_vector: np.ndarray,
#                                                layer_idx: int,
#                                                top_k_tokens: int = 20,
#                                                threshold: float = 0.1) -> Dict[str, Any]:
#         """
#         Enhanced reverse engineering with validation and confidence scores.
#         """
#         if len(activation_vector.shape) != 1:
#             raise ValueError(f"Expected 1D activation vector, got shape {activation_vector.shape}")
        
#         token_scores = defaultdict(float)
#         dimension_analysis = []
        
#         # Analyze each dimension in the activation vector
#         significant_dims = 0
#         for dim_idx, activation_strength in enumerate(activation_vector):
#             if abs(activation_strength) > threshold:
#                 significant_dims += 1
#                 ffn_key = (layer_idx, dim_idx)
                
#                 if ffn_key in self.ffn_projections:
#                     top_tokens = self.ffn_projections[ffn_key]
                    
#                     dimension_analysis.append({
#                         'dim': dim_idx,
#                         'activation_strength': activation_strength,
#                         'top_tokens': top_tokens[:10],
#                         'num_associated_tokens': len(top_tokens)
#                     })
                    
#                     # Score tokens with decay based on rank
#                     for rank, token in enumerate(top_tokens):
#                         # Higher activation strength and lower rank = higher score
#                         rank_weight = 1.0 / (rank + 1)  # 1.0, 0.5, 0.33, 0.25, ...
#                         token_scores[token] += abs(activation_strength) * rank_weight
        
#         # Sort and return results
#         sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        
#         return {
#             'top_tokens': sorted_tokens[:top_k_tokens],
#             'significant_dimensions': significant_dims,
#             'total_dimensions': len(activation_vector),
#             'dimension_analysis': dimension_analysis,
#             'confidence_score': significant_dims / len(activation_vector)
#         }
    
#     def validate_reverse_engineering(self, num_test_cases: int = 50) -> Dict[str, Any]:
#         """
#         Comprehensive validation of reverse engineering capabilities.
#         """
#         print(f"Running validation with {num_test_cases} test cases...")
        
#         validation_results = {
#             'test_cases': [],
#             'accuracy_metrics': {},
#             'error_analysis': [],
#             'summary_stats': {}
#         }
        
#         # Select random sentences for testing
#         test_indices = np.random.choice(len(self.model_df), min(num_test_cases, len(self.model_df)), replace=False)
        
#         hit_rates = []
#         precision_scores = []
#         recall_scores = []
        
#         for i, idx in enumerate(test_indices):
#             if i % 10 == 0:
#                 print(f"Processing test case {i+1}/{len(test_indices)}")
            
#             row = self.model_df.iloc[idx]
#             sentence = row['sent']
            
#             # Get actual tokens in the sentence
#             actual_tokens = set(self.tokenizer.decode([t]).strip() for t in self.tokenizer.encode(sentence))
#             actual_tokens = {t for t in actual_tokens if t in self.token_to_ffn_dims}  # Only tokens with FFN data
            
#             if not actual_tokens:
#                 continue
            
#             # Test 1: Forward prediction (sentence -> expected activations)
#             predicted_sentences = self.find_sentences_with_activation_pattern(
#                 list(actual_tokens), top_k=20, min_score_threshold=0.0
#             )
            
#             # Check if original sentence is in top predictions
#             original_found = any(pred['sentence'] == sentence for pred in predicted_sentences)
#             if predicted_sentences:
#                 original_rank = next((i for i, pred in enumerate(predicted_sentences) if pred['sentence'] == sentence), -1)
#             else:
#                 original_rank = -1
            
#             # Test 2: Reverse engineering from activation vectors
#             reverse_eng_results = {}
#             for layer_idx in range(min(3, len(row['layer_mlp_vec']))):  # Test first 3 layers
#                 if layer_idx < len(row['layer_mlp_vec']):
#                     mlp_vec = np.array(row['layer_mlp_vec'][layer_idx])
#                     reverse_result = self.reverse_engineer_from_activation_vector(mlp_vec, layer_idx)
#                     predicted_tokens = set(token for token, score in reverse_result['top_tokens'][:20])
                    
#                     # Calculate metrics
#                     intersection = actual_tokens & predicted_tokens
#                     precision = len(intersection) / len(predicted_tokens) if predicted_tokens else 0
#                     recall = len(intersection) / len(actual_tokens) if actual_tokens else 0
                    
#                     reverse_eng_results[layer_idx] = {
#                         'predicted_tokens': list(predicted_tokens),
#                         'intersection': list(intersection),
#                         'precision': precision,
#                         'recall': recall,
#                         'confidence': reverse_result['confidence_score']
#                     }
            
#             # Store test case results
#             test_case = {
#                 'sentence': sentence,
#                 'actual_tokens': list(actual_tokens),
#                 'original_found_in_predictions': original_found,
#                 'original_rank': original_rank,
#                 'reverse_engineering': reverse_eng_results,
#                 'avg_precision': np.mean([r['precision'] for r in reverse_eng_results.values()]) if reverse_eng_results else 0,
#                 'avg_recall': np.mean([r['recall'] for r in reverse_eng_results.values()]) if reverse_eng_results else 0
#             }
            
#             validation_results['test_cases'].append(test_case)
            
#             # Collect metrics
#             if reverse_eng_results:
#                 precision_scores.append(test_case['avg_precision'])
#                 recall_scores.append(test_case['avg_recall'])
#                 hit_rates.append(1 if original_found else 0)
        
#         # Calculate summary statistics
#         validation_results['summary_stats'] = {
#             'num_test_cases': len(validation_results['test_cases']),
#             'original_sentence_hit_rate': np.mean(hit_rates) if hit_rates else 0,
#             'avg_precision': np.mean(precision_scores) if precision_scores else 0,
#             'avg_recall': np.mean(recall_scores) if recall_scores else 0,
#             'precision_std': np.std(precision_scores) if precision_scores else 0,
#             'recall_std': np.std(recall_scores) if recall_scores else 0
#         }
        
#         # F1 score
#         if validation_results['summary_stats']['avg_precision'] + validation_results['summary_stats']['avg_recall'] > 0:
#             validation_results['summary_stats']['f1_score'] = (
#                 2 * validation_results['summary_stats']['avg_precision'] * validation_results['summary_stats']['avg_recall'] /
#                 (validation_results['summary_stats']['avg_precision'] + validation_results['summary_stats']['avg_recall'])
#             )
#         else:
#             validation_results['summary_stats']['f1_score'] = 0
        
#         return validation_results
    
#     def test_specific_token_patterns(self) -> Dict[str, Any]:
#         """
#         Test reverse engineering on specific, interpretable token patterns.
#         """
#         print("Testing specific token patterns...")
        
#         test_patterns = {
#             'common_words': ['the', 'and', 'of', 'to', 'a'],
#             'punctuation': ['.', ',', '!', '?', ';'],
#             'numbers': ['1', '2', '3', '10', '100'],
#             'articles': ['the', 'a', 'an'],
#             'prepositions': ['in', 'on', 'at', 'by', 'for'],
#             'conjunctions': ['and', 'or', 'but', 'so', 'yet']
#         }
        
#         pattern_results = {}
        
#         for pattern_name, tokens in test_patterns.items():
#             print(f"Testing pattern: {pattern_name}")
            
#             # Filter tokens that exist in our data
#             available_tokens = [t for t in tokens if t in self.token_to_ffn_dims]
#             if not available_tokens:
#                 pattern_results[pattern_name] = {'error': 'No tokens available in FFN data'}
#                 continue
            
#             # Find sentences with these patterns
#             sentences = self.find_sentences_with_activation_pattern(
#                 available_tokens, top_k=10, min_score_threshold=0.1
#             )
            
#             # Analyze results
#             pattern_analysis = {
#                 'available_tokens': available_tokens,
#                 'num_sentences_found': len(sentences),
#                 'avg_score': np.mean([s['score'] for s in sentences]) if sentences else 0,
#                 'sample_sentences': [s['sentence'] for s in sentences[:3]],
#                 'token_coverage': {}
#             }
            
#             # Check how well each token is represented
#             for token in available_tokens:
#                 token_appearances = sum(1 for s in sentences if token in s['sentence'].lower())
#                 pattern_analysis['token_coverage'][token] = {
#                     'appearances': token_appearances,
#                     'coverage_rate': token_appearances / len(sentences) if sentences else 0
#                 }
            
#             pattern_results[pattern_name] = pattern_analysis
        
#         return pattern_results
    
#     def comprehensive_testing_suite(self, save_results: bool = True) -> Dict[str, Any]:
#         """
#         Run the complete testing suite with detailed analysis.
#         """
#         print("="*60)
#         print("COMPREHENSIVE ACTIVATION REVERSE ENGINEERING TESTING")
#         print("="*60)
        
#         # Test 1: Dataset statistics and sanity checks
#         print("\n1. Dataset Statistics and Sanity Checks")
#         print("-" * 40)
#         self._print_dataset_stats()
        
#         # Test 2: Basic functionality tests
#         print("\n2. Basic Functionality Tests")
#         print("-" * 40)
#         basic_test_results = self._run_basic_functionality_tests()
        
#         # Test 3: Specific token pattern tests
#         print("\n3. Token Pattern Tests")
#         print("-" * 40)
#         pattern_test_results = self.test_specific_token_patterns()
#         self._print_pattern_test_results(pattern_test_results)
        
#         # Test 4: Comprehensive validation
#         print("\n4. Comprehensive Validation")
#         print("-" * 40)
#         validation_results = self.validate_reverse_engineering(num_test_cases=100)
#         self._print_validation_results(validation_results)
        
#         # Test 5: Edge case testing
#         print("\n5. Edge Case Testing")
#         print("-" * 40)
#         edge_case_results = self._test_edge_cases()
        
#         # Compile all results
#         all_results = {
#             'dataset_stats': self.dataset_stats,
#             'basic_tests': basic_test_results,
#             'pattern_tests': pattern_test_results,
#             'validation': validation_results,
#             'edge_cases': edge_case_results,
#             'overall_assessment': self._generate_overall_assessment(
#                 basic_test_results, pattern_test_results, validation_results, edge_case_results
#             )
#         }
        
#         self.test_results['tests'] = all_results
        
#         # Save results if requested
#         if save_results:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"reverse_engineering_test_results_{timestamp}.json"
            
#             # Convert numpy types to regular Python types for JSON serialization
#             json_serializable_results = self._make_json_serializable(self.test_results)
            
#             with open(filename, 'w') as f:
#                 json.dump(json_serializable_results, f, indent=2)
#             print(f"\nTest results saved to: {filename}")
        
#         return all_results
    
#     def _run_basic_functionality_tests(self) -> Dict[str, Any]:
#         """Run basic functionality tests."""
#         results = {}
        
#         # Test 1: Simple token lookup
#         test_token = "the"
#         if test_token in self.token_to_ffn_dims:
#             ffn_dims = self.token_to_ffn_dims[test_token]
#             results['token_lookup'] = {
#                 'token': test_token,
#                 'num_ffn_dimensions': len(ffn_dims),
#                 'layers_involved': len(set(dim[0] for dim in ffn_dims)),
#                 'status': 'PASS'
#             }
#         else:
#             results['token_lookup'] = {'status': 'FAIL', 'error': f"Token '{test_token}' not found"}
        
#         # Test 2: Sentence finding
#         try:
#             sentences = self.find_sentences_with_activation_pattern([test_token], top_k=5)
#             results['sentence_finding'] = {
#                 'num_sentences_found': len(sentences),
#                 'avg_score': np.mean([s['score'] for s in sentences]) if sentences else 0,
#                 'status': 'PASS' if sentences else 'FAIL'
#             }
#         except Exception as e:
#             results['sentence_finding'] = {'status': 'FAIL', 'error': str(e)}
        
#         # Test 3: Reverse engineering
#         try:
#             # Use a real activation vector from the dataset
#             if len(self.model_df) > 0:
#                 sample_row = self.model_df.iloc[0]
#                 if len(sample_row['layer_mlp_vec']) > 0:
#                     mlp_vec = np.array(sample_row['layer_mlp_vec'][0])
#                     reverse_result = self.reverse_engineer_from_activation_vector(mlp_vec, 0)
#                     results['reverse_engineering'] = {
#                         'num_tokens_predicted': len(reverse_result['top_tokens']),
#                         'confidence_score': reverse_result['confidence_score'],
#                         'significant_dimensions': reverse_result['significant_dimensions'],
#                         'status': 'PASS'
#                     }
#                 else:
#                     results['reverse_engineering'] = {'status': 'FAIL', 'error': 'No MLP vectors available'}
#             else:
#                 results['reverse_engineering'] = {'status': 'FAIL', 'error': 'No data available'}
#         except Exception as e:
#             results['reverse_engineering'] = {'status': 'FAIL', 'error': str(e)}
        
#         return results
    
#     def _test_edge_cases(self) -> Dict[str, Any]:
#         """Test edge cases and error handling."""
#         results = {}
        
#         # Test 1: Empty token list
#         try:
#             sentences = self.find_sentences_with_activation_pattern([], top_k=5)
#             results['empty_token_list'] = {
#                 'num_sentences': len(sentences),
#                 'status': 'PASS' if len(sentences) == 0 else 'UNEXPECTED'
#             }
#         except Exception as e:
#             results['empty_token_list'] = {'status': 'ERROR', 'error': str(e)}
        
#         # Test 2: Non-existent tokens
#         try:
#             sentences = self.find_sentences_with_activation_pattern(['NONEXISTENT_TOKEN_12345'], top_k=5)
#             results['nonexistent_tokens'] = {
#                 'num_sentences': len(sentences),
#                 'status': 'PASS'
#             }
#         except Exception as e:
#             results['nonexistent_tokens'] = {'status': 'ERROR', 'error': str(e)}
        
#         # Test 3: Invalid activation vector
#         try:
#             # Test with wrong shape
#             invalid_vec = np.random.randn(10, 10)  # 2D instead of 1D
#             reverse_result = self.reverse_engineer_from_activation_vector(invalid_vec, 0)
#             results['invalid_activation_vector'] = {'status': 'FAIL', 'error': 'Should have raised error'}
#         except ValueError as e:
#             results['invalid_activation_vector'] = {'status': 'PASS', 'error_caught': str(e)}
#         except Exception as e:
#             results['invalid_activation_vector'] = {'status': 'UNEXPECTED_ERROR', 'error': str(e)}
        
#         # Test 4: Very high threshold
#         try:
#             if len(self.model_df) > 0:
#                 sample_row = self.model_df.iloc[0]
#                 if len(sample_row['layer_mlp_vec']) > 0:
#                     mlp_vec = np.array(sample_row['layer_mlp_vec'][0])
#                     reverse_result = self.reverse_engineer_from_activation_vector(mlp_vec, 0, threshold=999.0)
#                     results['high_threshold'] = {
#                         'num_tokens': len(reverse_result['top_tokens']),
#                         'significant_dims': reverse_result['significant_dimensions'],
#                         'status': 'PASS'
#                     }
#         except Exception as e:
#             results['high_threshold'] = {'status': 'ERROR', 'error': str(e)}
        
#         return results
    
#     def _print_dataset_stats(self):
#         """Print dataset statistics in a readable format."""
#         stats = self.dataset_stats
#         print(f"Dataset size: {len(self.model_df)} sentences")
#         print(f"Total tokens: {stats['total_tokens']:,}")
#         print(f"Unique tokens: {stats['unique_tokens']:,}")
#         print(f"Average sentence length: {stats['avg_sentence_length']:.2f} tokens")
#         print(f"Tokens with FFN mappings: {len(self.token_to_ffn_dims):,}")
        
#         # Most common tokens
#         top_tokens = stats['token_frequency'].most_common(10)
#         print(f"Most frequent tokens: {', '.join([self.tokenizer.decode([t[0]]) for t in top_tokens[:5]])}")
    
#     def _print_pattern_test_results(self, pattern_results: Dict[str, Any]):
#         """Print pattern test results in a readable format."""
#         for pattern_name, results in pattern_results.items():
#             if 'error' in results:
#                 print(f"{pattern_name}: ERROR - {results['error']}")
#                 continue
            
#             print(f"\n{pattern_name.upper()}:")
#             print(f"  Available tokens: {results['available_tokens']}")
#             print(f"  Sentences found: {results['num_sentences_found']}")
#             print(f"  Average score: {results['avg_score']:.3f}")
            
#             if results['sample_sentences']:
#                 print(f"  Sample sentences:")
#                 for i, sent in enumerate(results['sample_sentences'], 1):
#                     print(f"    {i}. {sent[:80]}{'...' if len(sent) > 80 else ''}")
            
#             # Token coverage analysis
#             coverage_rates = [info['coverage_rate'] for info in results['token_coverage'].values()]
#             if coverage_rates:
#                 print(f"  Token coverage rate: {np.mean(coverage_rates):.2f} (avg)")
    
#     def _print_validation_results(self, validation_results: Dict[str, Any]):
#         """Print validation results in a readable format."""
#         stats = validation_results['summary_stats']
        
#         print(f"Validation completed on {stats['num_test_cases']} test cases")
#         print(f"Original sentence hit rate: {stats['original_sentence_hit_rate']:.3f}")
#         print(f"Average precision: {stats['avg_precision']:.3f} (±{stats['precision_std']:.3f})")
#         print(f"Average recall: {stats['avg_recall']:.3f} (±{stats['recall_std']:.3f})")
#         print(f"F1 score: {stats['f1_score']:.3f}")
        
#         # Performance interpretation
#         if stats['f1_score'] > 0.7:
#             performance = "EXCELLENT"
#         elif stats['f1_score'] > 0.5:
#             performance = "GOOD"
#         elif stats['f1_score'] > 0.3:
#             performance = "FAIR"
#         else:
#             performance = "POOR"
        
#         print(f"Overall performance: {performance}")
        
#         # Show some example results
#         print(f"\nSample test cases:")
#         for i, test_case in enumerate(validation_results['test_cases'][:3]):
#             print(f"  {i+1}. Sentence: {test_case['sentence'][:60]}...")
#             print(f"     Tokens: {test_case['actual_tokens'][:5]}")
#             print(f"     Found in predictions: {test_case['original_found_in_predictions']}")
#             print(f"     Avg precision: {test_case['avg_precision']:.3f}")
    
#     def _generate_overall_assessment(self, basic_tests, pattern_tests, validation, edge_cases) -> Dict[str, Any]:
#         """Generate an overall assessment of the reverse engineering system."""
        
#         # Count test passes
#         basic_passes = sum(1 for test in basic_tests.values() if test.get('status') == 'PASS')
#         edge_passes = sum(1 for test in edge_cases.values() if test.get('status') == 'PASS')
        
#         # Pattern test success rate
#         pattern_success_count = 0
#         pattern_total = 0
#         for pattern_name, results in pattern_tests.items():
#             if 'error' not in results:
#                 pattern_total += 1
#                 if results['num_sentences_found'] > 0:
#                     pattern_success_count += 1
        
#         pattern_success_rate = pattern_success_count / pattern_total if pattern_total > 0 else 0
        
#         # Overall metrics
#         f1_score = validation['summary_stats']['f1_score']
#         precision = validation['summary_stats']['avg_precision']
#         recall = validation['summary_stats']['avg_recall']
        
#         # Generate assessment
#         assessment = {
#             'basic_functionality_score': basic_passes / len(basic_tests),
#             'pattern_recognition_score': pattern_success_rate,
#             'validation_f1_score': f1_score,
#             'precision': precision,
#             'recall': recall,
#             'edge_case_handling_score': edge_passes / len(edge_cases) if edge_cases else 1.0,
#             'recommendations': [],
#             'strengths': [],
#             'weaknesses': []
#         }
        
#         # Overall score (weighted average)
#         overall_score = (
#             assessment['basic_functionality_score'] * 0.2 +
#             assessment['pattern_recognition_score'] * 0.3 +
#             assessment['validation_f1_score'] * 0.4 +
#             assessment['edge_case_handling_score'] * 0.1
#         )
#         assessment['overall_score'] = overall_score
        
#         # Determine overall grade
#         if overall_score >= 0.8:
#             assessment['grade'] = 'A'
#             assessment['status'] = 'EXCELLENT'
#         elif overall_score >= 0.7:
#             assessment['grade'] = 'B'
#             assessment['status'] = 'GOOD'
#         elif overall_score >= 0.6:
#             assessment['grade'] = 'C'
#             assessment['status'] = 'ACCEPTABLE'
#         elif overall_score >= 0.5:
#             assessment['grade'] = 'D'
#             assessment['status'] = 'NEEDS_IMPROVEMENT'
#         else:
#             assessment['grade'] = 'F'
#             assessment['status'] = 'POOR'
        
#         # Generate specific feedback
#         if assessment['basic_functionality_score'] < 0.8:
#             assessment['weaknesses'].append("Basic functionality issues detected")
#             assessment['recommendations'].append("Debug basic token lookup and sentence finding functions")
        
#         if assessment['pattern_recognition_score'] < 0.7:
#             assessment['weaknesses'].append("Poor pattern recognition performance")
#             assessment['recommendations'].append("Improve FFN dimension mapping or scoring algorithm")
        
#         if f1_score < 0.5:
#             assessment['weaknesses'].append("Low validation F1 score indicates poor reverse engineering accuracy")
#             assessment['recommendations'].append("Revise activation vector analysis or increase dataset quality")
        
#         if precision > 0.7:
#             assessment['strengths'].append("High precision - predictions are generally accurate")
        
#         if recall > 0.7:
#             assessment['strengths'].append("High recall - good coverage of relevant tokens")
        
#         if assessment['edge_case_handling_score'] > 0.8:
#             assessment['strengths'].append("Robust error handling and edge case management")
        
#         return assessment
    
#     def _make_json_serializable(self, obj):
#         """Convert numpy types and other non-serializable types to JSON-compatible types."""
#         if isinstance(obj, dict):
#             return {key: self._make_json_serializable(value) for key, value in obj.items()}
#         elif isinstance(obj, list):
#             return [self._make_json_serializable(item) for item in obj]
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, (np.integer, np.int64, np.int32)):
#             return int(obj)
#         elif isinstance(obj, (np.floating, np.float64, np.float32)):
#             return float(obj)
#         elif isinstance(obj, set):
#             return list(obj)
#         else:
#             return obj
    
#     def generate_diagnostic_report(self) -> str:
#         """Generate a comprehensive diagnostic report."""
#         if not self.test_results.get('tests'):
#             return "No test results available. Run comprehensive_testing_suite() first."
        
#         results = self.test_results['tests']
#         assessment = results['overall_assessment']
        
#         report = f"""
# ACTIVATION REVERSE ENGINEERING DIAGNOSTIC REPORT
# ===============================================
# Generated: {self.test_results['timestamp']}
# Model: {self.test_results['model_name']}
# Dataset Size: {self.test_results['dataset_size']} sentences

# OVERALL ASSESSMENT
# -----------------
# Grade: {assessment['grade']}
# Status: {assessment['status']}
# Overall Score: {assessment['overall_score']:.3f}/1.000

# DETAILED SCORES
# --------------
# Basic Functionality: {assessment['basic_functionality_score']:.3f}
# Pattern Recognition: {assessment['pattern_recognition_score']:.3f}
# Validation F1 Score: {assessment['validation_f1_score']:.3f}
# Edge Case Handling: {assessment['edge_case_handling_score']:.3f}

# PERFORMANCE METRICS
# ------------------
# Precision: {assessment['precision']:.3f}
# Recall: {assessment['recall']:.3f}
# F1 Score: {assessment['validation_f1_score']:.3f}

# STRENGTHS
# ---------"""
        
#         for strength in assessment['strengths']:
#             report += f"\n• {strength}"
        
#         if not assessment['strengths']:
#             report += "\n• None identified"
        
#         report += f"""

# WEAKNESSES
# ----------"""
        
#         for weakness in assessment['weaknesses']:
#             report += f"\n• {weakness}"
        
#         if not assessment['weaknesses']:
#             report += "\n• None identified"
        
#         report += f"""

# RECOMMENDATIONS
# --------------"""
        
#         for recommendation in assessment['recommendations']:
#             report += f"\n• {recommendation}"
        
#         if not assessment['recommendations']:
#             report += "\n• System is performing well"
        
#         # Add pattern test summary
#         report += f"""

# PATTERN TEST SUMMARY
# -------------------"""
        
#         for pattern_name, pattern_result in results['pattern_tests'].items():
#             if 'error' not in pattern_result:
#                 report += f"\n{pattern_name}: {pattern_result['num_sentences_found']} sentences found (avg score: {pattern_result['avg_score']:.3f})"
#             else:
#                 report += f"\n{pattern_name}: ERROR - {pattern_result['error']}"
        
#         report += f"""

# VALIDATION DETAILS
# -----------------
# Test Cases: {results['validation']['summary_stats']['num_test_cases']}
# Original Sentence Hit Rate: {results['validation']['summary_stats']['original_sentence_hit_rate']:.3f}
# Average Precision: {results['validation']['summary_stats']['avg_precision']:.3f} (±{results['validation']['summary_stats']['precision_std']:.3f})
# Average Recall: {results['validation']['summary_stats']['avg_recall']:.3f} (±{results['validation']['summary_stats']['recall_std']:.3f})

# CONCLUSION
# ----------"""
        
#         if assessment['overall_score'] >= 0.7:
#             report += "\nThe reverse engineering system is performing well and is suitable for production use."
#         elif assessment['overall_score'] >= 0.5:
#             report += "\nThe reverse engineering system shows promise but needs improvement before production use."
#         else:
#             report += "\nThe reverse engineering system requires significant improvements before it can be used reliably."
        
#         return report


# def main():
#     """
#     Enhanced example usage with comprehensive testing.
#     """
#     # Initialize the reverse engineer
#     try:
#         re = ActivationReverseEngineer(
#             ffn_projections_path="ffn_projections.pkl",
#             model_df_path="model_df_10k.pkl",
#             model_name="gpt2"
#         )
#     except FileNotFoundError as e:
#         print(f"Error: Could not find required data files. {e}")
#         print("Make sure you have generated 'ffn_projections.pkl' and 'model_df_10k.pkl' first.")
#         return
#     except Exception as e:
#         print(f"Error initializing reverse engineer: {e}")
#         return
    
#     # Run comprehensive testing suite
#     print("Running comprehensive testing suite...")
#     test_results = re.comprehensive_testing_suite(save_results=True)
    
#     # Generate and print diagnostic report
#     print("\n" + "="*80)
#     print("DIAGNOSTIC REPORT")
#     print("="*80)
#     diagnostic_report = re.generate_diagnostic_report()
#     print(diagnostic_report)
    
#     # Demonstrate specific functionality with detailed output
#     print("\n" + "="*80)
#     print("DETAILED FUNCTIONALITY DEMONSTRATION")
#     print("="*80)
    
#     # Example 1: Enhanced sentence finding with detailed analysis
#     print("\n1. ENHANCED SENTENCE FINDING")
#     print("-" * 40)
#     target_tokens = ["the", "and"]
#     results = re.find_sentences_with_activation_pattern(
#         target_tokens=target_tokens,
#         top_k=3,
#         min_score_threshold=1.0
#     )
    
#     for i, result in enumerate(results):
#         print(f"\nRank {i+1} (Score: {result['score']:.3f}):")
#         print(f"Sentence: {result['sentence']}")
        
#         # Detailed score breakdown
#         score_details = result['score_details']
#         print(f"Score breakdown:")
#         print(f"  - Activation strength: {score_details['activation_strength']:.3f}")
#         print(f"  - Token presence bonus: {score_details['token_presence_bonus']:.3f}")
#         print(f"  - Layer diversity bonus: {score_details['layer_diversity_bonus']:.3f}")
#         print(f"  - Prediction alignment: {score_details['prediction_alignment']:.3f}")
        
#         # Activation details
#         activation_details = result['activation_details']
#         print(f"Activation summary:")
#         print(f"  - Active layers: {activation_details['activation_summary']['num_active_layers']}")
#         print(f"  - Total activation: {activation_details['activation_summary']['total_activation_strength']:.3f}")
#         print(f"  - Max activation: {activation_details['activation_summary']['max_activation']:.3f}")
    
#     # Example 2: Enhanced reverse engineering demonstration
#     print("\n2. ENHANCED REVERSE ENGINEERING")
#     print("-" * 40)
    
#     if len(re.model_df) > 0:
#         sample_row = re.model_df.iloc[0]
#         if len(sample_row['layer_mlp_vec']) > 0:
#             sample_sentence = sample_row['sent']
#             print(f"Original sentence: {sample_sentence}")
            
#             mlp_vec = np.array(sample_row['layer_mlp_vec'][0])
#             reverse_result = re.reverse_engineer_from_activation_vector(mlp_vec, 0, top_k_tokens=10)
            
#             print(f"\nReverse engineering results for layer 0:")
#             print(f"Confidence score: {reverse_result['confidence_score']:.3f}")
#             print(f"Significant dimensions: {reverse_result['significant_dimensions']}/{reverse_result['total_dimensions']}")
            
#             print(f"\nTop predicted tokens:")
#             for i, (token, score) in enumerate(reverse_result['top_tokens'][:10], 1):
#                 print(f"  {i:2d}. '{token}' (score: {score:.3f})")
            
#             # Compare with actual tokens
#             actual_tokens = re.tokenizer.encode(sample_sentence)
#             actual_token_strs = [re.tokenizer.decode([t]).strip() for t in actual_tokens]
#             predicted_tokens = [token for token, score in reverse_result['top_tokens']]
            
#             intersection = set(actual_token_strs) & set(predicted_tokens)
#             print(f"\nActual tokens in sentence: {actual_token_strs}")
#             print(f"Intersection with predictions: {list(intersection)}")
#             print(f"Intersection rate: {len(intersection)/len(set(actual_token_strs)):.3f}")


# if __name__ == "__main__":
#     main()


'''
Loading FFN projections...
Loading model DataFrame...
Loaded 10000 sentences from WikiText
FFN projections available for 36864 (layer, dim) pairs
Building reverse mappings...
Built reverse mappings for 44494 unique tokens
Average FFN dimensions per token: 8.29
Analyzing dataset statistics...
Dataset contains 15621 unique tokens
Average sentence length: 14.29 tokens
Running comprehensive testing suite...
============================================================
COMPREHENSIVE ACTIVATION REVERSE ENGINEERING TESTING
============================================================

1. Dataset Statistics and Sanity Checks
----------------------------------------
Dataset size: 10000 sentences
Total tokens: 142,865
Unique tokens: 15,621
Average sentence length: 14.29 tokens
Tokens with FFN mappings: 44,494
Most frequent tokens:  the,  ,,  of,  and,  in

2. Basic Functionality Tests
----------------------------------------
Finding sentences with activation patterns for tokens: ['the']
Found 46 relevant FFN dimensions across 9 layers

3. Token Pattern Tests
----------------------------------------
Testing specific token patterns...
Testing pattern: common_words
Finding sentences with activation patterns for tokens: ['the', 'and', 'of', 'to', 'a']
Found 330 relevant FFN dimensions across 12 layers
Testing pattern: punctuation
Finding sentences with activation patterns for tokens: ['.', ',', '!', '?', ';']
Found 1748 relevant FFN dimensions across 12 layers
Testing pattern: numbers
Finding sentences with activation patterns for tokens: ['1', '2', '3', '10', '100']
Found 63 relevant FFN dimensions across 12 layers
Testing pattern: articles
Finding sentences with activation patterns for tokens: ['the', 'a', 'an']
Found 207 relevant FFN dimensions across 12 layers
Testing pattern: prepositions
Finding sentences with activation patterns for tokens: ['in', 'on', 'at', 'by', 'for']
Found 234 relevant FFN dimensions across 12 layers
Testing pattern: conjunctions
Finding sentences with activation patterns for tokens: ['and', 'or', 'but', 'so', 'yet']
Found 176 relevant FFN dimensions across 12 layers

COMMON_WORDS:
  Available tokens: ['the', 'and', 'of', 'to', 'a']
  Sentences found: 10
  Average score: 148.043
  Sample sentences:
    1. Toward the end of the video , they sit on a bench next
    2. However , cooler temperatures tend
    3. In 1885 , Andrew P. Morgan proposed that differences in microscopic characterist...
  Token coverage rate: 0.48 (avg)

PUNCTUATION:
  Available tokens: ['.', ',', '!', '?', ';']
  Sentences found: 10
  Average score: 3394.709
  Sample sentences:
    1. Development
    2. K
    3. Z
  Token coverage rate: 0.00 (avg)

NUMBERS:
  Available tokens: ['1', '2', '3', '10', '100']
  Sentences found: 10
  Average score: 55.447
  Sample sentences:
    1. The Mint most likely channeled its production through some favored Philadelphia ...
    2. Columbus had offered him a three @-@ year , $
    3. While there had been a petroleum industry in the Sarnia area since 1858 , the es...
  Token coverage rate: 0.14 (avg)

ARTICLES:
  Available tokens: ['the', 'a', 'an']
  Sentences found: 10
  Average score: 86.805
  Sample sentences:
    1. The Arkansas Gazette referred to the structure as "
    2. In 2013 , Entertainment Weekly named him the eighth @-@ greatest working directo...
    3. The island is oriented generally northeast @-@ southwest , with the Gulf of Mexi...
  Token coverage rate: 0.80 (avg)

PREPOSITIONS:
  Available tokens: ['in', 'on', 'at', 'by', 'for']
  Sentences found: 10
  Average score: 124.107
  Sample sentences:
    1. How it is handled depends primarily
    2. Ted Shen describes the male bonding shown in the film as bordering
    3. She 's relying
  Token coverage rate: 0.36 (avg)

CONJUNCTIONS:
  Available tokens: ['and', 'or', 'but', 'so', 'yet']
  Sentences found: 10
  Average score: 75.018
  Sample sentences:
    1. Townsend found it "
    2. Music Times writer Carolyn Menyes praised its composition for being "
    3. Thoth , as the overseer of time , was said to allot fixed lifespans to both huma...
  Token coverage rate: 0.10 (avg)

4. Comprehensive Validation
----------------------------------------
Running validation with 100 test cases...
Processing test case 1/100
Finding sentences with activation patterns for tokens: ['also', 'elsh', 'W']
Found 18 relevant FFN dimensions across 10 layers
Finding sentences with activation patterns for tokens: ['at', 'aw', 'A', 'ken', 'Atlanta', 'eh', 'We']
Found 142 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['In', ',', 'the', 'damage', 'and', 'was']
Found 1706 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['s', '.', 'Characters', 'C', 'Sh', 'akespeare', "'", 'Boy', ';', 'W']
Found 1266 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['gear', ',', 'three', 'eight', 'with', 'a', 'armed', 'eyed', 'The', '-', 'head', '@']
Found 2095 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['s', 'more', 'around', 'album', 'based', 'were', 'and', 'The', "'"]
Found 330 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['has']
Found 13 relevant FFN dimensions across 4 layers
Finding sentences with activation patterns for tokens: [',', 'sea', 'these', 'were', 'of', 'The', 'area', 'air', 'protection']
Found 1805 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['Description', '=']
Found 16 relevant FFN dimensions across 9 layers
Finding sentences with activation patterns for tokens: ['elsen', 'Ni', 'Dave']
Found 44 relevant FFN dimensions across 12 layers
Processing test case 11/100
Finding sentences with activation patterns for tokens: ['In', ',', 'in', 'Here', 'he', 'cl', 'the', 'again', 'ford', 'while', 'called', 'Fair', 'manager', 'Paul', '2005', 'for', 'up', 'with', 'back', 'was', 'by']
Found 1948 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['ission', ',', '-', 'specific', 'good', 'rem', 'on', 'or', 'less', 'not', 'groups', 'clear', 'the', 'changes', 'ogen', 'and', 'normal', 'person', 'into', '@', 'of', 'is', 'high', 'including', 'for', 'age', 'best', 'etics', ')', 'health', 'risk', 'AM', 'The', 'post', 'etic', 'L', '(']
Found 2558 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: ['She']
Found 4 relevant FFN dimensions across 3 layers
Finding sentences with activation patterns for tokens: ['two', 'failed', 'and', 'The', 'other', 'to']
Found 223 relevant FFN dimensions across 12 layers
Finding sentences with activation patterns for tokens: [',', 'the', 'press', 'turned', 'their', 'With', '"', 'and', 'public', 'to', 'political', 'case', 'this']
Found 1851 relevant FFN dimensions across 12 layers ....
'''
########### NOUN TEST

# import pickle
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional, Set
# from collections import defaultdict, Counter
# import torch
# import torch.nn.functional as F
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import nltk
# from nltk.corpus import wordnet
# import spacy
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# # Download required NLTK data
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')
#     nltk.download('averaged_perceptron_tagger')

# class EnhancedSemanticMapper:
#     """
#     Enhanced mapper for robust semantic words like nouns, with improved
#     semantic clustering and context-aware activation analysis.
#     """
    
#     def __init__(self, ffn_projections_path: str, model_df_path: str, model_name: str = "gpt2"):
#         self.model_name = model_name
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
        
#         # Load spaCy for better linguistic analysis
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
#             self.nlp = None
        
#         # Load data
#         print("Loading FFN projections...")
#         with open(ffn_projections_path, 'rb') as f:
#             self.ffn_projections = pickle.load(f)
        
#         print("Loading model DataFrame...")
#         with open(model_df_path, 'rb') as f:
#             self.model_df = pickle.load(f)
        
#         # Enhanced mappings for semantic analysis
#         self._build_enhanced_mappings()
#         self._categorize_semantic_tokens()
#         self._build_semantic_clusters()
    
#     def _build_enhanced_mappings(self):
#         """Build enhanced mappings with semantic awareness."""
#         print("Building enhanced semantic mappings...")
        
#         # Basic token to FFN mappings
#         self.token_to_ffn_dims = defaultdict(list)
#         self.layer_to_active_tokens = defaultdict(set)
        
#         # Enhanced semantic mappings
#         self.semantic_categories = {
#             'nouns': set(),
#             'proper_nouns': set(),
#             'verbs': set(),
#             'adjectives': set(),
#             'adverbs': set(),
#             'function_words': set(),
#             'numbers': set(),
#             'punctuation': set()
#         }
        
#         # Activation strength mappings
#         self.token_activation_profiles = {}  # token -> {layer: [activations]}
#         self.strong_activators = defaultdict(set)  # (layer, dim) -> {tokens}
        
#         for (layer, dim), top_tokens in self.ffn_projections.items():
#             for rank, token in enumerate(top_tokens):
#                 # Basic mapping
#                 self.token_to_ffn_dims[token].append((layer, dim, rank))
#                 self.layer_to_active_tokens[layer].add(token)
                
#                 # Track strong activators (top 10 tokens per dimension)
#                 if rank < 10:
#                     self.strong_activators[(layer, dim)].add(token)
                
#                 # Build activation profiles
#                 if token not in self.token_activation_profiles:
#                     self.token_activation_profiles[token] = defaultdict(list)
                
#                 # Weight by inverse rank (higher rank = lower weight)
#                 activation_strength = 1.0 / (rank + 1)
#                 self.token_activation_profiles[token][layer].append((dim, activation_strength))
    
#     def _categorize_semantic_tokens(self):
#         """Categorize tokens by semantic type using linguistic analysis."""
#         print("Categorizing tokens by semantic type...")
        
#         all_tokens = list(self.token_to_ffn_dims.keys())
        
#         for token in all_tokens:
#             # Clean token for analysis
#             clean_token = token.strip()
#             if not clean_token:
#                 continue
            
#             # Basic regex patterns
#             if re.match(r'^[0-9]+$', clean_token):
#                 self.semantic_categories['numbers'].add(token)
#             elif re.match(r'^[^\w\s]$', clean_token):
#                 self.semantic_categories['punctuation'].add(token)
#             elif len(clean_token) == 1 and not clean_token.isalpha():
#                 self.semantic_categories['punctuation'].add(token)
#             else:
#                 # Use spaCy for more sophisticated analysis
#                 if self.nlp:
#                     doc = self.nlp(clean_token)
#                     if doc:
#                         pos_tag = doc[0].pos_
#                         if pos_tag == 'NOUN':
#                             self.semantic_categories['nouns'].add(token)
#                         elif pos_tag == 'PROPN':
#                             self.semantic_categories['proper_nouns'].add(token)
#                         elif pos_tag == 'VERB':
#                             self.semantic_categories['verbs'].add(token)
#                         elif pos_tag == 'ADJ':
#                             self.semantic_categories['adjectives'].add(token)
#                         elif pos_tag == 'ADV':
#                             self.semantic_categories['adverbs'].add(token)
#                         else:
#                             self.semantic_categories['function_words'].add(token)
#                 else:
#                     # Fallback to NLTK POS tagging
#                     try:
#                         pos_tags = nltk.pos_tag([clean_token])
#                         pos = pos_tags[0][1]
                        
#                         if pos.startswith('NN'):
#                             if pos == 'NNP' or pos == 'NNPS':
#                                 self.semantic_categories['proper_nouns'].add(token)
#                             else:
#                                 self.semantic_categories['nouns'].add(token)
#                         elif pos.startswith('VB'):
#                             self.semantic_categories['verbs'].add(token)
#                         elif pos.startswith('JJ'):
#                             self.semantic_categories['adjectives'].add(token)
#                         elif pos.startswith('RB'):
#                             self.semantic_categories['adverbs'].add(token)
#                         else:
#                             self.semantic_categories['function_words'].add(token)
#                     except:
#                         # If all else fails, assume function word
#                         self.semantic_categories['function_words'].add(token)
        
#         # Print categorization results
#         for category, tokens in self.semantic_categories.items():
#             print(f"{category}: {len(tokens)} tokens")
    
#     def _build_semantic_clusters(self):
#         """Build semantic clusters based on activation patterns."""
#         print("Building semantic clusters...")
        
#         # Focus on nouns and content words for clustering
#         content_tokens = (self.semantic_categories['nouns'] | 
#                          self.semantic_categories['proper_nouns'] |
#                          self.semantic_categories['verbs'] |
#                          self.semantic_categories['adjectives'])
        
#         if len(content_tokens) < 10:
#             print("Warning: Not enough content tokens for clustering")
#             self.semantic_clusters = {}
#             return
        
#         # Create activation vectors for clustering
#         activation_vectors = []
#         token_list = []
        
#         for token in content_tokens:
#             if token in self.token_activation_profiles:
#                 # Create a unified activation vector across all layers
#                 vector = np.zeros(12 * 100)  # Assuming 12 layers, top 100 dims per layer
                
#                 for layer, activations in self.token_activation_profiles[token].items():
#                     if layer < 12:  # Safety check
#                         layer_start = layer * 100
#                         for dim, strength in activations[:100]:  # Top 100 activations
#                             if layer_start + dim < len(vector):
#                                 vector[layer_start + dim] = strength
                
#                 if np.sum(vector) > 0:  # Only include tokens with activations
#                     activation_vectors.append(vector)
#                     token_list.append(token)
        
#         if len(activation_vectors) > 10:
#             # Perform k-means clustering
#             n_clusters = min(20, len(activation_vectors) // 5)  # Adaptive cluster count
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             cluster_labels = kmeans.fit_predict(activation_vectors)
            
#             # Organize clusters
#             self.semantic_clusters = defaultdict(list)
#             for token, cluster_id in zip(token_list, cluster_labels):
#                 self.semantic_clusters[cluster_id].append(token)
            
#             print(f"Created {n_clusters} semantic clusters")
#         else:
#             self.semantic_clusters = {}
    
#     def find_semantic_similar_words(self, target_word: str, 
#                                    category: Optional[str] = None,
#                                    top_k: int = 10) -> List[Tuple[str, float]]:
#         """
#         Find semantically similar words based on activation patterns.
        
#         Args:
#             target_word: The word to find similarities for
#             category: Optional semantic category to restrict search
#             top_k: Number of similar words to return
#         """
#         if target_word not in self.token_activation_profiles:
#             print(f"Word '{target_word}' not found in activation profiles")
#             return []
        
#         # Get target word's activation profile
#         target_profile = self.token_activation_profiles[target_word]
#         target_vector = self._profile_to_vector(target_profile)
        
#         # Determine search space
#         if category and category in self.semantic_categories:
#             search_tokens = self.semantic_categories[category]
#         else:
#             # Default to content words for semantic similarity
#             search_tokens = (self.semantic_categories['nouns'] | 
#                            self.semantic_categories['proper_nouns'] |
#                            self.semantic_categories['verbs'] |
#                            self.semantic_categories['adjectives'])
        
#         # Calculate similarities
#         similarities = []
#         for token in search_tokens:
#             if token != target_word and token in self.token_activation_profiles:
#                 token_profile = self.token_activation_profiles[token]
#                 token_vector = self._profile_to_vector(token_profile)
                
#                 # Calculate cosine similarity
#                 similarity = cosine_similarity([target_vector], [token_vector])[0][0]
#                 if similarity > 0.1:  # Threshold for meaningful similarity
#                     similarities.append((token, similarity))
        
#         # Sort by similarity and return top-k
#         similarities.sort(key=lambda x: x[1], reverse=True)
#         return similarities[:top_k]
    
#     def _profile_to_vector(self, profile: Dict[int, List[Tuple[int, float]]]) -> np.ndarray:
#         """Convert activation profile to vector for similarity calculation."""
#         vector = np.zeros(12 * 1000)  # Larger vector for better representation
        
#         for layer, activations in profile.items():
#             if layer < 12:
#                 layer_start = layer * 1000
#                 for dim, strength in activations:
#                     if layer_start + dim < len(vector):
#                         vector[layer_start + dim] = strength
        
#         # Normalize vector
#         norm = np.linalg.norm(vector)
#         if norm > 0:
#             vector = vector / norm
        
#         return vector
    
#     def find_robust_noun_patterns(self, target_nouns: List[str], 
#                                  context_window: int = 5,
#                                  min_activation_threshold: float = 0.5) -> Dict[str, Any]:
#         """
#         Find sentences with robust activation patterns for meaningful nouns.
#         Enhanced with context awareness and semantic clustering.
#         """
#         print(f"Finding robust patterns for nouns: {target_nouns}")
        
#         # Filter to actual nouns in our data
#         valid_nouns = []
#         for noun in target_nouns:
#             if noun in self.semantic_categories['nouns'] or noun in self.semantic_categories['proper_nouns']:
#                 valid_nouns.append(noun)
#             else:
#                 # Try to find similar nouns
#                 similar = self.find_semantic_similar_words(noun, category='nouns', top_k=3)
#                 if similar:
#                     print(f"'{noun}' not found as noun, using similar: {[s[0] for s in similar[:1]]}")
#                     valid_nouns.extend([s[0] for s in similar[:1]])
        
#         if not valid_nouns:
#             return {'error': 'No valid nouns found in dataset'}
        
#         # Get relevant FFN dimensions with higher selectivity for nouns
#         relevant_dims = set()
#         noun_dim_mapping = {}
        
#         for noun in valid_nouns:
#             dims = []
#             if noun in self.token_to_ffn_dims:
#                 # Focus on top-ranked dimensions (more selective)
#                 for layer, dim, rank in self.token_to_ffn_dims[noun]:
#                     if rank < 5:  # Only top 5 positions per dimension
#                         dims.append((layer, dim))
#                         relevant_dims.add((layer, dim))
#             noun_dim_mapping[noun] = dims
        
#         print(f"Found {len(relevant_dims)} highly selective FFN dimensions for nouns")
        
#         # Enhanced sentence scoring with context awareness
#         sentence_results = []
        
#         for idx, row in self.model_df.iterrows():
#             sentence = row['sent']
            
#             # Check if sentence contains any target nouns
#             sentence_tokens = self.tokenizer.decode(self.tokenizer.encode(sentence)).split()
#             contains_target = any(noun.lower() in ' '.join(sentence_tokens).lower() for noun in valid_nouns)
            
#             if not contains_target:
#                 continue  # Skip sentences without target nouns
            
#             # Calculate enhanced activation score
#             score_info = self._calculate_robust_noun_score(
#                 row, relevant_dims, valid_nouns, noun_dim_mapping, sentence
#             )
            
#             if score_info['total_score'] >= min_activation_threshold:
#                 # Add context analysis
#                 context_info = self._analyze_noun_context(sentence, valid_nouns)
                
#                 sentence_results.append({
#                     'sentence': sentence,
#                     'score': score_info['total_score'],
#                     'score_breakdown': score_info,
#                     'context_analysis': context_info,
#                     'row_idx': idx
#                 })
        
#         # Sort by score and analyze patterns
#         sentence_results.sort(key=lambda x: x['score'], reverse=True)
        
#         # Cluster similar activation patterns
#         if len(sentence_results) > 5:
#             pattern_clusters = self._cluster_activation_patterns(sentence_results[:20])
#         else:
#             pattern_clusters = {}
        
#         return {
#             'target_nouns': valid_nouns,
#             'sentences_found': len(sentence_results),
#             'top_sentences': sentence_results[:10],
#             'pattern_clusters': pattern_clusters,
#             'activation_summary': self._summarize_noun_activations(sentence_results)
#         }
    
#     def _calculate_robust_noun_score(self, row: pd.Series, relevant_dims: Set[Tuple[int, int]], 
#                                    target_nouns: List[str], noun_dim_mapping: Dict[str, List],
#                                    sentence: str) -> Dict[str, Any]:
#         """Calculate enhanced scoring for noun patterns."""
#         score_info = {
#             'activation_strength': 0.0,
#             'noun_specificity': 0.0,
#             'context_coherence': 0.0,
#             'layer_consistency': 0.0,
#             'total_score': 0.0,
#             'active_layers': [],
#             'noun_matches': []
#         }
        
#         layers_with_activations = set()
#         layer_scores = defaultdict(float)
        
#         # 1. Activation strength (weighted by noun specificity)
#         for layer_idx in range(len(row['top_coef_idx'])):
#             top_indices = row['top_coef_idx'][layer_idx]
#             top_values = row['top_coef_vals'][layer_idx]
            
#             for dim_idx, coef_val in zip(top_indices, top_values):
#                 if (layer_idx, dim_idx) in relevant_dims:
#                     activation_strength = abs(coef_val)
                    
#                     # Weight by how specific this dimension is to nouns
#                     noun_specificity = self._calculate_noun_specificity(layer_idx, dim_idx, target_nouns)
#                     weighted_strength = activation_strength * (1 + noun_specificity)
                    
#                     score_info['activation_strength'] += weighted_strength
#                     layer_scores[layer_idx] += weighted_strength
#                     layers_with_activations.add(layer_idx)
        
#         # 2. Noun specificity bonus
#         for noun in target_nouns:
#             if noun.lower() in sentence.lower():
#                 score_info['noun_matches'].append(noun)
                
#                 # Check if this noun's specific dimensions are activated
#                 if noun in noun_dim_mapping:
#                     noun_dims = set(noun_dim_mapping[noun])
#                     active_noun_dims = 0
                    
#                     for layer_idx in range(len(row['top_coef_idx'])):
#                         top_indices = row['top_coef_idx'][layer_idx]
#                         for dim_idx in top_indices[:10]:  # Top 10 activations
#                             if (layer_idx, dim_idx) in noun_dims:
#                                 active_noun_dims += 1
                    
#                     score_info['noun_specificity'] += active_noun_dims * 0.5
        
#         # 3. Context coherence (semantic consistency)
#         score_info['context_coherence'] = self._calculate_context_coherence(sentence, target_nouns)
        
#         # 4. Layer consistency (activations across multiple layers)
#         score_info['layer_consistency'] = len(layers_with_activations) * 0.3
#         score_info['active_layers'] = list(layers_with_activations)
        
#         # Total score calculation
#         score_info['total_score'] = (
#             score_info['activation_strength'] * 1.0 +
#             score_info['noun_specificity'] * 1.5 +
#             score_info['context_coherence'] * 0.8 +
#             score_info['layer_consistency'] * 0.4
#         )
        
#         return score_info
    
#     def _calculate_noun_specificity(self, layer: int, dim: int, target_nouns: List[str]) -> float:
#         """Calculate how specific a dimension is to nouns vs other word types."""
#         if (layer, dim) not in self.ffn_projections:
#             return 0.0
        
#         top_tokens = self.ffn_projections[(layer, dim)][:20]  # Top 20 tokens
        
#         noun_count = 0
#         total_content_words = 0
        
#         for token in top_tokens:
#             if token in self.semantic_categories['nouns'] or token in self.semantic_categories['proper_nouns']:
#                 noun_count += 1
#                 total_content_words += 1
#             elif (token in self.semantic_categories['verbs'] or 
#                   token in self.semantic_categories['adjectives'] or
#                   token in self.semantic_categories['adverbs']):
#                 total_content_words += 1
        
#         if total_content_words == 0:
#             return 0.0
        
#         return noun_count / total_content_words
    
#     def _calculate_context_coherence(self, sentence: str, target_nouns: List[str]) -> float:
#         """Calculate semantic coherence of the sentence context."""
#         if not self.nlp:
#             return 0.5  # Default moderate score
        
#         doc = self.nlp(sentence)
        
#         # Look for semantic relationships
#         coherence_score = 0.0
        
#         # 1. Named entity recognition bonus
#         entities = [ent.text.lower() for ent in doc.ents]
#         entity_overlap = sum(1 for noun in target_nouns if noun.lower() in entities)
#         coherence_score += entity_overlap * 0.3
        
#         # 2. Dependency relationships
#         noun_tokens = [token for token in doc if token.text.lower() in [n.lower() for n in target_nouns]]
#         for noun_token in noun_tokens:
#             # Check for meaningful dependencies
#             if noun_token.dep_ in ['nsubj', 'dobj', 'pobj', 'compound']:
#                 coherence_score += 0.2
            
#             # Check for descriptive relationships
#             if any(child.pos_ == 'ADJ' for child in noun_token.children):
#                 coherence_score += 0.1
        
#         return min(coherence_score, 1.0)  # Cap at 1.0
    
#     def _analyze_noun_context(self, sentence: str, target_nouns: List[str]) -> Dict[str, Any]:
#         """Analyze the context around target nouns."""
#         context_info = {
#             'sentence_length': len(sentence.split()),
#             'noun_positions': [],
#             'surrounding_words': {},
#             'semantic_context': []
#         }
        
#         words = sentence.split()
        
#         for noun in target_nouns:
#             for i, word in enumerate(words):
#                 if noun.lower() in word.lower():
#                     context_info['noun_positions'].append({
#                         'noun': noun,
#                         'position': i,
#                         'word': word
#                     })
                    
#                     # Capture surrounding context
#                     start = max(0, i - 2)
#                     end = min(len(words), i + 3)
#                     context_info['surrounding_words'][noun] = ' '.join(words[start:end])
        
#         # Add semantic analysis if spaCy is available
#         if self.nlp:
#             doc = self.nlp(sentence)
#             context_info['semantic_context'] = [
#                 {'text': ent.text, 'label': ent.label_} 
#                 for ent in doc.ents
#             ]
        
#         return context_info
    
#     def _cluster_activation_patterns(self, sentence_results: List[Dict]) -> Dict[int, List[Dict]]:
#         """Cluster sentences by similar activation patterns."""
#         if len(sentence_results) < 3:
#             return {}
        
#         # Extract activation features for clustering
#         features = []
#         for result in sentence_results:
#             score_breakdown = result['score_breakdown']
#             feature_vector = [
#                 score_breakdown['activation_strength'],
#                 score_breakdown['noun_specificity'],
#                 score_breakdown['context_coherence'],
#                 score_breakdown['layer_consistency'],
#                 len(score_breakdown['active_layers']),
#                 len(score_breakdown['noun_matches'])
#             ]
#             features.append(feature_vector)
        
#         # Perform clustering
#         n_clusters = min(5, len(features) // 2)
#         if n_clusters < 2:
#             return {}
        
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         cluster_labels = kmeans.fit_predict(features)
        
#         # Organize results by cluster
#         clusters = defaultdict(list)
#         for result, cluster_id in zip(sentence_results, cluster_labels):
#             clusters[cluster_id].append(result)
        
#         return dict(clusters)
    
#     def _summarize_noun_activations(self, sentence_results: List[Dict]) -> Dict[str, Any]:
#         """Summarize activation patterns for noun analysis."""
#         if not sentence_results:
#             return {}
        
#         summary = {
#             'total_sentences': len(sentence_results),
#             'avg_score': np.mean([r['score'] for r in sentence_results]),
#             'score_std': np.std([r['score'] for r in sentence_results]),
#             'common_layers': [],
#             'frequent_nouns': [],
#             'context_patterns': []
#         }
        
#         # Analyze common activation layers
#         all_layers = []
#         for result in sentence_results:
#             all_layers.extend(result['score_breakdown']['active_layers'])
        
#         layer_counts = Counter(all_layers)
#         summary['common_layers'] = layer_counts.most_common(5)
        
#         # Analyze frequent nouns
#         all_nouns = []
#         for result in sentence_results:
#             all_nouns.extend(result['score_breakdown']['noun_matches'])
        
#         noun_counts = Counter(all_nouns)
#         summary['frequent_nouns'] = noun_counts.most_common(10)
        
#         return summary

# def demonstrate_enhanced_noun_mapping():
#     """Demonstration of enhanced noun mapping capabilities."""
    
#     # Initialize the enhanced mapper
#     mapper = EnhancedSemanticMapper(
#         ffn_projections_path="ffn_projections.pkl",
#         model_df_path="model_df_10k.pkl"
#     )
    
#     print("="*60)
#     print("ENHANCED SEMANTIC NOUN MAPPING DEMONSTRATION")
#     print("="*60)
    
#     # 1. Show semantic categorization results
#     print("\n1. SEMANTIC CATEGORIZATION RESULTS")
#     print("-" * 40)
#     for category, tokens in mapper.semantic_categories.items():
#         sample_tokens = list(tokens)[:10]
#         print(f"{category.upper()}: {len(tokens)} tokens")
#         print(f"  Sample: {sample_tokens}")
    
#     # 2. Find semantic similarities
#     print("\n2. SEMANTIC SIMILARITY ANALYSIS")
#     print("-" * 40)
    
#     test_words = ['king', 'house', 'water', 'computer']
#     for word in test_words:
#         if word in mapper.token_activation_profiles:
#             similar_words = mapper.find_semantic_similar_words(word, category='nouns', top_k=5)
#             print(f"\nWords similar to '{word}':")
#             for similar_word, similarity in similar_words:
#                 print(f"  {similar_word}: {similarity:.3f}")
    
#     # 3. Robust noun pattern analysis
#     print("\n3. ROBUST NOUN PATTERN ANALYSIS")
#     print("-" * 40)
    
#     target_nouns = ['king', 'house', 'water', 'time', 'world']
#     results = mapper.find_robust_noun_patterns(target_nouns, min_activation_threshold=1.0)
    
#     if 'error' not in results:
#         print(f"Analyzed nouns: {results['target_nouns']}")
#         print(f"Sentences found: {results['sentences_found']}")
        
#         print(f"\nTop activation patterns:")
#         for i, result in enumerate(results['top_sentences'][:3], 1):
#             print(f"\n{i}. Score: {result['score']:.3f}")
#             print(f"   Sentence: {result['sentence'][:100]}...")
#             print(f"   Nouns found: {result['score_breakdown']['noun_matches']}")
#             print(f"   Active layers: {result['score_breakdown']['active_layers']}")
#             print(f"   Context: {result['context_analysis']['surrounding_words']}")
        
#         # Show activation summary
#         if results['activation_summary']:
#             summary = results['activation_summary']
#             print(f"\nActivation Summary:")
#             print(f"  Average score: {summary['avg_score']:.3f}")
#             print(f"  Most active layers: {summary['common_layers'][:3]}")
#             print(f"  Most frequent nouns: {summary['frequent_nouns'][:5]}")
    
#     # 4. Pattern clustering analysis
#     if 'pattern_clusters' in results and results['pattern_clusters']:
#         print(f"\n4. ACTIVATION PATTERN CLUSTERS")
#         print("-" * 40)
        
#         for cluster_id, cluster_sentences in results['pattern_clusters'].items():
#             print(f"\nCluster {cluster_id}: {len(cluster_sentences)} sentences")
#             avg_score = np.mean([s['score'] for s in cluster_sentences])
#             print(f"  Average score: {avg_score:.3f}")
            
#             # Show representative sentence
#             representative = cluster_sentences[0]
#             print(f"  Representative: {representative['sentence'][:80]}...")

#     def build_semantic_word_embeddings(self) -> Dict[str, np.ndarray]:
#         """
#         Build semantic word embeddings based on FFN activation patterns.
#         More robust than simple activation vectors.
#         """
#         print("Building semantic word embeddings...")
        
#         word_embeddings = {}
        
#         # Focus on content words with rich semantic meaning
#         content_words = (self.semantic_categories['nouns'] | 
#                         self.semantic_categories['proper_nouns'] |
#                         self.semantic_categories['verbs'] |
#                         self.semantic_categories['adjectives'])
        
#         for word in content_words:
#             if word in self.token_activation_profiles:
#                 embedding = self._create_semantic_embedding(word)
#                 if embedding is not None:
#                     word_embeddings[word] = embedding
        
#         return word_embeddings
    
#     def _create_semantic_embedding(self, word: str) -> Optional[np.ndarray]:
#         """Create a rich semantic embedding for a word."""
#         profile = self.token_activation_profiles[word]
        
#         # Multi-scale embedding: combine different granularities
#         embedding_parts = []
        
#         # 1. Layer-specific activations (12 components)
#         layer_activations = np.zeros(12)
#         for layer in range(12):
#             if layer in profile:
#                 layer_activations[layer] = np.mean([strength for _, strength in profile[layer]])
#         embedding_parts.append(layer_activations)
        
#         # 2. Top dimension activations across all layers (50 components)
#         all_activations = []
#         for layer_acts in profile.values():
#             all_activations.extend([strength for _, strength in layer_acts])
        
#         if all_activations:
#             top_activations = sorted(all_activations, reverse=True)[:50]
#             top_activations.extend([0.0] * (50 - len(top_activations)))  # Pad if needed
#             embedding_parts.append(np.array(top_activations))
        
#         # 3. Semantic category features (10 components)
#         category_features = np.zeros(10)
#         for i, (cat_name, cat_tokens) in enumerate(self.semantic_categories.items()):
#             if i < 10 and word in cat_tokens:
#                 category_features[i] = 1.0
#         embedding_parts.append(category_features)
        
#         # 4. Co-occurrence features (number of shared dimensions with other words)
#         cooccurrence_features = self._calculate_cooccurrence_features(word)
#         embedding_parts.append(cooccurrence_features)
        
#         # Combine all parts
#         if embedding_parts:
#             full_embedding = np.concatenate(embedding_parts)
#             # Normalize
#             norm = np.linalg.norm(full_embedding)
#             if norm > 0:
#                 return full_embedding / norm
        
#         return None
    
#     def _calculate_cooccurrence_features(self, word: str, n_features: int = 20) -> np.ndarray:
#         """Calculate features based on shared FFN dimensions with other words."""
#         if word not in self.token_to_ffn_dims:
#             return np.zeros(n_features)
        
#         word_dims = set((layer, dim) for layer, dim, _ in self.token_to_ffn_dims[word])
        
#         # Count co-occurrences with different semantic categories
#         cooccurrence_counts = defaultdict(int)
        
#         for other_word, other_dims_info in self.token_to_ffn_dims.items():
#             if other_word != word:
#                 other_dims = set((layer, dim) for layer, dim, _ in other_dims_info)
#                 shared_dims = len(word_dims & other_dims)
                
#                 if shared_dims > 0:
#                     # Categorize the other word
#                     for cat_name, cat_tokens in self.semantic_categories.items():
#                         if other_word in cat_tokens:
#                             cooccurrence_counts[cat_name] += shared_dims
#                             break
        
#         # Convert to feature vector
#         features = np.zeros(n_features)
#         for i, cat_name in enumerate(self.semantic_categories.keys()):
#             if i < n_features:
#                 features[i] = cooccurrence_counts[cat_name]
        
#         # Add some statistical features
#         if len(features) > len(self.semantic_categories):
#             remaining_idx = len(self.semantic_categories)
#             if remaining_idx < n_features:
#                 features[remaining_idx] = len(word_dims)  # Total dimensions
#             if remaining_idx + 1 < n_features:
#                 features[remaining_idx + 1] = np.mean(list(cooccurrence_counts.values()) or [0])
        
#         return features
    
#     def find_conceptual_analogies(self, word_a: str, word_b: str, word_c: str, 
#                                  top_k: int = 5) -> List[Tuple[str, float]]:
#         """
#         Find analogical relationships: A is to B as C is to ?
#         Example: king is to queen as man is to woman
#         """
#         embeddings = self.build_semantic_word_embeddings()
        
#         if not all(word in embeddings for word in [word_a, word_b, word_c]):
#             missing = [w for w in [word_a, word_b, word_c] if w not in embeddings]
#             print(f"Missing embeddings for: {missing}")
#             return []
        
#         # Calculate analogy vector: B - A + C = ?
#         vec_a = embeddings[word_a]
#         vec_b = embeddings[word_b]
#         vec_c = embeddings[word_c]
        
#         target_vector = vec_b - vec_a + vec_c
        
#         # Find closest words to target vector
#         similarities = []
#         for word, embedding in embeddings.items():
#             if word not in [word_a, word_b, word_c]:  # Exclude input words
#                 similarity = cosine_similarity([target_vector], [embedding])[0][0]
#                 similarities.append((word, similarity))
        
#         similarities.sort(key=lambda x: x[1], reverse=True)
#         return similarities[:top_k]
    
#     def analyze_semantic_fields(self, seed_words: List[str], 
#                                expansion_threshold: float = 0.6) -> Dict[str, List[str]]:
#         """
#         Build semantic fields around seed words by finding related concepts.
#         """
#         embeddings = self.build_semantic_word_embeddings()
#         semantic_fields = {}
        
#         for seed_word in seed_words:
#             if seed_word not in embeddings:
#                 continue
            
#             # Find semantically related words
#             related_words = []
#             seed_embedding = embeddings[seed_word]
            
#             for word, embedding in embeddings.items():
#                 if word != seed_word:
#                     similarity = cosine_similarity([seed_embedding], [embedding])[0][0]
#                     if similarity >= expansion_threshold:
#                         related_words.append((word, similarity))
            
#             # Sort by similarity and extract words
#             related_words.sort(key=lambda x: x[1], reverse=True)
#             semantic_fields[seed_word] = [word for word, _ in related_words[:20]]
        
#         return semantic_fields
    
#     def detect_semantic_shifts(self, word: str, layer_range: Tuple[int, int] = (0, 11)) -> Dict[str, Any]:
#         """
#         Analyze how a word's semantic representation changes across layers.
#         Useful for understanding semantic processing in the model.
#         """
#         if word not in self.token_activation_profiles:
#             return {'error': f'Word "{word}" not found'}
        
#         profile = self.token_activation_profiles[word]
#         layer_representations = {}
        
#         # Get representation at each layer
#         for layer in range(layer_range[0], layer_range[1] + 1):
#             if layer in profile:
#                 # Create layer-specific embedding
#                 layer_activations = profile[layer]
#                 if layer_activations:
#                     # Use top activations as representation
#                     top_activations = sorted(layer_activations, key=lambda x: x[1], reverse=True)[:20]
#                     layer_vector = np.array([act[1] for act in top_activations])
#                     layer_representations[layer] = layer_vector
        
#         # Analyze changes between layers
#         semantic_shifts = {}
#         prev_layer = None
#         prev_vector = None
        
#         for layer in sorted(layer_representations.keys()):
#             current_vector = layer_representations[layer]
            
#             if prev_vector is not None:
#                 # Calculate similarity with previous layer
#                 similarity = cosine_similarity([prev_vector], [current_vector])[0][0]
#                 semantic_shifts[f'layers_{prev_layer}_to_{layer}'] = {
#                     'similarity': similarity,
#                     'change_magnitude': 1 - similarity
#                 }
            
#             prev_layer = layer
#             prev_vector = current_vector
        
#         # Find layers with biggest semantic shifts
#         if semantic_shifts:
#             biggest_shifts = sorted(
#                 semantic_shifts.items(), 
#                 key=lambda x: x[1]['change_magnitude'], 
#                 reverse=True
#             )[:3]
#         else:
#             biggest_shifts = []
        
#         return {
#             'word': word,
#             'layer_representations': {k: v.tolist() for k, v in layer_representations.items()},
#             'semantic_shifts': semantic_shifts,
#             'biggest_shifts': biggest_shifts,
#             'layers_analyzed': list(layer_representations.keys())
#         }

# def advanced_noun_analysis_demo():
#     """Demonstrate advanced semantic analysis capabilities."""
    
#     mapper = EnhancedSemanticMapper("ffn_projections.pkl", "model_df_10k.pkl")
    
#     print("="*60)
#     print("ADVANCED SEMANTIC ANALYSIS DEMONSTRATION")
#     print("="*60)
    
#     # 1. Semantic word embeddings
#     print("\n1. BUILDING SEMANTIC WORD EMBEDDINGS")
#     print("-" * 40)
#     embeddings = mapper.build_semantic_word_embeddings()
#     print(f"Built embeddings for {len(embeddings)} words")
    
#     # Show embedding dimensionality
#     if embeddings:
#         sample_word = list(embeddings.keys())[0]
#         print(f"Embedding dimensionality: {len(embeddings[sample_word])}")
    
#     # 2. Conceptual analogies
#     print("\n2. CONCEPTUAL ANALOGIES")
#     print("-" * 40)
    
#     analogy_tests = [
#         ("king", "queen", "man"),
#         ("Paris", "France", "London"),
#         ("write", "writer", "teach"),
#         ("big", "bigger", "small")
#     ]
    
#     for word_a, word_b, word_c in analogy_tests:
#         analogies = mapper.find_conceptual_analogies(word_a, word_b, word_c, top_k=3)
#         if analogies:
#             print(f"\n{word_a} : {word_b} :: {word_c} : ?")
#             for word, score in analogies:
#                 print(f"  {word} ({score:.3f})")
    
#     # 3. Semantic fields
#     print("\n3. SEMANTIC FIELD ANALYSIS")
#     print("-" * 40)
    
#     seed_words = ["king", "house", "water", "science"]
#     semantic_fields = mapper.analyze_semantic_fields(seed_words, expansion_threshold=0.5)
    
#     for seed, related_words in semantic_fields.items():
#         print(f"\nSemantic field for '{seed}':")
#         print(f"  Related: {related_words[:8]}")
    
#     # 4. Semantic shifts across layers
#     print("\n4. SEMANTIC SHIFTS ACROSS LAYERS")
#     print("-" * 40)
    
#     test_words = ["king", "water", "science"]
#     for word in test_words:
#         shifts = mapper.detect_semantic_shifts(word)
#         if 'error' not in shifts:
#             print(f"\nSemantic shifts for '{word}':")
#             print(f"  Layers analyzed: {shifts['layers_analyzed']}")
#             if shifts['biggest_shifts']:
#                 print(f"  Biggest shift: {shifts['biggest_shifts'][0][0]} "
#                       f"(magnitude: {shifts['biggest_shifts'][0][1]['change_magnitude']:.3f})")

# if __name__ == "__main__":
#     demonstrate_enhanced_noun_mapping()
#     print("\n" + "="*60)
#     advanced_noun_analysis_demo()

"""
nltk_data] Downloading package wordnet to /home/ubuntu/nltk_data...
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /home/ubuntu/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
Loading FFN projections...
Loading model DataFrame...
Building enhanced semantic mappings...
Categorizing tokens by semantic type...
nouns: 17955 tokens
proper_nouns: 10215 tokens
verbs: 8137 tokens
adjectives: 3450 tokens
adverbs: 1607 tokens
function_words: 2162 tokens
numbers: 751 tokens
punctuation: 195 tokens
Building semantic clusters...
Created 20 semantic clusters
============================================================
ENHANCED SEMANTIC NOUN MAPPING DEMONSTRATION
============================================================

1. SEMANTIC CATEGORIZATION RESULTS
----------------------------------------
NOUNS: 17955 tokens
  Sample: [' chees', ' Glover', ' screenshot', ' Murder', ' flakes', ' Towns', 'hemat', 'population', ' launches', ' KN']
PROPER_NOUNS: 10215 tokens
  Sample: ['APD', ' Sao', 'gemony', ' Nielsen', ' Tempest', 'CLA', 'inth', ' BAR', ' Myr', ' Pog']
VERBS: 8137 tokens
  Sample: [' Bun', ' chosen', ' Scouting', 'known', 'ufact', ' scrapped', ' regarded', ' undergoing', 'ater', 'Born']
ADJECTIVES: 3450 tokens
  Sample: [' noisy', ' fragile', 'sei', 'Bottom', ' wearable', ' elderly', ' secure', 'ample', ' free', ' irrational']
ADVERBS: 1607 tokens
  Sample: ['hw', ' discourse', 'ioxide', ' tougher', 'ully', 'especially', ' willingly', ' vehemently', 'Secondly', ' up']
FUNCTION_WORDS: 2162 tokens
  Sample: ['--------------------------------', 'š', 'oh', '/\u200b', ' Without', ' His', ';;', ' ILCS', '................................', ' fourteen']
NUMBERS: 751 tokens
  Sample: ['774', '25', ' 1930', ' 920', '172', ' 443', '6666', ' 2002', '55', ' 610']
PUNCTUATION: 195 tokens
  Sample: ['■', '@', ' •', ' █', '¨', '!', ' 🙂', '・', ' \xad', '\x7f']

2. SEMANTIC SIMILARITY ANALYSIS
----------------------------------------

Words similar to 'king':
   inequality: 0.577
   negativity: 0.369
   volatility: 0.333
   disadvantage: 0.315
  GHz: 0.288

Words similar to 'house':
   permission: 0.455
   thence: 0.400
   Lawn: 0.356
   immense: 0.318
   suggestions: 0.305

Words similar to 'water':
  sale: 0.360
  ifles: 0.285
  Links: 0.280
   Panda: 0.258
  adult: 0.195

Words similar to 'computer':
  Mu: 0.994
   Yi: 0.994
  Computer: 0.994
   computers: 0.616
  achusetts: 0.614

3. ROBUST NOUN PATTERN ANALYSIS
----------------------------------------
Finding robust patterns for nouns: ['king', 'house', 'water', 'time', 'world']
Found 52 highly selective FFN dimensions for nouns
Analyzed nouns: ['king', 'house', 'water', 'time', 'world']
Sentences found: 204

Top activation patterns:

1. Score: 43.537
   Sentence: Ross helped the Bruins finish first place in the league ten times and to win the Stanley Cup three...
   Nouns found: ['time']
   Active layers: [1, 5, 9]
   Context: {'time': 'league ten times and to'}

2. Score: 33.501
   Sentence: Even though he turned 40 during the season , he scored 20 or more points 42 times , 30...
   Nouns found: ['time']
   Active layers: [9]
   Context: {'time': 'points 42 times , 30'}

3. Score: 31.681
   Sentence: The lighthouse on North Island flanking Winyah Bay collapsed under high winds , and in Georgetown pr...
   Nouns found: ['king', 'house']
   Active layers: [10]
   Context: {'king': 'North Island flanking Winyah Bay', 'house': 'The lighthouse on North'}

Activation Summary:
  Average score: 8.760
  Most active layers: [(9, 44), (7, 37), (1, 35)]
  Most frequent nouns: [('time', 79), ('king', 69), ('world', 32), ('house', 21), ('water', 13)]

4. ACTIVATION PATTERN CLUSTERS
----------------------------------------

Cluster 2: 1 sentences
  Average score: 43.537
  Representative: Ross helped the Bruins finish first place in the league ten times and to win the...

Cluster 4: 4 sentences
  Average score: 31.695
  Representative: Even though he turned 40 during the season , he scored 20 or more points 42 time...

Cluster 0: 5 sentences
  Average score: 28.189
  Representative: He started in the final , whereby the team returned to The Football League for t...

Cluster 3: 5 sentences
  Average score: 21.807
  Representative: The outbreak of World War II in 1939 put a halt to all salvage operations , and ...

Cluster 1: 5 sentences
  Average score: 18.089
  Representative: Major religious works include the triptychs in oil , The Feeding of the Five Tho...

============================================================
Loading FFN projections...
Loading model DataFrame...
Building enhanced semantic mappings...
Categorizing tokens by semantic type...
nouns: 17955 tokens
proper_nouns: 10215 tokens
verbs: 8137 tokens
adjectives: 3450 tokens
adverbs: 1607 tokens
function_words: 2162 tokens
numbers: 751 tokens
punctuation: 195 tokens
Building semantic clusters...
Created 20 semantic clusters
============================================================
ADVANCED SEMANTIC ANALYSIS DEMONSTRATION
============================================================

1. BUILDING SEMANTIC WORD EMBEDDINGS
----------------------------------------
Traceback (most recent call last):
  File "/home/ubuntu/krishiv-llm/ffn-values/reverse_engineer.py", line 2534, in <module>
^^^^^^^^
  File "/home/ubuntu/krishiv-llm/ffn-values/reverse_engineer.py", line 2480, in advanced_noun_analysis_demo
    if other_word in cat_tokens:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'EnhancedSemanticMapper' object has no attribute 'build_semantic_word_embeddings'
"""

###### NEW Noun test 

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.corpus import wordnet
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class EnhancedSemanticMapper:
    """
    Enhanced mapper for robust semantic words like nouns, with improved
    semantic clustering and context-aware activation analysis.
    Fixed implementation with proper semantic embedding methods.
    """
    
    def __init__(self, ffn_projections_path: str, model_df_path: str, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Load spaCy for better linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load data
        print("Loading FFN projections...")
        with open(ffn_projections_path, 'rb') as f:
            self.ffn_projections = pickle.load(f)
        
        print("Loading model DataFrame...")
        with open(model_df_path, 'rb') as f:
            self.model_df = pickle.load(f)
        
        # Enhanced mappings for semantic analysis
        self._build_enhanced_mappings()
        self._categorize_semantic_tokens()
        self._build_semantic_clusters()
        
        # Cache for embeddings
        self._semantic_embeddings_cache = None
    
    def _build_enhanced_mappings(self):
        """Build enhanced mappings with semantic awareness."""
        print("Building enhanced semantic mappings...")
        
        # Basic token to FFN mappings
        self.token_to_ffn_dims = defaultdict(list)
        self.layer_to_active_tokens = defaultdict(set)
        
        # Enhanced semantic mappings
        self.semantic_categories = {
            'nouns': set(),
            'proper_nouns': set(),
            'verbs': set(),
            'adjectives': set(),
            'adverbs': set(),
            'function_words': set(),
            'numbers': set(),
            'punctuation': set()
        }
        
        # Activation strength mappings
        self.token_activation_profiles = {}  # token -> {layer: [activations]}
        self.strong_activators = defaultdict(set)  # (layer, dim) -> {tokens}
        
        # Dimension selectivity mapping - NEW: track how selective each dimension is
        self.dimension_selectivity = {}  # (layer, dim) -> selectivity_score
        
        for (layer, dim), top_tokens in self.ffn_projections.items():
            # Calculate dimension selectivity (inverse of activation spread)
            if len(top_tokens) > 0:
                # Higher selectivity = fewer strongly activated tokens
                selectivity = 1.0 / (len([t for t in top_tokens[:20]]) + 1)
                self.dimension_selectivity[(layer, dim)] = selectivity
            
            for rank, token in enumerate(top_tokens):
                # Basic mapping
                self.token_to_ffn_dims[token].append((layer, dim, rank))
                self.layer_to_active_tokens[layer].add(token)
                
                # Track strong activators (top 10 tokens per dimension)
                if rank < 10:
                    self.strong_activators[(layer, dim)].add(token)
                
                # Build activation profiles with selectivity weighting
                if token not in self.token_activation_profiles:
                    self.token_activation_profiles[token] = defaultdict(list)
                
                # Weight by inverse rank and dimension selectivity
                activation_strength = (1.0 / (rank + 1)) * selectivity
                self.token_activation_profiles[token][layer].append((dim, activation_strength))
    
    def _categorize_semantic_tokens(self):
        """Categorize tokens by semantic type using linguistic analysis."""
        print("Categorizing tokens by semantic type...")
        
        all_tokens = list(self.token_to_ffn_dims.keys())
        
        # Enhanced token cleaning
        def clean_token_for_analysis(token):
            # Remove common GPT-2 tokenization artifacts
            cleaned = token.strip()
            if cleaned.startswith('Ġ'):  # GPT-2 space prefix
                cleaned = cleaned[1:]
            if cleaned.startswith(' '):
                cleaned = cleaned[1:]
            return cleaned
        
        for token in all_tokens:
            clean_token = clean_token_for_analysis(token)
            if not clean_token:
                continue
            
            # Basic regex patterns
            if re.match(r'^[0-9]+$', clean_token):
                self.semantic_categories['numbers'].add(token)
            elif re.match(r'^[^\w\s]$', clean_token):
                self.semantic_categories['punctuation'].add(token)
            elif len(clean_token) == 1 and not clean_token.isalpha():
                self.semantic_categories['punctuation'].add(token)
            else:
                # Use spaCy for more sophisticated analysis
                if self.nlp:
                    doc = self.nlp(clean_token)
                    if doc and len(doc) > 0:
                        pos_tag = doc[0].pos_
                        if pos_tag == 'NOUN':
                            self.semantic_categories['nouns'].add(token)
                        elif pos_tag == 'PROPN':
                            self.semantic_categories['proper_nouns'].add(token)
                        elif pos_tag == 'VERB':
                            self.semantic_categories['verbs'].add(token)
                        elif pos_tag == 'ADJ':
                            self.semantic_categories['adjectives'].add(token)
                        elif pos_tag == 'ADV':
                            self.semantic_categories['adverbs'].add(token)
                        else:
                            self.semantic_categories['function_words'].add(token)
                    else:
                        self.semantic_categories['function_words'].add(token)
                else:
                    # Fallback to NLTK POS tagging
                    try:
                        pos_tags = nltk.pos_tag([clean_token])
                        pos = pos_tags[0][1]
                        
                        if pos.startswith('NN'):
                            if pos == 'NNP' or pos == 'NNPS':
                                self.semantic_categories['proper_nouns'].add(token)
                            else:
                                self.semantic_categories['nouns'].add(token)
                        elif pos.startswith('VB'):
                            self.semantic_categories['verbs'].add(token)
                        elif pos.startswith('JJ'):
                            self.semantic_categories['adjectives'].add(token)
                        elif pos.startswith('RB'):
                            self.semantic_categories['adverbs'].add(token)
                        else:
                            self.semantic_categories['function_words'].add(token)
                    except:
                        self.semantic_categories['function_words'].add(token)
        
        # Print categorization results
        for category, tokens in self.semantic_categories.items():
            print(f"{category}: {len(tokens)} tokens")
    
    def _build_semantic_clusters(self):
        """Build semantic clusters based on activation patterns."""
        print("Building semantic clusters...")
        
        # Focus on nouns and content words for clustering
        content_tokens = (self.semantic_categories['nouns'] | 
                         self.semantic_categories['proper_nouns'] |
                         self.semantic_categories['verbs'] |
                         self.semantic_categories['adjectives'])
        
        if len(content_tokens) < 10:
            print("Warning: Not enough content tokens for clustering")
            self.semantic_clusters = {}
            return
        
        # Create improved activation vectors for clustering
        activation_vectors = []
        token_list = []
        
        for token in content_tokens:
            if token in self.token_activation_profiles:
                vector = self._create_improved_activation_vector(token)
                
                if vector is not None and np.sum(np.abs(vector)) > 0:
                    activation_vectors.append(vector)
                    token_list.append(token)
        
        if len(activation_vectors) > 10:
            # Normalize vectors before clustering
            activation_vectors = np.array(activation_vectors)
            scaler = StandardScaler()
            normalized_vectors = scaler.fit_transform(activation_vectors)
            
            # Perform k-means clustering
            n_clusters = min(20, len(activation_vectors) // 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_vectors)
            
            # Organize clusters
            self.semantic_clusters = defaultdict(list)
            for token, cluster_id in zip(token_list, cluster_labels):
                self.semantic_clusters[cluster_id].append(token)
            
            print(f"Created {n_clusters} semantic clusters")
        else:
            self.semantic_clusters = {}
    
    def _create_improved_activation_vector(self, token: str, max_dims: int = 200) -> Optional[np.ndarray]:
        """Create improved activation vector with better representation."""
        if token not in self.token_activation_profiles:
            return None
        
        profile = self.token_activation_profiles[token]
        
        # Create multi-layer representation
        vector_parts = []
        
        # 1. Layer-wise max activations (12 dimensions)
        layer_maxes = np.zeros(12)
        for layer in range(12):
            if layer in profile:
                activations = [strength for _, strength in profile[layer]]
                if activations:
                    layer_maxes[layer] = max(activations)
        vector_parts.append(layer_maxes)
        
        # 2. Layer-wise activation counts (12 dimensions)
        layer_counts = np.zeros(12)
        for layer in range(12):
            if layer in profile:
                layer_counts[layer] = len(profile[layer])
        vector_parts.append(layer_counts)
        
        # 3. Top cross-layer activations
        all_activations = []
        for layer_acts in profile.values():
            all_activations.extend([strength for _, strength in layer_acts])
        
        if all_activations:
            top_activations = sorted(all_activations, reverse=True)[:50]
            # Pad if necessary
            while len(top_activations) < 50:
                top_activations.append(0.0)
            vector_parts.append(np.array(top_activations))
        
        # 4. Selectivity-weighted features
        selectivity_features = np.zeros(20)
        feature_idx = 0
        for layer, activations in profile.items():
            if feature_idx >= 20:
                break
            
            # Calculate average selectivity for this layer
            layer_selectivity = 0.0
            for dim, strength in activations:
                if (layer, dim) in self.dimension_selectivity:
                    layer_selectivity += self.dimension_selectivity[(layer, dim)] * strength
            
            if activations:
                selectivity_features[feature_idx] = layer_selectivity / len(activations)
            feature_idx += 1
        
        vector_parts.append(selectivity_features)
        
        # Combine and normalize
        if vector_parts:
            full_vector = np.concatenate(vector_parts)
            norm = np.linalg.norm(full_vector)
            if norm > 0:
                return full_vector / norm
        
        return None
    
    def build_semantic_word_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Build semantic word embeddings based on FFN activation patterns.
        More robust than simple activation vectors.
        """
        if self._semantic_embeddings_cache is not None:
            return self._semantic_embeddings_cache
        
        print("Building semantic word embeddings...")
        
        word_embeddings = {}
        
        # Focus on content words with rich semantic meaning
        content_words = (self.semantic_categories['nouns'] | 
                        self.semantic_categories['proper_nouns'] |
                        self.semantic_categories['verbs'] |
                        self.semantic_categories['adjectives'])
        
        for word in content_words:
            if word in self.token_activation_profiles:
                embedding = self._create_semantic_embedding(word)
                if embedding is not None:
                    word_embeddings[word] = embedding
        
        self._semantic_embeddings_cache = word_embeddings
        return word_embeddings
    
    def _create_semantic_embedding(self, word: str) -> Optional[np.ndarray]:
        """Create a rich semantic embedding for a word."""
        profile = self.token_activation_profiles[word]
        
        # Multi-scale embedding: combine different granularities
        embedding_parts = []
        
        # 1. Layer-specific activations (12 components) - IMPROVED
        layer_activations = np.zeros(12)
        layer_stds = np.zeros(12)  # Add standard deviation info
        
        for layer in range(12):
            if layer in profile:
                strengths = [strength for _, strength in profile[layer]]
                if strengths:
                    layer_activations[layer] = np.mean(strengths)
                    layer_stds[layer] = np.std(strengths) if len(strengths) > 1 else 0
        
        embedding_parts.extend([layer_activations, layer_stds])
        
        # 2. Top dimension activations with selectivity weighting (30 components)
        all_weighted_activations = []
        for layer, layer_acts in profile.items():
            for dim, strength in layer_acts:
                selectivity = self.dimension_selectivity.get((layer, dim), 0.5)
                weighted_strength = strength * selectivity
                all_weighted_activations.append(weighted_strength)
        
        if all_weighted_activations:
            top_activations = sorted(all_weighted_activations, reverse=True)[:30]
            top_activations.extend([0.0] * (30 - len(top_activations)))  # Pad if needed
            embedding_parts.append(np.array(top_activations))
        else:
            embedding_parts.append(np.zeros(30))
        
        # 3. Semantic category features (8 components)
        category_features = np.zeros(8)
        for i, (cat_name, cat_tokens) in enumerate(self.semantic_categories.items()):
            if i < 8 and word in cat_tokens:
                category_features[i] = 1.0
        embedding_parts.append(category_features)
        
        # 4. Co-occurrence features with improved calculation (15 components)
        cooccurrence_features = self._calculate_improved_cooccurrence_features(word, 15)
        embedding_parts.append(cooccurrence_features)
        
        # 5. WordNet semantic features if available (10 components)
        wordnet_features = self._get_wordnet_features(word, 10)
        embedding_parts.append(wordnet_features)
        
        # Combine all parts
        if embedding_parts:
            full_embedding = np.concatenate(embedding_parts)
            # Normalize
            norm = np.linalg.norm(full_embedding)
            if norm > 0:
                return full_embedding / norm
        
        return None
    
    def _calculate_improved_cooccurrence_features(self, word: str, n_features: int = 15) -> np.ndarray:
        """Calculate improved features based on shared FFN dimensions with other words."""
        if word not in self.token_to_ffn_dims:
            return np.zeros(n_features)
        
        word_dims = set((layer, dim) for layer, dim, _ in self.token_to_ffn_dims[word])
        
        # Count co-occurrences with different semantic categories (weighted by selectivity)
        cooccurrence_scores = defaultdict(float)
        
        for other_word, other_dims_info in self.token_to_ffn_dims.items():
            if other_word != word:
                other_dims = set((layer, dim) for layer, dim, _ in other_dims_info)
                shared_dims = word_dims & other_dims
                
                if shared_dims:
                    # Weight by dimension selectivity
                    shared_weight = sum(self.dimension_selectivity.get(dim_pair, 0.5) 
                                      for dim_pair in shared_dims)
                    
                    # Categorize the other word
                    for cat_name, cat_tokens in self.semantic_categories.items():
                        if other_word in cat_tokens:
                            cooccurrence_scores[cat_name] += shared_weight
                            break
        
        # Convert to feature vector
        features = np.zeros(n_features)
        
        # First 8 features: semantic category co-occurrences
        for i, cat_name in enumerate(self.semantic_categories.keys()):
            if i < min(8, n_features):
                features[i] = cooccurrence_scores[cat_name]
        
        # Additional statistical features
        if n_features > 8:
            remaining_idx = 8
            if remaining_idx < n_features:
                features[remaining_idx] = len(word_dims)  # Total dimensions
            if remaining_idx + 1 < n_features:
                avg_selectivity = np.mean([self.dimension_selectivity.get(dim_pair, 0.5) 
                                         for dim_pair in word_dims])
                features[remaining_idx + 1] = avg_selectivity
            if remaining_idx + 2 < n_features:
                features[remaining_idx + 2] = np.mean(list(cooccurrence_scores.values()) or [0])
        
        return features
    
    def _get_wordnet_features(self, word: str, n_features: int = 10) -> np.ndarray:
        """Extract WordNet-based semantic features."""
        features = np.zeros(n_features)
        
        try:
            # Clean word for WordNet lookup
            clean_word = word.strip().lower()
            if clean_word.startswith(' '):
                clean_word = clean_word[1:]
            
            synsets = wordnet.synsets(clean_word)
            if synsets:
                # Feature 0: Number of synsets (polysemy)
                features[0] = min(len(synsets), 1.0)  # Normalize
                
                # Feature 1-5: Presence in different POS categories
                pos_categories = ['noun', 'verb', 'adj', 'adv', 'other']
                for synset in synsets[:5]:  # Limit to first 5
                    pos = synset.pos()
                    if pos == 'n' and len(pos_categories) > 0:
                        features[1] = 1.0
                    elif pos == 'v' and len(pos_categories) > 1:
                        features[2] = 1.0
                    elif pos in ['a', 's'] and len(pos_categories) > 2:
                        features[3] = 1.0
                    elif pos == 'r' and len(pos_categories) > 3:
                        features[4] = 1.0
                
                # Feature 6-9: Semantic depth and relationships
                if len(synsets) > 0:
                    first_synset = synsets[0]
                    try:
                        # Hypernym depth
                        hypernym_paths = first_synset.hypernym_paths()
                        if hypernym_paths:
                            max_depth = max(len(path) for path in hypernym_paths)
                            features[5] = min(max_depth / 10.0, 1.0)  # Normalize
                        
                        # Number of hyponyms
                        hyponyms = first_synset.hyponyms()
                        features[6] = min(len(hyponyms) / 20.0, 1.0)  # Normalize
                        
                        # Number of hypernyms
                        hypernyms = first_synset.hypernyms()
                        features[7] = min(len(hypernyms) / 10.0, 1.0)  # Normalize
                        
                        # Number of similar words
                        similar_tos = first_synset.similar_tos()
                        features[8] = min(len(similar_tos) / 15.0, 1.0)  # Normalize
                        
                    except:
                        pass  # Skip if WordNet operations fail
                
        except:
            pass  # Return zero features if WordNet lookup fails
        
        return features
    
    def find_semantic_similar_words(self, target_word: str, 
                                   category: Optional[str] = None,
                                   top_k: int = 10,
                                   use_embeddings: bool = True) -> List[Tuple[str, float]]:
        """
        Find semantically similar words based on activation patterns.
        Improved with better embedding-based similarity.
        """
        if use_embeddings:
            embeddings = self.build_semantic_word_embeddings()
            
            if target_word not in embeddings:
                print(f"Word '{target_word}' not found in embeddings")
                return []
            
            target_embedding = embeddings[target_word]
            
            # Determine search space
            if category and category in self.semantic_categories:
                search_tokens = self.semantic_categories[category] & set(embeddings.keys())
            else:
                # Default to content words for semantic similarity
                content_words = (self.semantic_categories['nouns'] | 
                               self.semantic_categories['proper_nouns'] |
                               self.semantic_categories['verbs'] |
                               self.semantic_categories['adjectives'])
                search_tokens = content_words & set(embeddings.keys())
            
            # Calculate similarities using embeddings
            similarities = []
            for token in search_tokens:
                if token != target_word:
                    token_embedding = embeddings[token]
                    similarity = cosine_similarity([target_embedding], [token_embedding])[0][0]
                    if similarity > 0.1:  # Threshold for meaningful similarity
                        similarities.append((token, similarity))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        
        else:
            # Fallback to original activation-based similarity
            if target_word not in self.token_activation_profiles:
                print(f"Word '{target_word}' not found in activation profiles")
                return []
            
            target_profile = self.token_activation_profiles[target_word]
            target_vector = self._profile_to_vector(target_profile)
            
            # Determine search space
            if category and category in self.semantic_categories:
                search_tokens = self.semantic_categories[category]
            else:
                search_tokens = (self.semantic_categories['nouns'] | 
                               self.semantic_categories['proper_nouns'] |
                               self.semantic_categories['verbs'] |
                               self.semantic_categories['adjectives'])
            
            # Calculate similarities
            similarities = []
            for token in search_tokens:
                if token != target_word and token in self.token_activation_profiles:
                    token_profile = self.token_activation_profiles[token]
                    token_vector = self._profile_to_vector(token_profile)
                    
                    similarity = cosine_similarity([target_vector], [token_vector])[0][0]
                    if similarity > 0.1:
                        similarities.append((token, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    def _profile_to_vector(self, profile: Dict[int, List[Tuple[int, float]]]) -> np.ndarray:
        """Convert activation profile to vector for similarity calculation."""
        vector = np.zeros(12 * 1000)  # Larger vector for better representation
        
        for layer, activations in profile.items():
            if layer < 12:
                layer_start = layer * 1000
                for dim, strength in activations:
                    if layer_start + dim < len(vector):
                        vector[layer_start + dim] = strength
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def find_conceptual_analogies(self, word_a: str, word_b: str, word_c: str, 
                                 top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find analogical relationships: A is to B as C is to ?
        Example: king is to queen as man is to woman
        """
        embeddings = self.build_semantic_word_embeddings()
        
        if not all(word in embeddings for word in [word_a, word_b, word_c]):
            missing = [w for w in [word_a, word_b, word_c] if w not in embeddings]
            print(f"Missing embeddings for: {missing}")
            return []
        
        # Calculate analogy vector: B - A + C = ?
        vec_a = embeddings[word_a]
        vec_b = embeddings[word_b]
        vec_c = embeddings[word_c]
        
        target_vector = vec_b - vec_a + vec_c
        
        # Find closest words to target vector (excluding input words)
        similarities = []
        excluded_words = {word_a, word_b, word_c}
        
        for word, embedding in embeddings.items():
            if word not in excluded_words:
                similarity = cosine_similarity([target_vector], [embedding])[0][0]
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analyze_semantic_fields(self, seed_words: List[str], 
                               expansion_threshold: float = 0.3) -> Dict[str, List[str]]:
        """
        Build semantic fields around seed words by finding related concepts.
        Lowered threshold for more inclusive results.
        """
        embeddings = self.build_semantic_word_embeddings()
        semantic_fields = {}
        
        for seed_word in seed_words:
            if seed_word not in embeddings:
                print(f"Seed word '{seed_word}' not found in embeddings")
                continue
            
            # Find semantically related words
            related_words = []
            seed_embedding = embeddings[seed_word]
            
            for word, embedding in embeddings.items():
                if word != seed_word:
                    similarity = cosine_similarity([seed_embedding], [embedding])[0][0]
                    if similarity >= expansion_threshold:
                        related_words.append((word, similarity))
            
            # Sort by similarity and extract words
            related_words.sort(key=lambda x: x[1], reverse=True)
            semantic_fields[seed_word] = [word for word, _ in related_words[:20]]
        
        return semantic_fields
    
    def detect_semantic_shifts(self, word: str, layer_range: Tuple[int, int] = (0, 11)) -> Dict[str, Any]:
        """
        Analyze how a word's semantic representation changes across layers.
        """
        if word not in self.token_activation_profiles:
            return {'error': f'Word "{word}" not found'}
        
        profile = self.token_activation_profiles[word]
        layer_representations = {}
        
        # Get representation at each layer
        for layer in range(layer_range[0], layer_range[1] + 1):
            if layer in profile:
                layer_activations = profile[layer]
                if layer_activations:
                    # Use top activations as representation, weighted by selectivity
                    weighted_activations = []
                    for dim, strength in layer_activations:
                        selectivity = self.dimension_selectivity.get((layer, dim), 0.5)
                        weighted_activations.append(strength * selectivity)
                    
                    if weighted_activations:
                        # Take top 20 weighted activations
                        top_weighted = sorted(weighted_activations, reverse=True)[:20]
                        # Pad if necessary
                        while len(top_weighted) < 20:
                            top_weighted.append(0.0)
                        layer_representations[layer] = np.array(top_weighted)
        
        # Analyze changes between layers
        semantic_shifts = {}
        prev_layer = None
        prev_vector = None
        
        for layer in sorted(layer_representations.keys()):
            current_vector = layer_representations[layer]
            
            if prev_vector is not None:
                similarity = cosine_similarity([prev_vector], [current_vector])[0][0]
                semantic_shifts[f'layers_{prev_layer}_to_{layer}'] = {
                    'similarity': similarity,
                    'change_magnitude': 1 - similarity
                }
            
            prev_layer = layer
            prev_vector = current_vector
        
        # Find layers with biggest semantic shifts
        if semantic_shifts:
            biggest_shifts = sorted(
                semantic_shifts.items(), 
                key=lambda x: x[1]['change_magnitude'], 
                reverse=True
            )[:3]
        else:
            biggest_shifts = []
        
        return {
            'word': word,
            'layer_representations': {k: v.tolist() for k, v in layer_representations.items()},
            'semantic_shifts': semantic_shifts,
            'biggest_shifts': biggest_shifts,
            'layers_analyzed': list(layer_representations.keys())
        }
    
    def find_robust_noun_patterns(self, target_nouns: List[str], 
                                 context_window: int = 5,
                                 min_activation_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Find sentences with robust activation patterns for meaningful nouns.
        Enhanced with context awareness and semantic clustering.
        """
        print(f"Finding robust patterns for nouns: {target_nouns}")
        
        # Filter to actual nouns in our data with better matching
        valid_nouns = []
        for noun in target_nouns:
            # Direct match
            if noun in self.semantic_categories['nouns'] or noun in self.semantic_categories['proper_nouns']:
                valid_nouns.append(noun)
            else:
                # Try with space prefix (common in GPT-2 tokenization)
                space_noun = ' ' + noun
                if space_noun in self.semantic_categories['nouns'] or space_noun in self.semantic_categories['proper_nouns']:
                    valid_nouns.append(space_noun)
                    print(f"Found '{noun}' as '{space_noun}'")
                else:
                    # Try to find similar nouns using embeddings
                    similar = self.find_semantic_similar_words(noun, category='nouns', top_k=3)
                    if similar:
                        print(f"'{noun}' not found as noun, using similar: {[s[0] for s in similar[:1]]}")
                        valid_nouns.extend([s[0] for s in similar[:1]])
        
        if not valid_nouns:
            return {'error': 'No valid nouns found in dataset'}
        
        print(f"Valid nouns found: {valid_nouns}")
        
        # Get relevant FFN dimensions with higher selectivity for nouns
        relevant_dims = set()
        noun_dim_mapping = {}
        
        for noun in valid_nouns:
            dims = []
            if noun in self.token_to_ffn_dims:
                # Focus on top-ranked dimensions and high-selectivity dimensions
                for layer, dim, rank in self.token_to_ffn_dims[noun]:
                    selectivity = self.dimension_selectivity.get((layer, dim), 0.5)
                    if rank < 10 and selectivity > 0.3:  # More selective criteria
                        dims.append((layer, dim))
                        relevant_dims.add((layer, dim))
            noun_dim_mapping[noun] = dims
        
        print(f"Found {len(relevant_dims)} highly selective FFN dimensions for nouns")
        
        # Enhanced sentence scoring with context awareness
        sentence_results = []
        
        for idx, row in self.model_df.iterrows():
            sentence = row['sent']
            
            # Better noun detection in sentences
            sentence_lower = sentence.lower()
            contains_target = any(
                noun.strip().lower() in sentence_lower 
                for noun in valid_nouns
            )
            
            if not contains_target:
                continue
            
            # Calculate enhanced activation score
            score_info = self._calculate_robust_noun_score(
                row, relevant_dims, valid_nouns, noun_dim_mapping, sentence
            )
            
            if score_info['total_score'] >= min_activation_threshold:
                # Add context analysis
                context_info = self._analyze_noun_context(sentence, valid_nouns)
                
                sentence_results.append({
                    'sentence': sentence,
                    'score': score_info['total_score'],
                    'score_breakdown': score_info,
                    'context_analysis': context_info,
                    'row_idx': idx
                })
        
        # Sort by score and analyze patterns
        sentence_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Cluster similar activation patterns
        if len(sentence_results) > 5:
            pattern_clusters = self._cluster_activation_patterns(sentence_results[:20])
        else:
            pattern_clusters = {}
        
        return {
            'target_nouns': valid_nouns,
            'sentences_found': len(sentence_results),
            'top_sentences': sentence_results[:10],
            'pattern_clusters': pattern_clusters,
            'activation_summary': self._summarize_noun_activations(sentence_results)
        }
    
    def _calculate_robust_noun_score(self, row: pd.Series, relevant_dims: Set[Tuple[int, int]], 
                                   target_nouns: List[str], noun_dim_mapping: Dict[str, List],
                                   sentence: str) -> Dict[str, Any]:
        """Calculate enhanced scoring for noun patterns."""
        score_info = {
            'activation_strength': 0.0,
            'noun_specificity': 0.0,
            'context_coherence': 0.0,
            'layer_consistency': 0.0,
            'selectivity_bonus': 0.0,  # NEW: bonus for selective dimensions
            'total_score': 0.0,
            'active_layers': [],
            'noun_matches': []
        }
        
        layers_with_activations = set()
        layer_scores = defaultdict(float)
        
        # 1. Activation strength (weighted by selectivity)
        for layer_idx in range(len(row['top_coef_idx'])):
            top_indices = row['top_coef_idx'][layer_idx]
            top_values = row['top_coef_vals'][layer_idx]
            
            for dim_idx, coef_val in zip(top_indices, top_values):
                if (layer_idx, dim_idx) in relevant_dims:
                    activation_strength = abs(coef_val)
                    
                    # Weight by dimension selectivity
                    selectivity = self.dimension_selectivity.get((layer_idx, dim_idx), 0.5)
                    weighted_strength = activation_strength * (1 + selectivity)
                    
                    score_info['activation_strength'] += weighted_strength
                    score_info['selectivity_bonus'] += selectivity * activation_strength
                    layer_scores[layer_idx] += weighted_strength
                    layers_with_activations.add(layer_idx)
        
        # 2. Noun specificity bonus (improved detection)
        sentence_lower = sentence.lower()
        for noun in target_nouns:
            noun_clean = noun.strip().lower()
            if noun_clean in sentence_lower:
                score_info['noun_matches'].append(noun)
                
                # Check if this noun's specific dimensions are activated
                if noun in noun_dim_mapping:
                    noun_dims = set(noun_dim_mapping[noun])
                    active_noun_dims = 0
                    
                    for layer_idx in range(len(row['top_coef_idx'])):
                        top_indices = row['top_coef_idx'][layer_idx]
                        for dim_idx in top_indices[:15]:  # Expanded to top 15
                            if (layer_idx, dim_idx) in noun_dims:
                                # Weight by position in top activations
                                position_weight = 1.0 / (list(top_indices).index(dim_idx) + 1)
                                active_noun_dims += position_weight
                    
                    score_info['noun_specificity'] += active_noun_dims * 0.7
        
        # 3. Context coherence (improved calculation)
        score_info['context_coherence'] = self._calculate_context_coherence(sentence, target_nouns)
        
        # 4. Layer consistency (reward distributed activations)
        unique_layers = len(layers_with_activations)
        if unique_layers > 1:  # Bonus for multi-layer activation
            score_info['layer_consistency'] = unique_layers * 0.4
        
        score_info['active_layers'] = list(layers_with_activations)
        
        # 5. Total score calculation (rebalanced weights)
        score_info['total_score'] = (
            score_info['activation_strength'] * 1.0 +
            score_info['noun_specificity'] * 2.0 +  # Increased weight
            score_info['context_coherence'] * 1.2 +
            score_info['layer_consistency'] * 0.5 +
            score_info['selectivity_bonus'] * 0.8
        )
        
        return score_info
    
    def _calculate_noun_specificity(self, layer: int, dim: int, target_nouns: List[str]) -> float:
        """Calculate how specific a dimension is to nouns vs other word types."""
        if (layer, dim) not in self.ffn_projections:
            return 0.0
        
        top_tokens = self.ffn_projections[(layer, dim)][:25]  # Expanded to top 25
        
        noun_count = 0
        target_noun_count = 0
        total_content_words = 0
        
        for token in top_tokens:
            if token in self.semantic_categories['nouns'] or token in self.semantic_categories['proper_nouns']:
                noun_count += 1
                total_content_words += 1
                
                # Extra bonus if it's one of our target nouns
                if any(target.strip().lower() == token.strip().lower() for target in target_nouns):
                    target_noun_count += 1
                    
            elif (token in self.semantic_categories['verbs'] or 
                  token in self.semantic_categories['adjectives'] or
                  token in self.semantic_categories['adverbs']):
                total_content_words += 1
        
        if total_content_words == 0:
            return 0.0
        
        base_specificity = noun_count / total_content_words
        target_bonus = target_noun_count * 0.5  # Bonus for target noun presence
        
        return base_specificity + target_bonus
    
    def _calculate_context_coherence(self, sentence: str, target_nouns: List[str]) -> float:
        """Calculate semantic coherence of the sentence context."""
        if not self.nlp:
            return 0.5  # Default moderate score
        
        try:
            doc = self.nlp(sentence)
        except:
            return 0.5
        
        coherence_score = 0.0
        
        # 1. Named entity recognition bonus
        entities = [ent.text.lower() for ent in doc.ents]
        entity_overlap = sum(1 for noun in target_nouns 
                           if noun.strip().lower() in ' '.join(entities))
        coherence_score += entity_overlap * 0.4
        
        # 2. Dependency relationships
        target_tokens = []
        for token in doc:
            if any(noun.strip().lower() in token.text.lower() for noun in target_nouns):
                target_tokens.append(token)
        
        for noun_token in target_tokens:
            # Check for meaningful dependencies
            if noun_token.dep_ in ['nsubj', 'dobj', 'pobj', 'compound', 'ROOT']:
                coherence_score += 0.3
            
            # Check for descriptive relationships
            if any(child.pos_ == 'ADJ' for child in noun_token.children):
                coherence_score += 0.2
            
            # Check for verb relationships
            if any(child.pos_ == 'VERB' or parent.pos_ == 'VERB' 
                   for child in noun_token.children 
                   for parent in [noun_token.head]):
                coherence_score += 0.2
        
        # 3. Sentence complexity bonus (longer, more complex sentences often have richer context)
        sentence_length = len(doc)
        if sentence_length > 15:  # Longer sentences
            coherence_score += 0.1
        
        return min(coherence_score, 2.0)  # Allow higher scores for very coherent contexts
    
    def _analyze_noun_context(self, sentence: str, target_nouns: List[str]) -> Dict[str, Any]:
        """Analyze the context around target nouns with improved detection."""
        context_info = {
            'sentence_length': len(sentence.split()),
            'noun_positions': [],
            'surrounding_words': {},
            'semantic_context': [],
            'grammatical_roles': {}
        }
        
        words = sentence.split()
        sentence_lower = sentence.lower()
        
        # Improved noun detection
        for noun in target_nouns:
            noun_clean = noun.strip().lower()
            if noun_clean in sentence_lower:
                # Find all positions of this noun
                start_pos = 0
                while True:
                    pos = sentence_lower.find(noun_clean, start_pos)
                    if pos == -1:
                        break
                    
                    # Find word index
                    word_idx = len(sentence_lower[:pos].split()) - 1
                    if word_idx >= 0 and word_idx < len(words):
                        context_info['noun_positions'].append({
                            'noun': noun,
                            'position': word_idx,
                            'word': words[word_idx] if word_idx < len(words) else noun_clean
                        })
                        
                        # Capture surrounding context (expanded window)
                        start_ctx = max(0, word_idx - 3)
                        end_ctx = min(len(words), word_idx + 4)
                        context_info['surrounding_words'][noun] = ' '.join(words[start_ctx:end_ctx])
                    
                    start_pos = pos + 1
        
        # Enhanced semantic analysis with spaCy
        if self.nlp:
            try:
                doc = self.nlp(sentence)
                
                # Named entities
                context_info['semantic_context'] = [
                    {'text': ent.text, 'label': ent.label_, 'start': ent.start, 'end': ent.end} 
                    for ent in doc.ents
                ]
                
                # Grammatical roles for target nouns
                for token in doc:
                    token_lower = token.text.lower()
                    for noun in target_nouns:
                        if noun.strip().lower() in token_lower:
                            context_info['grammatical_roles'][noun] = {
                                'pos': token.pos_,
                                'dep': token.dep_,
                                'head': token.head.text,
                                'children': [child.text for child in token.children]
                            }
                            break
            except:
                pass  # Skip spaCy analysis if it fails
        
        return context_info
    
    def _cluster_activation_patterns(self, sentence_results: List[Dict]) -> Dict[int, List[Dict]]:
        """Cluster sentences by similar activation patterns with improved features."""
        if len(sentence_results) < 3:
            return {}
        
        # Extract enhanced activation features for clustering
        features = []
        for result in sentence_results:
            score_breakdown = result['score_breakdown']
            context_analysis = result['context_analysis']
            
            feature_vector = [
                score_breakdown['activation_strength'],
                score_breakdown['noun_specificity'],
                score_breakdown['context_coherence'],
                score_breakdown['layer_consistency'],
                score_breakdown.get('selectivity_bonus', 0),
                len(score_breakdown['active_layers']),
                len(score_breakdown['noun_matches']),
                context_analysis['sentence_length'],
                len(context_analysis['noun_positions']),
                len(context_analysis['semantic_context'])
            ]
            features.append(feature_vector)
        
        # Normalize features
        features = np.array(features)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Perform clustering
        n_clusters = min(5, len(features) // 2)
        if n_clusters < 2:
            return {}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Organize results by cluster
        clusters = defaultdict(list)
        for result, cluster_id in zip(sentence_results, cluster_labels):
            clusters[cluster_id].append(result)
        
        return dict(clusters)
    
    def _summarize_noun_activations(self, sentence_results: List[Dict]) -> Dict[str, Any]:
        """Summarize activation patterns for noun analysis with enhanced metrics."""
        if not sentence_results:
            return {}
        
        summary = {
            'total_sentences': len(sentence_results),
            'avg_score': np.mean([r['score'] for r in sentence_results]),
            'score_std': np.std([r['score'] for r in sentence_results]),
            'score_distribution': {
                'min': min(r['score'] for r in sentence_results),
                'max': max(r['score'] for r in sentence_results),
                'median': np.median([r['score'] for r in sentence_results])
            },
            'common_layers': [],
            'frequent_nouns': [],
            'context_patterns': [],
            'selectivity_analysis': {}
        }
        
        # Analyze common activation layers
        all_layers = []
        selectivity_scores = []
        
        for result in sentence_results:
            all_layers.extend(result['score_breakdown']['active_layers'])
            selectivity_scores.append(result['score_breakdown'].get('selectivity_bonus', 0))
        
        layer_counts = Counter(all_layers)
        summary['common_layers'] = layer_counts.most_common(8)
        
        # Analyze frequent nouns
        all_nouns = []
        for result in sentence_results:
            all_nouns.extend(result['score_breakdown']['noun_matches'])
        
        noun_counts = Counter(all_nouns)
        summary['frequent_nouns'] = noun_counts.most_common(10)
        
        # Analyze context patterns
        sentence_lengths = [r['context_analysis']['sentence_length'] for r in sentence_results]
        semantic_contexts = [len(r['context_analysis']['semantic_context']) for r in sentence_results]
        
        summary['context_patterns'] = {
            'avg_sentence_length': np.mean(sentence_lengths),
            'avg_semantic_entities': np.mean(semantic_contexts),
            'length_distribution': {
                'short': len([l for l in sentence_lengths if l < 15]),
                'medium': len([l for l in sentence_lengths if 15 <= l < 30]),
                'long': len([l for l in sentence_lengths if l >= 30])
            }
        }
        
        # Selectivity analysis
        summary['selectivity_analysis'] = {
            'avg_selectivity_bonus': np.mean(selectivity_scores),
            'high_selectivity_count': len([s for s in selectivity_scores if s > 1.0])
        }
        
        return summary


def demonstrate_enhanced_noun_mapping():
    """Demonstration of enhanced noun mapping capabilities with improvements."""
    
    # Initialize the enhanced mapper
    mapper = EnhancedSemanticMapper(
        ffn_projections_path="ffn_projections.pkl",
        model_df_path="model_df_10k.pkl"
    )
    
    print("="*60)
    print("ENHANCED SEMANTIC NOUN MAPPING DEMONSTRATION")
    print("="*60)
    
    # 1. Show semantic categorization results
    print("\n1. SEMANTIC CATEGORIZATION RESULTS")
    print("-" * 40)
    for category, tokens in mapper.semantic_categories.items():
        sample_tokens = list(tokens)[:10]
        print(f"{category.upper()}: {len(tokens)} tokens")
        print(f"  Sample: {sample_tokens}")
    
    # 2. Improved semantic similarities
    print("\n2. IMPROVED SEMANTIC SIMILARITY ANALYSIS")
    print("-" * 40)
    
    test_words = ['king', 'house', 'water', 'computer', 'time']
    for word in test_words:
        print(f"\nSemantic similarities for '{word}':")
        
        # Try both embedding-based and activation-based
        similar_words_emb = mapper.find_semantic_similar_words(
            word, category='nouns', top_k=5, use_embeddings=True
        )
        similar_words_act = mapper.find_semantic_similar_words(
            word, category='nouns', top_k=5, use_embeddings=False
        )
        
        if similar_words_emb:
            print(f"  Embedding-based:")
            for similar_word, similarity in similar_words_emb:
                print(f"    {similar_word}: {similarity:.3f}")
        
        if similar_words_act and similar_words_act != similar_words_emb:
            print(f"  Activation-based:")
            for similar_word, similarity in similar_words_act[:3]:
                print(f"    {similar_word}: {similarity:.3f}")
    
    # 3. Enhanced robust noun pattern analysis
    print("\n3. ENHANCED ROBUST NOUN PATTERN ANALYSIS")
    print("-" * 40)
    
    target_nouns = ['king', 'house', 'water', 'time', 'world']
    results = mapper.find_robust_noun_patterns(target_nouns, min_activation_threshold=0.8)
    
    if 'error' not in results:
        print(f"Analyzed nouns: {results['target_nouns']}")
        print(f"Sentences found: {results['sentences_found']}")
        
        print(f"\nTop activation patterns:")
        for i, result in enumerate(results['top_sentences'][:3], 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Sentence: {result['sentence'][:120]}...")
            print(f"   Nouns found: {result['score_breakdown']['noun_matches']}")
            print(f"   Active layers: {result['score_breakdown']['active_layers']}")
            print(f"   Selectivity bonus: {result['score_breakdown'].get('selectivity_bonus', 0):.3f}")
        
        # Show enhanced activation summary
        if results['activation_summary']:
            summary = results['activation_summary']
            print(f"\nEnhanced Activation Summary:")
            print(f"  Score range: {summary['score_distribution']['min']:.2f} - {summary['score_distribution']['max']:.2f}")
            print(f"  Average score: {summary['avg_score']:.3f} (±{summary['score_std']:.3f})")
            print(f"  Most active layers: {summary['common_layers'][:3]}")
            print(f"  Most frequent nouns: {summary['frequent_nouns'][:5]}")
            print(f"  Average selectivity bonus: {summary['selectivity_analysis']['avg_selectivity_bonus']:.3f}")
    else:
        print(f"Error: {results['error']}")


def advanced_noun_analysis_demo():
    """Demonstrate advanced semantic analysis capabilities with fixes."""
    
    mapper = EnhancedSemanticMapper("ffn_projections.pkl", "model_df_10k.pkl")
    
    print("="*60)
    print("ADVANCED SEMANTIC ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # 1. Semantic word embeddings
    print("\n1. BUILDING SEMANTIC WORD EMBEDDINGS")
    print("-" * 40)
    embeddings = mapper.build_semantic_word_embeddings()
    print(f"Built embeddings for {len(embeddings)} words")
    
    # Show embedding dimensionality and sample
    if embeddings:
        sample_word = list(embeddings.keys())[0]
        print(f"Embedding dimensionality: {len(embeddings[sample_word])}")
        print(f"Sample embedding stats for '{sample_word}':")
        emb = embeddings[sample_word]
        print(f"  Min: {np.min(emb):.4f}, Max: {np.max(emb):.4f}, Mean: {np.mean(emb):.4f}")
    
    # 2. Conceptual analogies
    print("\n2. CONCEPTUAL ANALOGIES")
    print("-" * 40)
    
    analogy_tests = [
        ("king", "queen", "man"),
        ("house", "home", "car"),
        ("water", "liquid", "air"),
        ("time", "temporal", "space")
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        analogies = mapper.find_conceptual_analogies(word_a, word_b, word_c, top_k=3)
        if analogies:
            print(f"\n{word_a} : {word_b} :: {word_c} : ?")
            for word, score in analogies:
                print(f"  {word} ({score:.3f})")
        else:
            print(f"\n{word_a} : {word_b} :: {word_c} : ? [No analogies found]")
    
    # 3. Semantic fields with improved threshold
    print("\n3. SEMANTIC FIELD ANALYSIS")
    print("-" * 40)
    
    seed_words = ["king", "house", "water", "time"]
    semantic_fields = mapper.analyze_semantic_fields(seed_words, expansion_threshold=0.25)
    
    for seed, related_words in semantic_fields.items():
        if related_words:
            print(f"\nSemantic field for '{seed}':")
            print(f"  Related: {related_words[:10]}")
        else:
            print(f"\nSemantic field for '{seed}': [No related words found]")
    
    # 4. Semantic shifts across layers
    print("\n4. SEMANTIC SHIFTS ACROSS LAYERS")
    print("-" * 40)
    
    test_words = ["king", "water", "time"]
    for word in test_words:
        shifts = mapper.detect_semantic_shifts(word)
        if 'error' not in shifts:
            print(f"\nSemantic shifts for '{word}':")
            print(f"  Layers analyzed: {shifts['layers_analyzed']}")
            if shifts['biggest_shifts']:
                for i, (shift_name, shift_data) in enumerate(shifts['biggest_shifts']):
                    print(f"  Shift {i+1}: {shift_name} (magnitude: {shift_data['change_magnitude']:.3f})")
        else:
            print(f"\nSemantic shifts for '{word}': {shifts['error']}")


if __name__ == "__main__":
    print("Starting Enhanced Semantic Mapper Demonstration...")
    try:
        demonstrate_enhanced_noun_mapping()
        print("\n" + "="*60)
        advanced_noun_analysis_demo()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

""" /home/ubuntu/miniconda3/envs/ffn_env/bin/python /home/ubuntu/krishiv-llm/ffn-values/reverse_engineer.py
[nltk_data] Downloading package wordnet to /home/ubuntu/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /home/ubuntu/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
Starting Enhanced Semantic Mapper Demonstration...
Loading FFN projections...
Loading model DataFrame...
Building enhanced semantic mappings...
Categorizing tokens by semantic type...
nouns: 17955 tokens
proper_nouns: 10215 tokens
verbs: 8137 tokens
adjectives: 3450 tokens
adverbs: 1607 tokens
function_words: 2162 tokens
numbers: 751 tokens
punctuation: 195 tokens
Building semantic clusters...
Created 20 semantic clusters
============================================================
ENHANCED SEMANTIC NOUN MAPPING DEMONSTRATION
============================================================

1. SEMANTIC CATEGORIZATION RESULTS
----------------------------------------
NOUNS: 17955 tokens
  Sample: [' grandmother', ' dexterity', 'Mission', ' entities', ' Mercury', ' boarding', ' entrants', ' coefficient', ' equival', ' Condition']
PROPER_NOUNS: 10215 tokens
  Sample: [' insepar', ' lightsaber', 'pt', 'nel', 'ordan', ' Halifax', 'uania', ' Crescent', 'otech', ' alien']
VERBS: 8137 tokens
  Sample: [' reconstruct', ' thinks', 'Roaming', ' automated', ' thanked', ' Hold', ' remains', 'sound', ' deserved', '大']
ADJECTIVES: 3450 tokens
  Sample: ['aye', ' Cum', ' Pebble', 'pathy', 'FREE', 'uge', 'third', ' Dominican', ' Shiite', 'ublic']
ADVERBS: 1607 tokens
  Sample: [' unex', ' extraordinarily', ' indefinitely', ' Near', ' externalToEVAOnly', ' shockingly', ' nationally', ' lately', ' instinctively', ' understandably']
FUNCTION_WORDS: 2162 tokens
  Sample: [' &&', ' amongst', ' ub', ' Ong', 'ヴァ', '(-', ' THEIR', ' seventy', 'If', ' dise']
NUMBERS: 751 tokens
  Sample: [' 0000', '301', '024', '802', ' 00', ' 7000', ' 530', ' 1899', ' 301', '41']
PUNCTUATION: 195 tokens
  Sample: [' @', '€', ' ¯', '●', '(', '+', '×', ' ✔', '。', ' ?']

2. IMPROVED SEMANTIC SIMILARITY ANALYSIS
----------------------------------------

Semantic similarities for 'king':
Building semantic word embeddings...
  Embedding-based:
    line: 1.000
     Thumbnails: 1.000
    UTC: 1.000
    ews: 1.000
    ales: 1.000
  Activation-based:
     inequality: 0.577
     negativity: 0.369
     volatility: 0.333

Semantic similarities for 'house':
  Embedding-based:
    oxide: 0.997
    Appearances: 0.997
    grass: 0.997
    letter: 0.996
    cap: 0.996
  Activation-based:
     permission: 0.455
     thence: 0.400
     Lawn: 0.356

Semantic similarities for 'water':
  Embedding-based:
     Flags: 0.999
     Trials: 0.998
     coefficient: 0.998
    flags: 0.998
     occupant: 0.998
  Activation-based:
    sale: 0.360
    ifles: 0.285
    Links: 0.280

Semantic similarities for 'computer':
  Embedding-based:
     Bell: 0.999
     whip: 0.998
     Citizens: 0.998
     Centers: 0.998
     doll: 0.997
  Activation-based:
    Mu: 0.994
    Computer: 0.994
     Yi: 0.994

Semantic similarities for 'time':
  Embedding-based:
    holders: 0.999
    points: 0.999
     behalf: 0.999
     heights: 0.999
    life: 0.999
  Activation-based:
    Time: 0.452
     time: 0.448
     retailers: 0.427

3. ENHANCED ROBUST NOUN PATTERN ANALYSIS
----------------------------------------
Finding robust patterns for nouns: ['king', 'house', 'water', 'time', 'world']
Valid nouns found: ['king', 'house', 'water', 'time', 'world']
Found 0 highly selective FFN dimensions for nouns
Analyzed nouns: ['king', 'house', 'water', 'time', 'world']
Sentences found: 156

Top activation patterns:

1. Score: 1.560
   Sentence: Richard H. Wilkinson , however , argues that some texts from the late New Kingdom suggest that , as beliefs about the go...
   Nouns found: ['king', 'world']
   Active layers: []
   Selectivity bonus: 0.000

2. Score: 1.560
   Sentence: Following World War I , Ali began managing his own affairs and toured the world , learning more...
   Nouns found: ['world']
   Active layers: []
   Selectivity bonus: 0.000

3. Score: 1.560
   Sentence: Erzherzog Ferdinand Max was also fitted with two above water 45 @-@ centimeter ( 17 @.@ 7...
   Nouns found: ['water', 'time']
   Active layers: []
   Selectivity bonus: 0.000

Enhanced Activation Summary:
  Score range: 0.84 - 1.56
  Average score: 0.963 (±0.154)
  Most active layers: []
  Most frequent nouns: [('world', 48), ('king', 45), ('time', 44), ('house', 20), ('water', 15)]
  Average selectivity bonus: 0.000

============================================================
Loading FFN projections...
Loading model DataFrame...
Building enhanced semantic mappings...
Categorizing tokens by semantic type...
nouns: 17955 tokens
proper_nouns: 10215 tokens
verbs: 8137 tokens
adjectives: 3450 tokens
adverbs: 1607 tokens
function_words: 2162 tokens
numbers: 751 tokens
punctuation: 195 tokens
Building semantic clusters...
Created 20 semantic clusters
============================================================
ADVANCED SEMANTIC ANALYSIS DEMONSTRATION
============================================================

1. BUILDING SEMANTIC WORD EMBEDDINGS
----------------------------------------
Building semantic word embeddings...
Built embeddings for 39757 words
Embedding dimensionality: 87
Sample embedding stats for ' reconstruct':
  Min: 0.0000, Max: 0.8077, Mean: 0.0296

2. CONCEPTUAL ANALOGIES
----------------------------------------
Missing embeddings for: ['queen']

king : queen :: man : ? [No analogies found]

house : home :: car : ?
   Firearms (0.995)
  trial (0.995)
   Markets (0.995)

water : liquid :: air : ?
   optic (0.957)
   chemical (0.955)
   SPECIAL (0.953)
Missing embeddings for: ['temporal']

time : temporal :: space : ? [No analogies found]

3. SEMANTIC FIELD ANALYSIS
----------------------------------------

Semantic field for 'king':
  Related: ['line', ' Thumbnails', 'UTC', 'ews', 'ales', 'son', 'folk', 'dash', 'ests', ' Journals']

Semantic field for 'house':
  Related: ['bridge', 'ray', ' POLITICO', 'Origin', 'SHARE', 'leg', 'rod', 'Runner', 'RAY', 'NES']

Semantic field for 'water':
  Related: [' Flags', ' Trials', ' coefficient', 'flags', ' occupant', ' autos', ' Views', 'skill', 'fights', 'Performance']

Semantic field for 'time':
  Related: ['holders', 'points', ' behalf', ' heights', 'life', 'leaders', ' night', ' shoulders', 'interest', ' events']

4. SEMANTIC SHIFTS ACROSS LAYERS
----------------------------------------

Semantic shifts for 'king':
  Layers analyzed: [0, 2, 3, 4, 6, 7, 8, 9, 10, 11]
  Shift 1: layers_6_to_7 (magnitude: 0.293)
  Shift 2: layers_3_to_4 (magnitude: 0.200)
  Shift 3: layers_4_to_6 (magnitude: 0.200)

Semantic shifts for 'water':
  Layers analyzed: [0, 1, 3, 6, 9]
  Shift 1: layers_6_to_9 (magnitude: 0.211)
  Shift 2: layers_0_to_1 (magnitude: 0.019)
  Shift 3: layers_1_to_3 (magnitude: 0.019)

Semantic shifts for 'time':
  Layers analyzed: [1, 3, 4, 7, 8, 9, 10, 11]
  Shift 1: layers_10_to_11 (magnitude: 0.219)
  Shift 2: layers_1_to_3 (magnitude: 0.126)
  Shift 3: layers_9_to_10 (magnitude: 0.120)"""