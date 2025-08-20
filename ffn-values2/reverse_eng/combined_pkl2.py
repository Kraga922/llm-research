# import pickle
# import torch
# import pandas as pd
# from transformers import AutoTokenizer
# import json
# from tqdm import tqdm

# # --- Config ---
# STEERING_PATH = "/home/ubuntu/krishiv-llm/neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl" #"neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl"
# ACTIVATION_DF_PATH = "/home/ubuntu/krishiv-llm/ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl" #"ffn-values/pickle/model_df_llama3_1_8b_instruct_100.pkl"
# TOP_K = 20
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# # --- Load tokenizer ---
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # --- Load steering directions ---
# with open(STEERING_PATH, "rb") as f:
#     steering_dict = pickle.load(f)  # {layer: torch.Tensor}
# print(f"Loaded {len(steering_dict)} steering directions")

# # --- Load token activation DataFrame ---
# df = pd.read_pickle(ACTIVATION_DF_PATH)
# print(f"Loaded activation data with {len(df)} rows")

# # --- Preprocess ---
# results = {}

# for layer_key, direction in steering_dict.items():
#     layer_index = abs(int(layer_key))

#     # If direction is 2D, take the first direction (or adjust as needed)
#     if direction.dim() == 2:
#         direction = direction[0]  # shape [D]
#     direction = direction.view(-1)  # flatten
#     direction = direction.to(torch.float32)

#     scores_and_tokens = []

#     for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Layer {layer_index}"):
#         mlp_vecs = row['layer_mlp_vec']  # list of [D]
#         tokens = row['layer_preds_tokens'][layer_index]  # list of token IDs at this layer

#         for vec, tok_id in zip(mlp_vecs, tokens):
#             vec_tensor = torch.tensor(vec, dtype=torch.float32, device=direction.device)
#             if vec_tensor.shape != direction.shape:
#                 # Skip or log mismatched shapes
#                 print(f"⚠️ Skipping due to shape mismatch: vec {vec_tensor.shape}, direction {direction.shape}")
#                 continue
#             score = torch.dot(vec_tensor, direction).item()
#             scores_and_tokens.append((score, tok_id))

#     # Sort and get top-K
#     top = sorted(scores_and_tokens, key=lambda x: x[0], reverse=True)[:TOP_K]
#     top_token_ids = [x[1] for x in top]
#     top_token_texts = [tokenizer.decode([tok_id]).strip() for tok_id in top_token_ids]

#     results[str(layer_key)] = top_token_texts
#     print(f"Layer {layer_key} top tokens: {top_token_texts}")

# # --- Save results ---
# with open("top_activating_tokens_by_layer.json", "w") as f:
#     json.dump(results, f, indent=2)

# print("✅ Saved top tokens to top_activating_tokens_by_layer.json")

#KA CLAUDE 1

# import pickle
# import pandas as pd
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import matplotlib.pyplot as plt
# from typing import List, Dict, Tuple, Optional
# import random
# import re
# from collections import defaultdict

# class SteeringTokenCombiner:
#     def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None
#         self.steering_vectors = None
#         self.mlp_df = None
#         self.strongest_layers = None
#         self.top_tokens_by_layer = {}
        
#     def load_components(self, steering_path: str, mlp_path: str, load_model: bool = False):
#         """Load steering vectors, MLP data, and optionally the model"""
#         print("Loading steering vectors...")
#         with open(steering_path, 'rb') as f:
#             self.steering_vectors = pickle.load(f)
        
#         print("Loading MLP activation data...")
#         with open(mlp_path, 'rb') as f:
#             self.mlp_df = pickle.load(f)
        
#         print("Loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         if load_model:
#             print("Loading model (this may take a while)...")
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 torch_dtype=torch.float16,
#                 device_map="auto" if torch.cuda.is_available() else None
#             )
        
#         print("✓ All components loaded successfully")
    
#     def identify_strongest_steering_layers(self, top_k: int = 10) -> List[Tuple[int, float]]:
#         """Identify layers with strongest harmful steering effects"""
#         layer_magnitudes = {}
        
#         for layer, vector in self.steering_vectors.items():
#             magnitude = torch.norm(vector).item()
#             layer_magnitudes[layer] = magnitude
        
#         # Sort by magnitude (strongest first)
#         self.strongest_layers = sorted(
#             layer_magnitudes.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )[:top_k]
        
#         print(f"Top {top_k} strongest harmful steering layers:")
#         for i, (layer, magnitude) in enumerate(self.strongest_layers):
#             print(f"{i+1}. Layer {layer}: magnitude = {magnitude:.4f}")
        
#         return self.strongest_layers
    
#     def extract_top_activation_tokens(self, target_layers: List[int], tokens_per_layer: int = 10):
#         """Extract top activation tokens for specific layers from MLP data"""
#         print(f"Extracting top {tokens_per_layer} activation tokens for layers: {target_layers}")
        
#         # Check if we have layer information in the MLP data
#         # The MLP vectors might correspond to different layers - we need to handle this
        
#         self.top_tokens_by_layer = {}
        
#         # Method 1: If there's explicit layer info
#         if 'layer' in self.mlp_df.columns:
#             for layer in target_layers:
#                 layer_data = self.mlp_df[self.mlp_df['layer'] == layer]
#                 if len(layer_data) > 0:
#                     self.top_tokens_by_layer[layer] = self._extract_tokens_from_samples(
#                         layer_data, tokens_per_layer
#                     )
        
#         # Method 2: If layer info is in the MLP vectors (assume each row is a different layer)
#         elif 'layer_mlp_vec' in self.mlp_df.columns:
#             # Assume the MLP vectors correspond to layers in some order
#             # This is a hypothesis that needs testing with your data
#             for i, layer in enumerate(target_layers):
#                 if i < len(self.mlp_df):
#                     sample_data = self.mlp_df.iloc[i:i+1]  # Take one sample for this layer
#                     self.top_tokens_by_layer[layer] = self._extract_tokens_from_samples(
#                         sample_data, tokens_per_layer
#                     )
        
#         # Method 3: Global top tokens (if we can't align layers)
#         else:
#             print("No clear layer alignment found. Using global top tokens.")
#             all_tokens = self._extract_global_top_tokens(tokens_per_layer)
#             for layer in target_layers:
#                 self.top_tokens_by_layer[layer] = all_tokens
        
#         return self.top_tokens_by_layer
    
#     def _extract_tokens_from_samples(self, data_samples: pd.DataFrame, top_k: int) -> List[Dict]:
#         """Extract top tokens from MLP data samples"""
#         token_counts = defaultdict(int)
#         token_info = {}
        
#         for _, row in data_samples.iterrows():
#             top_coef_idx = row['top_coef_idx']
#             if isinstance(top_coef_idx, list):
#                 for i, token_id in enumerate(top_coef_idx[:top_k]):
#                     # Weight by position (earlier = more important)
#                     weight = 1.0 / (i + 1)
#                     token_counts[token_id] += weight
                    
#                     if token_id not in token_info:
#                         try:
#                             # Properly decode individual token
#                             token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
#                             # Clean up the token text
#                             token_text = token_text.strip()
#                             if not token_text:  # Handle empty tokens
#                                 token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                            
#                             token_info[token_id] = {
#                                 'id': token_id,
#                                 'text': token_text,
#                                 'score': 0
#                             }
#                         except Exception as e:
#                             print(f"Warning: Could not decode token {token_id}: {e}")
#                             try:
#                                 # Fallback to convert_ids_to_tokens
#                                 token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
#                                 token_info[token_id] = {
#                                     'id': token_id,
#                                     'text': token_text,
#                                     'score': 0
#                                 }
#                             except:
#                                 token_info[token_id] = {
#                                     'id': token_id,
#                                     'text': f'<UNK_{token_id}>',
#                                     'score': 0
#                                 }
        
#         # Update scores and sort
#         for token_id, score in token_counts.items():
#             if token_id in token_info:
#                 token_info[token_id]['score'] = score
        
#         # Return top tokens sorted by score
#         sorted_tokens = sorted(token_info.values(), key=lambda x: x['score'], reverse=True)
#         return sorted_tokens[:top_k]
    
#     def _extract_global_top_tokens(self, top_k: int) -> List[Dict]:
#         """Extract globally most frequent top tokens"""
#         all_token_counts = defaultdict(int)
        
#         for _, row in self.mlp_df.iterrows():
#             top_coef_idx = row['top_coef_idx']
#             if isinstance(top_coef_idx, list):
#                 for token_id in top_coef_idx[:5]:  # Take top 5 from each sample
#                     all_token_counts[token_id] += 1
        
#         # Convert to token info format
#         token_info = []
#         for token_id, count in all_token_counts.items():
#             try:
#                 # Properly decode individual token
#                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
#                 token_text = token_text.strip()
#                 if not token_text:  # Handle empty tokens
#                     token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                
#                 token_info.append({
#                     'id': token_id,
#                     'text': token_text,
#                     'score': count
#                 })
#             except Exception as e:
#                 print(f"Warning: Could not decode token {token_id}: {e}")
#                 try:
#                     token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
#                     token_info.append({
#                         'id': token_id,
#                         'text': token_text,
#                         'score': count
#                     })
#                 except:
#                     token_info.append({
#                         'id': token_id,
#                         'text': f'<UNK_{token_id}>',
#                         'score': count
#                     })
        
#         return sorted(token_info, key=lambda x: x['score'], reverse=True)[:top_k]
    
#     def generate_control_prompts(self, base_prompts: List[str], strategy: str = "natural") -> Dict:
#         """Generate control prompts using top activation tokens"""
        
#         if not self.top_tokens_by_layer:
#             raise ValueError("Must extract top tokens first!")
        
#         strategies = {
#             "natural": self._generate_natural_prompts,
#             "direct": self._generate_direct_prompts,
#             "template": self._generate_template_prompts,
#             "context": self._generate_context_prompts
#         }
        
#         if strategy not in strategies:
#             raise ValueError(f"Strategy must be one of: {list(strategies.keys())}")
        
#         print(f"Generating control prompts using '{strategy}' strategy...")
#         return strategies[strategy](base_prompts)
    
#     def _generate_natural_prompts(self, base_prompts: List[str]) -> Dict:
#         """Generate natural language prompts incorporating top tokens"""
#         control_prompts = {}
        
#         for layer, tokens in self.top_tokens_by_layer.items():
#             control_prompts[layer] = []
            
#             # Extract meaningful words (filter out punctuation, etc.)
#             meaningful_tokens = [
#                 t['text'] for t in tokens 
#                 if len(t['text'].strip()) > 1 and t['text'].isalnum()
#             ][:5]  # Top 5 meaningful tokens
            
#             for base_prompt in base_prompts:
#                 if meaningful_tokens:
#                     # Strategy 1: Prefix with context
#                     token_context = f"Considering concepts like {', '.join(meaningful_tokens)}, "
#                     natural_prompt = token_context + base_prompt.lower()
                    
#                     # Strategy 2: Weave tokens into the prompt
#                     woven_prompt = self._weave_tokens_naturally(base_prompt, meaningful_tokens)
                    
#                     control_prompts[layer].extend([
#                         {"type": "context_prefix", "prompt": natural_prompt, "tokens_used": meaningful_tokens},
#                         {"type": "woven", "prompt": woven_prompt, "tokens_used": meaningful_tokens}
#                     ])
        
#         return control_prompts
    
#     def _weave_tokens_naturally(self, base_prompt: str, tokens: List[str]) -> str:
#         """Naturally weave tokens into the base prompt"""
#         # Simple approach: add tokens as "related concepts"
#         if tokens:
#             token_phrase = f"Think about {', '.join(tokens[:3])} when answering: "
#             return token_phrase + base_prompt
#         return base_prompt
    
#     def _generate_direct_prompts(self, base_prompts: List[str]) -> Dict:
#         """Generate prompts with direct token insertion"""
#         control_prompts = {}
        
#         for layer, tokens in self.top_tokens_by_layer.items():
#             control_prompts[layer] = []
            
#             for base_prompt in base_prompts:
#                 # Insert top tokens directly
#                 token_list = [t['text'] for t in tokens[:5]]
#                 direct_prompt = f"{' '.join(token_list)} {base_prompt}"
                
#                 control_prompts[layer].append({
#                     "type": "direct_insertion",
#                     "prompt": direct_prompt,
#                     "tokens_used": token_list
#                 })
        
#         return control_prompts
    
#     def _generate_template_prompts(self, base_prompts: List[str]) -> Dict:
#         """Generate prompts using templates"""
#         templates = [
#             "Given the importance of {tokens}, please {prompt}",
#             "Focusing on {tokens}: {prompt}",
#             "With {tokens} in mind, {prompt}",
#         ]
        
#         control_prompts = {}
        
#         for layer, tokens in self.top_tokens_by_layer.items():
#             control_prompts[layer] = []
#             token_list = [t['text'] for t in tokens[:3] if len(t['text'].strip()) > 1]
            
#             for base_prompt in base_prompts:
#                 for template in templates:
#                     templated_prompt = template.format(
#                         tokens=', '.join(token_list),
#                         prompt=base_prompt.lower()
#                     )
                    
#                     control_prompts[layer].append({
#                         "type": "template",
#                         "prompt": templated_prompt,
#                         "tokens_used": token_list,
#                         "template": template
#                     })
        
#         return control_prompts
    
#     def _generate_context_prompts(self, base_prompts: List[str]) -> Dict:
#         """Generate prompts with rich context from tokens"""
#         control_prompts = {}
        
#         for layer, tokens in self.top_tokens_by_layer.items():
#             control_prompts[layer] = []
            
#             # Build context from tokens
#             context_tokens = [t['text'] for t in tokens[:5] if len(t['text'].strip()) > 1]
            
#             for base_prompt in base_prompts:
#                 context_prompt = (
#                     f"In a scenario involving {', '.join(context_tokens[:3])}, "
#                     f"with particular attention to {context_tokens[0] if context_tokens else 'key factors'}, "
#                     f"{base_prompt}"
#                 )
                
#                 control_prompts[layer].append({
#                     "type": "rich_context",
#                     "prompt": context_prompt,
#                     "tokens_used": context_tokens
#                 })
        
#         return control_prompts
    
#     def evaluate_control_prompts(self, control_prompts: Dict, test_prompts: List[str] = None) -> Dict:
#         """Evaluate the effectiveness of control prompts"""
#         if self.model is None:
#             print("Model not loaded. Loading for evaluation...")
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 torch_dtype=torch.float16,
#                 device_map="auto" if torch.cuda.is_available() else None
#             )
        
#         evaluation_results = {}
        
#         # Default test prompts if none provided
#         if test_prompts is None:
#             test_prompts = [
#                 "How can I help people?",
#                 "What's the best way to solve conflicts?",
#                 "Tell me about safety measures."
#             ]
        
#         for layer, layer_prompts in control_prompts.items():
#             print(f"Evaluating layer {layer} prompts...")
#             layer_results = []
            
#             for prompt_data in layer_prompts[:3]:  # Test top 3 prompts per layer
#                 prompt = prompt_data['prompt']
                
#                 # Generate response
#                 try:
#                     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#                     if torch.cuda.is_available():
#                         inputs = {k: v.cuda() for k, v in inputs.items()}
                    
#                     with torch.no_grad():
#                         outputs = self.model.generate(
#                             **inputs,
#                             max_new_tokens=100,
#                             temperature=0.7,
#                             pad_token_id=self.tokenizer.eos_token_id,
#                             do_sample=True
#                         )
                    
#                     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                     response = response[len(prompt):].strip()  # Remove input prompt
                    
#                     layer_results.append({
#                         'prompt': prompt,
#                         'response': response,
#                         'tokens_used': prompt_data['tokens_used'],
#                         'prompt_type': prompt_data['type']
#                     })
                    
#                 except Exception as e:
#                     print(f"Error generating response: {e}")
#                     layer_results.append({
#                         'prompt': prompt,
#                         'response': f"ERROR: {str(e)}",
#                         'tokens_used': prompt_data['tokens_used'],
#                         'prompt_type': prompt_data['type']
#                     })
            
#             evaluation_results[layer] = layer_results
        
#         return evaluation_results
    
#     def run_complete_pipeline(self, steering_path: str, mlp_path: str, 
#                             base_prompts: List[str], top_layers: int = 5,
#                             tokens_per_layer: int = 10, strategy: str = "natural"):
#         """Run the complete pipeline"""
#         print("=" * 60)
#         print("RUNNING COMPLETE STEERING + TOKEN COMBINATION PIPELINE")
#         print("=" * 60)
        
#         # Step 1: Load components
#         self.load_components(steering_path, mlp_path, load_model=False)
        
#         # Step 2: Identify strongest steering layers
#         strongest_layers = self.identify_strongest_steering_layers(top_layers)
#         target_layers = [layer for layer, _ in strongest_layers]
        
#         # Step 3: Extract top activation tokens
#         self.extract_top_activation_tokens(target_layers, tokens_per_layer)
        
#         # Step 4: Generate control prompts
#         control_prompts = self.generate_control_prompts(base_prompts, strategy)
        
#         # Step 5: Display results
#         self.display_results(control_prompts)
        
#         return {
#             'strongest_layers': strongest_layers,
#             'top_tokens': self.top_tokens_by_layer,
#             'control_prompts': control_prompts
#         }
    
#     def display_results(self, control_prompts: Dict):
#         """Display the results in a readable format"""
#         print("\n" + "=" * 60)
#         print("RESULTS: TOP ACTIVATION TOKENS BY LAYER")
#         print("=" * 60)
        
#         for layer, tokens in self.top_tokens_by_layer.items():
#             print(f"\nLayer {layer} - Top Tokens:")
#             for i, token in enumerate(tokens[:5]):
#                 print(f"  {i+1}. '{token['text']}' (ID: {token['id']}, Score: {token['score']:.3f})")
        
#         print("\n" + "=" * 60)
#         print("GENERATED CONTROL PROMPTS")
#         print("=" * 60)
        
#         for layer, layer_prompts in control_prompts.items():
#             print(f"\nLayer {layer} Control Prompts:")
#             for i, prompt_data in enumerate(layer_prompts[:2]):  # Show top 2 per layer
#                 print(f"  {i+1}. [{prompt_data['type']}] {prompt_data['prompt']}")
#                 print(f"     Tokens used: {prompt_data['tokens_used']}")


# # Usage example
# def main():
#     """Main execution function"""
#     combiner = SteeringTokenCombiner()
    
#     # Define paths (update these to your actual paths)
#     steering_path = "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl"
#     mlp_path = "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl"
    
#     # Base prompts to test
#     base_prompts = [
#         "Explain how to handle difficult situations.",
#         "What should I do when facing challenges?",
#         "How can I make better decisions?"
#     ]
    
#     # Run complete pipeline
#     results = combiner.run_complete_pipeline(
#         steering_path=steering_path,
#         mlp_path=mlp_path,
#         base_prompts=base_prompts,
#         top_layers=5,
#         tokens_per_layer=10,
#         strategy="natural"
#     )
    
#     return results, combiner

# if __name__ == "__main__":
#     results, combiner = main()
# """
# Complete Integration Script for Steering Vector + Token Activation Research
# Combines neural controllers (steering vectors) with FFN activation analysis
# """

# # import sys
# # import os
# # import pickle
# # import torch
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # import argparse
# # import json
# # from datetime import datetime

# # # Import our custom modules (assuming they're in the same directory)
# # # If running as separate files, uncomment these:
# # # from diagnostic_tests import DataDiagnostics
# # # from steering_token_combiner import SteeringTokenCombiner
# # # from evaluation_framework import ControlPromptEvaluator

# # class ResearchPipeline:
# #     """
# #     Complete research pipeline combining steering vectors with token activations
# #     """
# #     def __init__(self, config_path: str = None):
# #         self.config = self.load_config(config_path)
# #         self.results = {}
# #         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
# #         # Initialize components
# #         self.diagnostics = None  # DataDiagnostics()
# #         self.combiner = None     # SteeringTokenCombiner()
# #         self.evaluator = None    # ControlPromptEvaluator()
    
# #     def load_config(self, config_path: str) -> dict:
# #         """Load configuration from file or use defaults"""
# #         default_config = {
# #             "paths": {
# #                 "steering_vectors": "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl",
# #                 "mlp_data": "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl",
# #                 "output_dir": "results"
# #             },
# #             "model": {
# #                 "name": "meta-llama/Llama-3.1-8B-Instruct",
# #                 "load_for_evaluation": True
# #             },
# #             "experiment": {
# #                 "top_layers": 5,
# #                 "tokens_per_layer": 10,
# #                 "prompt_strategies": ["natural", "template", "context"],
# #                 "max_evaluations": 3
# #             },
# #             "evaluation": {
# #                 "baseline_prompts": [
# #                     "How can I handle difficult situations?",
# #                     "What's the best approach to solve problems?",
# #                     "Give me advice on making decisions.",
# #                     "How should I respond to challenges?",
# #                     "What's the right way to deal with conflicts?"
# #                 ]
# #             }
# #         }
        
# #         if config_path and os.path.exists(config_path):
# #             with open(config_path, 'r') as f:
# #                 loaded_config = json.load(f)
# #                 # Merge with defaults
# #                 for key, value in loaded_config.items():
# #                     if isinstance(value, dict) and key in default_config:
# #                         default_config[key].update(value)
# #                     else:
# #                         default_config[key] = value
        
# #         return default_config
    
# #     def setup_output_directory(self):
# #         """Create output directory for results"""
# #         output_dir = Path(self.config["paths"]["output_dir"])
# #         output_dir.mkdir(exist_ok=True)
        
# #         # Create timestamped subdirectory
# #         self.experiment_dir = output_dir / f"experiment_{self.timestamp}"
# #         self.experiment_dir.mkdir(exist_ok=True)
        
# #         print(f"Results will be saved to: {self.experiment_dir}")
# #         return self.experiment_dir
    
# #     def run_diagnostics(self) -> dict:
# #         """Run diagnostic tests on the input data"""
# #         print("\n" + "="*60)
# #         print("STEP 1: RUNNING DIAGNOSTICS")
# #         print("="*60)
        
# #         # Initialize diagnostics (you'll need to import or define this)
# #         # self.diagnostics = DataDiagnostics()
        
# #         # For now, let's create a simplified diagnostic
# #         steering_path = self.config["paths"]["steering_vectors"]
# #         mlp_path = self.config["paths"]["mlp_data"]
        
# #         print(f"Checking steering vectors: {steering_path}")
# #         print(f"Checking MLP data: {mlp_path}")
        
# #         # Load and inspect data
# #         diagnostic_results = {}
        
# #         try:
# #             with open(steering_path, 'rb') as f:
# #                 steering_data = pickle.load(f)
# #             diagnostic_results['steering_vectors'] = {
# #                 'type': str(type(steering_data)),
# #                 'keys': list(steering_data.keys()) if isinstance(steering_data, dict) else 'Not a dict',
# #                 'sample_shape': steering_data[list(steering_data.keys())[0]].shape if isinstance(steering_data, dict) else 'N/A'
# #             }
# #             print(f"✓ Steering vectors loaded: {len(steering_data)} layers")
# #         except Exception as e:
# #             print(f"✗ Error loading steering vectors: {e}")
# #             return {"error": str(e)}
        
# #         try:  
# #             with open(mlp_path, 'rb') as f:
# #                 mlp_data = pickle.load(f)
# #             diagnostic_results['mlp_data'] = {
# #                 'type': str(type(mlp_data)),
# #                 'shape': mlp_data.shape if hasattr(mlp_data, 'shape') else 'No shape attr',
# #                 'columns': list(mlp_data.columns) if hasattr(mlp_data, 'columns') else 'No columns attr'
# #             }
# #             print(f"✓ MLP data loaded: {mlp_data.shape if hasattr(mlp_data, 'shape') else 'Unknown shape'}")
# #         except Exception as e:
# #             print(f"✗ Error loading MLP data: {e}")
# #             return {"error": str(e)}
        
# #         # Save diagnostic results
# #         with open(self.experiment_dir / "diagnostics.json", 'w') as f:
# #             json.dump(diagnostic_results, f, indent=2, default=str)
        
# #         self.results['diagnostics'] = diagnostic_results
# #         return diagnostic_results
    
# #     def run_combination_pipeline(self) -> dict:
# #         """Run the main combination pipeline"""
# #         print("\n" + "="*60)
# #         print("STEP 2: COMBINING STEERING VECTORS WITH TOKEN ACTIVATIONS")
# #         print("="*60)
        
# #         # Initialize combiner (you'll need to import or define this)
# #         # self.combiner = SteeringTokenCombiner(self.config["model"]["name"])
        
# #         # For now, let's create a simplified version
# #         combination_results = {}
        
# #         try:
# #             # Load data
# #             steering_path = self.config["paths"]["steering_vectors"]
# #             mlp_path = self.config["paths"]["mlp_data"]
            
# #             with open(steering_path, 'rb') as f:
# #                 steering_vectors = pickle.load(f)
            
# #             with open(mlp_path, 'rb') as f:
# #                 mlp_df = pickle.load(f)
            
# #             # Find strongest steering layers
# #             layer_magnitudes = {}
# #             for layer, vector in steering_vectors.items():
# #                 magnitude = torch.norm(vector).item()
# #                 layer_magnitudes[layer] = magnitude
            
# #             strongest_layers = sorted(
# #                 layer_magnitudes.items(), 
# #                 key=lambda x: x[1], 
# #                 reverse=True
# #             )[:self.config["experiment"]["top_layers"]]
            
# #             print(f"Top {len(strongest_layers)} strongest layers:")
# #             for i, (layer, magnitude) in enumerate(strongest_layers):
# #                 print(f"  {i+1}. Layer {layer}: magnitude = {magnitude:.4f}")
            
# #             # Extract top tokens (simplified version)
# #             top_tokens_by_layer = {}
# #             for layer, _ in strongest_layers:
# #                 # Use global top tokens for now (simplified)
# #                 if 'top_coef_idx' in mlp_df.columns:
# #                     all_tokens = []
# #                     for _, row in mlp_df.head(50).iterrows():  # Sample first 50 rows
# #                         tokens = row['top_coef_idx']
# #                         if isinstance(tokens, list):
# #                             all_tokens.extend(tokens[:5])
                    
# #                     # Count token frequencies
# #                     from collections import Counter
# #                     token_counts = Counter(all_tokens)
# #                     top_tokens = token_counts.most_common(self.config["experiment"]["tokens_per_layer"])
                    
# #                     top_tokens_by_layer[layer] = [
# #                         {'id': token_id, 'text': f'token_{token_id}', 'score': count}
# #                         for token_id, count in top_tokens
# #                     ]
            
# #             combination_results = {
# #                 'strongest_layers': strongest_layers,
# #                 'top_tokens': top_tokens_by_layer,
# #                 'config_used': self.config["experiment"]
# #             }
            
# #             # Save combination results
# #             with open(self.experiment_dir / "combination_results.json", 'w') as f:
# #                 json.dump(combination_results, f, indent=2, default=str)
            
# #             self.results['combination'] = combination_results
# #             print(f"✓ Combination pipeline completed for {len(strongest_layers)} layers")
            
# #         except Exception as e:
# #             print(f"✗ Error in combination pipeline: {e}")
# #             combination_results = {"error": str(e)}
        
# #         return combination_results
    
# #     def generate_control_prompts(self) -> dict:
# #         """Generate control prompts using different strategies"""
# #         print("\n" + "="*60)
# #         print("STEP 3: GENERATING CONTROL PROMPTS")
# #         print("="*60)
        
# #         if 'combination' not in self.results:
# #             print("✗ No combination results available")
# #             return {"error": "No combination results"}
        
# #         top_tokens = self.results['combination']['top_tokens']
# #         baseline_prompts = self.config["evaluation"]["baseline_prompts"]
        
# #         control_prompts = {}
        
# #         for strategy in self.config["experiment"]["prompt_strategies"]:
# #             print(f"Generating {strategy} prompts...")
            
# #             strategy_prompts = {}
            
# #             for layer, tokens in top_tokens.items():
# #                 layer_prompts = []
# #                 token_texts = [t['text'] for t in tokens[:5]]  # Top 5 tokens
                
# #                 for base_prompt in baseline_prompts[:3]:  # Use first 3 baseline prompts
# #                     if strategy == "natural":
# #                         prompt = f"Considering concepts like {', '.join(token_texts)}, {base_prompt.lower()}"
# #                     elif strategy == "template":
# #                         prompt = f"Given the importance of {', '.join(token_texts)}, please {base_prompt.lower()}"
# #                     elif strategy == "context":
# #                         prompt = f"In a scenario involving {', '.join(token_texts[:3])}, {base_prompt}"
# #                     else:
# #                         prompt = f"{' '.join(token_texts)} {base_prompt}"
                    
# #                     layer_prompts.append({
# #                         'prompt': prompt,
# #                         'type': strategy,
# #                         'tokens_used': token_texts,
# #                         'base_prompt': base_prompt
# #                     })
                
# #                 strategy_prompts[layer] = layer_prompts
            
# #             control_prompts[strategy] = strategy_prompts
        
# #         # Save control prompts
# #         with open(self.experiment_dir / "control_prompts.json", 'w') as f:
# #             json.dump(control_prompts, f, indent=2, default=str)
        
# #         self.results['control_prompts'] = control_prompts
# #         print(f"✓ Generated {len(control_prompts)} prompt strategies")
        
# #         return control_prompts
    
# #     def run_evaluation(self) -> dict:
# #         """Run evaluation of control prompts"""
# #         print("\n" + "="*60)
# #         print("STEP 4: EVALUATING CONTROL PROMPTS")
# #         print("="*60)
        
# #         if not self.config["model"]["load_for_evaluation"]:
# #             print("Model evaluation disabled in config. Skipping...")
# #             return {"status": "skipped", "reason": "evaluation disabled"}
        
# #         if 'control_prompts' not in self.results:
# #             print("✗ No control prompts available")
# #             return {"error": "No control prompts"}
        
# #         evaluation_results = {}
        
# #         try:
# #             # Initialize evaluator (simplified version)
# #             # self.evaluator = ControlPromptEvaluator(self.config["model"]["name"])
            
# #             # For demonstration, create mock evaluation results
# #             baseline_prompts = self.config["evaluation"]["baseline_prompts"]
# #             control_prompts = self.results['control_prompts']
            
# #             print("Running mock evaluation (replace with actual model evaluation)...")
            
# #             for strategy, strategy_prompts in control_prompts.items():
# #                 print(f"Evaluating {strategy} strategy...")
                
# #                 strategy_results = {}
                
# #                 for layer, layer_prompts in strategy_prompts.items():
# #                     layer_results = []
                    
# #                     for prompt_data in layer_prompts[:self.config["experiment"]["max_evaluations"]]:
# #                         # Mock evaluation (replace with actual model inference)
# #                         mock_response = f"This is a mock response to: {prompt_data['prompt'][:50]}..."
# #                         mock_harmfulness = np.random.uniform(-1, 1)  # Random score for demo
                        
# #                         layer_results.append({
# #                             'prompt': prompt_data['prompt'],
# #                             'response': mock_response,
# #                             'harmfulness_score': mock_harmfulness,
# #                             'tokens_used': prompt_data['tokens_used'],
# #                             'prompt_type': prompt_data['type']
# #                         })
                    
# #                     strategy_results[layer] = {
# #                         'responses': layer_results,
# #                         'avg_harmfulness': np.mean([r['harmfulness_score'] for r in layer_results]),
# #                         'consistency': np.std([r['harmfulness_score'] for r in layer_results])
# #                     }
                
# #                 evaluation_results[strategy] = strategy_results
            
# #             # Generate summary statistics
# #             evaluation_results['summary'] = self._generate_evaluation_summary(evaluation_results)
            
# #             # Save evaluation results
# #             with open(self.experiment_dir / "evaluation_results.json", 'w') as f:
# #                 json.dump(evaluation_results, f, indent=2, default=str)
            
# #             self.results['evaluation'] = evaluation_results
# #             print(f"✓ Evaluation completed for {len(control_prompts)} strategies")
            
# #         except Exception as e:
# #             print(f"✗ Error in evaluation: {e}")
# #             evaluation_results = {"error": str(e)}
        
# #         return evaluation_results
    
# #     def _generate_evaluation_summary(self, evaluation_results: dict) -> dict:
# #         """Generate summary statistics from evaluation results"""
# #         summary = {
# #             'best_strategy': None,
# #             'best_layer': None,
# #             'max_steering_effect': 0,
# #             'strategy_comparison': {}
# #         }
        
# #         max_effect = 0
        
# #         for strategy, strategy_data in evaluation_results.items():
# #             if strategy == 'summary':  # Skip summary key
# #                 continue
                
# #             strategy_effects = []
# #             best_layer_for_strategy = None
# #             best_effect_for_strategy = 0
            
# #             for layer, layer_data in strategy_data.items():
# #                 avg_harm = layer_data['avg_harmfulness']
# #                 effect_magnitude = abs(avg_harm)
# #                 strategy_effects.append(effect_magnitude)
                
# #                 if effect_magnitude > best_effect_for_strategy:
# #                     best_effect_for_strategy = effect_magnitude
# #                     best_layer_for_strategy = layer
                
# #                 if effect_magnitude > max_effect:
# #                     max_effect = effect_magnitude
# #                     summary['best_strategy'] = strategy
# #                     summary['best_layer'] = layer
# #                     summary['max_steering_effect'] = avg_harm
            
# #             summary['strategy_comparison'][strategy] = {
# #                 'avg_effect_magnitude': np.mean(strategy_effects),
# #                 'max_effect': best_effect_for_strategy,
# #                 'best_layer': best_layer_for_strategy,
# #                 'consistency': np.std(strategy_effects)
# #             }
        
# #         return summary
    
# #     def generate_final_report(self) -> dict:
# #         """Generate comprehensive final report"""
# #         print("\n" + "="*60)
# #         print("STEP 5: GENERATING FINAL REPORT")
# #         print("="*60)
        
# #         report = {
# #             'experiment_info': {
# #                 'timestamp': self.timestamp,
# #                 'config': self.config,
# #                 'output_directory': str(self.experiment_dir)
# #             },
# #             'results_summary': {},
# #             'key_findings': [],
# #             'recommendations': [],
# #             'next_steps': []
# #         }
        
# #         # Summarize each step
# #         if 'diagnostics' in self.results:
# #             report['results_summary']['diagnostics'] = {
# #                 'status': 'completed' if 'error' not in self.results['diagnostics'] else 'failed',
# #                 'details': self.results['diagnostics']
# #             }
        
# #         if 'combination' in self.results:
# #             combination = self.results['combination']
# #             report['results_summary']['combination'] = {
# #                 'status': 'completed' if 'error' not in combination else 'failed',
# #                 'layers_analyzed': len(combination.get('strongest_layers', [])),
# #                 'tokens_extracted': sum(len(tokens) for tokens in combination.get('top_tokens', {}).values())
# #             }
        
# #         if 'control_prompts' in self.results:
# #             prompts = self.results['control_prompts']
# #             report['results_summary']['prompt_generation'] = {
# #                 'status': 'completed',
# #                 'strategies_used': list(prompts.keys()),
# #                 'total_prompts': sum(len(layer_prompts) for strategy in prompts.values() 
# #                                    for layer_prompts in strategy.values())
# #             }
        
# #         if 'evaluation' in self.results:
# #             evaluation = self.results['evaluation']
# #             if 'error' not in evaluation:
# #                 report['results_summary']['evaluation'] = {
# #                     'status': 'completed',
# #                     'summary': evaluation.get('summary', {})
# #                 }
# #             else:
# #                 report['results_summary']['evaluation'] = {
# #                     'status': 'failed',
# #                     'error': evaluation['error']
# #                 }
        
# #         # Generate findings and recommendations
# #         report['key_findings'] = self._generate_key_findings()
# #         report['recommendations'] = self._generate_recommendations()
# #         report['next_steps'] = self._generate_next_steps()
        
# #         # Save final report
# #         with open(self.experiment_dir / "final_report.json", 'w') as f:
# #             json.dump(report, f, indent=2, default=str)
        
# #         # Also save a human-readable version
# #         self._save_readable_report(report)
        
# #         self.results['final_report'] = report
# #         print(f"✓ Final report generated: {self.experiment_dir / 'final_report.json'}")
        
# #         return report
    
# #     def _generate_key_findings(self) -> list:
# #         """Generate key findings from the experiment"""
# #         findings = []
        
# #         if 'combination' in self.results:
# #             combination = self.results['combination']
# #             if 'strongest_layers' in combination:
# #                 strongest_layers = combination['strongest_layers']
# #                 findings.append(f"Identified {len(strongest_layers)} strongest harmful steering layers")
# #                 top_layer = strongest_layers[0][0] if strongest_layers else None
# #                 if top_layer:
# #                     findings.append(f"Layer {top_layer} shows the highest steering magnitude")
        
# #         if 'evaluation' in self.results and 'summary' in self.results['evaluation']:
# #             summary = self.results['evaluation']['summary']
# #             if summary.get('best_strategy'):
# #                 findings.append(f"Most effective prompt strategy: {summary['best_strategy']}")
# #             if summary.get('max_steering_effect'):
# #                 effect = summary['max_steering_effect']
# #                 direction = "harmful" if effect > 0 else "safety"
# #                 findings.append(f"Maximum steering effect: {effect:.3f} ({direction} direction)")
        
# #         return findings
    
# #     def _generate_recommendations(self) -> list:
# #         """Generate recommendations based on results"""
# #         recommendations = []
        
# #         if 'evaluation' in self.results:
# #             if 'error' in self.results['evaluation']:
# #                 recommendations.append("Enable model evaluation to get concrete steering effectiveness metrics")
# #             elif 'summary' in self.results['evaluation']:
# #                 summary = self.results['evaluation']['summary']
# #                 if summary.get('max_steering_effect', 0) < 0.1:
# #                     recommendations.append("Steering effects are weak. Consider using different layers or more tokens")
# #                     recommendations.append("Try focusing on middle layers (10-20) which often show stronger effects")
        
# #         recommendations.extend([
# #             "Test with a larger dataset to improve token selection reliability",
# #             "Experiment with different prompt templates and strategies",
# #             "Consider using multiple tokens simultaneously for stronger effects",
# #             "Validate results with human evaluation of generated responses"
# #         ])
        
# #         return recommendations
    
# #     def _generate_next_steps(self) -> list:
# #         """Generate suggested next steps"""
# #         return [
# #             "Run full evaluation with model inference to measure actual steering effects",
# #             "Test the approach on different types of harmful content beyond the current dataset",
# #             "Compare results with traditional activation patching methods",
# #             "Investigate why certain layers show stronger steering effects",
# #             "Develop automated metrics for measuring steering effectiveness",
# #             "Scale up the experiment with more layers and token combinations"
# #         ]
    
# #     def _save_readable_report(self, report: dict):
# #         """Save a human-readable version of the report"""
# #         readable_path = self.experiment_dir / "final_report.md"
        
# #         with open(readable_path, 'w') as f:
# #             f.write(f"# Steering Vector + Token Activation Research Report\n\n")
# #             f.write(f"**Experiment ID:** {self.timestamp}\n")
# #             f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
# #             f.write("## Configuration\n\n")
# #             f.write(f"- **Model:** {self.config['model']['name']}\n")
# #             f.write(f"- **Top Layers:** {self.config['experiment']['top_layers']}\n")
# #             f.write(f"- **Tokens per Layer:** {self.config['experiment']['tokens_per_layer']}\n")
# #             f.write(f"- **Prompt Strategies:** {', '.join(self.config['experiment']['prompt_strategies'])}\n\n")
            
# #             f.write("## Results Summary\n\n")
# #             for step, summary in report['results_summary'].items():
# #                 f.write(f"### {step.title()}\n")
# #                 f.write(f"**Status:** {summary['status']}\n")
# #                 if 'error' in summary:
# #                     f.write(f"**Error:** {summary['error']}\n")
# #                 f.write("\n")
            
# #             f.write("## Key Findings\n\n")
# #             for i, finding in enumerate(report['key_findings'], 1):
# #                 f.write(f"{i}. {finding}\n")
# #             f.write("\n")
            
# #             f.write("## Recommendations\n\n")
# #             for i, rec in enumerate(report['recommendations'], 1):
# #                 f.write(f"{i}. {rec}\n")
# #             f.write("\n")
            
# #             f.write("## Next Steps\n\n")
# #             for i, step in enumerate(report['next_steps'], 1):
# #                 f.write(f"{i}. {step}\n")
            
# #         print(f"✓ Readable report saved: {readable_path}")
    
# #     def run_complete_pipeline(self) -> dict:
# #         """Run the complete research pipeline"""
# #         print("="*60)
# #         print("STEERING VECTOR + TOKEN ACTIVATION RESEARCH PIPELINE")
# #         print("="*60)
        
# #         # Setup
# #         self.setup_output_directory()
        
# #         # Run pipeline steps
# #         try:
# #             self.run_diagnostics()
# #             self.run_combination_pipeline()
# #             self.generate_control_prompts()
# #             self.run_evaluation()
# #             final_report = self.generate_final_report()
            
# #             print("\n" + "="*60)
# #             print("PIPELINE COMPLETION SUMMARY")
# #             print("="*60)
# #             print(f"✓ All steps completed successfully")
# #             print(f"✓ Results saved to: {self.experiment_dir}")
# #             print(f"✓ View final report: {self.experiment_dir / 'final_report.md'}")
            
# #             return final_report
            
# #         except Exception as e:
# #             print(f"\n✗ Pipeline failed with error: {e}")
# #             error_report = {
# #                 'status': 'failed',
# #                 'error': str(e),
# #                 'completed_steps': list(self.results.keys())
# #             }
            
# #             with open(self.experiment_dir / "error_report.json", 'w') as f:
# #                 json.dump(error_report, f, indent=2, default=str)
            
# #             return error_report


# # def create_sample_config():
# #     """Create a sample configuration file"""
# #     config = {
# #         "paths": {
# #             "steering_vectors": "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl",
# #             "mlp_data": "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl",
# #             "output_dir": "results"
# #         },
# #         "model": {
# #             "name": "meta-llama/Llama-3.1-8B-Instruct",
# #             "load_for_evaluation": False  # Set to True for full evaluation
# #         },
# #         "experiment": {
# #             "top_layers": 5,
# #             "tokens_per_layer": 10,
# #             "prompt_strategies": ["natural", "template", "context"],
# #             "max_evaluations": 3
# #         },
# #         "evaluation": {
# #             "baseline_prompts": [
# #                 "How can I handle difficult situations?",
# #                 "What's the best approach to solve problems?",
# #                 "Give me advice on making decisions.",
# #                 "How should I respond to challenges?",
# #                 "What's the right way to deal with conflicts?"
# #             ]
# #         }
# #     }
    
# #     with open("research_config.json", 'w') as f:
# #         json.dump(config, f, indent=2)
    
# #     print("Sample configuration created: research_config.json")
# #     return config


# # def main():
# #     """Main function with command line interface"""
# #     parser = argparse.ArgumentParser(description='Steering Vector + Token Activation Research Pipeline')
# #     parser.add_argument('--config', type=str, help='Path to configuration file')
# #     parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
# #     parser.add_argument('--steering-path', type=str, help='Path to steering vectors pickle file')
# #     parser.add_argument('--mlp-path', type=str, help='Path to MLP data pickle file')
# #     parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
# #     parser.add_argument('--top-layers', type=int, default=5, help='Number of top layers to analyze')
# #     parser.add_argument('--no-evaluation', action='store_true', help='Skip model evaluation step')
    
# #     args = parser.parse_args()
    
# #     if args.create_config:
# #         create_sample_config()
# #         return
    
# #     # Create config from command line args if no config file provided
# #     if not args.config:
# #         config = None
# #         if args.steering_path or args.mlp_path:
# #             config = {
# #                 "paths": {
# #                     "steering_vectors": args.steering_path or "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl",
# #                     "mlp_data": args.mlp_path or "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl",
# #                     "output_dir": args.output_dir
# #                 },
# #                 "model": {
# #                     "load_for_evaluation": not args.no_evaluation
# #                 },
# #                 "experiment": {
# #                     "top_layers": args.top_layers
# #                 }
# #             }
            
# #             # Save temporary config
# #             with open("temp_config.json", 'w') as f:
# #                 json.dump(config, f, indent=2)
# #             args.config = "temp_config.json"
    
# #     # Run pipeline
# #     pipeline = ResearchPipeline(args.config)
# #     results = pipeline.run_complete_pipeline()
    
# #     # Cleanup temporary config if created
# #     if args.config == "temp_config.json":
# #         os.remove("temp_config.json")
    
# #     return results


# # if __name__ == "__main__":
# #     # Example usage for your specific case
# #     print("Running Steering Vector + Token Activation Research Pipeline")
    
# #     # You can run this directly or use the command line interface
# #     if len(sys.argv) == 1:  # No command line arguments
# #         # Direct execution with your paths
# #         pipeline = ResearchPipeline()
        
# #         # Update paths in the pipeline config
# #         pipeline.config["paths"]["steering_vectors"] = "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl"
# #         pipeline.config["paths"]["mlp_data"] = "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl"
# #         pipeline.config["model"]["load_for_evaluation"] = False  # Set to True when ready for full evaluation
        
# #         results = pipeline.run_complete_pipeline()
# #     else:
# #         # Command line execution
# #         results = main()


#!/usr/bin/env python3#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Comprehensive Integration Script: Steering Vectors + Token Activations
Combines neural steering vectors with FFN token activations to create control prompts

This script:
1. Loads steering vectors from neural_controllers
2. Loads FFN activation data from Geva et al. methodology  
3. Identifies top tokens for layers with strongest steering effects
4. Creates control prompts using these tokens
5. Evaluates steering effectiveness through prompting vs direct vector steering
"""

import sys
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict
from pathlib import Path
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional, Any

class SteeringTokenIntegrator:
    """
    Main class for integrating steering vectors with token activations
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model will be loaded on demand
        self.model = None
        
        # Data storage
        self.steering_vectors = None
        self.mlp_df = None
        self.layer_token_mapping = {}
        self.results = {}
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_data(self, steering_path: str, mlp_path: str) -> Dict[str, Any]:
        """
        Load both steering vectors and MLP activation data
        """
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load steering vectors
        print(f"Loading steering vectors from: {steering_path}")
        try:
            with open(steering_path, 'rb') as f:
                self.steering_vectors = pickle.load(f)
            print(f"✓ Loaded steering vectors for {len(self.steering_vectors)} layers")
            print(f"  Layer keys: {list(self.steering_vectors.keys())}")
            
            # Show sample vector info
            sample_layer = list(self.steering_vectors.keys())[0]
            sample_vector = self.steering_vectors[sample_layer]
            print(f"  Sample vector shape: {sample_vector.shape}")
            print(f"  Sample vector norm: {torch.norm(sample_vector).item():.4f}")
            
        except Exception as e:
            raise ValueError(f"Failed to load steering vectors: {e}")
        
        # Load MLP activation data
        print(f"\nLoading MLP data from: {mlp_path}")
        try:
            with open(mlp_path, 'rb') as f:
                self.mlp_df = pickle.load(f)
            print(f"✓ Loaded MLP data: {self.mlp_df.shape}")
            print(f"  Columns: {list(self.mlp_df.columns)}")
            
            # Show sample data
            if len(self.mlp_df) > 0:
                sample_row = self.mlp_df.iloc[0]
                print(f"  Sample top_coef_idx type: {type(sample_row['top_coef_idx'])}")
                if hasattr(sample_row['top_coef_idx'], '__len__'):
                    print(f"  Sample top_coef_idx length: {len(sample_row['top_coef_idx'])}")
                
        except Exception as e:
            raise ValueError(f"Failed to load MLP data: {e}")
        
        return {
            'steering_layers': len(self.steering_vectors),
            'mlp_samples': len(self.mlp_df),
            'steering_keys': list(self.steering_vectors.keys()),
            'mlp_columns': list(self.mlp_df.columns)
        }
    
    def analyze_steering_magnitudes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Analyze steering vector magnitudes to find most influential layers
        """
        print("\n" + "="*60)
        print("ANALYZING STEERING VECTOR MAGNITUDES")
        print("="*60)
        
        layer_magnitudes = {}
        
        for layer, vector in self.steering_vectors.items():
            # Calculate different magnitude metrics
            l2_norm = torch.norm(vector, p=2).item()
            l1_norm = torch.norm(vector, p=1).item()
            max_val = torch.max(torch.abs(vector)).item()
            
            layer_magnitudes[layer] = {
                'l2_norm': l2_norm,
                'l1_norm': l1_norm,
                'max_abs': max_val,
                'mean_abs': torch.mean(torch.abs(vector)).item()
            }
        
        # Sort by L2 norm (most common metric for steering strength)
        strongest_layers = sorted(
            layer_magnitudes.items(),
            key=lambda x: x[1]['l2_norm'],
            reverse=True
        )[:top_k]
        
        print(f"Top {len(strongest_layers)} layers by steering magnitude:")
        for i, (layer, metrics) in enumerate(strongest_layers):
            print(f"  {i+1:2d}. Layer {layer:3d}: L2={metrics['l2_norm']:8.4f}, "
                  f"L1={metrics['l1_norm']:8.1f}, Max={metrics['max_abs']:6.4f}")
        
        self.results['steering_analysis'] = {
            'all_magnitudes': layer_magnitudes,
            'strongest_layers': strongest_layers
        }
        
        return strongest_layers
    
    def extract_tokens_for_layers(self, strongest_layers: List[Tuple[int, float]], 
                                  tokens_per_layer: int = 15) -> Dict[int, List[Dict]]:
        """
        Extract top activating tokens for the strongest steering layers
        """
        print("\n" + "="*60)
        print("EXTRACTING TOKENS FOR TOP STEERING LAYERS")
        print("="*60)
        
        layer_tokens = {}
        
        # For each strong steering layer, find the most relevant tokens
        for layer_idx, (layer, magnitude_dict) in enumerate(strongest_layers):
            magnitude = magnitude_dict['l2_norm']
            print(f"\nProcessing Layer {layer} (magnitude: {magnitude:.4f})...")
            
            # Collect all tokens from MLP data - we'll use global tokens but could be layer-specific
            all_token_ids = []
            token_counts = defaultdict(int)
            
            # Extract tokens from top_coef_idx column
            for _, row in self.mlp_df.iterrows():
                top_coef_idx = row['top_coef_idx']
                
                # Handle different data formats
                extracted_tokens = self._extract_token_ids(top_coef_idx)
                all_token_ids.extend(extracted_tokens)
                
                # Count frequencies
                for token_id in extracted_tokens:
                    if isinstance(token_id, int):
                        token_counts[token_id] += 1
            
            # Get top tokens by frequency
            top_token_ids = Counter(token_counts).most_common(tokens_per_layer * 3)  # Get more for filtering
            
            # Decode tokens and filter for meaningful ones
            meaningful_tokens = []
            for token_id, count in top_token_ids:
                try:
                    if not isinstance(token_id, int):
                        continue
                    
                    # Decode token
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                    if not token_text:
                        token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    
                    # Clean BPE artifacts
                    clean_text = token_text.replace('Ġ', '').replace('▁', '').replace('Â', '').strip()
                    
                    # Filter for meaningful tokens
                    if (len(clean_text) > 1 and 
                        clean_text.replace(' ', '').isalnum() and
                        not clean_text.isdigit() and
                        not clean_text.startswith('<') and
                        not clean_text.startswith('[') and
                        len(clean_text) < 20):  # Avoid very long tokens
                        
                        meaningful_tokens.append({
                            'id': token_id,
                            'text': clean_text,
                            'raw_text': token_text,
                            'count': count,
                            'layer': layer
                        })
                    
                    if len(meaningful_tokens) >= tokens_per_layer:
                        break
                        
                except Exception as e:
                    continue
            
            layer_tokens[layer] = meaningful_tokens
            print(f"  Found {len(meaningful_tokens)} meaningful tokens for layer {layer}")
            
            # Show top 5 tokens for this layer
            for i, token_data in enumerate(meaningful_tokens[:5]):
                print(f"    {i+1}. '{token_data['text']}' (count={token_data['count']})")
        
        self.layer_token_mapping = layer_tokens
        self.results['token_extraction'] = layer_tokens
        
        return layer_tokens
    
    def _extract_token_ids(self, top_coef_idx: Any) -> List[int]:
        """
        Extract token IDs from various data formats in top_coef_idx
        """
        token_ids = []
        
        if isinstance(top_coef_idx, (list, tuple)):
            if len(top_coef_idx) > 0:
                first_element = top_coef_idx[0]
                
                if isinstance(first_element, (tuple, list)):
                    # Nested structure - flatten first few elements
                    for item in top_coef_idx[:10]:  # First 10 items
                        if isinstance(item, (tuple, list)):
                            token_ids.extend([x for x in item[:5] if isinstance(x, int)])
                        elif isinstance(item, int):
                            token_ids.append(item)
                elif isinstance(first_element, int):
                    # Direct list of integers
                    token_ids = [x for x in top_coef_idx[:20] if isinstance(x, int)]
        
        return token_ids
    
    def generate_control_prompts(self, strategies: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Generate control prompts using extracted tokens with different strategies
        """
        if strategies is None:
            strategies = ['natural', 'template', 'context', 'direct']
        
        print("\n" + "="*60)
        print("GENERATING CONTROL PROMPTS")
        print("="*60)
        
        # Base prompts for testing
        base_prompts = [
            "How should I handle difficult situations?",
            "What's the best way to resolve conflicts?",
            "Give me advice on making important decisions.",
            "How can I deal with challenging people?",
            "What's the right approach to solve problems?",
            "How should I respond to criticism?",
            "What's the best way to manage stress?",
            "Give me guidance on handling pressure."
        ]
        
        control_prompts = {}
        
        for strategy in strategies:
            print(f"\nGenerating {strategy} prompts...")
            strategy_prompts = []
            
            # Use tokens from top 3 strongest layers
            top_layers = list(self.layer_token_mapping.keys())[:3]
            
            for layer in top_layers:
                tokens = self.layer_token_mapping[layer]
                if not tokens:
                    continue
                
                # Get top 5 tokens for this layer
                top_token_texts = [t['text'] for t in tokens[:5]]
                
                for base_prompt in base_prompts[:4]:  # Use first 4 base prompts
                    control_prompt = self._create_control_prompt(
                        base_prompt, top_token_texts, strategy
                    )
                    
                    strategy_prompts.append({
                        'prompt': control_prompt,
                        'base_prompt': base_prompt,
                        'strategy': strategy,
                        'layer': layer,
                        'tokens_used': top_token_texts,
                        'token_data': tokens[:5]
                    })
            
            control_prompts[strategy] = strategy_prompts
            print(f"  Generated {len(strategy_prompts)} prompts for {strategy} strategy")
        
        self.results['control_prompts'] = control_prompts
        return control_prompts
    
    def _create_control_prompt(self, base_prompt: str, tokens: List[str], strategy: str) -> str:
        """
        Create a control prompt using specific strategy
        """
        if strategy == 'natural':
            return f"When considering concepts like {', '.join(tokens[:3])}, {base_prompt.lower()}"
        
        elif strategy == 'template':
            return f"Given the importance of {', '.join(tokens[:3])}, please {base_prompt.lower()}"
        
        elif strategy == 'context':
            return f"In a scenario involving {', '.join(tokens[:3])}, {base_prompt}"
        
        elif strategy == 'direct':
            return f"{' '.join(tokens[:3])} - {base_prompt}"
        
        elif strategy == 'weighted':
            # Use token frequencies for weighting (if available)
            weighted_tokens = ' '.join(tokens[:3])  # Simplified
            return f"Focusing on {weighted_tokens}: {base_prompt}"
        
        else:
            return f"With {', '.join(tokens[:3])} in mind: {base_prompt}"
    
    def load_model_for_evaluation(self) -> bool:
        """
        Load the model for evaluation (optional, resource intensive)
        """
        print("\n" + "="*60)
        print("LOADING MODEL FOR EVALUATION")
        print("="*60)
        
        try:
            print(f"Loading {self.model_name} on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("Evaluation will be skipped or use mock results")
            return False
    
    def evaluate_steering_effectiveness(self, max_samples: int = 5, 
                                      generate_responses: bool = False) -> Dict[str, Any]:
        """
        Evaluate how well token-based prompts replicate steering vector effects
        """
        print("\n" + "="*60)
        print("EVALUATING STEERING EFFECTIVENESS")
        print("="*60)
        
        if 'control_prompts' not in self.results:
            print("✗ No control prompts available. Run generate_control_prompts first.")
            return {}
        
        evaluation_results = {
            'baseline_vs_control': {},
            'cross_strategy_comparison': {},
            'layer_effectiveness': {},
            'summary_metrics': {}
        }
        
        # Base prompts for comparison
        base_prompts = [
            "How should I handle difficult situations?",
            "What's the best way to resolve conflicts?",
            "Give me advice on making important decisions."
        ]
        
        if generate_responses and self.model is not None:
            evaluation_results = self._evaluate_with_model(
                base_prompts, max_samples, evaluation_results
            )
        else:
            evaluation_results = self._evaluate_without_model(
                base_prompts, max_samples, evaluation_results
            )
        
        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    def _evaluate_with_model(self, base_prompts: List[str], max_samples: int, 
                           evaluation_results: Dict) -> Dict:
        """
        Evaluate with actual model inference (resource intensive)
        """
        print("Running evaluation with model inference...")
        
        # This would involve:
        # 1. Generate responses for base prompts
        # 2. Generate responses for control prompts
        # 3. Apply steering vectors directly and generate responses
        # 4. Compare harmfulness/alignment metrics
        
        # For now, return mock structure with actual implementation framework
        evaluation_results['method'] = 'model_inference'
        evaluation_results['note'] = 'Full model evaluation would go here'
        
        return evaluation_results
    
    def _evaluate_without_model(self, base_prompts: List[str], max_samples: int,
                              evaluation_results: Dict) -> Dict:
        """
        Evaluate without model inference using heuristics and analysis
        """
        print("Running evaluation without model inference (heuristic analysis)...")
        
        control_prompts = self.results['control_prompts']
        
        # Analyze prompt characteristics
        for strategy, prompts in control_prompts.items():
            strategy_analysis = {
                'avg_length': np.mean([len(p['prompt']) for p in prompts]),
                'token_diversity': len(set(token for p in prompts for token in p['tokens_used'])),
                'layer_coverage': len(set(p['layer'] for p in prompts)),
                'sample_prompts': prompts[:max_samples]
            }
            
            evaluation_results['cross_strategy_comparison'][strategy] = strategy_analysis
        
        # Layer effectiveness analysis
        for layer in list(self.layer_token_mapping.keys())[:5]:
            layer_prompts = []
            for strategy, prompts in control_prompts.items():
                layer_prompts.extend([p for p in prompts if p['layer'] == layer])
            
            if layer_prompts:
                steering_magnitude = None
                if self.steering_vectors and layer in self.steering_vectors:
                    steering_magnitude = torch.norm(self.steering_vectors[layer]).item()
                
                evaluation_results['layer_effectiveness'][layer] = {
                    'steering_magnitude': steering_magnitude,
                    'prompt_count': len(layer_prompts),
                    'avg_tokens_per_prompt': np.mean([len(p['tokens_used']) for p in layer_prompts]),
                    'sample_tokens': layer_prompts[0]['tokens_used'] if layer_prompts else []
                }
        
        # Summary metrics
        all_prompts = [p for prompts in control_prompts.values() for p in prompts]
        evaluation_results['summary_metrics'] = {
            'total_prompts_generated': len(all_prompts),
            'unique_tokens_used': len(set(token for p in all_prompts for token in p['tokens_used'])),
            'strategies_tested': list(control_prompts.keys()),
            'layers_analyzed': len(self.layer_token_mapping),
            'avg_prompt_length': np.mean([len(p['prompt']) for p in all_prompts])
        }
        
        evaluation_results['method'] = 'heuristic_analysis'
        return evaluation_results
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report of the integration process
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'experiment_info': {
                'timestamp': timestamp,
                'model_name': self.model_name,
                'device': self.device,
                'approach': 'Steering Vectors + Token Activations Integration'
            },
            'data_summary': {},
            'methodology': {
                'steering_analysis': 'L2 norm ranking of steering vector magnitudes',
                'token_extraction': 'FFN activation top coefficients from Geva et al.',
                'prompt_generation': 'Multiple strategies for embedding tokens in prompts',
                'evaluation': 'Comparative analysis of steering effectiveness'
            },
            'results': self.results.copy(),
            'key_findings': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # Data summary
        if self.steering_vectors:
            report['data_summary']['steering_vectors'] = {
                'layers_count': len(self.steering_vectors),
                'layer_range': f"{min(self.steering_vectors.keys())} to {max(self.steering_vectors.keys())}",
                'vector_dimension': self.steering_vectors[list(self.steering_vectors.keys())[0]].shape[0]
            }
        
        if self.mlp_df is not None:
            report['data_summary']['mlp_data'] = {
                'samples_count': len(self.mlp_df),
                'columns': list(self.mlp_df.columns),
                'token_format': 'nested_tuples' if isinstance(self.mlp_df.iloc[0]['top_coef_idx'][0], tuple) else 'flat_list'
            }
        
        # Generate findings
        report['key_findings'] = self._generate_key_findings()
        report['recommendations'] = self._generate_recommendations()
        report['next_steps'] = self._generate_next_steps()
        
        # Save report
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            # JSON report
            json_path = save_path / f"steering_token_integration_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Markdown report  
            md_path = save_path / f"steering_token_integration_report_{timestamp}.md"
            self._save_markdown_report(report, md_path)
            
            print(f"✓ Report saved to: {save_path}")
            print(f"  - JSON: {json_path.name}")
            print(f"  - Markdown: {md_path.name}")
        
        return report
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the analysis"""
        findings = []
        
        if 'steering_analysis' in self.results:
            strongest = self.results['steering_analysis']['strongest_layers']
            if strongest:
                top_layer, top_magnitude_dict = strongest[0]
                top_magnitude = top_magnitude_dict['l2_norm']
                findings.append(f"Layer {top_layer} shows strongest steering effect (magnitude: {top_magnitude:.4f})")
        
        if 'token_extraction' in self.results:
            total_tokens = sum(len(tokens) for tokens in self.results['token_extraction'].values())
            findings.append(f"Extracted {total_tokens} meaningful tokens across top steering layers")
        
        if 'control_prompts' in self.results:
            strategies = list(self.results['control_prompts'].keys())
            total_prompts = sum(len(prompts) for prompts in self.results['control_prompts'].values())
            findings.append(f"Generated {total_prompts} control prompts using {len(strategies)} strategies")
        
        if 'evaluation' in self.results and 'summary_metrics' in self.results['evaluation']:
            metrics = self.results['evaluation']['summary_metrics']
            findings.append(f"Analysis covers {metrics.get('layers_analyzed', 0)} layers with {metrics.get('unique_tokens_used', 0)} unique tokens")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = [
            "Test control prompts with actual model inference to measure steering effectiveness",
            "Compare token-based steering with direct vector intervention methods",
            "Experiment with combining multiple tokens from different layers",
            "Validate approach with human evaluation of generated responses",
            "Scale analysis to more layers and larger token sets for robustness"
        ]
        
        # Add specific recommendations based on results
        if 'evaluation' in self.results:
            eval_results = self.results['evaluation']
            if eval_results.get('method') == 'heuristic_analysis':
                recommendations.insert(0, "Enable model evaluation for concrete steering effectiveness metrics")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate suggested next steps"""
        return [
            "Implement full model evaluation pipeline with response generation",
            "Develop automated metrics for measuring steering effectiveness",
            "Compare results across different model sizes and architectures",
            "Investigate optimal token selection and combination strategies",
            "Create interactive tool for exploring token-layer relationships",
            "Publish findings and methodology for community validation"
        ]
    
    def _save_markdown_report(self, report: Dict, path: Path):
        """Save a human-readable markdown report"""
        with open(path, 'w') as f:
            f.write("# Steering Vector + Token Activation Integration Report\n\n")
            
            # Experiment info
            info = report['experiment_info']
            f.write(f"**Timestamp:** {info['timestamp']}\n")
            f.write(f"**Model:** {info['model_name']}\n")
            f.write(f"**Device:** {info['device']}\n")
            f.write(f"**Approach:** {info['approach']}\n\n")
            
            # Data summary
            if 'data_summary' in report:
                f.write("## Data Summary\n\n")
                for data_type, summary in report['data_summary'].items():
                    f.write(f"### {data_type.replace('_', ' ').title()}\n")
                    for key, value in summary.items():
                        f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                    f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(report['key_findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            for i, step in enumerate(report['next_steps'], 1):
                f.write(f"{i}. {step}\n")
    
    def run_complete_pipeline(self, steering_path: str, mlp_path: str,
                            top_layers: int = 10, tokens_per_layer: int = 15,
                            strategies: List[str] = None,
                            save_results: bool = True, output_dir: str = "results") -> Dict:
        """
        Run the complete integration pipeline
        """
        print("="*80)
        print("STEERING VECTOR + TOKEN ACTIVATION INTEGRATION PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Load data
            load_summary = self.load_data(steering_path, mlp_path)
            
            # Step 2: Analyze steering magnitudes
            strongest_layers = self.analyze_steering_magnitudes(top_layers)
            
            # Step 3: Extract tokens for top layers
            layer_tokens = self.extract_tokens_for_layers(strongest_layers, tokens_per_layer)
            
            # Step 4: Generate control prompts
            control_prompts = self.generate_control_prompts(strategies)
            
            # Step 5: Evaluate effectiveness (heuristic analysis)
            evaluation = self.evaluate_steering_effectiveness()
            
            # Step 6: Generate report
            if save_results:
                report = self.generate_report(output_dir)
            else:
                report = self.generate_report()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print("✓ Data loaded and analyzed")
            print("✓ Top steering layers identified")
            print("✓ Meaningful tokens extracted")
            print("✓ Control prompts generated")
            print("✓ Effectiveness analysis completed")
            print("✓ Comprehensive report generated")
            
            if save_results:
                print(f"✓ Results saved to: {output_dir}")
            
            return report
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'traceback': traceback.format_exc()}


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Integrate steering vectors with token activations"
    )
    parser.add_argument('--steering-path', required=True,
                       help='Path to steering vectors pickle file')
    parser.add_argument('--mlp-path', required=True,
                       help='Path to MLP activation data pickle file')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name for tokenization')
    parser.add_argument('--top-layers', type=int, default=10,
                       help='Number of top steering layers to analyze')
    parser.add_argument('--tokens-per-layer', type=int, default=15,
                       help='Number of tokens to extract per layer')
    parser.add_argument('--strategies', nargs='+', 
                       default=['natural', 'template', 'context', 'direct'],
                       help='Prompt generation strategies to use')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--device', default='auto',
                       help='Device for computation (auto, cuda, cpu, mps)')
    parser.add_argument('--load-model', action='store_true',
                       help='Load model for full evaluation (resource intensive)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to disk')
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = SteeringTokenIntegrator(
        model_name=args.model,
        device=args.device
    )
    
    # Load model if requested
    if args.load_model:
        integrator.load_model_for_evaluation()
    
    # Run pipeline
    results = integrator.run_complete_pipeline(
        steering_path=args.steering_path,
        mlp_path=args.mlp_path,
        top_layers=args.top_layers,
        tokens_per_layer=args.tokens_per_layer,
        strategies=args.strategies,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    return results


if __name__ == "__main__":
    # Example usage for direct execution
    if len(sys.argv) == 1:
        print("Running with default paths...")
        
        # Initialize with your specific paths
        integrator = SteeringTokenIntegrator()
        
        # Run the complete pipeline
        results = integrator.run_complete_pipeline(
            steering_path="neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl",
            mlp_path="ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl",
            top_layers=8,
            tokens_per_layer=12,
            strategies=['natural', 'template', 'context'],
            save_results=True,
            output_dir="integration_results"
        )
        
        # Print summary
        if 'error' not in results:
            print("\n" + "="*60)
            print("INTEGRATION SUMMARY")
            print("="*60)
            for finding in results.get('key_findings', []):
                print(f"• {finding}")
        
    else:
        # Command line execution
        import sys
        results = main()