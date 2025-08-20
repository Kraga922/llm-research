import pickle

# # Replace with your actual file path
# pkl_path = "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl"
pkl_path = "/home/ubuntu/krishiv-llm/neural_controllers/directions/logistic_prose_llama_3_8b_it.pkl"
# pkl_path = "neural_controllers/directions/rfm_harmful_llama_3_8b_it_detector.pkl"
# pkl_path = "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl"
# # pkl_path = "ffn-values/pickle/model_df_10k.pkl"



# # Load the pickle file
# with open(pkl_path, "rb") as f:
#     data = pickle.load(f)

# # Display the top-level structure
# print("Type of loaded object:", type(data))

# # Optionally inspect the structure
# if isinstance(data, dict):
#     print("Top-level keys:", list(data.keys()))
# elif isinstance(data, list):
#     print("Length of list:", len(data))
#     print("First item type:", type(data[0]))
#     print("First item content (truncated):", data[0])
# else:
#     print("Data preview:", data)


with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, dict):
    print(data.keys())
    print(type(data[list(data.keys())[0]]))
elif isinstance(data, list):
    print(type(data[0]))
    print(data[0])


############IGNORE BELOW THIS LINE - OLD CODE FOR REFERENCE############
# import pickle
# import pandas as pd
# import torch
# import numpy as np
# from transformers import AutoTokenizer
# import matplotlib.pyplot as plt
# import seaborn as sns

# class DataDiagnostics:
#     def __init__(self):
#         self.steering_vectors = None
#         self.mlp_df = None
#         self.tokenizer = None
        
#     def load_data(self, steering_path, mlp_path):
#         """Load both pickle files and analyze their structure"""
#         print("=== LOADING DATA ===")
        
#         # Load steering vectors
#         print(f"Loading steering vectors from: {steering_path}")
#         with open(steering_path, 'rb') as f:
#             self.steering_vectors = pickle.load(f)
        
#         print(f"Steering vectors type: {type(self.steering_vectors)}")
#         print(f"Steering vector keys: {list(self.steering_vectors.keys())}")
        
#         if isinstance(self.steering_vectors, dict):
#             sample_key = list(self.steering_vectors.keys())[0]
#             sample_vector = self.steering_vectors[sample_key]
#             print(f"Sample vector shape: {sample_vector.shape}")
#             print(f"Sample vector type: {type(sample_vector)}")
        
#         # Load MLP DataFrame
#         print(f"\nLoading MLP data from: {mlp_path}")
#         with open(mlp_path, 'rb') as f:
#             self.mlp_df = pickle.load(f)
        
#         print(f"MLP data type: {type(self.mlp_df)}")
#         print(f"DataFrame shape: {self.mlp_df.shape}")
#         print(f"DataFrame columns: {list(self.mlp_df.columns)}")
        
#         return self.steering_vectors, self.mlp_df
    
#     def analyze_steering_vectors(self):
#         """Analyze steering vectors to find strongest harmful layers"""
#         print("\n=== ANALYZING STEERING VECTORS ===")
        
#         if self.steering_vectors is None:
#             print("Please load data first!")
#             return
        
#         # Calculate vector magnitudes for each layer
#         layer_magnitudes = {}
#         for layer, vector in self.steering_vectors.items():
#             magnitude = torch.norm(vector).item()
#             layer_magnitudes[layer] = magnitude
            
#         # Sort by magnitude (strongest first)
#         sorted_layers = sorted(layer_magnitudes.items(), key=lambda x: x[1], reverse=True)
        
#         print("Top 10 layers by steering vector magnitude:")
#         for i, (layer, magnitude) in enumerate(sorted_layers[:10]):
#             print(f"{i+1}. Layer {layer}: magnitude = {magnitude:.4f}")
        
#         # Plot magnitudes
#         layers = list(layer_magnitudes.keys())
#         magnitudes = list(layer_magnitudes.values())
        
#         plt.figure(figsize=(12, 6))
#         plt.plot(layers, magnitudes, 'bo-')
#         plt.xlabel('Layer Index')
#         plt.ylabel('Steering Vector Magnitude')
#         plt.title('Harmful Steering Vector Magnitudes by Layer')
#         plt.grid(True)
#         plt.show()
        
#         return sorted_layers
    
#     def analyze_mlp_data(self):
#         """Analyze MLP DataFrame structure"""
#         print("\n=== ANALYZING MLP DATA ===")
        
#         if self.mlp_df is None:
#             print("Please load data first!")
#             return
        
#         # Show sample rows
#         print("Sample rows:")
#         print(self.mlp_df.head(3))
        
#         # Analyze columns
#         for col in self.mlp_df.columns:
#             print(f"\nColumn '{col}':")
#             sample_value = self.mlp_df[col].iloc[0]
#             print(f"  Sample value type: {type(sample_value)}")
#             print(f"  Sample value: {str(sample_value)[:200]}...")
            
#             if isinstance(sample_value, list):
#                 print(f"  List length: {len(sample_value)}")
#                 if len(sample_value) > 0:
#                     print(f"  First element type: {type(sample_value[0])}")
        
#         # Check if there's layer information
#         potential_layer_cols = [col for col in self.mlp_df.columns if 'layer' in col.lower()]
#         print(f"\nPotential layer columns: {potential_layer_cols}")
        
#         return self.mlp_df.dtypes
    
#     def test_tokenizer_compatibility(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
#         """Test if we can load the appropriate tokenizer"""
#         print(f"\n=== TESTING TOKENIZER: {model_name} ===")
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             print("✓ Tokenizer loaded successfully")
            
#             # Test token decoding with sample from MLP data
#             if self.mlp_df is not None:
#                 sample_tokens = self.mlp_df['top_coef_idx'].iloc[0]
#                 if isinstance(sample_tokens, list) and len(sample_tokens) > 0:
#                     print(f"\nTesting token decoding:")
#                     print(f"Sample token indices: {sample_tokens[:10]}")
                    
#                     decoded_tokens = []
#                     for token_id in sample_tokens[:10]:
#                         try:
#                             token = self.tokenizer.decode([token_id])
#                             decoded_tokens.append(f"{token_id}->'{token}'")
#                         except:
#                             decoded_tokens.append(f"{token_id}->'ERROR'")
                    
#                     print(f"Decoded tokens: {decoded_tokens}")
            
#             return True
            
#         except Exception as e:
#             print(f"✗ Failed to load tokenizer: {e}")
#             return False
    
#     def check_layer_alignment(self):
#         """Check how to align layers between steering vectors and MLP data"""
#         print("\n=== CHECKING LAYER ALIGNMENT ===")
        
#         if self.steering_vectors is None or self.mlp_df is None:
#             print("Please load both datasets first!")
#             return
        
#         steering_layers = list(self.steering_vectors.keys())
#         print(f"Steering vector layers: {steering_layers}")
#         print(f"Steering layers range: {min(steering_layers)} to {max(steering_layers)}")
        
#         # Check if MLP data has layer information
#         if 'layer' in self.mlp_df.columns:
#             mlp_layers = self.mlp_df['layer'].unique()
#             print(f"MLP layers: {sorted(mlp_layers)}")
#         else:
#             print("No explicit layer column found in MLP data")
            
#             # Check if layer info is embedded in other columns
#             for col in self.mlp_df.columns:
#                 if 'layer' in col.lower():
#                     sample = self.mlp_df[col].iloc[0]
#                     if isinstance(sample, list):
#                         print(f"Column '{col}' might contain layer info - sample length: {len(sample)}")
    
#     def generate_diagnostic_report(self, steering_path, mlp_path):
#         """Generate a complete diagnostic report"""
#         print("=" * 60)
#         print("DIAGNOSTIC REPORT FOR STEERING + MLP COMBINATION")
#         print("=" * 60)
        
#         # Load and analyze data
#         self.load_data(steering_path, mlp_path)
        
#         # Analyze components
#         strongest_layers = self.analyze_steering_vectors()
#         mlp_info = self.analyze_mlp_data()
#         tokenizer_ok = self.test_tokenizer_compatibility()
#         self.check_layer_alignment()
        
#         print("\n" + "=" * 60)
#         print("SUMMARY AND RECOMMENDATIONS")
#         print("=" * 60)
        
#         print(f"✓ Steering vectors loaded: {len(self.steering_vectors)} layers")
#         print(f"✓ MLP data loaded: {len(self.mlp_df)} samples")
#         print(f"{'✓' if tokenizer_ok else '✗'} Tokenizer compatibility")
        
#         if strongest_layers:
#             top_3_layers = [layer for layer, _ in strongest_layers[:3]]
#             print(f"\nTop 3 strongest harmful layers: {top_3_layers}")
        
#         return {
#             'steering_vectors': self.steering_vectors,
#             'mlp_df': self.mlp_df,
#             'tokenizer': self.tokenizer,
#             'strongest_layers': strongest_layers if 'strongest_layers' in locals() else None
#         }

# # Usage example:
# def run_diagnostics():
#     """Run diagnostic tests on your data"""
#     diagnostics = DataDiagnostics()
    
#     # Update these paths to your actual file locations
#     steering_path = "neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl"
#     mlp_path = "ffn-values/pickle/model_df_llama3_1_8b_instruct_1000.pkl"
    
#     results = diagnostics.generate_diagnostic_report(steering_path, mlp_path)
#     return results

# if __name__ == "__main__":
#     # Run the diagnostic tests
#     results = run_diagnostics()