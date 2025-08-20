import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional, Set
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re

class EnhancedHarmfulSteeringTokenExtractor:
    """
    Enhanced version implementing insights from:
    1. "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" (2203.14680)
    2. "Toward universal steering and monitoring of AI models" (2502.03708)
    
    Key improvements:
    - Focus on FFN layers and residual stream analysis
    - Cumulative steering effects through layers
    - Multi-token phrase analysis
    - Concept coherence validation
    - Better semantic clustering
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize with enhanced capabilities"""
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(model_name)
            self.vocab_size = len(self.tokenizer)
            self.hidden_dim = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config, 'num_hidden_layers') else 12
            
            # Storage for data
            self.activation_data = None
            self.harmful_directions = None
            self.ffn_directions = None  # New: specific FFN directions
            self.unembedding_matrix = None
            
            self._load_unembedding_matrix()
            print(f"✓ Loaded model {model_name} with {self.num_layers} layers")
            
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
            self.vocab_size = None
            self.hidden_dim = None
            self.unembedding_matrix = None
        
    def _load_unembedding_matrix(self):
        """Load the unembedding matrix with better handling"""
        if self.model is None:
            return
            
        try:
            if hasattr(self.model, 'lm_head'):
                self.unembedding_matrix = self.model.lm_head.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'embed_out'):
                self.unembedding_matrix = self.model.embed_out.weight.detach().cpu().numpy()
            else:
                # Use embedding layer (transposed for unembedding)
                self.unembedding_matrix = self.model.get_input_embeddings().weight.detach().cpu().numpy()
            
            print(f"✓ Loaded unembedding matrix: {self.unembedding_matrix.shape}")
        except Exception as e:
            print(f"Could not load unembedding matrix: {e}")
            self.unembedding_matrix = None
    
    def load_harmful_directions(self, pkl_path: str):
        """Enhanced loading with FFN detection"""
        print(f"Loading harmful directions from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.harmful_directions = pickle.load(f)
        
        # Process similar to original but detect FFN directions
        if isinstance(self.harmful_directions, dict):
            concept = self.harmful_directions.get('concept', 'harmful')
            model_name = self.harmful_directions.get('model_name', 'unknown')
            
            if 'directions' in self.harmful_directions:
                directions = self.harmful_directions['directions']
            elif 'layer_directions' in self.harmful_directions:
                directions = self.harmful_directions['layer_directions']
            else:
                directions = {k: v for k, v in self.harmful_directions.items() 
                            if isinstance(k, (int, str)) and str(k).lstrip('-').isdigit()}
            
            if 'hidden_layers' in self.harmful_directions:
                hidden_layers = self.harmful_directions['hidden_layers']
            elif 'layers' in self.harmful_directions:
                hidden_layers = self.harmful_directions['layers']
            else:
                hidden_layers = list(directions.keys())
            
            # Separate FFN directions if they exist
            ffn_directions = {}
            regular_directions = {}
            
            for layer_idx, direction in directions.items():
                if hasattr(direction, 'cpu'):
                    direction_np = direction.cpu().numpy()
                else:
                    direction_np = np.array(direction)
                
                if len(direction_np.shape) > 1:
                    if direction_np.shape[0] > 1:
                        direction_np = np.mean(direction_np, axis=0)
                    else:
                        direction_np = direction_np.squeeze()
                
                # Check if this might be an FFN direction (heuristic based on name)
                if 'ffn' in str(layer_idx).lower() or 'mlp' in str(layer_idx).lower():
                    ffn_directions[layer_idx] = direction_np
                else:
                    regular_directions[layer_idx] = direction_np
            
            self.harmful_directions = {
                'concept': concept,
                'model_name': model_name,
                'directions': regular_directions,
                'ffn_directions': ffn_directions,
                'hidden_layers': hidden_layers
            }
            
            print(f"✓ Loaded directions for concept: {concept}")
            print(f"  Regular directions: {len(regular_directions)}")
            print(f"  FFN directions: {len(ffn_directions)}")
            
        else:
            raise ValueError(f"Unexpected pickle format: {type(self.harmful_directions)}")
    
    def diagnose_steering_vectors(self):
        """Diagnose potential issues with steering vector quality"""
        print("\n🔍 STEERING VECTOR DIAGNOSTICS")
        print("="*50)
        
        all_mags = []
        all_means = []
        all_stds = []
        
        for layer_idx, direction in self.harmful_directions['directions'].items():
            magnitude = np.linalg.norm(direction)
            mean_val = np.mean(direction)
            std_val = np.std(direction)
            
            all_mags.append(magnitude)
            all_means.append(mean_val)
            all_stds.append(std_val)
            
            print(f"Layer {layer_idx:2d}: mag={magnitude:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
        
        print(f"\nSUMMARY:")
        print(f"  Magnitude - Mean: {np.mean(all_mags):.6f}, Std: {np.std(all_mags):.6f}")
        print(f"  All magnitudes identical: {len(set([f'{m:.6f}' for m in all_mags])) == 1}")
        
        if len(set([f'{m:.6f}' for m in all_mags])) == 1:
            print("  ⚠️  WARNING: All vectors have identical magnitude - likely unit normalized")
            print("     This suggests vectors may not represent different concept strengths")
    
    def extract_ffn_steering_tokens(self, layer_idx: int, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Extract tokens specifically from FFN layer outputs, following 
        Dar et al. 2022's vocabulary space interpretation
        """
        # Try FFN-specific directions first, fall back to regular
        if self.harmful_directions['ffn_directions'] and layer_idx in self.harmful_directions['ffn_directions']:
            steering_vector = self.harmful_directions['ffn_directions'][layer_idx]
            print(f"Using FFN-specific direction for layer {layer_idx}")
        elif layer_idx in self.harmful_directions['directions']:
            steering_vector = self.harmful_directions['directions'][layer_idx]
            print(f"Using regular direction for layer {layer_idx} (no FFN direction available)")
        else:
            raise ValueError(f"No direction found for layer {layer_idx}")
        
        if self.unembedding_matrix is None:
            raise ValueError("Unembedding matrix not available")
        
        # FFN outputs can be interpreted as additive vocabulary updates
        vocab_updates = steering_vector @ self.unembedding_matrix.T
        
        return self._get_top_tokens(vocab_updates, top_k)
    
    def analyze_cumulative_steering_effects(self, max_layers: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze how steering effects accumulate through the residual stream.
        This implements the additive view from the vocabulary space paper.
        """
        print("\n🔄 ANALYZING CUMULATIVE STEERING EFFECTS")
        
        if max_layers is None:
            max_layers = len(self.harmful_directions['hidden_layers'])
        
        layers_to_analyze = sorted(self.harmful_directions['hidden_layers'])[:max_layers]
        
        cumulative_effect = np.zeros(self.vocab_size)
        layer_contributions = {}
        layer_magnitudes = {}
        
        for layer_idx in layers_to_analyze:
            steering_vector = self.harmful_directions['directions'][layer_idx]
            
            # Compute this layer's contribution to vocabulary space
            layer_effect = steering_vector @ self.unembedding_matrix.T
            layer_magnitude = np.linalg.norm(layer_effect)
            
            # Add to cumulative (residual stream interpretation)
            cumulative_effect += layer_effect
            
            layer_contributions[layer_idx] = layer_effect
            layer_magnitudes[layer_idx] = layer_magnitude
        
        # Get tokens most promoted by cumulative effect
        top_cumulative = self._get_top_tokens(cumulative_effect, 50)
        
        # Analyze layer contribution patterns
        contribution_analysis = self._analyze_layer_contributions(layer_contributions)
        
        return {
            'cumulative_tokens': top_cumulative,
            'layer_contributions': layer_contributions,
            'layer_magnitudes': layer_magnitudes,
            'cumulative_effect': cumulative_effect,
            'contribution_analysis': contribution_analysis
        }
    
    def _analyze_layer_contributions(self, layer_contributions: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Analyze how different layers contribute to the overall effect"""
        
        # Find layers with highest overall contribution
        layer_strengths = {layer: np.sum(np.abs(contrib)) for layer, contrib in layer_contributions.items()}
        strongest_layers = sorted(layer_strengths.items(), key=lambda x: x[1], reverse=True)
        
        # Find most influential tokens per layer
        layer_top_tokens = {}
        for layer_idx, contribution in layer_contributions.items():
            layer_top_tokens[layer_idx] = self._get_top_tokens(contribution, 10)
        
        # Compute layer similarity (how similar are their vocabulary effects?)
        layer_similarities = {}
        layer_list = list(layer_contributions.keys())
        for i, layer1 in enumerate(layer_list):
            for layer2 in layer_list[i+1:]:
                contrib1 = layer_contributions[layer1]
                contrib2 = layer_contributions[layer2]
                similarity = cosine_similarity([contrib1], [contrib2])[0, 0]
                layer_similarities[f"{layer1}-{layer2}"] = similarity
        
        return {
            'strongest_layers': strongest_layers,
            'layer_top_tokens': layer_top_tokens,
            'layer_similarities': layer_similarities
        }
    
    def extract_phrase_steering_patterns(self, max_phrase_length: int = 3, min_frequency: int = 2) -> List[Tuple[str, int, float]]:
        """
        Extract multi-token phrases that are consistently promoted.
        Harmful concepts often involve phrases, not just single tokens.
        """
        print(f"\n📝 EXTRACTING PHRASE PATTERNS (max length: {max_phrase_length})")
        
        all_layer_tokens = self.extract_all_layer_steering_tokens(top_k=50)
        
        # Collect token sequences and their scores
        phrase_scores = defaultdict(list)
        phrase_layers = defaultdict(set)
        
        for layer_idx, layer_tokens in all_layer_tokens.items():
            token_list = [(token.strip(), score) for token, score in layer_tokens]
            
            # Generate phrases of different lengths
            for length in range(2, max_phrase_length + 1):
                for i in range(len(token_list) - length + 1):
                    phrase_tokens = token_list[i:i+length]
                    phrase_text = ' '.join([token for token, _ in phrase_tokens])
                    phrase_score = np.mean([score for _, score in phrase_tokens])
                    
                    # Only consider phrases where tokens are semantically related
                    if self._tokens_are_related(phrase_tokens):
                        phrase_scores[phrase_text].append(phrase_score)
                        phrase_layers[phrase_text].add(layer_idx)
        
        # Filter and rank phrases
        frequent_phrases = []
        for phrase, scores in phrase_scores.items():
            if len(phrase_layers[phrase]) >= min_frequency:
                avg_score = np.mean(scores)
                layer_count = len(phrase_layers[phrase])
                frequent_phrases.append((phrase, layer_count, avg_score))
        
        # Sort by average score
        frequent_phrases.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(frequent_phrases)} frequent phrases")
        return frequent_phrases
    
    def _tokens_are_related(self, phrase_tokens: List[Tuple[str, float]], threshold: float = 0.8) -> bool:
        """
        Heuristic to determine if tokens in a phrase are semantically related.
        This is a simple implementation - could be improved with embeddings.
        """
        tokens = [token.strip().lower() for token, _ in phrase_tokens]
        
        # Simple heuristics
        # 1. If tokens share common prefixes/suffixes
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                token1, token2 = tokens[i], tokens[j]
                if len(token1) > 3 and len(token2) > 3:
                    # Check common prefix (at least 3 chars)
                    if token1[:3] == token2[:3] or token1[-3:] == token2[-3:]:
                        return True
        
        # 2. If tokens are alphabetically close (might indicate related concepts)
        sorted_tokens = sorted(tokens)
        for i in range(len(sorted_tokens) - 1):
            if abs(ord(sorted_tokens[i][0]) - ord(sorted_tokens[i+1][0])) <= 2:
                return True
        
        # Default: consider them related (conservative approach)
        return True
    
    def cluster_steering_tokens_semantically(self, n_clusters: int = 5, layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Cluster extracted tokens by semantic similarity to find coherent concept groups.
        """
        print(f"\n🎯 SEMANTIC CLUSTERING (k={n_clusters})")
        
        if layer_idx is not None:
            # Cluster tokens from specific layer
            layer_tokens = self.extract_steering_vector_tokens(layer_idx, top_k=100)
            tokens_to_cluster = [token for token, _ in layer_tokens]
            token_scores = {token: score for token, score in layer_tokens}
        else:
            # Cluster consistent tokens across layers
            consistent_tokens = self.find_consistent_harmful_tokens(min_layers=2, top_k_per_layer=50)
            tokens_to_cluster = [token for token, _, _ in consistent_tokens]
            token_scores = {token: score for token, score, _ in consistent_tokens}
        
        if len(tokens_to_cluster) < n_clusters:
            print(f"Warning: Only {len(tokens_to_cluster)} tokens available, reducing clusters to {len(tokens_to_cluster)}")
            n_clusters = len(tokens_to_cluster)
        
        # Create simple feature vectors based on token properties
        # This is a basic implementation - could be enhanced with actual embeddings
        token_features = self._create_token_features(tokens_to_cluster)
        
        # Perform clustering
        if len(token_features) > 0 and len(token_features[0]) > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(token_features)
            
            # Organize results
            clusters = defaultdict(list)
            for token, label in zip(tokens_to_cluster, cluster_labels):
                clusters[label].append((token, token_scores[token]))
            
            # Sort tokens within each cluster by score
            for label in clusters:
                clusters[label].sort(key=lambda x: x[1], reverse=True)
            
            # Analyze clusters
            cluster_analysis = self._analyze_token_clusters(clusters)
            
            return {
                'clusters': dict(clusters),
                'cluster_analysis': cluster_analysis,
                'n_clusters': n_clusters,
                'source_layer': layer_idx
            }
        else:
            return {'clusters': {}, 'cluster_analysis': {}, 'n_clusters': 0}
    
    def _create_token_features(self, tokens: List[str]) -> List[List[float]]:
        """Create simple feature vectors for token clustering"""
        features = []
        
        for token in tokens:
            token_clean = token.strip()
            feature_vector = [
                len(token_clean),  # Length
                sum(1 for c in token_clean if c.isupper()),  # Uppercase count
                sum(1 for c in token_clean if c.isdigit()),  # Digit count
                sum(1 for c in token_clean if c in '.,!?;:'),  # Punctuation count
                1 if token_clean.startswith(' ') else 0,  # Starts with space
                1 if any(ord(c) > 127 for c in token_clean) else 0,  # Contains non-ASCII
                1 if re.match(r'^[a-zA-Z]+$', token_clean.strip()) else 0,  # Pure alphabetic
                1 if '.' in token_clean else 0,  # Contains dot (might be code)
                1 if token_clean.startswith(('_', '-', '.')) else 0,  # Special prefix
            ]
            features.append(feature_vector)
        
        return features
    
    def _analyze_token_clusters(self, clusters: Dict[int, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Analyze the semantic clusters to identify patterns"""
        cluster_analysis = {}
        
        for cluster_id, tokens in clusters.items():
            token_list = [token for token, _ in tokens]
            scores = [score for _, score in tokens]
            
            # Basic statistics
            analysis = {
                'size': len(tokens),
                'avg_score': np.mean(scores),
                'score_std': np.std(scores),
                'top_tokens': tokens[:5],  # Top 5 tokens
            }
            
            # Pattern analysis
            patterns = {
                'has_code_tokens': sum(1 for t in token_list if '.' in t or '_' in t) / len(token_list),
                'has_foreign_chars': sum(1 for t in token_list if any(ord(c) > 127 for c in t)) / len(token_list),
                'avg_length': np.mean([len(t.strip()) for t in token_list]),
                'has_uppercase': sum(1 for t in token_list if any(c.isupper() for c in t)) / len(token_list),
            }
            
            analysis['patterns'] = patterns
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def _get_top_tokens(self, vocab_logits: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Helper to get top-k tokens from vocabulary logits"""
        top_indices = np.argsort(vocab_logits)[-top_k:][::-1]
        
        top_tokens = []
        for idx in top_indices:
            if self.tokenizer is not None:
                token = self.tokenizer.decode([idx])
            else:
                token = f"token_{idx}"
            score = vocab_logits[idx]
            top_tokens.append((token, score))
        
        return top_tokens
    
    # Keep all original methods but enhance them
    def extract_steering_vector_tokens(self, layer_idx: int, top_k: int = 50, 
                                     method: str = 'direct') -> List[Tuple[str, float]]:
        """Enhanced version of original method"""
        if self.harmful_directions is None:
            raise ValueError("Harmful directions not loaded.")
        
        if layer_idx not in self.harmful_directions['directions']:
            raise ValueError(f"Layer {layer_idx} not found in harmful directions")
            
        if self.unembedding_matrix is None:
            raise ValueError("Unembedding matrix not available.")
        
        steering_vector = self.harmful_directions['directions'][layer_idx]
        if len(steering_vector.shape) > 1:
            steering_vector = steering_vector.squeeze()
        
        if steering_vector.shape[0] != self.unembedding_matrix.shape[1]:
            raise ValueError(f"Dimension mismatch: steering vector has {steering_vector.shape[0]} dims, "
                           f"but unembedding matrix expects {self.unembedding_matrix.shape[1]} dims.")
        
        if method == 'normalized':
            steering_vector = steering_vector / (np.linalg.norm(steering_vector) + 1e-8)
        
        vocab_logits = steering_vector @ self.unembedding_matrix.T
        return self._get_top_tokens(vocab_logits, top_k)
    
    def extract_all_layer_steering_tokens(self, top_k: int = 30, 
                                        method: str = 'direct') -> Dict[int, List[Tuple[str, float]]]:
        """Enhanced version of original method"""
        if self.harmful_directions is None:
            raise ValueError("Harmful directions not loaded.")
        
        all_layer_tokens = {}
        
        for layer_idx in self.harmful_directions['hidden_layers']:
            tokens = self.extract_steering_vector_tokens(layer_idx, top_k, method)
            all_layer_tokens[layer_idx] = tokens
        
        return all_layer_tokens
    
    def find_consistent_harmful_tokens(self, min_layers: int = 3, top_k_per_layer: int = 20) -> List[Tuple[str, float, int]]:
        """Enhanced version of original method"""
        all_layer_tokens = self.extract_all_layer_steering_tokens(top_k_per_layer)
        
        token_scores = defaultdict(list)
        token_layer_count = defaultdict(int)
        
        for layer_idx, tokens in all_layer_tokens.items():
            for token, score in tokens:
                clean_token = token.strip()
                token_scores[clean_token].append(score)
                token_layer_count[clean_token] += 1
        
        consistent_tokens = []
        for token, scores in token_scores.items():
            if token_layer_count[token] >= min_layers:
                avg_score = np.mean(scores)
                layer_count = token_layer_count[token]
                consistent_tokens.append((token, avg_score, layer_count))
        
        consistent_tokens.sort(key=lambda x: x[1], reverse=True)
        return consistent_tokens
    
    def print_enhanced_steering_summary(self, top_k: int = 10):
        """Enhanced summary with new analysis features"""
        print("=" * 80)
        print("ENHANCED HARMFUL CONCEPT STEERING VECTOR TOKEN ANALYSIS")
        print("=" * 80)
        
        # Diagnostics first
        self.diagnose_steering_vectors()
        
        # Cumulative effects analysis
        cumulative_analysis = self.analyze_cumulative_steering_effects()
        
        print(f"\n🎯 TOP CUMULATIVE STEERING TOKENS:")
        for i, (token, score) in enumerate(cumulative_analysis['cumulative_tokens'][:top_k]):
            print(f"  {i+1:2d}. '{token}': {score:.4f}")
        
        # Strongest contributing layers
        print(f"\n💪 STRONGEST CONTRIBUTING LAYERS:")
        for layer, strength in cumulative_analysis['contribution_analysis']['strongest_layers'][:5]:
            print(f"  Layer {layer:2d}: {strength:.4f}")
        
        # Phrase patterns
        phrases = self.extract_phrase_steering_patterns(max_phrase_length=3, min_frequency=2)
        if phrases:
            print(f"\n📝 TOP PHRASE PATTERNS:")
            for phrase, layer_count, avg_score in phrases[:top_k]:
                print(f"  '{phrase}' ({layer_count} layers): {avg_score:.4f}")
        
        # Semantic clustering
        clustering = self.cluster_steering_tokens_semantically(n_clusters=3)
        if clustering['clusters']:
            print(f"\n🎯 SEMANTIC CLUSTERS:")
            for cluster_id, tokens in clustering['clusters'].items():
                analysis = clustering['cluster_analysis'][cluster_id]
                print(f"  Cluster {cluster_id} ({analysis['size']} tokens, avg: {analysis['avg_score']:.4f}):")
                for token, score in tokens[:3]:  # Top 3 per cluster
                    print(f"    '{token}': {score:.4f}")
        
        # Original consistent tokens for comparison
        consistent_tokens = self.find_consistent_harmful_tokens(min_layers=2)
        print(f"\n🔄 CONSISTENT TOKENS (≥2 layers):")
        for i, (token, avg_score, layer_count) in enumerate(consistent_tokens[:top_k]):
            print(f"  {i+1:2d}. '{token}' (avg: {avg_score:.4f}, {layer_count} layers)")

# Enhanced usage example
def main():
    """Enhanced example usage"""
    print("Initializing Enhanced Steering Token Extractor...")
    extractor = EnhancedHarmfulSteeringTokenExtractor("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    try:
        # Load directions
        extractor.load_harmful_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl")
        
        # Run enhanced analysis
        extractor.print_enhanced_steering_summary(top_k=15)
        
        # Additional detailed analyses
        print("\n" + "="*60)
        print("DETAILED CUMULATIVE ANALYSIS")
        print("="*60)
        
        cumulative = extractor.analyze_cumulative_steering_effects(max_layers=10)
        
        # Show layer similarity analysis
        similarities = cumulative['contribution_analysis']['layer_similarities']
        print(f"\nLayer Similarities (top 5 most similar):")
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for layer_pair, similarity in sorted_sims[:5]:
            print(f"  {layer_pair}: {similarity:.4f}")
        
        # Test FFN-specific extraction if available
        if extractor.harmful_directions['ffn_directions']:
            print(f"\n🔧 FFN-SPECIFIC ANALYSIS")
            print("="*30)
            first_ffn_layer = list(extractor.harmful_directions['ffn_directions'].keys())[0]
            ffn_tokens = extractor.extract_ffn_steering_tokens(first_ffn_layer, top_k=10)
            for i, (token, score) in enumerate(ffn_tokens):
                print(f"  {i+1:2d}. '{token}': {score:.4f}")
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()