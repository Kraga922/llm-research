import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re

class SteeringTokenExtractor:
    """
    Extract the most effective tokens for steering using comprehensive filtering.
    Combines multiple metrics to identify tokens that will be most effective for
    manual prompt steering rather than just highest raw scores.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the extractor with a specific model"""
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(model_name)
            self.vocab_size = len(self.tokenizer)
            self.hidden_dim = self.model.config.hidden_size
            
            self.activation_data = None
            self.steering_directions = None
            self.unembedding_matrix = None
            
            # Cache for expensive computations
            self._all_layer_tokens_cache = None
            self._effectiveness_cache = {}
            
            self._load_unembedding_matrix()
            
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
            self.vocab_size = None
            self.hidden_dim = None
            self.unembedding_matrix = None
        
    def _load_unembedding_matrix(self):
        """Load the unembedding matrix (W_U) for projecting to vocabulary space"""
        if self.model is None:
            return
            
        try:
            if hasattr(self.model, 'lm_head'):
                self.unembedding_matrix = self.model.lm_head.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'embed_out'):
                self.unembedding_matrix = self.model.embed_out.weight.detach().cpu().numpy()
            else:
                self.unembedding_matrix = self.model.get_input_embeddings().weight.detach().cpu().numpy()
            
            print(f"Loaded unembedding matrix: {self.unembedding_matrix.shape}")
        except Exception as e:
            print(f"Could not load unembedding matrix: {e}")
            self.unembedding_matrix = None
        
    def load_steering_directions(self, pkl_path: str):
        """Load steering directions from pickle file (generalized from poetry-specific)"""
        print(f"Loading steering directions from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.steering_directions = pickle.load(f)
        
        print(f"Pickle file structure: {type(self.steering_directions)}")
        
        if isinstance(self.steering_directions, dict):
            concept = self.steering_directions.get('concept', 'unknown_concept')
            model_name = self.steering_directions.get('model_name', 'unknown')
            
            if 'directions' in self.steering_directions:
                directions = self.steering_directions['directions']
            elif 'layer_directions' in self.steering_directions:
                directions = self.steering_directions['layer_directions']
            else:
                directions = {k: v for k, v in self.steering_directions.items() 
                            if isinstance(k, (int, str)) and str(k).lstrip('-').isdigit()}
            
            if 'hidden_layers' in self.steering_directions:
                hidden_layers = self.steering_directions['hidden_layers']
            elif 'layers' in self.steering_directions:
                hidden_layers = self.steering_directions['layers']
            else:
                hidden_layers = list(directions.keys()) if directions else []
            
            self.steering_directions = {
                'concept': concept,
                'model_name': model_name,
                'directions': directions,
                'hidden_layers': hidden_layers
            }
            
            print(f"Loaded steering directions for concept: {concept}")
            print(f"Model: {model_name}")
            print(f"Hidden layers: {hidden_layers}")
            
            # Process direction vectors
            processed_directions = {}
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
                
                processed_directions[layer_idx] = direction_np
            
            self.steering_directions['directions'] = processed_directions
        else:
            raise ValueError(f"Unexpected pickle format: {type(self.steering_directions)}")
    
    def _calculate_token_quality_score(self, token: str) -> float:
        """
        Calculate token quality based on linguistic and practical considerations.
        Higher scores indicate better steering potential.
        """
        clean_token = token.strip()
        
        # Filter out clearly bad tokens
        if len(clean_token) == 0:
            return 0.0
        
        score = 1.0
        
        # Length penalty for very short tokens (often function words)
        if len(clean_token) <= 1:
            score *= 0.1
        elif len(clean_token) == 2:
            score *= 0.3
        
        # Penalize common function words heavily
        function_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'a', 'an', 'is', 'are',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        if clean_token.lower() in function_words:
            score *= 0.1
        
        # Penalize pure punctuation
        if re.match(r'^[^\w\s]+$', clean_token):
            score *= 0.2
        
        # Penalize pure numbers
        if clean_token.isdigit():
            score *= 0.3
        
        # Penalize tokens that are mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in clean_token) / len(clean_token)
        if alpha_ratio < 0.5:
            score *= 0.5
        
        # Bonus for tokens that look like content words
        if len(clean_token) >= 4 and alpha_ratio > 0.8:
            score *= 1.2
        
        # Penalize tokens with weird characters or encoding issues
        if any(ord(c) > 127 for c in clean_token):
            score *= 0.7
        
        return score
    
    def _calculate_consistency_score(self, token: str, all_layer_tokens: Dict) -> float:
        """Calculate how consistently a token appears across layers"""
        clean_token = token.strip()
        appearances = 0
        total_score = 0.0
        
        for layer_tokens in all_layer_tokens.values():
            for t, score in layer_tokens:
                if t.strip() == clean_token:
                    appearances += 1
                    total_score += score
                    break
        
        if appearances == 0:
            return 0.0
        
        # Consistency score: (appearances / total_layers) * average_score
        consistency = appearances / len(all_layer_tokens)
        avg_score = total_score / appearances
        
        return consistency * avg_score
    
    def _calculate_signal_to_noise_ratio(self, token: str, layer_idx: int) -> float:
        """Calculate signal-to-noise ratio for a token in a specific layer"""
        # Get promoted and suppressed tokens for this layer
        promoted_tokens = self.extract_steering_vector_tokens(layer_idx, top_k=200)
        suppressed_tokens = self.get_antipodal_tokens(layer_idx, top_k=200)
        
        # Find this token's scores
        promoted_score = 0.0
        suppressed_score = 0.001  # Small default to avoid division by zero
        
        clean_token = token.strip()
        
        for t, score in promoted_tokens:
            if t.strip() == clean_token:
                promoted_score = score
                break
        
        for t, score in suppressed_tokens:
            if t.strip() == clean_token:
                suppressed_score = abs(score)
                break
        
        return promoted_score / suppressed_score
    
    def _calculate_concept_specificity(self, token: str, all_layer_tokens: Dict) -> float:
        """
        Calculate how specific a token is to the concept vs being generally high-scoring.
        This is a proxy for avoiding generic "good" tokens.
        """
        clean_token = token.strip()
        
        # Find the token's average rank across layers (lower rank = more specific)
        ranks = []
        scores = []
        
        for layer_tokens in all_layer_tokens.values():
            for rank, (t, score) in enumerate(layer_tokens, 1):
                if t.strip() == clean_token:
                    ranks.append(rank)
                    scores.append(score)
                    break
        
        if not ranks:
            return 0.0
        
        # Tokens that consistently rank high (low rank numbers) are more concept-specific
        avg_rank = np.mean(ranks)
        rank_consistency = 1.0 / (np.std(ranks) + 1.0)  # Lower std = more consistent
        
        # Combine: prefer low average rank with high consistency
        specificity = (50.0 / (avg_rank + 1.0)) * rank_consistency
        
        return min(specificity, 1.0)  # Cap at 1.0
    
    def extract_steering_vector_tokens(self, layer_idx: int, top_k: int = 50, 
                                     method: str = 'direct') -> List[Tuple[str, float]]:
        """Extract tokens promoted by steering vector at specific layer"""
        if self.steering_directions is None:
            raise ValueError("Steering directions not loaded.")
        
        if layer_idx not in self.steering_directions['directions']:
            raise ValueError(f"Layer {layer_idx} not found in steering directions")
            
        if self.unembedding_matrix is None:
            raise ValueError("Unembedding matrix not available.")
        
        steering_vector = self.steering_directions['directions'][layer_idx]
        if len(steering_vector.shape) > 1:
            steering_vector = steering_vector.squeeze()
        
        if steering_vector.shape[0] != self.unembedding_matrix.shape[1]:
            raise ValueError(f"Dimension mismatch: steering vector has {steering_vector.shape[0]} dims, "
                           f"but unembedding matrix expects {self.unembedding_matrix.shape[1]} dims.")
        
        if method == 'normalized':
            steering_vector = steering_vector / (np.linalg.norm(steering_vector) + 1e-8)
        
        vocab_logits = steering_vector @ self.unembedding_matrix.T
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
    
    def get_antipodal_tokens(self, layer_idx: int, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get tokens most suppressed by the steering vector"""
        if layer_idx not in self.steering_directions['directions']:
            raise ValueError(f"Layer {layer_idx} not found")
        
        steering_vector = self.steering_directions['directions'][layer_idx]
        if len(steering_vector.shape) > 1:
            steering_vector = steering_vector.squeeze()
        
        vocab_logits = steering_vector @ self.unembedding_matrix.T
        bottom_indices = np.argsort(vocab_logits)[:top_k]
        
        antipodal_tokens = []
        for idx in bottom_indices:
            token = self.tokenizer.decode([idx])
            score = vocab_logits[idx]
            antipodal_tokens.append((token, score))
        
        return antipodal_tokens
    
    def extract_all_layer_steering_tokens(self, top_k: int = 100) -> Dict[int, List[Tuple[str, float]]]:
        """Extract steering tokens for all layers (cached for efficiency)"""
        if self._all_layer_tokens_cache is not None:
            return self._all_layer_tokens_cache
        
        if self.steering_directions is None:
            raise ValueError("Steering directions not loaded.")
        
        all_layer_tokens = {}
        for layer_idx in self.steering_directions['hidden_layers']:
            tokens = self.extract_steering_vector_tokens(layer_idx, top_k)
            all_layer_tokens[layer_idx] = tokens
        
        self._all_layer_tokens_cache = all_layer_tokens
        return all_layer_tokens
    
    def calculate_comprehensive_effectiveness_score(self, token: str) -> float:
        """
        Calculate comprehensive effectiveness score combining multiple metrics.
        This is the key method for identifying the best steering tokens.
        """
        if token in self._effectiveness_cache:
            return self._effectiveness_cache[token]
        
        all_layer_tokens = self.extract_all_layer_steering_tokens()
        
        # 1. Token Quality Score (linguistic/practical considerations)
        quality_score = self._calculate_token_quality_score(token)
        if quality_score < 0.1:  # Early exit for clearly bad tokens
            self._effectiveness_cache[token] = 0.0
            return 0.0
        
        # 2. Consistency Score (appears across multiple layers)
        consistency_score = self._calculate_consistency_score(token, all_layer_tokens)
        
        # 3. Concept Specificity (high rank in concept, not just generally high)
        specificity_score = self._calculate_concept_specificity(token, all_layer_tokens)
        
        # 4. Signal-to-Noise Ratio (average across layers where token appears)
        snr_scores = []
        for layer_idx in self.steering_directions['hidden_layers']:
            # Only calculate SNR for layers where token appears in top results
            layer_tokens = all_layer_tokens[layer_idx]
            if any(t.strip() == token.strip() for t, _ in layer_tokens[:50]):
                snr = self._calculate_signal_to_noise_ratio(token, layer_idx)
                snr_scores.append(snr)
        
        avg_snr = np.mean(snr_scores) if snr_scores else 0.0
        normalized_snr = min(avg_snr / 10.0, 1.0)  # Normalize and cap
        
        # Combine all scores with weights
        # These weights prioritize consistency and quality over raw scores
        final_score = (
            quality_score * 0.3 +        # Must be a reasonable token
            consistency_score * 0.35 +   # Must appear across layers  
            specificity_score * 0.2 +    # Must be concept-specific
            normalized_snr * 0.15        # Must have good signal-to-noise
        )
        
        self._effectiveness_cache[token] = final_score
        return final_score
    
    def get_top_steering_tokens(self, top_k: int = 15, min_layers: int = 2) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Get the absolute best tokens for steering with detailed breakdown.
        
        Args:
            top_k: Number of top tokens to return
            min_layers: Minimum layers a token must appear in
            
        Returns:
            List of (token, effectiveness_score, score_breakdown) tuples
        """
        print(f"🔍 Analyzing tokens for steering effectiveness...")
        
        all_layer_tokens = self.extract_all_layer_steering_tokens()
        
        # Collect all candidate tokens that appear in at least min_layers
        token_layer_count = defaultdict(int)
        candidate_tokens = set()
        
        for layer_tokens in all_layer_tokens.values():
            for token, score in layer_tokens[:100]:  # Consider top 100 per layer
                clean_token = token.strip()
                token_layer_count[clean_token] += 1
                if token_layer_count[clean_token] >= min_layers:
                    candidate_tokens.add(clean_token)
        
        print(f"Found {len(candidate_tokens)} candidate tokens appearing in ≥{min_layers} layers")
        
        # Calculate effectiveness scores
        token_scores = []
        for i, token in enumerate(candidate_tokens):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(candidate_tokens)} tokens...")
            
            effectiveness = self.calculate_comprehensive_effectiveness_score(token)
            
            if effectiveness > 0.1:  # Only keep reasonably effective tokens
                # Get detailed breakdown
                quality = self._calculate_token_quality_score(token)
                consistency = self._calculate_consistency_score(token, all_layer_tokens)
                specificity = self._calculate_concept_specificity(token, all_layer_tokens)
                
                breakdown = {
                    'quality': quality,
                    'consistency': consistency,
                    'specificity': specificity,
                    'layer_count': token_layer_count[token]
                }
                
                token_scores.append((token, effectiveness, breakdown))
        
        # Sort by effectiveness and return top-k
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Identified {len(token_scores)} effective tokens, returning top {top_k}")
        
        return token_scores[:top_k]
    
    def print_steering_token_analysis(self, top_k: int = 15):
        """Print comprehensive analysis of top steering tokens"""
        print("=" * 90)
        print(f"{self.steering_directions['concept'].upper()} CONCEPT - TOP STEERING TOKENS")
        print("=" * 90)
        
        top_tokens = self.get_top_steering_tokens(top_k)
        
        print(f"\n🎯 TOP {top_k} TOKENS FOR MANUAL STEERING:")
        print(f"{'Rank':<4} {'Token':<20} {'Score':<8} {'Quality':<8} {'Consistency':<12} {'Specificity':<12} {'Layers':<6}")
        print("-" * 90)
        
        for i, (token, score, breakdown) in enumerate(top_tokens, 1):
            print(f"{i:<4} '{token}'<{20-len(token)-2} {score:<8.3f} "
                  f"{breakdown['quality']:<8.3f} {breakdown['consistency']:<12.3f} "
                  f"{breakdown['specificity']:<12.3f} {breakdown['layer_count']:<6}")
        
        # Extract just the tokens for easy copy-paste
        print(f"\n📋 COPY-PASTE TOKEN LIST (for prompt steering):")
        token_list = [token for token, _, _ in top_tokens]
        print("Tokens:", ", ".join(f"'{token}'" for token in token_list))
        
        # Show concept vs anti-concept comparison
        print(f"\n🔄 CONCEPT vs ANTI-CONCEPT COMPARISON (Layer {self.steering_directions['hidden_layers'][0]}):")
        first_layer = self.steering_directions['hidden_layers'][0]
        promoted = self.extract_steering_vector_tokens(first_layer, top_k=10)
        suppressed = self.get_antipodal_tokens(first_layer, top_k=10)
        
        print(f"{'Promoted (' + self.steering_directions['concept'] + ')':<30} | {'Suppressed (opposite)':<30}")
        print("-" * 63)
        for (p_token, p_score), (s_token, s_score) in zip(promoted, suppressed):
            print(f"'{p_token}' ({p_score:.3f}){' ' * (28-len(p_token)-len(f'{p_score:.3f}'))} | "
                  f"'{s_token}' ({s_score:.3f})")
        
        return top_tokens
    
    def export_steering_analysis(self, output_path: str, top_k: int = 50):
        """Export detailed steering analysis to CSV"""
        top_tokens = self.get_top_steering_tokens(top_k)
        all_layer_tokens = self.extract_all_layer_steering_tokens()
        
        rows = []
        for token, effectiveness, breakdown in top_tokens:
            # Get per-layer details
            layer_details = {}
            for layer_idx, layer_tokens in all_layer_tokens.items():
                for t, score in layer_tokens:
                    if t.strip() == token:
                        layer_details[f'layer_{layer_idx}_score'] = score
                        layer_details[f'layer_{layer_idx}_rank'] = layer_tokens.index((t, score)) + 1
                        break
            
            row = {
                'token': token,
                'effectiveness_score': effectiveness,
                'quality_score': breakdown['quality'],
                'consistency_score': breakdown['consistency'],
                'specificity_score': breakdown['specificity'],
                'layer_count': breakdown['layer_count'],
                'concept': self.steering_directions['concept'],
                **layer_details
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"📊 Exported detailed analysis to {output_path}")
        return df

# Updated main function
def main():
    """Example usage focusing on getting the best steering tokens"""
    print("🚀 Initializing Steering Token Extractor...")
    
    # Initialize with your model
    extractor = SteeringTokenExtractor("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    try:
        # Load steering directions
        # extractor.load_steering_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl")
        # extractor.load_steering_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/logistic_prose_llama_3_8b_it.pkl")
        extractor.load_steering_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/rfm_python_javascript_llama_3_8b_it.pkl")
        
        
        # Get the absolute best tokens for steering
        print("\n" + "="*60)
        print("EXTRACTING OPTIMAL STEERING TOKENS")
        print("="*60)
        
        top_tokens = extractor.print_steering_token_analysis(top_k=15)
        
        # Export detailed analysis
        df = extractor.export_steering_analysis("steering_tokens_analysis_programing1.csv", top_k=30)
        
        # Quick access to just the token names
        best_tokens = [token for token, _, _ in top_tokens[:10]]
        print(f"\n🎯 YOUR TOP 10 STEERING TOKENS:")
        for i, token in enumerate(best_tokens, 1):
            print(f"  {i:2d}. '{token}'")
        
        print(f"\n💡 Usage suggestion:")
        print(f"Add these tokens to your prompts like:")
        print(f"'Please write in a {', '.join(best_tokens[:3])} style...'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your pickle file path is correct and model matches the directions.")

if __name__ == "__main__":
    main()