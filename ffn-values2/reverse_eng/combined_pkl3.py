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
from sklearn.cluster import KMeans, AgglomerativeClustering
import re

class GevaSteeringAnalyzer:
    """
    Analyze steering vectors using Geva et al.'s theoretical framework.
    
    Applies the FFN sub-update decomposition methodology to steering vectors,
    treating steering directions as collections of sub-updates that can be
    interpreted in vocabulary space using the embedding matrix.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the analyzer with a specific model"""
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(model_name)
            self.vocab_size = len(self.tokenizer)
            self.hidden_dim = self.model.config.hidden_size
            
            self.steering_directions = None
            self.embedding_matrix = None  # E in Geva et al.
            
            # Caches for expensive computations
            self._sub_updates_cache = {}
            self._concept_cache = {}
            
            self._load_embedding_matrix()
            
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
            self.vocab_size = None
            self.hidden_dim = None
            self.embedding_matrix = None
    
    def _load_embedding_matrix(self):
        """Load the embedding matrix E for projecting to vocabulary space (Geva et al. method)"""
        if self.model is None:
            return
            
        try:
            # Get input embeddings (E matrix in Geva et al.)
            self.embedding_matrix = self.model.get_input_embeddings().weight.detach().cpu().numpy()
            print(f"Loaded embedding matrix E: {self.embedding_matrix.shape}")
            
            # Transpose to match Geva et al. notation: E ∈ R^(|V| × d)
            if self.embedding_matrix.shape[0] == self.hidden_dim:
                self.embedding_matrix = self.embedding_matrix.T
                
            print(f"Embedding matrix shape (|V| × d): {self.embedding_matrix.shape}")
            
        except Exception as e:
            print(f"Could not load embedding matrix: {e}")
            self.embedding_matrix = None
    
    def load_steering_directions(self, pkl_path: str):
        """Load steering directions from pickle file"""
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
    
    def decompose_steering_to_sub_updates(self, layer_idx: int, num_components: int = 50) -> List[Tuple[np.ndarray, float]]:
        """
        Decompose steering vector into sub-updates similar to Geva et al.'s FFN decomposition.
        
        Since we don't have the original FFN structure, we use PCA to decompose the steering
        vector into meaningful components that can be analyzed as "sub-updates".
        
        Args:
            layer_idx: Layer index
            num_components: Number of components to extract
            
        Returns:
            List of (sub_update_vector, coefficient) tuples
        """
        if layer_idx not in self.steering_directions['directions']:
            raise ValueError(f"Layer {layer_idx} not found in steering directions")
        
        cache_key = f"{layer_idx}_{num_components}"
        if cache_key in self._sub_updates_cache:
            return self._sub_updates_cache[cache_key]
        
        steering_vector = self.steering_directions['directions'][layer_idx]
        
        # Method 1: Decompose based on embedding matrix structure
        # Similar to how Geva et al. decompose FFN outputs
        
        # Project steering vector onto embedding space and back to find principal directions
        # E @ E.T gives us the vocabulary-space projection structure
        if self.embedding_matrix is not None:
            vocab_projection = self.embedding_matrix @ steering_vector  # Shape: (vocab_size,)
            
            # Find the most significant directions by taking top-k vocabulary elements
            # and their corresponding embedding vectors
            top_indices = np.argsort(np.abs(vocab_projection))[-num_components:][::-1]
            
            sub_updates = []
            for i, vocab_idx in enumerate(top_indices):
                # Each sub-update is the embedding vector scaled by its coefficient
                embedding_vec = self.embedding_matrix[vocab_idx]  # Shape: (hidden_dim,)
                coefficient = vocab_projection[vocab_idx]
                
                sub_updates.append((embedding_vec, coefficient))
            
        else:
            # Fallback: Simple PCA decomposition if embedding matrix not available
            # Reshape steering vector for PCA (treat as single sample with multiple features)
            steering_reshaped = steering_vector.reshape(1, -1)
            
            # Use PCA to find principal components
            pca = PCA(n_components=min(num_components, steering_vector.shape[0]))
            pca.fit(steering_reshaped.T)  # Transpose to treat dimensions as samples
            
            sub_updates = []
            for i in range(pca.n_components_):
                component = pca.components_[i]
                coefficient = pca.explained_variance_ratio_[i] * np.dot(steering_vector, component)
                sub_updates.append((component, coefficient))
        
        self._sub_updates_cache[cache_key] = sub_updates
        return sub_updates
    
    def project_sub_update_to_vocab(self, sub_update_vector: np.ndarray) -> np.ndarray:
        """
        Project sub-update vector to vocabulary space using embedding matrix.
        Following Geva et al.: r_i = E @ v_i
        
        Args:
            sub_update_vector: Sub-update vector (shape: hidden_dim)
            
        Returns:
            Vocabulary projection scores (shape: vocab_size)
        """
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix not available")
        
        # Geva et al. notation: r_i = E @ v_i
        # E shape: (vocab_size, hidden_dim), v_i shape: (hidden_dim,)
        vocab_scores = self.embedding_matrix @ sub_update_vector
        return vocab_scores
    
    def analyze_sub_update_concepts(self, layer_idx: int, sub_update_idx: int, top_k: int = 30) -> Dict[str, Any]:
        """
        Analyze what concepts a sub-update promotes, following Geva et al.'s methodology.
        
        Args:
            layer_idx: Layer index
            sub_update_idx: Index of sub-update within the layer
            top_k: Number of top tokens to analyze
            
        Returns:
            Dictionary with concept analysis results
        """
        sub_updates = self.decompose_steering_to_sub_updates(layer_idx)
        
        if sub_update_idx >= len(sub_updates):
            raise ValueError(f"Sub-update {sub_update_idx} not found (max: {len(sub_updates)-1})")
        
        sub_update_vector, coefficient = sub_updates[sub_update_idx]
        
        # Project to vocabulary space
        vocab_scores = self.project_sub_update_to_vocab(sub_update_vector)
        
        # Get top-k tokens
        top_indices = np.argsort(vocab_scores)[-top_k:][::-1]
        top_tokens = []
        
        for idx in top_indices:
            if self.tokenizer is not None:
                token = self.tokenizer.decode([idx])
                score = vocab_scores[idx]
                top_tokens.append((token, score, idx))
            else:
                top_tokens.append((f"token_{idx}", vocab_scores[idx], idx))
        
        # Effective score calculation (following Geva et al.'s Equation 2)
        # The effective score for token w is: e_w · (m_i * v_i) = coefficient * (e_w · v_i)
        effective_scores = [(token, coefficient * score, idx) for token, score, idx in top_tokens]
        
        return {
            'layer_idx': layer_idx,
            'sub_update_idx': sub_update_idx,
            'coefficient': coefficient,
            'top_tokens': top_tokens,
            'effective_scores': effective_scores,
            'sub_update_norm': np.linalg.norm(sub_update_vector)
        }
    
    def analyze_promotion_vs_elimination(self, layer_idx: int, reference_tokens: List[str] = None) -> Dict[str, Any]:
        """
        Analyze whether steering vector works via promotion or elimination mechanism.
        Replicates Geva et al.'s Section 5 analysis.
        
        Args:
            layer_idx: Layer to analyze
            reference_tokens: Specific tokens to analyze (if None, uses top promoted tokens)
            
        Returns:
            Analysis results showing promotion vs elimination patterns
        """
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix required for promotion analysis")
        
        steering_vector = self.steering_directions['directions'][layer_idx]
        
        # Get direct vocabulary projection (like Geva et al.'s full FFN output)
        full_vocab_projection = self.embedding_matrix @ steering_vector
        
        # Get sub-updates for detailed analysis
        sub_updates = self.decompose_steering_to_sub_updates(layer_idx, num_components=10)
        
        # If no reference tokens provided, use top promoted tokens
        if reference_tokens is None:
            top_indices = np.argsort(full_vocab_projection)[-20:][::-1]
            reference_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
        
        analysis_results = {
            'layer_idx': layer_idx,
            'reference_tokens': reference_tokens,
            'promotion_scores': [],
            'elimination_scores': [],
            'sub_update_contributions': []
        }
        
        # Analyze each reference token
        for token in reference_tokens:
            try:
                # Get token ID
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if not token_ids:
                    continue
                token_id = token_ids[0]
                
                # Full steering vector score for this token
                full_score = full_vocab_projection[token_id]
                
                # Sub-update contributions
                sub_contributions = []
                total_sub_score = 0
                
                for i, (sub_vec, coeff) in enumerate(sub_updates):
                    sub_vocab_proj = self.project_sub_update_to_vocab(sub_vec)
                    sub_token_score = coeff * sub_vocab_proj[token_id]
                    sub_contributions.append({
                        'sub_update_idx': i,
                        'coefficient': coeff,
                        'token_score': sub_vocab_proj[token_id],
                        'effective_score': sub_token_score
                    })
                    total_sub_score += sub_token_score
                
                token_analysis = {
                    'token': token,
                    'token_id': token_id,
                    'full_score': full_score,
                    'sub_updates_total': total_sub_score,
                    'sub_contributions': sub_contributions
                }
                
                # Classify as promotion or elimination based on score
                if full_score > 0:
                    analysis_results['promotion_scores'].append(token_analysis)
                else:
                    analysis_results['elimination_scores'].append(token_analysis)
                    
            except Exception as e:
                print(f"Error analyzing token '{token}': {e}")
                continue
        
        return analysis_results
    
    def cluster_sub_updates_by_concept(self, layer_idx: int, num_clusters: int = 10) -> Dict[str, Any]:
        """
        Cluster sub-updates by the concepts they promote, following Geva et al.'s clustering approach.
        
        Args:
            layer_idx: Layer to analyze
            num_clusters: Number of clusters to create
            
        Returns:
            Clustering results with concept interpretation
        """
        sub_updates = self.decompose_steering_to_sub_updates(layer_idx, num_components=50)
        
        # Create feature matrix from vocabulary projections
        vocab_projections = []
        coefficients = []
        
        for sub_vec, coeff in sub_updates:
            vocab_proj = self.project_sub_update_to_vocab(sub_vec)
            vocab_projections.append(vocab_proj)
            coefficients.append(coeff)
        
        vocab_projections = np.array(vocab_projections)
        
        # Cluster based on cosine similarity of vocabulary projections
        # Following Geva et al.'s approach
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(vocab_projections)
        
        # Analyze each cluster
        clusters = {}
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Get representative tokens for this cluster
            cluster_vocab_proj = np.mean(vocab_projections[cluster_indices], axis=0)
            top_token_indices = np.argsort(cluster_vocab_proj)[-30:][::-1]
            
            top_tokens = []
            for idx in top_token_indices:
                token = self.tokenizer.decode([idx])
                score = cluster_vocab_proj[idx]
                top_tokens.append((token, score))
            
            clusters[cluster_id] = {
                'sub_update_indices': cluster_indices.tolist(),
                'size': len(cluster_indices),
                'top_tokens': top_tokens,
                'avg_coefficient': np.mean([coefficients[i] for i in cluster_indices]),
                'coefficient_std': np.std([coefficients[i] for i in cluster_indices])
            }
        
        return {
            'layer_idx': layer_idx,
            'num_clusters': num_clusters,
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def generate_geva_style_analysis_report(self, layer_idx: int, output_path: str = None) -> str:
        """
        Generate a comprehensive analysis report following Geva et al.'s methodology.
        
        Args:
            layer_idx: Layer to analyze
            output_path: Path to save the report (optional)
            
        Returns:
            Analysis report as string
        """
        report = []
        report.append("=" * 80)
        report.append(f"GEVA-STYLE STEERING VECTOR ANALYSIS")
        report.append(f"Concept: {self.steering_directions['concept']}")
        report.append(f"Layer: {layer_idx}")
        report.append(f"Model: {self.model_name}")
        report.append("=" * 80)
        
        # 1. Sub-update decomposition analysis
        report.append(f"\n📊 SUB-UPDATE DECOMPOSITION ANALYSIS")
        report.append("-" * 50)
        
        sub_updates = self.decompose_steering_to_sub_updates(layer_idx, num_components=10)
        report.append(f"Decomposed steering vector into {len(sub_updates)} sub-updates")
        
        # Analyze top 5 sub-updates
        for i in range(min(5, len(sub_updates))):
            concept_analysis = self.analyze_sub_update_concepts(layer_idx, i, top_k=10)
            report.append(f"\nSub-update {i} (coefficient: {concept_analysis['coefficient']:.3f}):")
            
            top_tokens = concept_analysis['top_tokens'][:10]
            token_str = ", ".join([f"'{token}' ({score:.2f})" for token, score, _ in top_tokens])
            report.append(f"  Top tokens: {token_str}")
        
        # 2. Promotion vs Elimination Analysis
        report.append(f"\n🎯 PROMOTION vs ELIMINATION ANALYSIS")
        report.append("-" * 50)
        
        promo_analysis = self.analyze_promotion_vs_elimination(layer_idx)
        
        report.append(f"Promoted tokens: {len(promo_analysis['promotion_scores'])}")
        report.append(f"Eliminated tokens: {len(promo_analysis['elimination_scores'])}")
        
        if promo_analysis['promotion_scores']:
            report.append("\nTop promoted tokens:")
            for token_info in promo_analysis['promotion_scores'][:5]:
                report.append(f"  '{token_info['token']}': {token_info['full_score']:.3f}")
        
        if promo_analysis['elimination_scores']:
            report.append("\nTop eliminated tokens:")
            for token_info in promo_analysis['elimination_scores'][:5]:
                report.append(f"  '{token_info['token']}': {token_info['full_score']:.3f}")
        
        # 3. Concept clustering
        report.append(f"\n🎨 CONCEPT CLUSTERING ANALYSIS")
        report.append("-" * 50)
        
        cluster_analysis = self.cluster_sub_updates_by_concept(layer_idx, num_clusters=5)
        
        for cluster_id, cluster_info in cluster_analysis['clusters'].items():
            report.append(f"\nCluster {cluster_id} ({cluster_info['size']} sub-updates):")
            top_tokens = cluster_info['top_tokens'][:10]
            token_str = ", ".join([f"'{token}'" for token, _ in top_tokens])
            report.append(f"  Representative tokens: {token_str}")
            report.append(f"  Avg coefficient: {cluster_info['avg_coefficient']:.3f}")
        
        # 4. Summary statistics
        report.append(f"\n📈 SUMMARY STATISTICS")
        report.append("-" * 50)
        
        steering_vector = self.steering_directions['directions'][layer_idx]
        full_vocab_projection = self.embedding_matrix @ steering_vector
        
        report.append(f"Steering vector norm: {np.linalg.norm(steering_vector):.3f}")
        report.append(f"Vocabulary projection range: [{full_vocab_projection.min():.3f}, {full_vocab_projection.max():.3f}]")
        report.append(f"Positive projections: {np.sum(full_vocab_projection > 0)} / {len(full_vocab_projection)}")
        report.append(f"Mean projection: {full_vocab_projection.mean():.3f}")
        report.append(f"Std projection: {full_vocab_projection.std():.3f}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text
    
    def compare_layers_geva_style(self, layers: List[int] = None) -> Dict[str, Any]:
        """
        Compare steering behavior across layers using Geva et al.'s analysis framework.
        
        Args:
            layers: List of layers to compare (if None, uses all available)
            
        Returns:
            Comparative analysis results
        """
        if layers is None:
            layers = self.steering_directions['hidden_layers']
        
        comparison = {
            'layers': layers,
            'layer_analyses': {},
            'cross_layer_patterns': {}
        }
        
        # Analyze each layer
        for layer_idx in layers:
            print(f"Analyzing layer {layer_idx}...")
            
            # Get basic statistics
            steering_vector = self.steering_directions['directions'][layer_idx]
            vocab_projection = self.embedding_matrix @ steering_vector
            
            # Promotion/elimination analysis
            promo_analysis = self.analyze_promotion_vs_elimination(layer_idx)
            
            # Clustering analysis
            cluster_analysis = self.cluster_sub_updates_by_concept(layer_idx, num_clusters=5)
            
            comparison['layer_analyses'][layer_idx] = {
                'vector_norm': np.linalg.norm(steering_vector),
                'vocab_projection_stats': {
                    'mean': float(vocab_projection.mean()),
                    'std': float(vocab_projection.std()),
                    'min': float(vocab_projection.min()),
                    'max': float(vocab_projection.max())
                },
                'promotion_count': len(promo_analysis['promotion_scores']),
                'elimination_count': len(promo_analysis['elimination_scores']),
                'num_clusters': len(cluster_analysis['clusters'])
            }
        
        return comparison

# Example usage function
def main():
    """Example usage demonstrating Geva et al.'s methodology applied to steering vectors"""
    print("🚀 Initializing Geva-Style Steering Vector Analyzer...")
    
    # Initialize analyzer
    analyzer = GevaSteeringAnalyzer("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    try:
        # Load steering directions
        # analyzer.load_steering_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/logistic_prose_llama_3_8b_it.pkl")
        analyzer.load_steering_directions("/home/ubuntu/krishiv-llm/neural_controllers/directions/rfm_harmful_llama_3_8b_it.pkl")
        
        
        
        # Pick a layer to analyze
        layer_to_analyze = analyzer.steering_directions['hidden_layers'][0]
        print(f"\n🔍 Analyzing layer {layer_to_analyze}...")
        
        # Generate comprehensive Geva-style analysis
        report = analyzer.generate_geva_style_analysis_report(layer_to_analyze, "geva_style_analysis_harmful1.txt")
        print(report)
        
        # Detailed sub-update analysis
        print(f"\n🧩 DETAILED SUB-UPDATE ANALYSIS")
        print("=" * 60)
        
        for i in range(3):  # Analyze top 3 sub-updates
            concept_analysis = analyzer.analyze_sub_update_concepts(layer_to_analyze, i)
            print(f"\nSub-update {i}:")
            print(f"  Coefficient: {concept_analysis['coefficient']:.3f}")
            print(f"  Vector norm: {concept_analysis['sub_update_norm']:.3f}")
            print(f"  Top promoted tokens:")
            
            for token, score, _ in concept_analysis['top_tokens'][:10]:
                print(f"    '{token}': {score:.3f}")
        
        # Cross-layer comparison
        print(f"\n🔄 CROSS-LAYER COMPARISON")
        print("=" * 60)
        
        comparison = analyzer.compare_layers_geva_style(analyzer.steering_directions['hidden_layers'][:3])
        
        for layer_idx, analysis in comparison['layer_analyses'].items():
            print(f"\nLayer {layer_idx}:")
            print(f"  Vector norm: {analysis['vector_norm']:.3f}")
            print(f"  Vocab projection mean: {analysis['vocab_projection_stats']['mean']:.3f}")
            print(f"  Promoted tokens: {analysis['promotion_count']}")
            print(f"  Eliminated tokens: {analysis['elimination_count']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your pickle file path is correct and model matches the directions.")

if __name__ == "__main__":
    main()