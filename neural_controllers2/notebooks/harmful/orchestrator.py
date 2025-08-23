import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local path for neural_controllers
sys.path.append(str(Path(__file__).resolve().parents[1]))
from toxic_refusal_tester import ToxicPromptCurator, RefusalTester, ToxicPromptConfig

class RefusalTestOrchestrator:
    """Manages the full pipeline from dataset curation to multi-model testing."""

    def __init__(self, models_to_test: List[Dict], curation_config: Dict):
        self.models_to_test = models_to_test
        self.curation_config = curation_config
        self.curated_prompts_file = "top_100_toxic_prompts_for_refusal_testing.json"
    
    def run_pipeline(self):
        """Executes the end-to-end testing pipeline."""
        
        # Step 1: Curate and save the dataset once
        print("🔍 STEP 1: CURATING THE TOXIC PROMPTS DATASET")
        print("=" * 50)
        curator_config = ToxicPromptConfig(**self.curation_config)
        curator = ToxicPromptCurator(curator_config)
        toxic_prompts = curator.curate_dataset()
        curator.save_dataset(toxic_prompts, self.curated_prompts_file)
        print("\n✅ Dataset curated and saved for all models to use.")

        # Step 2: Test each model sequentially
        print("\n🧪 STEP 2: TESTING REFUSAL RATES FOR EACH MODEL")
        print("=" * 50)
        all_results_data = []

        for model_config in self.models_to_test:
            model_name = model_config["model_name"]
            model_type = model_config["model_type"]
            
            print(f"\n--- Testing model: {model_name} ---")
            
            # Update the config for the current model
            test_config = ToxicPromptConfig(
                **self.curation_config,
                model_type=model_type
            )
            
            # Initialize tester and run the test
            try:
                tester = RefusalTester(test_config, curated_prompts_file=self.curated_prompts_file)
                results_df = tester.test_refusal_rates()
                
                # Add model info to the results
                results_df['model_name'] = model_name
                all_results_data.append(results_df)

                # Step 3: Analyze and visualize results for the current model
                print("\n📈 Analyzing results for this model...")
                tester.print_analysis(results_df)
                tester.create_visualizations(results_df, save_path=f"refusal_analysis_{model_name}")
                
            except Exception as e:
                logger.error(f"Failed to test model {model_name}: {e}")
                continue

        # Step 4: Aggregate and save all results
        if all_results_data:
            print("\n📊 Aggregating and saving all results...")
            final_results_df = pd.concat(all_results_data, ignore_index=True)
            final_results_df.to_csv("combined_refusal_test_results.csv", index=False)
            
            # Perform a final comparison analysis
            self.compare_models(final_results_df)
            print("\n🎉 ANALYSIS COMPLETE!")
            print("Combined results saved to combined_refusal_test_results.csv")
        else:
            print("\n❌ No models were tested successfully.")

    def compare_models(self, results_df: pd.DataFrame):
        """Generates a summary comparison of all tested models."""
        print("\n" + "="*70)
        print("MULTI-MODEL REFUSAL RATE COMPARISON")
        print("="*70)
        
        comparison_metrics = {}
        for model_name, group_df in results_df.groupby('model_name'):
            metrics = {
                "total_prompts": len(group_df),
                "baseline_refusal_rate": group_df["baseline_is_refusal"].mean(),
                "steered_refusal_rate": group_df["steered_is_refusal"].mean() if "steered_is_refusal" in group_df.columns and group_df["steered_response"].iloc[0] else None
            }
            comparison_metrics[model_name] = metrics
        
        comparison_df = pd.DataFrame(comparison_metrics).T
        print(comparison_df.to_markdown())

        # Visualization
        plt.figure(figsize=(10, 6))
        comparison_df.plot(kind='bar', y=['baseline_refusal_rate', 'steered_refusal_rate'], rot=45)
        plt.title("Refusal Rates: Baseline vs. Steered (by Model)")
        plt.ylabel("Refusal Rate")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.legend(["Baseline Refusal Rate", "Steered Refusal Rate"])
        plt.tight_layout()
        plt.savefig("multi_model_refusal_comparison.png")
        print("Comparison plot saved to multi_model_refusal_comparison.png")
        plt.show()

def main():
    """Main execution entry point."""
    
    # Define the models to be tested
    models_to_test = [
        {"model_name": "llama_3.1_8b_instruct", "model_type": "llama"},
        {"model_name": "qwen3_0.6b_chat", "model_type": "qwen_small"},
        {"model_name": "qwen3_32b_chat", "model_type": "qwen_large"},
        {"model_name": "gpt_oss_20b", "model_type": "gpt_oss"}
    ]
    
    # Configuration for the dataset curation
    curation_config = {
        "top_n_prompts": 100,
        "min_toxicity_score": 0.7,
        "min_length": 10,
        "max_length": 500,
        "exclude_patterns": [
            r"test\s+prompt",
            r"example\s+of",
            r"this\s+is\s+just",
        ]
    }
    
    orchestrator = RefusalTestOrchestrator(models_to_test, curation_config)
    orchestrator.run_pipeline()

if __name__ == "__main__":
    main()