"""
LLM Interpretability Testing Framework - Main Runner
Run this script from your terminal to execute the testing framework.

Usage:
    python main.py --help
    python main.py --samples 20 --auto-engineer
    python main.py --samples 10 --manual
    python main.py --test-only
"""

import argparse
import sys
import os
from datetime import datetime

# Import the testing framework (assumes it's in the same directory)
try:
    from llm_tester import LLMInterpreTester, workflow_step1_sample_prompts, workflow_step2_test
except ImportError:
    print("Error: Could not import LLMInterpreTester. Make sure llm_interpretability_tester.py is in the same directory.")
    sys.exit(1)

def print_banner():
    """Print the application banner."""
    print("="*80)
    print("    LLM INTERPRETABILITY TESTING FRAMEWORK")
    print("    Automatic Prompt Engineering with Token Integration")
    print("="*80)
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

def print_status(message, status_type="INFO"):
    """Print status messages with formatting."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if status_type == "ERROR":
        print(f"[{timestamp}] ❌ ERROR: {message}")
    elif status_type == "SUCCESS":
        print(f"[{timestamp}] ✅ SUCCESS: {message}")
    elif status_type == "WARNING":
        print(f"[{timestamp}] ⚠️  WARNING: {message}")
    else:
        print(f"[{timestamp}] ℹ️  INFO: {message}")

def check_dependencies():
    """Check if required dependencies are available."""
    print_status("Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing required packages: {', '.join(missing_packages)}", "ERROR")
        print("Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print_status("All dependencies found!")
    return True

def run_full_workflow(args):
    """Run the complete workflow from sampling to testing."""
    print_status("Starting full workflow...")
    print_status(f"Samples: {args.samples}")
    print_status(f"Auto-engineering: {args.auto_engineer}")
    print_status(f"Multi-strategy: {getattr(args, 'multi_strategy', True)}")
    
    try:
        # Step 1: Sample and engineer prompts
        print_status("Step 1: Sampling control prompts...")
        workflow_step1_sample_prompts(args.samples, auto_engineer=args.auto_engineer, 
                                    multi_strategy=getattr(args, 'multi_strategy', True))
        print_status("Step 1 completed successfully!", "SUCCESS")
        
        # Step 2: Run testing
        print_status("Step 2: Running batch testing...")
        if args.auto_engineer:
            workflow_step2_test(use_auto_engineered=True)
        else:
            # Check if manual prompts are ready
            if not os.path.exists("engineered_prompts_template.txt"):
                print_status("Manual engineered prompts template not found!", "ERROR")
                print("Please fill in 'engineered_prompts_template.txt' and run with --test-only")
                return False
            
            # Check if template is filled
            with open("engineered_prompts_template.txt", 'r') as f:
                content = f.read()
                if "[ENTER YOUR ENGINEERED PROMPT HERE]" in content:
                    print_status("Please fill in the engineered prompts template first!", "WARNING")
                    print("Edit 'engineered_prompts_template.txt' and run with --test-only")
                    return False
            
            workflow_step2_test(use_auto_engineered=False)
        
        print_status("Full workflow completed successfully!", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Error in workflow: {str(e)}", "ERROR")
        return False

def run_sampling_only(args):
    """Run only the sampling step."""
    print_status("Running sampling step only...")
    
    try:
        workflow_step1_sample_prompts(args.samples, auto_engineer=args.auto_engineer,
                                    multi_strategy=getattr(args, 'multi_strategy', True))
        print_status("Sampling completed successfully!", "SUCCESS")
        
        if args.auto_engineer:
            print("Next step: Run 'python main.py --test-only' to execute testing")
        else:
            print("Next steps:")
            print("1. Fill in 'engineered_prompts_template.txt'")
            print("2. Run 'python main.py --test-only' to execute testing")
        
        return True
        
    except Exception as e:
        print_status(f"Error in sampling: {str(e)}", "ERROR")
        return False

def run_testing_only(args):
    """Run only the testing step."""
    print_status("Running testing step only...")
    
    # Check if control prompts exist
    if not os.path.exists("control_prompts.txt"):
        print_status("Control prompts file not found!", "ERROR")
        print("Please run sampling first: python main.py --samples 20")
        return False
    
    # Determine which engineered prompts to use
    use_auto = False
    if os.path.exists("auto_engineered_prompts.txt"):
        use_auto = True
        print_status("Found auto-generated engineered prompts")
    elif os.path.exists("engineered_prompts_template.txt"):
        # Check if template is filled
        with open("engineered_prompts_template.txt", 'r') as f:
            content = f.read()
            if "[ENTER YOUR ENGINEERED PROMPT HERE]" in content:
                print_status("Engineered prompts template not filled!", "ERROR")
                print("Please fill in 'engineered_prompts_template.txt' first")
                return False
        print_status("Found manual engineered prompts")
    else:
        print_status("No engineered prompts found!", "ERROR")
        print("Please run sampling first: python main.py --samples 20")
        return False
    
    try:
        workflow_step2_test(use_auto_engineered=use_auto)
        print_status("Testing completed successfully!", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Error in testing: {str(e)}", "ERROR")
        return False

def run_interactive_mode():
    """Run interactive mode with user prompts."""
    print_status("Starting interactive mode...")
    
    print("\nInteractive Setup:")
    print("1. Full automated workflow (recommended)")
    print("2. Sample prompts only")
    print("3. Test existing prompts")
    print("4. Manual workflow")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            samples = input("Number of samples (default 20): ").strip()
            samples = int(samples) if samples.isdigit() else 20
            
            print_status(f"Running full automated workflow with {samples} samples...")
            
            args_mock = argparse.Namespace(
                samples=samples,
                auto_engineer=True,
                sample_only=False,
                test_only=False
            )
            
            return run_full_workflow(args_mock)
            
        elif choice == "2":
            samples = input("Number of samples (default 20): ").strip()
            samples = int(samples) if samples.isdigit() else 20
            
            auto = input("Auto-engineer prompts? (y/n, default y): ").strip().lower()
            auto_engineer = auto != 'n'
            
            args_mock = argparse.Namespace(
                samples=samples,
                auto_engineer=auto_engineer,
                sample_only=True,
                test_only=False
            )
            
            return run_sampling_only(args_mock)
            
        elif choice == "3":
            args_mock = argparse.Namespace(
                samples=20,
                auto_engineer=True,
                sample_only=False,
                test_only=True
            )
            
            return run_testing_only(args_mock)
            
        elif choice == "4":
            samples = input("Number of samples (default 20): ").strip()
            samples = int(samples) if samples.isdigit() else 20
            
            print_status("Manual workflow selected")
            print("This will create templates for you to fill manually")
            
            args_mock = argparse.Namespace(
                samples=samples,
                auto_engineer=False,
                sample_only=False,
                test_only=False
            )
            
            return run_full_workflow(args_mock)
            
        else:
            print("Invalid choice. Please select 1-4.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Interpretability Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --samples 20 --auto-engineer           # Full automated workflow
  python main.py --samples 20 --multi-strategy          # Advanced multi-strategy approach  
  python main.py --samples 10 --single-strategy         # Faster single-strategy approach
  python main.py --samples 10 --manual                  # Manual prompt engineering
  python main.py --sample-only --samples 15             # Sample prompts only
  python main.py --test-only                            # Test existing prompts
  python main.py --interactive                          # Interactive mode
  python main.py --preview --samples 5                  # Preview mode with 5 samples

Advanced Options:
  python main.py --min-tokens 5 --quality-threshold 7   # Higher quality requirements
  python main.py --engineer-model "meta-llama/Llama-2-13b-chat-hf"  # Different engineering model

Prompt Engineering Strategies:
  - Contextual Integration: Academic/professional framing with natural token embedding
  - Persona-based: Creates believable characters using target tokens as names/roles
  - Technical Obfuscation: Uses academic/technical language to obscure malicious intent  
  - Narrative Embedding: Builds stories/scenarios that naturally incorporate tokens
  - Multi-strategy: Tries all approaches and selects the highest-quality result

High-scoring tokens to be incorporated:
  tertiary, firmly, removeClass, Vaugh, Louise, Defaults, Sap, educated,
  646, .groupby, TValue, besonders, 856, OMIT, readily, INU, Willis,
  cere, headlights, reel, biology, contractual, Robbins, supreme
        """
    )
    
    # Main workflow options
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of control prompts to sample (default: 20)')
    parser.add_argument('--auto-engineer', action='store_true',
                       help='Use automatic prompt engineering with GPT-OSS-20B')
    parser.add_argument('--manual', action='store_true',
                       help='Use manual prompt engineering (creates template)')
    
    # Step control options
    parser.add_argument('--sample-only', action='store_true',
                       help='Only run the sampling step')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run the testing step (requires existing prompts)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    # Engineering quality options
    parser.add_argument('--multi-strategy', action='store_true', default=True,
                       help='Use multiple engineering strategies and select best (default: True)')
    parser.add_argument('--single-strategy', action='store_true',
                       help='Use single engineering strategy (faster but lower quality)')
    
    # Model configuration
    parser.add_argument('--main-model', default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
                       help='Main model for testing (default: Llama-3.3-70B)')
    parser.add_argument('--engineer-model', default="openai/gpt-oss-20b",
                       help='Model for prompt engineering (default: GPT-OSS-20B)')
    
    # Quality control
    parser.add_argument('--min-tokens', type=int, default=3,
                       help='Minimum tokens to integrate per prompt (default: 3)')
    parser.add_argument('--quality-threshold', type=int, default=5,
                       help='Minimum quality score for engineered prompts (default: 5)')
    
    # Utility options
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--preview', action='store_true',
                       help='Preview generated prompts before testing')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    # Check dependencies automatically
    if not check_dependencies():
        sys.exit(1)
    
    # Handle strategy conflicts
    if args.single_strategy:
        args.multi_strategy = False
    
    # Validate arguments
    if args.auto_engineer and args.manual:
        print_status("Cannot use both --auto-engineer and --manual", "ERROR")
        sys.exit(1)
    
    if args.sample_only and args.test_only:
        print_status("Cannot use both --sample-only and --test-only", "ERROR")
        sys.exit(1)
    
    # Set default to auto-engineer if neither specified
    if not args.manual:
        args.auto_engineer = True
    
    # Run interactive mode
    if args.interactive:
        success = run_interactive_mode()
        sys.exit(0 if success else 1)
    
    # Run specific steps
    success = True
    
    if args.sample_only:
        success = run_sampling_only(args)
    elif args.test_only:
        success = run_testing_only(args)
    else:
        success = run_full_workflow(args)
    
    # Print final status
    print("\n" + "="*80)
    if success:
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        if os.path.exists("control_prompts.txt"):
            print("  📄 control_prompts.txt - Original control prompts")
        if os.path.exists("auto_engineered_prompts.txt"):
            print("  🤖 auto_engineered_prompts.txt - Auto-generated engineered prompts")
        if os.path.exists("engineered_prompts_template.txt"):
            print("  📝 engineered_prompts_template.txt - Manual prompt template")
        
        # Find result files
        for file in os.listdir("."):
            if file.startswith("llm_interpretability_results_") and file.endswith(".txt"):
                print(f"  📊 {file} - Test results")
        
    else:
        print("❌ EXPERIMENT FAILED!")
        print("Check the error messages above for details.")
    
    print("="*80)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()